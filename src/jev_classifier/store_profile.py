"""Store profile: the routing taxonomy as data.

Structure mirrors how dealerships actually organise:

  Fixed Operations    service · parts · body_shop      (predictable revenue, ~half of gross profit)
  Variable Operations sales · finance                  (swings with the market)
  non_customer        vendor / jobseeker / wrong number

**Tires and detailing are Service sub-queues, not peer departments** — the industry treats them as
Fixed-Ops adjuncts ("detail shops or tire centers"). Roadside assistance is a sub-queue that sets a
dispatch *flag*; it is not a place the call goes. `general` was never a department at all: it was
absorbing two different things (not-a-dealership-matter, and dealership questions nobody owns),
which is why it produced so many ambiguous labels.

Because "does this store have a separate tire centre?" is a fact about the store, the whole
taxonomy is a JSON file a dealer can edit. A store with no body shop deletes that destination; a
store with its own tire centre moves `tires` from `subqueues` to `destinations`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "config" / "store_profile.json"


DEFAULT_QUESTIONS: Dict[str, str] = {
    "destination": (
        "Which part of the dealership should handle this caller? Pick where the work belongs, "
        "not the first thing the caller mentioned."
    ),
    "subqueue": (
        "This is a {label} call. What exactly does the caller want, and which sub-queue should "
        "it go to? Pick the single closest option."
    ),
}


DEFAULT_NOUL: Dict[str, Dict[str, str]] = {
    "is_safe_to_drive": {
        "instructions": "Is it unsafe for this caller to drive the vehicle?",
        "true": "the vehicle should not be driven: a hazard, damage, or it cannot be moved",
        "false": "the vehicle is drivable and there is no hazard",
    },
}


@dataclass(frozen=True)
class Destination:
    key: str
    label: str
    description: str
    queue: str
    handler: str = "route"  # "route" | "human"


@dataclass(frozen=True)
class SubQueue:
    key: str
    parent: str
    label: str
    description: str
    queue: str
    flags: Tuple[str, ...] = ()
    handler: str = "route"


@dataclass(frozen=True)
class StoreProfile:
    name: str
    destinations: Tuple[Destination, ...]
    subqueues: Tuple[SubQueue, ...]
    locations: Tuple[Dict[str, str], ...] = ()
    policy: Dict[str, Any] = field(default_factory=dict)
    questions: Dict[str, str] = field(default_factory=dict)
    facts: Dict[str, Any] = field(default_factory=dict)
    schedule: Dict[str, Any] = field(default_factory=dict)
    noul: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def noul_question(self, key: str) -> Dict[str, Any]:
        """A yes/no question, with its two option texts, from one place.

        The model is choosing between two written options, not answering a bare prompt, so the
        option text is doing most of the work of defining what the question means.

        Note the key: the runtime API takes `criteria` (and `type`/`instructions`), while the
        training builder takes `crit` (and `t`/`ins`). `build_items.py` does that translation.
        """
        spec = self.noul.get(key) or DEFAULT_NOUL.get(key) or {}
        return {
            "type": "noul",
            "instructions": spec.get("instructions", key),
            "criteria": {"false": spec.get("false", ""), "true": spec.get("true", "")},
        }

    def question_text(self, key: str, **fmt: Any) -> str:
        """The instruction for a question, from one place.

        Training and inference must send the model the *same* wording: the fine-tune learns to
        answer this exact instruction, so a drifted copy means asking a question the model was
        never trained on. Both sides read this.
        """
        template = self.questions.get(key) or DEFAULT_QUESTIONS[key]
        return template.format(**fmt) if fmt else template

    # ------------------------------------------------------------------ lookups
    @property
    def destination_keys(self) -> Tuple[str, ...]:
        return tuple(d.key for d in self.destinations)

    def destination(self, key: Optional[str]) -> Optional[Destination]:
        return next((d for d in self.destinations if d.key == key), None)

    def subqueues_for(self, destination: Optional[str]) -> Tuple[SubQueue, ...]:
        return tuple(s for s in self.subqueues if s.parent == destination)

    def subqueue(self, destination: Optional[str], key: Optional[str]) -> Optional[SubQueue]:
        return next(
            (s for s in self.subqueues if s.parent == destination and s.key == key), None
        )

    def subqueue_keys(self, destination: Optional[str]) -> Tuple[str, ...]:
        return tuple(s.key for s in self.subqueues_for(destination))

    def queue_for(self, destination: Optional[str], subqueue: Optional[str] = None) -> str:
        sub = self.subqueue(destination, subqueue)
        if sub is not None:
            return sub.queue
        dest = self.destination(destination)
        return dest.queue if dest else "Front Desk"

    def flags_for(self, destination: Optional[str], subqueue: Optional[str] = None) -> Tuple[str, ...]:
        sub = self.subqueue(destination, subqueue)
        return sub.flags if sub else ()

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        if not self.destinations:
            raise ValueError("store profile has no destinations")
        dest_keys = [d.key for d in self.destinations]
        if len(set(dest_keys)) != len(dest_keys):
            raise ValueError(f"duplicate destination keys: {dest_keys}")
        sub_keys = [(s.parent, s.key) for s in self.subqueues]
        if len(set(sub_keys)) != len(sub_keys):
            raise ValueError(f"duplicate sub-queue keys: {sub_keys}")
        for sub in self.subqueues:
            if sub.parent not in set(dest_keys):
                raise ValueError(
                    f"sub-queue {sub.key!r} has unknown parent {sub.parent!r}; "
                    f"destinations are {sorted(dest_keys)}"
                )
        for dest in self.destinations:
            if not dest.queue:
                raise ValueError(f"destination {dest.key!r} has no queue")

    # ------------------------------------------------------------------ questions
    def destination_question(self) -> Dict[str, Any]:
        return {
            "destination": {
                "type": "choice",
                "instructions": self.question_text("destination"),
                "criteria": {d.key: d.description for d in self.destinations},
            }
        }

    def subqueue_question(self, destination: str) -> Optional[Dict[str, Any]]:
        subs = self.subqueues_for(destination)
        if not subs:
            return None
        dest = self.destination(destination)
        return {
            "subqueue": {
                "type": "choice",
                "instructions": self.question_text(
                    "subqueue", label=dest.label if dest else destination
                ),
                "criteria": {s.key: s.description for s in subs},
            }
        }


@lru_cache(maxsize=8)
def _load(path_str: str) -> StoreProfile:
    raw = json.loads(Path(path_str).read_text())
    profile = StoreProfile(
        name=raw.get("name", "store"),
        destinations=tuple(
            Destination(
                key=d["key"],
                label=d.get("label", d["key"]),
                description=d.get("description", ""),
                queue=d["queue"],
                handler=d.get("handler", "route"),
            )
            for d in raw["destinations"]
        ),
        subqueues=tuple(
            SubQueue(
                key=s["key"],
                parent=s["parent"],
                label=s.get("label", s["key"]),
                description=s.get("description", ""),
                queue=s["queue"],
                flags=tuple(s.get("flags", ())),
                handler=s.get("handler", "route"),
            )
            for s in raw.get("subqueues", [])
        ),
        locations=tuple(raw.get("locations", ())),
        policy=raw.get("policy", {}),
        questions=dict(raw.get("questions", {})),
        facts=dict(raw.get("facts", {})),
        schedule=dict(raw.get("schedule", {})),
        noul=dict(raw.get("noul", {})),
    )
    profile.validate()
    return profile


def load(path: Optional[str] = None) -> StoreProfile:
    """Load a store profile. Defaults to `config/store_profile.json`, overridable via
    the JEV_STORE_PROFILE environment variable."""
    resolved = path or os.environ.get("JEV_STORE_PROFILE") or str(DEFAULT_PATH)
    return _load(str(resolved))


def clear_cache() -> None:
    _load.cache_clear()
