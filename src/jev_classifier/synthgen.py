"""Synthetic utterance generation and near-duplicate detection.

Two jobs:

1. Ask the teacher for realistic caller utterances for a given (department, intent) pair, varied
   across explicit *style axes*. Without those axes you get four thousand paraphrases of fifty
   sentences, and a model trained on that learns the phrasing rather than the task.
2. Keep the corpus honest — normalise for exact dedup, and Jaccard-match on word shingles for
   near-duplicates, both within the corpus and against the held-out hand-labelled set.

Everything the generator emits is a *candidate*. It only becomes a training example after two
independent labelling passes agree (see `scripts/generate_training.py`).
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any, Dict, Iterable, List, Set

from . import dealership as D
from . import llm

# Diversity axes. Each generation call is given one, and they rotate, so the corpus covers
# register and shape rather than repeating one voice.
STYLES = [
    "short and clipped, the way someone talks when they are in a hurry",
    "chatty and polite, with a greeting and some small talk before the point",
    "non-native English speaker, slightly imperfect grammar but clear meaning",
    "an older caller who describes the problem at length and vaguely",
    "someone reading from a written note, formal and precise",
    "frustrated and blunt, bordering on rude but not abusive",
    "confused and unsure, asking whether this is the right place",
    "naming specific make, model and year",
    "using informal or regional words for car parts",
    "giving the problem but burying it after an unrelated detail",
    "one long run-on sentence with no punctuation",
    "answering as if mid-conversation, with no greeting",
]

GENERATION_SYSTEM = (
    "You write realistic transcripts of what a customer says when they phone a car dealership.\n\n"
    "You will be told the department the call should go to and what the caller wants. Write "
    "things a real person would actually say out loud. Do not name the department. Do not use the "
    "words the dealership would use internally unless a customer would. Vary sentence length and "
    "opening. Never include a caller name, phone number or address.\n\n"
    "Return JSON of the form {\"utterances\": [\"...\", \"...\"]}."
)

_WORD = re.compile(r"[a-z0-9']+")
_WS = re.compile(r"\s+")


def generation_prompt(
    department: str, intent: str, n: int, style: str, vague: bool = False, mode: str = "normal"
) -> str:
    if mode == "offtopic":
        # The synthetic set is all "plausible calls about a car or the dealership", so it contains
        # no negatives. Measured consequence: the fine-tuned model routes the neighbour's-dog call
        # to Service Department, losing the base model's (weak but present) habit of answering
        # `general` for things that are simply not the dealership's business.
        return (
            f"Write {n} different things a caller might say to a car dealership's phone line that "
            "are NOT about a car, a repair, a purchase, or anything the dealership sells or does.\n\n"
            "They must still sound like real people phoning a business by mistake or for an "
            "unrelated reason — wrong number, a complaint about something else entirely, "
            "a personal matter, a completely different kind of company. Do not mention cars, "
            "repairs, dealerships or vehicle problems. "
            f"Write them {style}. Return JSON only."
        )
    if vague:
        # `other` is a residual class: it is what the branch falls back to when nothing specific
        # fits. Asking for "an example of other" produces utterances that clearly belong to a
        # specific intent, and the labelling pass rightly relabels them — measured, that is why
        # every undersized intent in the first run was an `other`. To generate real positives you
        # ask for the *shape* that lands there: too vague, too mixed, or off-topic to place.
        return (
            f"Department: {department} — {D.DEPARTMENTS.get(department, '')}\n"
            f"Category: {intent} — {D.INTENTS.get(department, {}).get(intent, '')}\n\n"
            f"Write {n} different things a caller might say that a {department.replace('_', ' ')} "
            "switchboard could NOT confidently place in a specific category. Make them genuinely "
            "unclear, too vague, mixed across several problems, or only tangentially related. "
            "They must still be plausible phone calls about a car or the dealership. "
            f"Write them {style}. Return JSON only."
        )
    return (
        f"Department: {department} — {D.DEPARTMENTS.get(department, '')}\n"
        f"What the caller wants: {intent} — {D.INTENTS.get(department, {}).get(intent, '')}\n\n"
        f"Write {n} different things a caller might say that belong in this category. "
        f"Write them {style}. Each must be self-contained (the caller's first utterance). "
        "Return JSON only."
    )


def generate(
    department: str,
    intent: str,
    n: int,
    style: str,
    *,
    provider: str = "deepseek",
    model: str | None = None,
    thinking: bool = True,
    mode: str = "auto",
) -> tuple[List[str], Dict[str, Any]]:
    """One generation call. Returns (utterances, call metadata).

    `mode="auto"` picks: off-topic calls for the general department's residual class, the vaguer
    prompt for other residual classes, and the normal prompt otherwise.
    """
    if mode == "auto":
        if department == "general" and intent == "other":
            mode = "offtopic"
        elif intent == "other":
            mode = "vague"
        else:
            mode = "normal"
    vague = mode == "vague"
    call = llm.chat_json(
        GENERATION_SYSTEM,
        generation_prompt(department, intent, n, style, vague=vague, mode=mode),
        {
            "type": "object",
            "properties": {"utterances": {"type": "array", "items": {"type": "string"}}},
            "required": ["utterances"],
            "additionalProperties": False,
        },
        provider=provider,
        model=model,
        thinking=thinking,
    )
    data = call["data"] if isinstance(call.get("data"), dict) else {}
    raw = data.get("utterances") or []
    out = [normalise_ws(u) for u in raw if isinstance(u, str) and u.strip()]
    return out, call


def normalise_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def fingerprint(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for exact dedup."""
    return " ".join(_WORD.findall(text.lower()))


def shingles(text: str, k: int = 4) -> Set[str]:
    words = _WORD.findall(text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class Deduper:
    """Rejects exact repeats, near-duplicates within the corpus, and anything close to the
    held-out hand-labelled set (which must never leak into training).

    Thread-safe: the generator runs many pairs concurrently against one deduper, and an
    unguarded `seen` dict raised "dictionary changed size during iteration".
    """

    def __init__(self, protected: Iterable[str] = (), threshold: float = 0.6):
        self.threshold = threshold
        self.seen: dict[str, Set[str]] = {}
        self.protected = [(t, shingles(t)) for t in protected]
        self._lock = threading.Lock()

    def why_reject(self, text: str) -> str | None:
        fp = fingerprint(text)
        if not fp:
            return "empty"
        sh = shingles(text)
        with self._lock:
            if fp in self.seen:
                return "exact-duplicate"
            for other, osh in self.protected:
                if jaccard(sh, osh) >= self.threshold:
                    return f"near-duplicate-of-heldout:{other[:40]}"
            for other, osh in self.seen.items():
                if jaccard(sh, osh) >= self.threshold:
                    return f"near-duplicate:{other[:40]}"
            self.seen[fp] = sh
        return None
