"""External generality suites: can the model still answer questions it was never trained on?

The whole project's evaluation so far is synthetic-on-synthetic. This is the external check.

Laya's defining property is that the *option space is defined at request time*, so a new schema
needs no retraining. We then fine-tuned the encoder hard on 43 fixed intents for 4 epochs. If that
destroyed the property, then "a store adds a Fleet queue and it just works" is false, and per-store
configuration becomes per-store retraining — a different product.

Two suites, both real human text, both fetched as plain CSV/JSON so this needs no new dependency:

  Banking77   13,083 real customer-service queries, 77 intents. Laya publishes 0.425 on this,
              so there is a reference number to check our harness against.
  CLINC150    150 intents plus a real out-of-domain split (4,500 in-domain / 1,000 OOD).

Neither is dealership data. They measure *transfer and generality*, not domain correctness.
"""

from __future__ import annotations

import csv
import io
import json
import statistics
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "external"

BANKING77_TRAIN = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
BANKING77_TEST = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
BANKING77_LABELS = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/categories.json"
CLINC150_FULL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"

OOD_LABEL = "out_of_domain"
OOD_DESCRIPTION = "the request is not about this domain at all"


def _fetch(url: str, name: str, binary: bool = False):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.is_file():
        print(f"  downloading {name} ...", flush=True)
        with urllib.request.urlopen(url, timeout=120) as resp:
            path.write_bytes(resp.read())
    return path.read_bytes() if binary else path.read_text()


def _humanise(label: str) -> str:
    return label.replace("_", " ")


def load_banking77(limit: int = 0) -> Tuple[List[str], List[str], List[str]]:
    """Real customer-service queries. Returns (texts, gold labels, option labels).

    The CSVs carry the intent as a string in a `category` column, so the option list is taken from
    `categories.json` (all 77) rather than from whatever a sample happened to contain.
    """
    all_labels = json.loads(_fetch(BANKING77_LABELS, "banking77_categories.json"))
    raw = _fetch(BANKING77_TEST, "banking77_test.csv")
    rows = list(csv.DictReader(io.StringIO(raw)))
    if limit:
        # Even sample: take every Nth so all 77 intents appear.
        step = max(1, len(rows) // limit)
        rows = rows[::step][:limit]
    texts = [r["text"] for r in rows]
    gold = [r["category"] for r in rows]
    options = sorted(set(all_labels) | set(gold))
    return texts, gold, options


def load_clinc150(limit: int = 0) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """Returns (in_texts, in_gold, ood_texts, ood_gold, option_labels)."""
    data = json.loads(_fetch(CLINC150_FULL, "clinc150_full.json"))
    intents = sorted({label for _, label in data["train"] if label != "oos"})
    options = intents + [OOD_LABEL]

    def rows(key: str) -> List[Tuple[str, str]]:
        items = data[key]
        if limit:
            step = max(1, len(items) // limit)
            items = items[::step][:limit]
        return [(t, l) for t, l in items]

    in_rows = rows("test")
    ood_rows = rows("oos_test")
    return (
        [t for t, _ in in_rows],
        [l for _, l in in_rows],
        [t for t, _ in ood_rows],
        [l for _, l in ood_rows],
        options,
    )


# --------------------------------------------------------------------------- running
def ask_choice(agent, texts: Sequence[str], options: Sequence[str], *, batch_report: int = 0) -> List[str]:
    """One `choice` question per text, with the option space supplied at request time."""
    criteria = {o: _humanise(o) for o in options}
    question = {
        "intent": {
            "type": "choice",
            "instructions": "Which of these best describes what the person wants?",
            "criteria": criteria,
        }
    }
    preds: List[str] = []
    for i, text in enumerate(texts, 1):
        try:
            res = agent.predict({"text": text}, question)
            preds.append(res["answers"]["intent"]["choice"])
        except Exception:  # noqa: BLE001 - record as a miss rather than aborting the suite
            preds.append("<error>")
        if batch_report and i % batch_report == 0:
            print(f"    {i}/{len(texts)} ...", flush=True)
    return preds


def ask_noul(agent, texts: Sequence[str], instruction: str) -> List[float]:
    question = {"q": {"type": "noul", "instructions": instruction}}
    out: List[float] = []
    for text in texts:
        try:
            res = agent.predict({"text": text}, question)
            out.append(float(res["answers"]["q"]["noul"]))
        except Exception:  # noqa: BLE001
            out.append(0.5)
    return out


def accuracy(preds: Sequence[str], gold: Sequence[str]) -> float:
    if not gold:
        return 0.0
    return round(sum(p == g for p, g in zip(preds, gold)) / len(gold), 4)


def top_confusions(preds: Sequence[str], gold: Sequence[str], n: int = 8) -> List[tuple]:
    counts: Dict[tuple, int] = {}
    for p, g in zip(preds, gold):
        if p != g:
            counts[(g, p)] = counts.get((g, p), 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:n]


def configure(agent, *, head_max_len: int = 0, max_len: int = 0) -> None:
    """Override the option/state token budget in place.

    The fine-tuned checkpoint ships max_len=1024/head_max_len=256 while the base ships 512/192, so
    a head-to-head comparison has to pin both to the same budget or the token allocation differs.
    """
    if head_max_len:
        agent.cfg["head_max_len"] = head_max_len
    if max_len:
        agent.cfg["max_len"] = max_len
