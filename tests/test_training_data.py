"""The training data contract, checked before it is uploaded rather than on the GPU.

`build_items.py` runs only on Kaggle, so a schema mismatch between what the generators write and
what it reads would surface as a failed GPU run - the most expensive place to find out. These
tests pin the shape of every file that gets uploaded.

They skip when a file is absent rather than failing, because the datasets are generated artefacts
and a fresh clone has none of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_classifier import dealership as D, labels, store_profile as SP

ROOT = Path(__file__).resolve().parents[1]
CALLS = ROOT / "data" / "calls"


def rows(name: str) -> list:
    path = CALLS / name
    if not path.is_file():
        pytest.skip(f"{name} not generated yet")
    data = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert data, f"{name} exists but is empty"
    return data


# --------------------------------------------------------------------------- routing labels
def test_synthetic_rows_validate_against_the_profile():
    for row in rows("synthetic.jsonl"):
        assert row["text"].strip()
        assert labels.validate_label(row["destination"], row.get("subqueue"))[2], row


# --------------------------------------------------------------------------- yes/no labels
def test_severity_rows_carry_at_least_one_boolean_label():
    """The noul training data. A row with no label teaches nothing and wastes a forward pass."""
    for row in rows("severity_train.jsonl"):
        assert row["text"].strip()
        present = [k for k in ("is_safe_to_drive", "needs_human") if k in row]
        assert present, f"row has no label: {row['text'][:60]}"
        for key in present:
            assert isinstance(row[key], bool), f"{key} must be a bool, got {row[key]!r}"


def test_severity_rows_are_not_dominated_by_one_class():
    """A prior nothing like a real switchboard cannot be undone by a threshold.

    The generators deliberately over-represent hazards and complaints, so the cap is what keeps the
    rate learnable without distorting the base rate beyond what the operating point can absorb.
    """
    data = rows("severity_train.jsonl")
    for key in ("is_safe_to_drive", "needs_human"):
        labelled = [r for r in data if key in r]
        if len(labelled) < 50:
            continue
        rate = sum(1 for r in labelled if r[key]) / len(labelled)
        assert 0.2 <= rate <= 0.55, f"{key} is {rate:.0%} positive"


def test_every_noul_question_has_both_option_texts():
    """The model chooses between the two texts, so they carry the definition.

    `build_items.py` reads the same block, and the runtime normalises `criteria` to `crit` - an
    empty option text would silently train on a question with no meaning.
    """
    for key in ("is_safe_to_drive", "needs_human"):
        q = SP.load().noul_question(key)
        assert q["instructions"].strip()
        assert q["criteria"]["true"].strip()
        assert q["criteria"]["false"].strip()


# --------------------------------------------------------------------------- acceptance
def test_acceptance_rows_are_self_consistent():
    for row in rows("acceptance_train.jsonl"):
        assert row["text"].strip()
        assert row["choice"] in row["options"], row["choice"]
        assert "none_of_these" in row["options"]
        assert "unclear" in row["options"]
        assert row["options"][row["choice"]].strip()


def test_acceptance_covers_every_slot_position():
    """Slot 1/2/3 differ only by the time text, so an uneven split teaches a positional bias."""
    data = rows("acceptance_train.jsonl")
    counts = {c: sum(1 for r in data if r["choice"] == c) for c in
              ("slot_1", "slot_2", "slot_3", "none_of_these", "unclear")}
    assert all(n > 0 for n in counts.values()), counts
    slots = [counts["slot_1"], counts["slot_2"], counts["slot_3"]]
    assert max(slots) <= min(slots) * 1.5, f"slot classes unbalanced: {counts}"


def test_acceptance_state_looks_like_the_transcript_the_cascade_sends():
    """The state has to match inference, or the model is trained on a different task.

    The cascade sends "Caller: ...\\nAgent: ..." ending with the offer, so the training state must
    contain the offered times for a reply to be matchable against them.
    """
    data = rows("acceptance_train.jsonl")
    for row in data[:40]:
        assert "Caller:" in row["text"] and "Agent:" in row["text"]
        assert "I can get you in at" in row["text"]


def test_acceptance_instructions_come_from_the_profile():
    """Same one-source rule as every other question."""
    q = D.SLOT_QUESTIONS  # forces the profile to load
    assert SP.load().question_text("acceptance").strip()
    assert "{label}" not in SP.load().question_text("acceptance")


# --------------------------------------------------------------------------- generated artefacts
def _wanted(source: str) -> set:
    import re

    m = re.search(r"WANTED = \(([^)]*)\)", source)
    return set(re.findall(r'"([^"]+)"', m.group(1))) if m else set()


def test_the_notebook_recipe_matches_the_committed_config():
    """The epoch count lived in `JEV_EPOCHS` for one run, and regenerating the notebook without it
    set silently reset the run to 4 epochs - a whole GPU run four epochs short, with nothing to
    show for it. It now comes from `training/run_config.json`, and this is the guard.
    """
    recipe = json.loads((ROOT / "training" / "run_config.json").read_text())
    notebook = json.loads((ROOT / "notebooks" / "laya_finetune_dealership_kaggle.ipynb").read_text())
    src = "\n".join("".join(c.get("source", [])) for c in notebook["cells"])
    assert f"EPOCHS = {recipe['epochs']}" in src, (
        "the notebook's epoch count disagrees with training/run_config.json - "
        "regenerate with: uv run python training/make_notebook.py"
    )


def test_the_notebook_copies_every_file_the_dataset_ships():
    """The notebook is a generated artefact, and changing its generator without regenerating it
    cost a GPU run.

    The pushed notebook copied four dataset files into `/kaggle/working`, so `build_items.py` found
    no severity or acceptance data and silently trained choice questions only - 2,512 items instead
    of ~4,600, with no error anywhere. This is the same one-fact-in-two-places failure this project
    keeps meeting, so it gets a test rather than a note to be careful.
    """
    generator = (ROOT / "training" / "make_notebook.py").read_text()
    packaging = (ROOT / "training" / "make_kaggle_dataset.py").read_text()

    shipped = set(__import__("re").findall(r'\("data/[^"]+",\s*"([^"]+)"\)', packaging))
    assert shipped, "could not read the dataset manifest"

    notebook = json.loads((ROOT / "notebooks" / "laya_finetune_dealership_kaggle.ipynb").read_text())
    notebook_src = "\n".join("".join(c.get("source", [])) for c in notebook["cells"])

    assert _wanted(generator) == _wanted(notebook_src), (
        "the notebook is out of sync with its generator - "
        "regenerate with: uv run python training/make_notebook.py"
    )
    assert shipped <= _wanted(generator), (
        f"the dataset ships {sorted(shipped - _wanted(generator))} but the notebook does not copy it"
    )


# --------------------------------------------------------------------------- held-out measurement
def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def test_the_acceptance_eval_set_holds_out_its_phrasings():
    """The acceptance eval read the training file and reported 1.000 accuracy.

    Every reply is a template, so a row-level split would still leak - "Yes, {t} works for me."
    would sit in both halves. The split holds out whole phrasings; this asserts the two files never
    share one.
    """
    train = rows("acceptance_train.jsonl")
    dev = rows("acceptance_dev.jsonl")
    train_phrases = {r["reply_template"] for r in train}
    dev_phrases = {r["reply_template"] for r in dev}
    assert not (train_phrases & dev_phrases), sorted(train_phrases & dev_phrases)
    assert dev_phrases, "the dev split is empty, which measures nothing"


def test_the_acceptance_eval_does_not_default_to_the_training_file():
    """A default pointing at the training set is a leak waiting to be reported as a win."""
    src = (ROOT / "scripts" / "eval_acceptance.py").read_text()
    assert 'DATA = ROOT / "data" / "calls" / "acceptance_dev.jsonl"' in src
    assert "assert_held_out(" in src


def test_no_eval_set_is_a_substring_of_a_training_row():
    """Exact-text disjointness is not enough, which is how `sev-04` hid.

    "there's smoke coming from under the hood." is a held-out severity case *and* a substring of two
    training rows. The exact-text check passed while the case was effectively trained on. Either
    direction counts as leakage, and short utterances are ignored because they match by accident.
    """
    for eval_name, train_name in (
        ("routing.jsonl", "synthetic.jsonl"),
        ("severity.jsonl", "severity_train.jsonl"),
    ):
        ev = [_norm(r["text"]) for r in rows(eval_name)]
        tr = [_norm(r["text"]) for r in rows(train_name)]
        tr_set = set(tr)
        assert not (set(ev) & tr_set), f"{eval_name} has exact-text overlap with {train_name}"

        leaked = [
            (e, t)
            for e in ev
            if len(e) >= 20
            for t in tr
            if e in t
        ]
        assert not leaked, (
            f"{eval_name} cases appear inside {train_name} rows: {leaked[:3]}"
        )
