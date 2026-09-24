# Experiment ledger

Every experiment gets a directory containing:

- `hypothesis.md` — the question and the predeclared success criteria;
- `manifest.json` — code/data/config/runtime provenance;
- `results.json` — measurements, including negative and null results;
- `decision.md` — what was promoted, rejected, or left unresolved.

A run is not finished because its numbers look good. It is finished when the next reader can
reconstruct what changed, what was held out, and why the decision was made.

The first Phase-A entry is `phase-a-v7-diagnostics/`. It used no hosted model and no GPU training.
