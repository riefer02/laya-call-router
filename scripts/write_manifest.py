"""Write a self-verifying experiment manifest without capturing credentials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jev_classifier.provenance import build_run_manifest, write_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--input", action="append", required=True, help="LABEL=path")
    ap.add_argument("--config", default="{}", help="JSON config text or path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inputs = {}
    for item in args.input:
        if "=" not in item:
            raise SystemExit(f"--input must be LABEL=path, got {item!r}")
        label, raw_path = item.split("=", 1)
        inputs[label] = Path(raw_path)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text() if config_path.is_file() else args.config)
    manifest = build_run_manifest(
        experiment_id=args.experiment_id,
        inputs=inputs,
        config=config,
        model_id=args.model,
        model_revision=args.revision,
        seed=args.seed,
    )
    write_manifest(Path(args.out), manifest)
    print(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
