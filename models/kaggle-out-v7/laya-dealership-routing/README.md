# Dealership routing fine-tune (v7)

This is the checkpoint used by the jev-classifier demo. It fine-tunes
[Laya](https://huggingface.co/convaiinnovations/laya) for typed decisions in a car-dealership
switchboard. The model chooses among answers supplied with each question. It does not generate the
switchboard's spoken replies; the application handles dialogue, scheduling, and booking separately.

The checkpoint includes the weights, encoder configuration, tokenizer, and agent configuration
needed for inference. `models/active` points here. Git LFS stores `model.safetensors`; run
`git lfs pull` if the file in your checkout is only a small pointer.

## Origin and license

Base model: `convaiinnovations/laya`, distributed under Apache-2.0. This directory includes a copy
of the upstream Apache-2.0 license. The weights were changed by the jev-classifier v7 fine-tune.
The frozen training inputs and their SHA-256 manifest are in the parent `kaggle-out-v7` directory.
See the repository's [training guide](../../../training/README.md) for the recipe and
[results guide](../../../results/README.md) for the evaluation reports.

## What was measured

On 81 hand-labelled routing cases, v7 answered both destination and request type correctly for
71 cases; the `gpt-5.4-nano` and `deepseek-flash` comparison arms scored 72 and 73. On 27 scripted
calls, v7 chose the right final team for 25. It caught 18 of 18 labelled hazards in a separate
safety set and falsely flagged 3 of 27 safe cases. These are small, domain-specific samples; the
model should be evaluated on new labels and full conversations before another application uses it.

## Running it

The included application serves these weights through MLX during development on Apple Silicon.
The `safetensors` checkpoint is a separate artifact for a future compatible hosted runner. The
repository does not yet contain a hosted inference deployment. Start with the root README and run
`uv run python scripts/check_demo.py` to verify the bundled example.
