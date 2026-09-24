# Laya Call Router

An inspectable call-routing research demo for a car dealership, built on Laya. A fine-tuned Laya model reads a
caller's words and chooses typed answers, such as a department or vehicle type. Rules decide what
to ask next, check available appointments, and file a booking. The debugger shows each step. The
checkpoint, frozen training snapshot, evaluation cases, and lessons are here so you can run the example and
adapt the process to your own domain.

This is an independent research demo built on Laya; it is not an official Laya product.

The demo uses **scripted text calls** so each decision can be replayed. Short reply templates use
confirmed caller and store facts; the model chooses typed answers while the dialogue layer speaks.

![Call-routing debugger](docs/overview.png)

## See a call

The four-turn “Buying a car” example starts with an electric-car request. The switchboard asks
which showroom and day work, offers times from the store schedule, and books the time the caller
accepts. “Unavailable appointment time” asks for Sunday at 3am after an offer, so you can see the
system decline that time and keep the original appointment type.

The graph shows which steps came from the caller, the model, a rule, or the scheduler. Select any
box to see its answer and evidence. The Evidence tab shows the measured results and names the
report files behind them. [DEMO.md](DEMO.md) is a short presentation guide.

## Run the demo

Install [Git LFS](https://git-lfs.com/) before cloning so the fine-tuned weights download with the
repository. This development runner uses MLX on macOS with Apple Silicon; the checkpoint is a
separate artifact that can be served from another compatible runtime later. You also need Python
3.12, `uv`, and Node.js. The current demo is not a Linux/Windows runtime.

```bash
git lfs pull
uv sync
cd web && npm ci && npm run build && cd ..
git config core.hooksPath .githooks
uv run python scripts/doctor.py
uv run uvicorn jev_classifier.api:app --port 8765
```

The `core.hooksPath` setting enables the repository's secret-scanning and Git LFS hooks. It is
local Git configuration, so run it once after cloning.

The local Laya demo does not need hosted-model credentials. If you want to run the hosted comparison
or teacher-labelling scripts, copy [`.env.example`](.env.example) to `.env` and fill in only the
provider keys you need:

```bash
cp .env.example .env
```

Kaggle authentication is separate. The [current CLI authentication
instructions](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication) list
`kaggle auth login`, a `KAGGLE_API_TOKEN` environment variable, `~/.kaggle/access_token`, and the
legacy `~/.kaggle/kaggle.json` file. `KAGGLE_USERNAME` in the example is only an optional default
owner for `training/make_kaggle_dataset.py`. Submitting a Kaggle job can consume quota and is not
part of the local demo.

Open <http://127.0.0.1:8765>. For frontend work, run `cd web && npm run dev` in a second terminal
and open <http://localhost:5173>.

`models/active` points to the bundled v7 checkpoint. If you clone without Git LFS, run
`git lfs pull` before starting the server. Set `JEV_MODEL` to a different checkpoint directory to
compare your own fine-tune. The training workflow is in [training/README.md](training/README.md).

Check the scripted calls before a demo:

```bash
uv run python scripts/check_demo.py
uv run pytest -q
```

`check_demo.py` uses a temporary booking store. It checks the final team, completion, appointment
type, spoken confirmation, and key question sequences. The app itself saves local run recordings
under `results/runs/` and bookings under `data/bookings.jsonl`; both are ignored by Git.

## What is doing the work?

| Part | What it does | Where to look |
| --- | --- | --- |
| Store profile | Defines departments, request types, questions, safety thresholds, hours, and availability | [config/store_profile.json](config/store_profile.json) |
| Classifier | Chooses among supplied answers; it never generates reply text | [src/jev_classifier/agent.py](src/jev_classifier/agent.py) and [dealership.py](src/jev_classifier/dealership.py) |
| Call policy | Reuses known facts, asks for missing ones, and decides when to book or hand off | [src/jev_classifier/call.py](src/jev_classifier/call.py) |
| Scheduler | Offers real, future openings and files a local booking | [src/jev_classifier/schedule.py](src/jev_classifier/schedule.py) |
| Spoken replies | Turns confirmed decisions and facts into short sentences | [src/jev_classifier/dialogue.py](src/jev_classifier/dialogue.py) |
| Debugger | Replays a call and exposes the evidence for each decision | [web/src](web/src) |

The parts can be tested separately. The model chooses a label; the policy decides the next action;
the dialogue layer says it plainly. The spoken-path check now covers the actual caller experience,
including which question was asked, whether an offered time was accepted, and whether the final
confirmation matches the filed appointment.

## What we measured

The active **v7** report used a frozen 81-case legacy/development routing benchmark, 27 scripted
evaluation calls, and a separate 45-case safety set. The evidence is small and synthetic-heavy; it is
not a real-world dealership validation set. The fine-tune landed close to the hosted models on this
small set. The sample is too small to establish a quality ranking.

| Measure | Local fine-tune | gpt-5.4-nano | deepseek-flash |
| --- | ---: | ---: | ---: |
| Department and request type both right | 71/81 | 72/81 | 73/81 |
| Final team right across 27 scripted calls | 25/27 | 26/27 | 26/27 |
| Median time for one routing case | 21 ms | 651 ms | 1,451 ms |
| API cost per routing case | — | $0.000031 | $0.000124 |
| API cost for 81 routing cases | — | $0.0025 | $0.0100 |

The hosted figures are reported API spend for the routing cases in `results/eval_v7.json`. The local
MLX arm has no per-call API fee; its hardware, electricity, and operations costs are not included.
These are not full-conversation or production infrastructure costs.

These are **routing-case** times, not full conversation times. The debugger's model-time counter
adds classifier work across every turn. Serving costs depend on where you host the model; this MLX
development setup has no per-call API fee. Training used teacher calls and cloud GPU time.

The current demo gate passes **12 of 13 scenarios**, including five bookings. On the safety set,
v7 caught 18/18 labelled hazards and falsely flagged 3/27 safe cases. The remaining demo miss is
an off-topic caller flagged for roadside help before they can clarify. See
[results/README.md](results/README.md) for report provenance and [LEARNINGS.md](LEARNINGS.md) for
the full account of experiments and corrections.

## Apply the process to another domain

The reusable idea is to make the choices and success criteria explicit before fine-tuning. For a
library help desk, for example, you might replace dealership departments with accounts, loans,
events, and building information, then use real help-desk messages to test the new boundaries.
This repository's measured scores would no longer apply.

1. **Define the work.** List the destinations, the specific requests within each one, and which
   facts a person must provide. Keep the vocabulary in one profile and make ambiguous boundaries
   explicit.
2. **Write a held-out test set first.** Label real or realistic short messages and complete
   conversations. Check the final action as well as the classifier's label.
3. **Validate any teacher.** Compare its labels with the held-out set before using it to create
   training examples. See [scripts/validate_teacher.py](scripts/validate_teacher.py). This step
   calls an external model and can incur a fee.
4. **Generate and audit training data.** Require valid labels, enough examples per destination,
   and no exact or contained copies of test examples. See
   [scripts/generate_training.py](scripts/generate_training.py) and
   [scripts/audit_snapshots.py](scripts/audit_snapshots.py). Generated rows are hypotheses until
   they pass these checks.
5. **Train from a recorded recipe.** The notebook is generated from code, and
   [training/run_config.json](training/run_config.json) holds the settings. Audit the packaged
   snapshot beside the weights, since that is what the model actually saw.
6. **Compare the same cases and inspect failures.** [scripts/eval.py](scripts/eval.py) compares
   routing and complete calls; [scripts/eval_severity.py](scripts/eval_severity.py) tests safety
   decisions. For local-only evaluation with a checkpoint, use
   `uv run python scripts/eval.py --skip-llm --finetuned PATH_TO_CHECKPOINT --out /tmp/jev-eval.json`.
7. **Test the user-visible path.** Check the words spoken, the facts used, the final action, and
   what happens when the person disagrees or changes course. This is what
   [scripts/check_demo.py](scripts/check_demo.py) does for the dealership example.

The experiment reports in `results/` are versioned; some older ones measure different policies
or contaminated checkpoints. The top of [LEARNINGS.md](LEARNINGS.md) describes the active result,
and the rest explains how the project got there. [NEXT.md](NEXT.md) lists the remaining work.

## Scope and next steps

- No audio, phone integration, dropped-call recovery, or production dispatch.
- Safety false positives and a weak human-handoff classifier remain open problems.
- The demo covers 13 scripted situations. Typed input is useful for exploration, but its outcomes
  have not been measured on a broad set of natural conversations.
- Opening-hours questions can use the store schedule. Other factual questions still go to a person.
- The bundled v7 checkpoint reproduces inference, and `models/kaggle-out-v7/` includes the frozen
  training snapshot and hashes. A new training run can still differ because of GPU execution.

The next step is broader conversational evaluation using a frozen typed-conversation stress set. A
hosted inference adapter is optional future work if operational deployment becomes a goal. This is a
working, inspectable example to build on: use your own labels, facts, and fresh evaluation set when
applying it elsewhere.

## Security and local-only boundary

This is a local research demo, not an authenticated service. Keep the server bound to loopback:

```bash
uv run uvicorn jev_classifier.api:app --host 127.0.0.1 --port 8765
```

Do not expose it to a public interface. Hosted comparison and teacher scripts read credentials from
`.env`; keep that file local and use `.env.example` only as a template. The repository does not
contain real booking records or hosted-model credentials.

The first run of the external generality evaluation downloads public datasets into the ignored
`data/external/` directory and requires internet access. The local dealership demo does not.

## License

Repository code and documentation are available under [Apache-2.0](LICENSE). The bundled
fine-tuned weights derive from [Laya](https://huggingface.co/convaiinnovations/laya); their
[model card](models/kaggle-out-v7/laya-dealership-routing/README.md) includes attribution and the
upstream license. Check the source terms of any new training data you bring to another domain.
