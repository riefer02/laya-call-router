# Web and debugger instructions

These instructions apply to the `web/` directory.

## Commands

Install and build with:

```bash
cd web
npm install
npm run build
```

For local development:

```bash
cd web
npm run dev
```

Do not edit `web/dist/`; it is generated output.

## UI contracts

- Preserve the event vocabulary consumed by the graph: caller/agent turns, model decisions,
  deterministic policy steps, extraction steps, and terminal routes.
- Keep model decisions visually distinct from rules, scheduler actions, and regex extraction.
- Do not make a failed scenario look successful by hiding its terminal result.
- The Evidence tab must read the reports belonging to the checkpoint currently served by the API.
- Keep the known off-topic failure visible until a trained scope decision and a new evaluation
  justify changing it.
- Preserve keyboard access, labels, and `aria` attributes when adding controls.
- Use the existing frontend dependencies and visual language unless there is a measured reason to
  add another.

## Verification

After frontend changes:

```bash
npm run build
cd ..
uv run pytest -q
uv run python scripts/check_demo.py
```

The demo gate currently has one documented known failure. Report it honestly; do not weaken the
scenario or fixture merely to obtain a green result.

## Current research behavior

The debugger is an observability surface, not an additional source of model truth. When displaying
probabilities, distinguish raw or answer confidence from legacy entropy confidence when the
underlying model exposes both. Do not imply that a saturated probability proves correctness.
