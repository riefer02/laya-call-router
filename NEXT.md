# Next steps

## Phase A status — local infrastructure, no generation or GPU

Phase A is implemented locally. The optional scope and safety-applicability questions now live in
the store profile and have policy tests, but they are not added to the active v7 pass because v7
was not trained on them. The bundled v7 raw-logit diagnostic reproduces 71/81 joint routing and
finds four option-order-sensitive destination cases; its ledger is in
`experiments/phase-a-v7-diagnostics/`.

The trainer now has seeded runtime controls, a single token-budget source, strict build mode,
optional validation/calibration item files, validation history, a best-validation checkpoint, and
a run manifest. A future run should set `JEV_REQUIRE_SPLITS=1`; the legacy calibration fallback is
explicitly marked untrusted.

The next implementation step is a small frozen typed-conversation stress set and a clean handoff
between the Evidence page, the README, and the active checkpoint report. This should stay local and
should not expand into a hosted service or cost-reporting project unless operational deployment
becomes an explicit goal.

The current demo serves the v7 fine-tune, replays inspectable calls, and checks 13 scripted
scenarios. The checkpoint is bundled through Git LFS. [LEARNINGS.md](LEARNINGS.md) records the
experiments and corrections; [results/README.md](results/README.md) identifies the report files.

## Make the conversation stronger

1. Add varied caller turns for each outcome: changed plans, corrections, missing details,
   interruptions, and replies that do not answer the current question.
2. Score the spoken path as well as the route: one useful reply per turn, no repeated filler,
   suitable offers, and a final confirmation that matches the filed action.
3. Fix the off-topic caller that is still flagged for roadside help and recheck all genuine
   roadside cases. The current demo gate passes 12 of 13 scenarios.

## Strengthen the ML evidence

1. Expand the hand-labelled routing and full-call sets beyond the current 81 cases and 27 calls.
   Gather examples from the setting where the model will actually be used.
2. Improve the `needs_human` question and measure safety errors at realistic base rates. The v7
   safety set has 18/18 hazard recall and 3/27 safe-case false alarms, with limited precision
   because the sample is small.
3. Calibrate the confidence gate on held-out decisions. Current routing errors can be confident;
   a high score alone cannot be treated as permission to automate.
4. Automate a rerun from the bundled frozen v7 snapshot and compare its measured outputs with the
   included checkpoint. The current packaging command uses the newer working data by default.

## Keep deployment optional

A hosted inference adapter, usage gateway, and operating-cost report are deliberately not active
research tasks for this repository. They become relevant only if operational deployment becomes an
explicit goal and there is a real workload to measure.

## Make the example easier to reuse

1. Move dealership-specific dialogue and scheduling facts behind the store profile where that
   makes a new domain simpler to implement and test.
2. Add a second small domain with its own profile, labels, and complete-call evaluations. This
   will test whether the process described in the README transfers beyond this dealership.
