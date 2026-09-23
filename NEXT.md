# Next steps

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

## Make the example easier to reuse

1. Add a hosted inference adapter for the packaged checkpoint and measure its own latency and
   operating cost. MLX is the current development runner, not a required final deployment shape.
2. Move dealership-specific dialogue and scheduling facts behind the store profile where that
   makes a new domain simpler to implement and test.
3. Add a second small domain with its own profile, labels, and complete-call evaluations. This
   will test whether the process described in the README transfers beyond this dealership.
