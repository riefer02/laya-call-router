# A small model you can inspect, train, and serve

**Laya Call Router is a car-dealership call-routing demo built around typed decisions and Laya.** Its
fine-tuned Laya model chooses from answers supplied with each question. A separate call policy
decides what to ask and when to book; the dialogue layer turns confirmed facts into plain replies.
The debugger lets you replay a call and inspect the source and evidence for every step. This is an independent research demo built on Laya, not an official Laya product.

## Why show this

The useful experiment is whether a small specialist model can handle a narrow decision task well
enough to support a real workflow. Fine-tuning raised joint routing accuracy from 42/81 to 71/81
on our frozen 81-case legacy/development benchmark. In the same evaluation, `gpt-5.4-nano` scored
72/81 and `deepseek-flash` 73/81. The three results are close on this sample; a larger test would
be needed to rank them. The fine-tune's median routing-case time was 21 ms in the MLX development setup,
against 651 ms and 1,451 ms for the two API calls. A whole call takes longer and involves more
than one decision.

The checkpoint is included through Git LFS so people can run the same model, inspect the recorded
reports, and try new calls. MLX is the current development runner. A compatible hosted runtime
would be an optional future deployment option; where and how the weights are hosted would determine
operating cost and latency.

## What to demonstrate

1. Open the debugger and run **Buying a car**. The caller names a showroom and a day; the
   switchboard offers matching openings, then files the accepted appointment.
2. Select a model decision. Show the question, choices, probabilities, checkpoint, and timing.
   Select a rule or scheduler step to show which parts are ordinary application code.
3. Run **Unavailable appointment time**. The caller requests a time that was never offered; the
   switchboard keeps the appointment open and asks for an available choice.
4. Open **Evidence**. It reads versioned report files, including the 81 routing cases, 27 scripted
   calls, and 45 safety cases. The measured failure cases remain visible.

The scripted demo currently passes 12 of 13 scenario checks. It includes five bookings, two
roadside cases, three transfers, one hours answer, and one wrong-number close. The open miss is an
off-topic caller incorrectly flagged for roadside help. The app records a dispatch flag in that
case; it does not contact an actual tow service.

## What others can reuse

The dealership is one example of the process. The store profile holds destinations, request
types, questions, and hours. Training code builds typed questions from labelled messages, the
evaluation harness compares models on the same cases, and the conversation checks verify the
user-visible result. Someone applying it to a library or clinic would define their own profile,
collect their own held-out examples, train a checkpoint, and test complete conversations.

Start with the [README](README.md) to run it, [LEARNINGS.md](LEARNINGS.md) for the experiment
history, [training/README.md](training/README.md) for the training recipe, and
[NEXT.md](NEXT.md) for the next development steps.
