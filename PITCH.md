# The elevator pitch

## In one sentence

A car-dealership phone switchboard that runs on a laptop, decides where a call should go in about
20 milliseconds, costs nothing per call, and shows you every decision it made.

---

## The problem

A dealership's phone line has to work out, from what a caller says, whether they want Service or
Sales or Parts, whether they're stranded on a motorway, whether they need a person rather than a
booking, and then actually book them in. That is a routing problem, and the usual answer is to send
every call to a large language model.

That works. It is also slow, it costs money per call, and it gives you a different answer each time.

## The technology

This is built on **Laya**, and the interesting thing about Laya is what it *doesn't* do: **it never
writes text.**

You give it a question and a list of possible answers. It hands back one of them, with a
probability. That's it.

Everything good follows from that one property:

| because it doesn't generate text | you get |
| --- | --- |
| nothing is sampled | **the same answer every time** |
| nothing is written out | **~20 ms, not ~1.5 seconds** |
| no API is called | **$0 per call, and the call never leaves the building** |
| the answers are passed in per request | **a store can add a queue by editing a config file — no retraining** |

That last row is the one people miss. The categories aren't baked into the model's weights. They're
supplied with each question, so the taxonomy is *data* — which is what makes this work for a
different dealership without rebuilding anything.

## What we built

A working switchboard and a debugger for it.

The debugger is the part worth seeing. Every call is drawn as a grid: **rows are conversation turns,
columns are decisions, left to right.** Click any box and it tells you the question it asked, every
option it considered with the probability it gave each one, which model answered, and how long it
took. Most boxes are marked `policy` or `regex`, because most decisions are simple rules — and the
tool never pretends a model decided something a rule decided.

## The result

Fine-tuning took the model from **42 of 81** routing cases to **71 of 81.** That's the headline: a
small local model that started out useless at this task and learned it.

Against the frontier models, we are **level.** On the same 81 cases:

| | ours | gpt-5.4-nano | deepseek-flash |
| --- | --- | --- | --- |
| routing cases | 71/81 | 72/81 | 73/81 |
| per call | **21 ms** | 651 ms | 1.5 s |
| per call | **$0** | $0.0025 | $0.010 |
| same answer twice | **always** | 98% | 99% |

**One case apart is not a difference** — on a set this size, the error bars are about six cases wide.
So we don't claim to have won. We claim to be level with them, at roughly **70 times the speed, for
nothing, with a number that never moves.** Their scores wobble by several points between identical
runs; ours hasn't moved once.

## What it costs

| | |
| --- | --- |
| Building the training data (teacher-model calls) | ~$3.40 |
| Fine-tuning | **$0** — free cloud GPU, about 40 minutes |
| Running it | **$0 per call**, forever, on a laptop |
| **The whole project** | **~$4.40** |

`deepseek-flash` charges $0.010 a call. So the entire project — the data, the training, every
experiment — cost about as much as **400 calls** to the frontier model. The next 400,000 cost nothing.

And the cost per call *falls* with volume, because the expensive part was one-off. It doesn't rise.

## If someone asks how it works

**"Isn't it slow, asking the model about every box?"** No — and the node count is misleading. A real
six-turn booking call draws 51 boxes, but it only asks **17 questions across 6 passes**, 191 ms in
total. The questions are batched into a single pass per turn, because the transcript is the expensive
part, not the questions. Seven questions against one transcript costs one pass; asking them one at a
time would cost seven times as much for the same answer.

The rest of the boxes are policy and regex. They're free.

## What we would not claim

- **Not "we beat them."** One case apart on 81 is not a result, and their own scores move more than
  that between runs.
- **Not "the safety classifier is fixed."** It catches every stranded caller, but the set we measure
  it on is deliberately 40% emergencies when reality is nearer 2%, so most of its alarms would still
  be false ones in a real switchboard.
- **Not "the threshold is tuned."** The probabilities are so confident that moving the bar from 0.3
  to 0.8 changes nothing. It isn't a dial we actually have.
- **Not "1.000 booking accuracy"** without saying it's measured on phrasings the model hasn't read,
  not on whole conversations it hasn't had.

## The honest footnote

Most of what we learned was about **measurement**, not models. Three of our own evaluations turned
out to be reading the data they were trained on. Every checkpoint had been scored partly on cases it
had already seen. Our safety numbers were flattered by the set they were measured on.

Eight of those silent mistakes now have tests. That is the part I'd actually defend — a number you
can't trace is a number nobody can check.
