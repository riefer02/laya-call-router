# The elevator pitch

## In one sentence

A car-dealership phone switchboard that runs on a laptop, decides where a call should go in about
20 milliseconds, costs nothing per call, and shows you every decision it made.

---

## The problem

A dealership's phone line has to work out, from what a caller says, whether they want Service or
Sales or Parts, whether they're stranded on a motorway, whether they need a person rather than a
booking, and then actually book them in.

The usual answer is to send every call to a large language model. That works — and it is slow, it
costs money every time someone dials, and it gives you a different answer each time you ask.

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

That last row is the one people miss. The categories aren't baked into the model's weights — they're
supplied with each question. That makes the taxonomy *data*, which is what lets this work for a
different dealership without rebuilding anything.

## What we built

A working switchboard, and a debugger for it.

The debugger is the part worth seeing. Every call is drawn as a grid: **rows are conversation turns,
columns are decisions, left to right.** Click any box and it tells you the question it asked, every
option it considered with the probability it gave each one, which model answered, and how long it
took. Most boxes are marked `policy` or `regex`, because most decisions are simple rules — and the
tool never pretends a model decided something a rule decided.

## The result

Fine-tuning took the model from **42 of 81** routing cases to **71 of 81.** A small local model that
started out useless at this task learned it.

On the same 81 cases it finished close to the two API models: 71 correct, versus 72 for Nano and
73 for DeepSeek Flash. That small sample cannot establish equal quality.

So: **close on this set, about 70 times faster per routing case, with no per-call API fee.**

---

## What this gets you

**A phone line that never makes anyone wait.** 21 milliseconds to decide where a call goes. The
frontier model takes 1.5 seconds, and a caller can hear that.

**Nothing per call, forever.** Zero marginal cost, on hardware you already own. A busy switchboard at
10,000 calls a day would pay `deepseek-flash` about **$1,700 a year** and `gpt-5.4-nano` about **$435**
— and this costs about two dollars, in electricity. The saving is modest in absolute terms for one
dealership; it becomes material across several stores, or over a few years, and it never appears on a
per-call invoice.

**The same answer every time you ask.** Not 98% of the time — always. That means you can write tests
against it, replay a customer complaint, and reproduce a bug. The frontier arms wobble by several
points between identical runs. Ours has never moved once.

**You can see why it decided.** Click any box in a call and you get the question, every option with
its probability, which checkpoint answered, and the milliseconds it took. When someone disputes a
routing decision, you show them instead of guessing.

**It does not generate facts.** Appointment times and callback numbers come from the scheduler and
caller text, not model prose. It can still choose a wrong label with high confidence; the off-topic
roadside failure is a clear example.

**It declines rather than guesses.** Asked to book, it checks that the caller named a time it actually
offered. Say "Tuesday" when the only times on offer are Wednesday, and it asks again rather than
filing an appointment nobody agreed to. Our booking scenario has it refuse three times, then book.

**It caught all 18 hazards in the current labelled set**, with priority raised and roadside selected.
It also falsely selected roadside on three of 27 safe cases, and an ambiguous off-topic demo call
still misfires. This is a routing prototype; the Evidence tab shows the full tradeoff.

**A new queue starts as a config change.** Categories are supplied with each question rather than
hard-coded in the output layer. A store can add a Fleet department in the profile, then must test
that queue on its own calls before relying on it; the current scores cover this store's taxonomy.

**The call never leaves the building.** No API call, no data leaving, no third party. For phone
numbers and call recordings, that is usually a requirement rather than a nicety.

**Every number is traceable.** The debugger reads its measurements from the same files the README
quotes. We found eight places where that wasn't true — including three evaluations that were reading
their own training data — fixed them, and gave each one a test. A number you can't trace is a number
nobody can check.

---

## What it costs

| | |
| --- | --- |
| Building the training data (teacher-model calls) | ~$3.40 |
| Fine-tuning | **$0** — free cloud GPU, about 40 minutes |
| Running it | **$0 per call**, forever, on a laptop |
| **The whole project** | **~$4.40** |

At 10,000 calls a day — a busy multi-line switchboard — that works out as:

| per 10k calls/day | gpt-5.4-nano | deepseek-flash | **this** |
| --- | --- | --- | --- |
| per decision | $0.000031 | $0.000124 | **$0** |
| per call (4 turns) | $0.000119 | $0.000476 | **$0** |
| per day | $1.19 | $4.76 | **$0.00** |
| per month | $36 | $143 | **$0.20** |
| per year | $435 | $1,737 | **~$2** |

Two honest notes. **The saving is real but modest in absolute terms** — a few hundred to under two
thousand dollars a year at this volume, which for a dealership is not the reason to do this. Where it
becomes material is at multi-store or contact-centre scale, or over years.

And **the fixed cost matters at low volume**: this runs on a laptop you already have, but if you had
to buy a dedicated machine for it, that machine would cost more than the API calls would at a few
hundred calls a day. The honest version of the cost story is that inference is free *once you have
the hardware* — not that the hardware is free.

The whole project — the data, the training, every experiment — came to $4.40, which is about **9,000
`deepseek-flash` calls**. At 10k calls a day that is **less than one day** of the API bill.

## If someone asks how it works

**"Isn't it slow, asking the model about every box?"** No — and the node count is misleading. A real
six-turn booking call draws 51 boxes, but it asks only **17 questions across 6 passes**, 191 ms in
total. The questions are batched into one pass per turn, because the transcript is the expensive part,
not the questions. Seven questions against one transcript is one pass; asking them one at a time would
cost seven times as much for the same answer.

The rest of the boxes are policy and regex. They're free.
