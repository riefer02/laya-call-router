# Show the call-routing demo

This is a short path through the car-dealership example. The debugger replays scripted text calls
and shows what the model, call policy, and scheduler each decided. The bundled v7 checkpoint makes
the example runnable from a checkout with Git LFS.

## Prepare

Follow the [README](README.md) to install dependencies and start the server. Then check the demo
gate:

```bash
uv run python scripts/check_demo.py
```

It currently passes 12 of 13 scenarios. The off-topic caller remains a visible miss: the model
flags roadside help too early. A dispatch here is a recorded decision and queue change, not a
connection to a tow service.

## A five-minute walkthrough

1. **Run “Buying a car.”** The caller asks about an electric car, chooses a showroom and day,
   receives available times, and accepts one. Open the booking card to check that the appointment
   type, person, place, and spoken confirmation match the filed record.
2. **Click a model decision.** The Inspector shows the question, available answers, probabilities,
   checkpoint, and time spent. Then click a rule or scheduler box to show the separate source of
   that decision.
3. **Run “Unavailable appointment time.”** The caller proposes a time outside the offered slots.
   The switchboard keeps the booking open and asks them to choose an available time.
4. **Try a different ending.** “Hours” answers from the store schedule; “Part order” transfers to
   the parts team; “No start” raises the roadside flag. These cases show the same model feeding
   different application actions.
5. **Open Evidence.** The routing comparison uses 81 labelled cases; the full-call comparison uses
   27 scripted calls; the safety report has 45 cases. The page names the report files it reads.

Playback controls step through the recorded events without rerunning the backend. The caller input
at the bottom lets you try new text; each submission recomputes the full typed call. Those new
conversations are exploratory until they are labelled and added to an evaluation set.

## What the experiment shows

The fine-tuned model got both destination and request type right for 71/81 routing cases. In the
same report, `gpt-5.4-nano` got 72/81 and `deepseek-flash` 73/81. The fine-tune chose the right
final team in 25/27 scripted calls. The routing-case median was 21 ms in the MLX development
runner. These results put a small specialist model close to the hosted models on this sample and
make every step of its application visible. The sample is too small to rank close scores.

The checkpoint is packaged for readers to run and inspect. MLX is the current development
runtime; a compatible hosted runner is a next step. The [pitch](PITCH.md) gives a concise story,
[LEARNINGS.md](LEARNINGS.md) explains the experiments, and [NEXT.md](NEXT.md) lists the work ahead.
