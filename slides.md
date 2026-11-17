---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-family: Arial, sans-serif; }
  h1, h2 { color: #155e75; }
  strong { color: #9a3412; }
---

<!--
EDITING: This is the editable source; no slide service account is required.
Preview with the VS Code Marp extension.
Optional CLI (Node.js required; downloads Marp):
  npx @marp-team/marp-cli slides.md -o slides.html
  npx @marp-team/marp-cli slides.md --pdf
PowerPoint export is possible, but the Markdown remains the reliably editable
source; some export modes render slides as images.
Speaker notes are these HTML comments. Keep them out of the projected content.
-->

# Can AI Predict the Future?
## Exploring Time Series Foundation Models

**PyLadies | 90 minutes hands-on + 30 minutes questions**

One dataset. Sensible baselines. A fair test.

<!-- 0:00–0:02. Welcome. No prior deep learning expected. Pairing is welcome.
State that the goal is a defensible comparison, not making AI win.
Before arrival, rehearse the exact Colab path and bundle real model artifacts. -->

---

# Vote before you fit

Who will forecast best?

**A.** Repeat last week  
**B.** A classical statistical model  
**C.** A pretrained transformer

Write one reason. Keep your answer.

<!-- 0:02–0:05. Hands or a simple poll, no extra service required.
Ask two people for different reasons. Promise to revisit the vote.
Use green/yellow/red help signals; no public speed leaderboard. -->

---

# Today's route

**Notebook 1:** see a pattern → split time → forecast → plot

**Notebook 2:** backtest → compare → explain a decision

RUN THIS = prewritten  
YOUR TURN = your code, with hints and solutions

**Using a hint counts as progress.**

<!-- Setup should be done beforehand. If a participant is blocked, timebox
troubleshooting to two minutes, pair them, or use the saved-results route.
Explain that progress badges are private and celebrate learning, not ranking. -->

---

# Forecasting is conditional

**What** are we predicting?  
**When** do we make the prediction?  
**How far ahead**?  
**What information** do we know then?

No model gets to see its own future.

<!-- 0:05–0:07. Ask the group for a forecasting example. Keep the live dataset
fixed: daily aggregate U.S. births, not individual medical predictions. -->

---

# One small, daily series

420 days of aggregate U.S. births  
Daily frequency; a possible **7-day seasonal cycle**

We reserve the last **42 days** before exploration.

Why not weekly? Aggregation would erase the weekday pattern.

<!-- 0:07–0:12. Notebook 1 data plot and weekday exercise.
Attribution: FiveThirtyEight / SSA, CC BY 4.0; full provenance in data/README.md.
Ask what might explain weekends. Distinguish description from causation.
Do not project the final evaluation period yet. -->

---

# YOUR TURN: build a time machine

```text
168 days we know | 14 days we predict
                 ^
               cutoff
```

**Context:** observed history  
**Horizon:** days 1 through 14

Random splitting would break the task.

<!-- 0:12–0:22. Notebook 1 split exercise. Give 3–4 minutes of quiet work.
Ask why horizon zero is wrong. The demo is within development; the final
42 days remain reserved for notebook 2. -->

---

# Meet the model to beat

## Seasonal naive

**"Next Monday will look like last Monday."**

```python
NaiveForecaster(strategy="last", sp=7)
```

For two weeks ahead, repeat the **available** week twice.

<!-- 0:22–0:27. Baseline exercise. Ask where the second forecast week's
values come from. They cannot come from the first week's actual future. -->

---

# A plot before a leaderboard

Do the timestamps line up?  
Does the weekly shape make sense?  
Where does the model miss?

**MAE = mean absolute error**  
Here: births per day. Lower is better.

<!-- 0:27–0:32. Display the notebook's baseline forecast graph.
Ask participants to identify one miss before reading the score.
No plot embedded here: use the actual notebook, not an illustrative curve. -->

---

# What makes a model "foundation"?

Pretrained on many series → reused on our series

**Zero-shot:** no task-specific weight training here

`fit()` can load weights and store context.  
It does not necessarily train a neural network.

**Pretrained does not mean guaranteed accurate.**

<!-- 0:32–0:35. Use "patterns learned elsewhere" as intuition.
Public historical data may have appeared in pretraining. Zero-shot does not
certify unseen data. Avoid claims about universality or "understanding time". -->

---

# Three architectures, one interface

| Model | Core idea |
|---|---|
| Chronos-2 | Patched encoder + time/group attention |
| TimesFM 2.5 | Patched decoder-only transformer |
| t0-gamma | Patch transformer + time/covariate attention |

**Native capability ≠ wrapper capability**

Our experiment uses one target and no covariates.

<!-- 0:35–0:38. Stay at intuition level; TUTORIAL.md has model sizes,
context/horizon limits, native versus sktime details, and papers.
Chronos-2 wrapper lacks interval output; TimesFM wrapper lacks XReg;
t0's multi-column wrapper processes columns independently. -->

---

# RUN THIS: one foundation forecast

Choose one prepared model.

**Live inference** or **saved real inference**—label which.

Compare its curve with actuals and seasonal naive.

If a model failed: **not evaluated**, not "zero error".

<!-- 0:38–0:45. Notebook 1 model cell and reflection.
If no actual foundation results were produced in rehearsal, say the evidence
is unavailable. Never present a synthetic curve as model output.
Do not spend more than two minutes troubleshooting a live download.
At minute 45, driver/navigator swap and open notebook 2. -->

---

# One lucky split is not enough

## Backtest three forecast origins

168-day context → 14-day forecast  
Move forward 14 days → repeat  
Move forward 14 days → repeat

Earlier test outcomes can become **later history**.

<!-- 0:45–0:53. Notebook 2 splitter exercise and schedule table.
No overlap in test windows. Fixed-length rolling context, not expanding.
Ask why using the first test window in a later training window is valid. -->

---

# Agree on the scoring rules

**MAE:** typical absolute error, in original units

**Seasonal MASE:** MAE divided by a training-only seasonal scale

MASE < 1 does **not** prove a test-set win over seasonal naive.

Direct comparison:  
`100 × (1 − model MAE / naive MAE)`

<!-- 0:53–0:57. Keep the denominator interpretation explicit.
Near-zero values cause trouble for percentage metrics.
Use the notebook formula rather than a long derivation. -->

---

# Same track for every contestant

Seasonal naive · SARIMA · Exponential smoothing

Chronos-2 · TimesFM 2.5 · t0-gamma

**One `sktime.evaluate` path**  
Same history, cutoffs, horizon, and metrics

Prophet is an optional extra, not an installation hurdle.

<!-- 0:57–1:05. Run baseline benchmark and load real instructor artifacts.
Assign models across tables rather than every person running every model.
Explain refit: classical parameters change; zero-shot weights do not.
Runtime includes different cold/warm effects—do not overinterpret timings. -->

---

# YOUR TURN: trust, but inspect

Build the leaderboard.

Does the winner change by fold?  
Where is the largest miss?  
Is a small improvement worth more setup and runtime?

**A baseline win is a useful result.**

<!-- 1:05–1:17. Leaderboard exercise, per-fold bars, actual forecast graph.
Ask each pair for one observation, not just the winning model name.
The late-year windows include calendar effects: don't secretly add future
features to just one contestant after seeing the errors. -->

---

# Your recommendation needs boundaries

> For **this series, horizon, and three cutoffs**, I would choose ___
> because ___. Before deployment I would test ___.

One series is not a universal benchmark.  
Three folds are not a significance study.  
Public data may overlap with pretraining.

<!-- 1:17–1:25. Pair discussion and written recommendation.
If no foundation artifacts were available, explicitly state the
foundation-versus-baseline question remains unanswered. -->

---

# Finish line

Did you change your opening vote?

**Exit ticket**

One leakage risk  
One useful baseline  
One thing this experiment cannot establish

**You can now run a fair forecasting comparison.**

<!-- 1:25–1:30. Re-run poll, celebrate justified changes of mind.
Save/download notebooks. Move to optional slides only after the 90-minute core. -->

---

# Q&A: choose a direction

**Uncertainty:** are 80% intervals calibrated?  
**Covariates:** what will truly be known in advance?  
**Robustness:** other seasons, horizons, and series?  
**Operations:** latency, monitoring, licenses, drift?

30 minutes for your questions and use cases.

<!-- 1:30–2:00. Let participants choose the order. If they want code,
optional Prophet is already in notebook 2. No need to add a new notebook.
Keep uncertainty claims separate from point-error rankings. -->

---

# Beyond the core: Kumo Forecast

Time-series forecasting with a transformer backbone  
Optional context retrieval / DARR and interpretability

**Tutorial-only today:** another environment and more runtime complexity

Current Kumo-TS supports CPU fallback;  
it is **not strictly NVIDIA-GPU-only**.

<!-- Optional Q&A slide. Distinguish Kumo-TS from the GPU-native Structured
Data Models route. Use TUTORIAL.md for paper and official setup links.
If using retrieval memory, its continuations must respect the cutoff. -->

---

# Kumo Relational is a different problem

Customers + orders + products → related tables / graph

Row embeddings → graph message passing → in-context prediction

**Kumo Relational:** pretrained relational model  
**Kumo Engine:** context-generation/training infrastructure  
**Kumo Forecast:** time-series forecasting

An endpoint client can need no local GPU; the server still does inference.

<!-- Optional Q&A slide. Do not suggest uploading participants' employer data.
Kumo Tabular and Kumo Anomaly are other family members.
Exact implementation and papers are linked in TUTORIAL.md. -->

---

# Take this workflow with you

Define the decision → establish the baseline → respect time

Forecast → backtest → inspect misses → qualify the conclusion

**Read next:** `TUTORIAL.md`  
**Run later:** `SETUP.md`  
**Help a friend prepare:** `PARTICIPANT_PREP.md`

Thank you!

<!-- Sources for architectural claims, model cards, and papers are collected
with the corresponding claims in TUTORIAL.md. All slides remain editable.
Do not extend the session past two hours; leave unresolved topics as questions. -->