# Can AI Predict the Future? A practical model guide

**Reading date: 15 November 2026.** This guide covers the three models we use
in the workshop: Chronos-2, TimesFM **2.5** and **t0-gamma**. It does not cover
whichever checkpoint is newest. Package versions and wrapper behavior refer to
sktime **1.3.0**. Software, model cards, access rules and hosted services can
change.

The official repositories and source code were inspected. Papers and model
cards are linked below. Hugging Face and some other documentation sites could
not be reached from where this guide was written. So not every link was opened,
and not every pretrained model was run end to end.

## 1. What question are we answering?

A forecast is an estimate of future values, based on what we know at a given
time. It is not knowledge of the future. Before you choose a model, write down
the target, how often it is measured, the cutoff, the horizon, what you know
at the cutoff, and what a wrong forecast costs.

Our question: **given the previous 168 daily U.S. birth counts, forecast the
next 14 days.** We compare models at three later cutoffs, with the same
training data and no extra variables (covariates). The dataset is small,
complete and has a clear source. It is bundled with the workshop, so a data
download cannot break the session.

Daily data and weekly seasonality are different things. Adding the days up to
weekly totals would leave fewer observations and remove the weekday and
weekend pattern. Daily data with a short context makes both the task and the
baseline easier to follow.

## 2. What a foundation model is, and what it is not

A **local model** learns its numbers (parameters) from the one series you give
it. A **global model** learns from many series. A **foundation model** is
trained once on a very large and varied collection of series, and then reused
on new data. These groups overlap. "Foundation" does not mean "better".

**Zero-shot forecasting** means we use the pretrained model as it is, without
training it further on our task. The model still reads a window of recent
history. It may rescale the values, cut them into patches, look across the
window, and return one forecast or quantiles. Loading a model is not training
it. A method called `fit` does not mean the model is being fine-tuned.

**Training data overlap:** old public datasets can be part of what a model was
trained on. We cannot say that this series was not. Our leak-free splits stop
*us* from looking ahead. They cannot rule out that the model has seen these
days before.

A **patch** is a short block of consecutive values that the model treats as one
unit. **Attention** lets the model look at other relevant parts of its input.
**Time attention** looks across time. **Group or variate attention** can look
across related variables. This is not the same as splitting text into tokens,
and not every time-series foundation model works like the original Chronos.

## 3. Do not skip the classical models

| Model | How it forecasts | What we use | Strengths and limits |
|---|---|---|---|
| Seasonal naive | Repeats the last full season | `NaiveForecaster(strategy="last", sp=7)` | Almost no setup. Strong when weeks repeat. Cannot see new trends or special days |
| ARIMA/SARIMA | Predicts from recent values and recent errors; the series can be differenced | `SARIMAX(order=(1,0,1), seasonal_order=(1,0,0,7), trend="c")` | A clear classical baseline. You have to choose the orders, and the series needs to behave in a steady way (stationarity) |
| Exponential smoothing (ETS) | Keeps updating a level, a trend and a seasonal pattern, and gives recent days more weight | Damped additive trend + additive weekly pattern | Fast, and good for smooth, slowly changing patterns. Unusual events stay hard |
| Prophet (bonus) | A trend plus smooth seasonal curves, with optional holiday and extra-variable terms | Weekly on; daily and yearly off | Convenient for calendar patterns. Needs Stan and tuning, so it is optional |

The "I" in ARIMA means *integrated*: the series is differenced. Our fixed
example uses `d=0,D=0`, so nothing is differenced, and it is still an ARIMA
model. We do not run an expensive automatic search for the best settings. The
sktime `SARIMAX` wrapper uses statsmodels, so we do not need the extra pmdarima
package. This experiment does not compare against the best possible tuned
classical model.

Prophet accepts extra variables, but you need real values for them over the
forecast period. Observed future weather is not available at the cutoff, so it
cannot be used. Calendar features that are known in advance are fine.

Sources: [sktime naive](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.naive.NaiveForecaster.html),
[statsmodels SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html),
[exponential smoothing](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html),
[Prophet getting started](https://facebook.github.io/prophet/docs/quick_start.html),
[Forecasting at Scale paper](https://peerj.com/preprints/3190/).
For more background: [Forecasting: Principles and Practice](https://otexts.com/fpp3/).

## 4. Chronos-2

**Developer:** Amazon. **Workshop checkpoint:** `amazon/chronos-2`, 120M
parameters. Do not mix it up with the original Chronos-T5 or Chronos-Bolt.

### How it works and what it predicts

Chronos-2 is a **patch-based, encoder-only transformer**. It combines attention
over time with group attention, which lets related series and extra variables
influence each other. One pretrained model handles a single series, several
series, and series with covariates, and it returns quantile forecasts natively.
The published default context is up to 8192 observations. That is a maximum,
not a recommendation to always use all of it.

The original Chronos scaled the values and turned them into discrete tokens,
like a language model, and sampled forecast paths. Chronos-Bolt switched to
patches and predicts several steps directly, with quantiles. Calling Chronos-2
"T5 predicting discretized numbers" mixes up different generations.

### What the model can do, and what our wrapper does

The Chronos-2 model natively handles several target series, covariates seen
only in the past, and covariates known for the future. Learning across several
series is an extra option. It is not the same as forecasting every unrelated
series together.

The sktime 1.3.0 `Chronos2Forecaster` supports several series and extra
variables (`X`). It does **not** support prediction intervals or missing
values. So our notebook only asks it for point forecasts on complete data. Do
not assume `predict_interval` works just because the model itself produces
quantiles.

The notebook uses a 168-day context, batch size 1 and float32, to keep things
simple. A native default may use another data type. Tuning precision and batch
size is not a beginner exercise.

**Start and sources**

Official code and architecture/version descriptions:
https://github.com/amazon-science/chronos-forecasting

Official quickstart:
https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb

Checkpoint/license card:
https://huggingface.co/amazon/chronos-2

Chronos-2 report, *From Univariate to Universal Forecasting*:
https://arxiv.org/abs/2510.15821

Original Chronos paper, for historical comparison:
https://arxiv.org/abs/2403.07815

Pinned wrapper source:
https://github.com/sktime/sktime/blob/v1.3.0/sktime/forecasting/chronos2.py

## 5. TimesFM 2.5

**Developer:** Google Research. **Workshop checkpoint:**
`google/timesfm-2.5-200m-transformers`. The 2.5 base model has 200M parameters.
The upstream project also describes an optional quantile head of about 30M
parameters. How the checkpoint is packaged, and whether the quantile head is
on, both matter.

### How it works and what it predicts

TimesFM is a **decoder-only, patched transformer** built for forecasting.
TimesFM 2.5 has fewer base parameters than 2.0 (200M instead of 500M) and a
longer context (up to 16k observations). It no longer needs the frequency
indicator that earlier versions had. Its optional quantile head supports
horizons of about 1k observations. That does not mean it can forecast any
horizon reliably.

The original TimesFM paper describes the decoder-only family, not every detail
added in 2.5. Cite the official 2.5 release notes as well.

### Extra variables and wrapper limits

The native TimesFM 2.5 library supports covariates through **XReg**, an
external regression step. That is not the same as the cross-variable attention
in a joint multivariate model.

Our `TimesFM2Forecaster` uses Hugging Face transformers. It returns point
forecasts and quantiles. In sktime 1.3.0 it has **no support for extra
variables**, so passing `X` does not switch on the native XReg. Forecasting
several columns one by one is also not evidence that the model learned how the
series affect each other. We use one target series to keep this simple.

The adapter works with both the 2.0 and the 2.5 checkpoint families. Its
minimum transformers version is too broad to guarantee that the 2.5 class
exists, so the code checks for a newer transformers version. Our pinned
versions mean we do not rely on the old transformers that Colab ships with.

**Start and sources**

Official code and 2.5 release notes:
https://github.com/google-research/timesfm

Transformers checkpoint:
https://huggingface.co/google/timesfm-2.5-200m-transformers

Transformers 2.5 documentation:
https://huggingface.co/docs/transformers/model_doc/timesfm2_5

Paper, *A decoder-only foundation model for time-series forecasting*:
https://arxiv.org/abs/2310.10688

Pinned wrapper:
https://github.com/sktime/sktime/blob/v1.3.0/sktime/forecasting/timesfm2.py

**Version warning:** TimesFM 3.0 now exists. It adds native multivariate and
covariate support. We deliberately stay on 2.5. Do not upgrade without
checking: features and licenses differ. The official repository says weights up
to 2.5 stay Apache-2.0, while the downloaded 3.0 weights have a separate
non-commercial license. Check the current terms for your own use.

## 6. t0-gamma

**Developer:** The Forecasting Company. **Checkpoint:**
`theforecastingcompany/t0-gamma`, about 256M parameters. This is not t0-alpha
(about 102M). Use `tfc-t0[torch]>=0.6.0`; earlier releases do not read all the
gamma settings.

### How it works and what it predicts

t0 is a **decoder-style patch transformer that alternates time attention and
covariate attention**. The gamma description lists 24 layers, an embedding size
of 1024, 8 attention heads and a patch size of 32. It predicts many steps ahead
in parallel, up to 1024 steps in one pass, and continues step by step beyond
that.

Gamma has 21 trained quantile levels from 0.01 to 0.99. The native code
interpolates between them and uses exponential tails beyond them. So treat very
extreme quantiles with extra care.

Native t0 supports multivariate history, covariates known for the future,
batches of different lengths and missing values. Current native inference also
offers options based on the history, such as clipping to nonnegative values,
rounding to integers and symmetric averaging. These options change the
experiment. We do not claim to use them through an adapter that does not offer
them.

### What sktime offers

`T0Forecaster` defaults to **alpha**, so our factory picks gamma explicitly.
The adapter's point forecast is the median. It has quantile and interval
methods, and it supports missing values and covariates known for the future.
For `X`, give it values for the history and for the whole forecast horizon.

With several target columns it forecasts each column **on its own**. The
multivariate tag does not mean the wrapper offers the native model's joint
multivariate attention. Our single-series exercise does not pretend to test
that.

The wrapper needs `license_accepted=True`. Its text still describes the older
gated access, while the current official tfc-t0 README says alpha, beta and
gamma are public, download without login and are Apache-2.0. We keep the
explicit acceptance, do not ask for a token by default, and tell participants
to check the current terms. Outdated access text in a wrapper does not show
that a model is gated today.

**Start and sources**

Official code, architecture table, license, and gamma version requirement:
https://github.com/theforecastingcompany/tfc-t0

Official inference quickstart:
https://github.com/theforecastingcompany/tfc-t0/blob/main/notebooks/01_inference_quickstart.ipynb

Checkpoint:
https://huggingface.co/theforecastingcompany/t0-gamma

Report, *t0: A Time-Series Foundation Model for Forecasting with Context*:
https://arxiv.org/abs/2609.24559

Pinned wrapper:
https://github.com/sktime/sktime/blob/v1.3.0/sktime/forecasting/t0.py

## 7. The three models at a glance

This table is specific to these versions and interfaces. It is not a ranking of
features. What a model can do natively does not automatically carry over into
every wrapper.

| Property | Chronos-2 | TimesFM 2.5 | t0-gamma |
|---|---|---|---|
| Architecture | Encoder-only patch transformer; time and group attention | Decoder-only patch transformer | Decoder-style patch transformer; time and covariate attention |
| Approximate base size | 120M | 200M; the optional quantile head adds more | 256M |
| Native probabilistic output | Quantiles | Quantiles, with the right head and configuration | Quantiles, including the median |
| Native multi-variable and covariate support | Joint multivariate targets and covariates | XReg covariates (not the same as joint multivariate attention) | Multivariate history and covariates known for the future |
| sktime intervals here | No | Yes | Yes |
| sktime extra variables here | Yes | No | Yes, with values for the history and the whole forecast horizon |
| Main wrapper caveat | Native quantiles are not offered as intervals | Use the transformers 2.5 checkpoint; no XReg through this adapter | Defaults to alpha; several target columns are processed independently |
| In the workshop | Your chosen model, run live or from saved results | Your chosen model, run live or from saved results | Your chosen model, run live or from saved results; license terms apply |

A CPU can run these libraries, but "can run" and "runs comfortably for every
participant within 90 minutes" are different things. Speed depends on the
hardware, the precision, the context length, the package versions, and whether
the model is loaded for the first time or already cached. All models get the
same raw history. Each architecture patches and scales it differently inside.

## 8. Kumo Forecast: not in the workshop notebooks

**NVIDIA Kumo-Forecast** is part of **Kumo-TS** (formerly NV-Tesseract). It is a
time-series model and library. Do not confuse it with Kumo Relational.

The implementation we looked at uses a patched **T5-encoder / MOMENT-style
backbone** with a forecasting head. The model builder refers to
`AutonLab/MOMENT-1-large`. The SDK supports multivariate input and optional
cross-channel attention. The published forecast checkpoint typically uses a
512-step context and predicts 72 steps natively. Longer horizons repeat the
forecast step by step. These settings belong to that checkpoint.

An extra **DARR** mode combines direct predictions with similar stretches of
past history that are retrieved (nearest neighbors), together with what
followed them. What you allow it to retrieve is part of the forecasting system.
In a fair backtest the retrieval memory may only hold what was known at each
cutoff, including the retrieved continuations. If a retrieved example continues
into the test days, that is leakage.

The SDK also has preprocessing, optional fine-tuning and interpretability tools,
plus direct and hybrid point forecasts. Do not assume the interpretability
output is a calibrated uncertainty interval. Check how it handles missing data
and where its standardizer comes from before you use it on your own data.

### Why it is not in the live notebooks

**It does not strictly need an NVIDIA GPU.** The current Kumo-TS documentation
describes CUDA and MPS use and a CPU fallback. Its forecasting requirements
recommend a GPU but allow a CPU. NVIDIA's separate Structured Data Models route
is built for GPUs. These are different ways to install it, so do not mix them up.

The upstream forecasting guide recommends a lot of disk space, and says that
DARR combined with interpretability can need far more memory than basic
forecasting. The setup, the extra files, the modes and the runtime needs make it
a poor extra dependency for a 90-minute session. We left it out to keep the
session small and reliable. We are not saying it cannot run on a CPU.

### Getting started after the workshop

Use a separate environment and follow the official Kumo-Forecast guide. Start
with the basic DataFrame forecast and leave DARR, interpretability and
fine-tuning off. Look at the format of the output and match it to the exact
dates in this workshop before you try the same three-fold backtest. The
official quickstart uses:

```python
# Tutorial only: requires the separately installed Kumo-Forecast environment.
from sdk.forecasting import ForecastingConfig, perform_forecasting

config = ForecastingConfig(
    timestamp_column="timestamp",
    target_column="target",
    forecast_horizon=14,
)
# training_frame must contain ONLY history available at the forecast cutoff.
result = perform_forecasting(df=training_frame, config=config)
```

This is a sketch of the official entry point. It is not a runnable workshop
cell, and not a tested drop-in sktime model.

Official project:
https://github.com/NVIDIA/Kumo-TS

Installation, hardware requirements, and quickstart:
https://github.com/NVIDIA/Kumo-TS/blob/main/Kumo-Forecast/README.md

SDK and DARR details:
https://github.com/NVIDIA/Kumo-TS/blob/main/Kumo-Forecast/sdk/README.md

Checkpoint:
https://huggingface.co/nvidia/Kumo-Forecast

Forecasting paper linked by NVIDIA, *Memory-Augmented Forecasting:
Scalability and Generalization Across Temporal Domains*:
https://ieeexplore.ieee.org/abstract/document/11402442

Backbone-family background, *MOMENT: A Family of Open Time-series Foundation Models*:
https://arxiv.org/abs/2402.03885

Do not cite the Kumo-Anomaly diffusion paper as if it were the architecture
paper for Kumo-Forecast.

## 9. Kumo Relational and the wider family

**Kumo Relational** predicts from related tables, not just from a regularly
sampled series. Example: from tables of customers, orders and products, predict
which customers will buy in the next 30 days. Keys link the rows into a graph.
The task is defined by the entity, the prediction time, the query and the
relational context.

The NVIDIA `KumoRelational` implementation we looked at has three stages:
**row embeddings inside each table**, **message passing between related rows
with a graph neural network (GNN) that works on any schema**, and
**in-context learning over the whole dataset**, using labeled context examples
and query examples. Its source calls it an adaptation of KumoRFM-2 and uses
TabICLv2-style embedding and in-context-learning parts. This is a different
architecture and a different data problem from a single-series patch
forecaster. In that interface it can return classification logits or
regression quantiles.

The `kumo-relational-client` library calls a **NIM inference endpoint**. The
client downloads no model weights and needs no local GPU: the server running
the endpoint provides the hardware. Hosted endpoints can need an account or an
API key, and they receive the data you send. Self-hosting has its own GPU,
deployment and license requirements. Do not send private relational data to a
service just to try a workshop example.

Keep these apart: **Kumo Engine** is infrastructure that builds relational
context and can train a GNN on your data. **Kumo Relational** is a pretrained
model. Neither is another name for Kumo Forecast.

The NVIDIA family also includes **Kumo Tabular** for tables and **Kumo Anomaly**
for finding anomalies in time series. Beyond that, there are other forecasting
foundation models, for example Moirai, Toto, TiRex, Chronos-Bolt and newer
TimesFM versions. Their tasks, covariate support, licenses and ways of adapting
them differ. A longer list of models does not make a better workshop.

Official relational client and endpoint instructions:
https://github.com/kumo-ai/kumo-relational-client

NVIDIA implementation and model-family/paper references:
https://github.com/NVIDIA/structured-data-models

Architecture source:
https://github.com/NVIDIA/structured-data-models/blob/main/sdm/models/kumo/relational/model.py

Paper, *KumoRFM-2: Scaling Foundation Models for Relational Learning*:
https://arxiv.org/abs/2604.12596

Product/serving documentation:
https://docs.nvidia.com/sdgm/rfm/overview

Engine, not the same thing as the pretrained relational model:
https://github.com/NVIDIA/kumo-engine

### Timestral (Mistral AI): mentioned, not used

Mistral AI's **Timestral** is a family of pretrained forecasting models with
5M, 28M, 123M and 209M parameters. It is a patch-based transformer with time
and variate attention. It reads up to 32,768 steps of context, returns 21
quantiles, and supports covariates seen only in the past and covariates known
for the future. Mistral reports state-of-the-art zero-shot results on
GIFT-Eval, TIME and fev-bench. These are the vendor's claims, not our
benchmark, and the tech report is listed as "coming soon".

**Many of us may not be allowed to use it, possibly including the workshop
instructor.** It has a "Modified MIT" license. Under it, you are not allowed to
use any of the rights if the global consolidated monthly revenue of your
company, or of your employer, was above $20 million in the previous month. The
limit depends on the size of the company, not on whether your use is
commercial. So it can apply to employees who use it for personal or learning
purposes. Read the license yourself and ask your legal team before you use it.
Larger companies are sent to Mistral for a commercial license or its hosted
service.

**It is also not in sktime yet.** sktime 1.3.0, the version we use, has no
Timestral wrapper, so it cannot go through our shared `evaluate` benchmark. A
wrapper has been proposed (issue #11568, pull request #11569, both open when
this was written). Check their status for what is available now. For these two
reasons we neither run nor teach it in the workshop.

Model card: https://huggingface.co/mistralai/Timestral-209M

License text: https://huggingface.co/mistralai/Timestral-123M/blob/main/LICENSE

Announcement: https://mistral.ai/news/timestral/

Code: https://github.com/mistralai/timestral/

sktime proposal: https://github.com/sktime/sktime/issues/11568 and
https://github.com/sktime/sktime/pull/11569

## 10. Fair evaluation and what to conclude

Use **time-based** splits, never random ones. Scaling, filling gaps, choosing
parameters, building features and any retrieval memory must only use what was
known at the cutoff. Do not use actual future values of a covariate when, in
real life, you would only have had a forecast of it.

The notebook uses a **rolling context** (a fixed 168 days), not a growing one.
The three 14-day test windows do not overlap. `evaluate` uses the `refit`
strategy: the model is fitted again in every fold. Metrics are computed on
every day of the horizon. We average the MAE of the folds and the
fold-specific seasonal MASE, look at each fold, and plot the forecasts.
Section 3.2 of the notebook compares MAE, RMSE, MAPE, WAPE and MASE.

For the training values `y[0], ..., y[n-1]` of a fold, the MASE denominator is:

```text
scale = mean(abs(y[t] - y[t-7])) for t = 7,...,n-1
MASE = test_MAE / scale
```

A MASE below 1 compares with the **seasonal-difference scale on the training
data**, not with the seasonal-naive forecast on the test days. For a direct
comparison on the test days:

```text
improvement (%) = 100 * (1 - model_test_MAE / seasonal_naive_test_MAE)
```

A constant series has a MASE scale of zero. A real benchmark needs a clear rule
for that, not silent infinities. Our data has nonzero scales. Folds of equal
length make the mean MAE directly comparable. Timings differ with hardware and
cache state, so treat them as rough.

Evaluating probabilistic forecasts is a useful extension: pinball loss,
interval coverage and interval width answer different questions than the point
MAE. An 80% interval should cover about 80% of the actual values over many
suitable cases. It is not a guarantee for a single point, and not necessarily
an interval for the whole forecast path at once.

What not to claim:

- Do not call the model with the lowest point-forecast MAE "best at
  uncertainty".
- Do not call three folds a significance test.
- Do not tune on these folds and then report the better score as a clean final
  test.
- Do not rank a model that failed as having zero error. **Not evaluated** is a
  valid result.

The final question is a decision: **is the improvement, if any, worth the run
time, the dependencies, the effort to operate it and the uncertainty, for your
use case?**

### Extra variables (covariates) to try next

The workshop data has only the date and the number of births. The original
FiveThirtyEight file adds nothing more: year, month, day of month and weekday
are all calendar fields that we can compute from the date.

An extra variable only helps if you **know its value in advance**, for the whole
forecast horizon. Ideas:

- **Weekday, month, day of month.** Computed from the date. The weekly pattern
  is already in the seasonal models, but an explicit weekday column is the
  simplest first try.
- **Public holidays.** The best idea for this data. pandas has
  `pandas.tseries.holiday.USFederalHolidayCalendar`, so nothing needs to be
  downloaded. Two holidays fall in our test days: Thanksgiving (27 November
  2014, fold 1) and Christmas Day (25 December 2014, fold 3). Births were about
  7,400 and 6,700 on those days, against about 12,500 and 12,800 on the same
  weekday a week earlier. Every model in our backtest over-forecast them by
  4,400 to 5,700. Add a holiday flag, and maybe "day before" and "day after"
  flags: 24 and 26 December were low too.
- **Long weekends and bridge days**, for example when July 4 falls on a Friday.

Not allowed: weather, or any other series, observed during the forecast period.
At the cutoff you would only have a forecast of it.

In our setup, ARIMA (`SARIMAX`), Chronos-2 and t0-gamma accept extra variables.
Seasonal naive, ETS and the TimesFM 2.5 adapter do not. For a fair comparison,
give every model that can use them the same information, and say clearly which
models could not.

Evaluation sources:
https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.model_evaluation.evaluate.html

Temporal cross-validation background:
https://otexts.com/fpp3/tscv.html

Accuracy measures:
https://otexts.com/fpp3/accuracy.html