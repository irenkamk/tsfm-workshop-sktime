# Can AI Predict the Future? A practical model guide

**Reading date: 15 November 2026.** This guide deliberately teaches the
requested Chronos-2, TimesFM **2.5**, and **t0-gamma**, not whichever checkpoint
is newest. Package pins and wrapper behavior refer to sktime **1.3.0**.
Upstream software, model cards, access rules, and hosted services can change.
Official repositories/source were inspected; links to papers and model cards
are supplied below. Hugging Face and some external documentation hosts were
unreachable from the authoring environment, so this is not a claim that every
linked page or pretrained inference path was independently exercised.

## 1. What question are we actually answering?

A forecast is a conditional estimate of future outcomes given the information
available at a particular time. It is not knowledge of the future.
Before choosing an algorithm, specify the target, frequency, forecast origin,
horizon, information available at that origin, and the cost of being wrong.

Our teaching question: **given the preceding 168 daily U.S. birth counts,
forecast the next 14 days**. We compare models on three later origins with
the same context and no external covariates. We use one small, complete,
attributed dataset, bundled so data access cannot derail the session.

Daily sampling is not the same as weekly seasonality. Aggregating to weekly
totals would reduce the number of observations but remove the weekday/weekend
pattern. Daily data plus a short context makes both the task and the baseline
more intuitive.

## 2. Foundation model does not mean "no assumptions"

A **local model** learns its parameters from the target series. A **global
model** learns across multiple series. A **foundation model** is pretrained
on a broad collection and designed for reuse across tasks or datasets.
These categories overlap; "foundation" is not a guarantee of superiority.

**Zero-shot forecasting** here means applying pretrained weights without
task-specific gradient updates. The model still consumes a context window.
It may scale values, split them into patches, attend over the context, and
produce points or quantiles. Loading a checkpoint is not training.
Calling an API method named `fit` does not by itself imply fine-tuning.

**Pretraining contamination:** old public datasets can appear in training
corpora. We cannot certify that this series was unseen during pretraining.
Leak-free workshop splits prevent our own look-ahead, not unknown exposure
in a vendor's pretraining pipeline.

**Patches** group consecutive numeric observations into tokens. **Attention**
lets representations use other relevant tokens. **Time attention** exchanges
information across time; **group/variate attention** can exchange it across
related variables. This is not the same as text tokenization, and not every
time-series foundation model follows the original Chronos design.

## 3. Don't skip the classical contestants

| Model | How it forecasts | What we use | Strengths and limitations |
|---|---|---|---|
| Seasonal naive | Repeat the last observed seasonal cycle | `NaiveForecaster(strategy="last", sp=7)` | Almost no setup; strong when weeks repeat; cannot anticipate new trends or special days |
| ARIMA/SARIMA | Linear autoregressive and moving-average dynamics; differencing is optional | `SARIMAX(order=(1,0,1), seasonal_order=(1,0,0,7), trend="c")` | Interpretable classical baseline; order selection and stationarity assumptions matter |
| Exponential smoothing | Recursively update level, trend and seasonal components, favoring recent observations | Additive damped trend + additive weekly seasonality | Fast and useful for smooth evolving seasonal patterns; unusual events remain difficult |
| Prophet (bonus) | Additive trend plus Fourier seasonalities and optional holiday/regressor terms | Weekly on; daily/yearly off | Convenient calendar model; Stan dependency and tuning overhead make it optional |

ARIMA's "I" means integrated/differenced; our fixed example uses `d=0,D=0`.
It is still in the ARIMA family. We deliberately avoid an expensive automatic
search. The sktime `SARIMAX` wrapper uses statsmodels and avoids an additional
pmdarima dependency. The experiment is not a comparison against the best
possible tuned classical model.

Prophet supports additional regressors, but they need suitable future values.
Observed future weather is not a legitimate known-future regressor. Calendar
features genuinely known at the origin are different.

Sources: [sktime naive](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.naive.NaiveForecaster.html),
[statsmodels SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html),
[exponential smoothing](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html),
[Prophet getting started](https://facebook.github.io/prophet/docs/quick_start.html),
[Forecasting at Scale paper](https://peerj.com/preprints/3190/).
For more background: [Forecasting: Principles and Practice](https://otexts.com/fpp3/).

## 4. Chronos-2

**Developer:** Amazon. **Workshop checkpoint:** `amazon/chronos-2`, 120M
parameters. Do not confuse it with original Chronos-T5 or Chronos-Bolt.

### Architecture and predictions

Chronos-2 is a **patch-based, encoder-only transformer**. It combines temporal
modeling with a group-attention mechanism that allows related targets and
covariates to interact. It supports univariate, multivariate, and
covariate-informed forecasting in one pretrained model, and natively produces
quantile forecasts. The published default context is up to 8192 observations.
That maximum is not a recommendation to use all of it for every task.

Original Chronos scaled and quantized numeric values into discrete tokens
and used language-model-style training and sampled forecast paths.
Chronos-Bolt moved to patching and direct multi-step quantiles.
Describing Chronos-2 simply as "T5 predicting discretized numbers" would
conflate different generations.

### Native capabilities versus our wrapper

Native Chronos-2 handles multiple target variables, past-only covariates and
known-future covariates. Cross-learning across series is an additional option,
not automatically equivalent to joint forecasting every unrelated series.

The sktime 1.3.0 `Chronos2Forecaster` exposes multivariate and exogenous support,
but advertises **no prediction-interval capability** and **no missing-value
capability**. Therefore our notebook asks it only for complete-data point
forecasts. Do not infer that `predict_interval` works just because the
underlying model has quantiles.

The notebook uses an explicit 168-day context, batch size 1, and float32
for simplicity. A native default may choose another dtype. Optimizing
precision and batch size is not a beginner exercise.

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
`google/timesfm-2.5-200m-transformers`. The 2.5 base has 200M parameters;
upstream describes an optional approximately 30M continuous-quantile head.
The checkpoint packaging and quantile-head configuration matter.

### Architecture and predictions

TimesFM is a **decoder-only, patched time-series transformer** trained for
forecasting. TimesFM 2.5 reduces the base parameter count relative to 2.0
(200M versus 500M) while extending supported context to 16k observations.
It drops the earlier explicit frequency indicator. Its optional continuous
quantile head supports horizons up to about 1k observations; do not interpret
that as an unlimited long-horizon guarantee.

The foundational TimesFM paper describes the decoder-only family, not every
implementation detail added in 2.5. Cite the official 2.5 release notes as well.

### Covariates and wrapper limits

The native TimesFM 2.5 library has **XReg** support: an external regression
mechanism for covariates. That is not the same as native cross-variate
attention in a joint multivariate model.

Our `TimesFM2Forecaster` uses Hugging Face transformers, supports point and
quantile predictions, and advertises **no exogenous support** in sktime 1.3.0.
Passing `X` is not a way to activate native-library XReg here.
Forecasting columns independently is also not evidence of learned cross-series
interactions. We use a single target so the distinction stays manageable.

The adapter can support both 2.0 and 2.5 checkpoint families. Its broad
transformers dependency minimum is insufficient to guarantee that the 2.5
class exists; the code specifically checks newer transformers support.
Our pins avoid relying on an old Colab transformers install.

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

**Version warning:** TimesFM 3.0 now exists and adds native multivariate and
covariate functionality. We intentionally retain 2.5. Do not silently upgrade:
capabilities and licenses differ. The official repository says weights up to
2.5 remain Apache-2.0, whereas 3.0 downloaded weights have a separate
non-commercial license. Review current terms for your deployment.

## 6. t0-gamma

**Developer:** The Forecasting Company. **Checkpoint:**
`theforecastingcompany/t0-gamma`; approximately 256M parameters.
This is not t0-alpha (approximately 102M). Use `tfc-t0[torch]>=0.6.0`;
earlier releases do not read all gamma configuration conventions.

### Architecture and predictions

t0 is a **decoder-style patch transformer alternating time and covariate
attention**. The current gamma description gives 24 layers, embedding
dimension 1024, 8 attention heads, and patch size 32. It decodes multiple
horizons in parallel, up to 1024 steps in one forward pass, with
autoregressive rollout beyond that.

Gamma has 21 trained quantile levels spanning 0.01–0.99. The native
implementation interpolates between trained quantiles and uses exponential
tails beyond them. More extreme quantiles therefore deserve particular
scrutiny rather than automatic confidence.

Native t0 supports historical multivariate context and known-future
covariates, variable-length batching, and missing observations. Current native
inference also offers history-based options such as nonnegative clipping,
integer snapping, and symmetric averaging. These options change the
experiment; we do not claim to use them through an adapter that does not
expose them.

### What sktime actually exposes

`T0Forecaster` defaults to **alpha**, so the factory explicitly selects gamma.
The adapter's point prediction is the median. It provides quantile and
interval methods, and supports missing values and known-future exogenous
inputs. For `X`, provide historical and future values with complete horizon
coverage.

Crucially, its multi-column target handling forecasts columns **independently**.
The multivariate capability tag does not imply the wrapper exposes the native
model's full joint multivariate attention. Our univariate exercise avoids
pretending to test that capability.

The wrapper requires `license_accepted=True`. Its text still describes older
gated access, while the current official tfc-t0 README says alpha, beta, and
gamma are public and download without authentication and identifies
Apache-2.0 licensing. We keep explicit acceptance, do not demand a token by
default, and direct participants to check the current terms. A wrapper's
outdated access text is not evidence that a model is currently gated.

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

## 7. Practical comparison at a glance

This table is **version- and interface-specific**, not a universal feature
ranking. Native capabilities do not automatically flow through every wrapper.

| Property | Chronos-2 | TimesFM 2.5 | t0-gamma |
|---|---|---|---|
| Architecture | Encoder-only patched transformer, time/group attention | Decoder-only patched transformer | Decoder-style patch transformer, time/covariate attention |
| Approximate base size | 120M | 200M; optional quantile head adds parameters | 256M |
| Native probabilistic output | Quantiles | Quantiles with appropriate head/configuration | Quantiles, including median |
| Native cross-variable/context support | Joint multivariate targets and covariates | XReg covariates; distinguish from joint multivariate attention | Multivariate historical context and known-future covariates |
| sktime interval interface here | No | Yes | Yes |
| sktime exogenous interface here | Yes | No | Yes, with required history/future coverage |
| Important wrapper caveat | Native quantiles not exposed as intervals | Use the transformers 2.5 checkpoint; no XReg through this adapter | Defaults to alpha; multi-column y processed independently |
| Live classroom plan | One assigned model or saved results | One assigned model or saved results | One assigned model or saved results, explicit license opt-in |

CPU execution can be possible for these libraries, but "possible" and
"comfortable for every learner in 90 minutes" are different requirements.
Actual speed depends on hardware, precision, context length, package versions,
and cold versus warm loading. The experiment supplies identical raw context;
each architecture has different internal patching and scaling.

## 8. Kumo Forecast: tutorial-only extension

**NVIDIA Kumo-Forecast**, part of **Kumo-TS** (formerly NV-Tesseract), is a
time-series model/library. It should not be confused with Kumo Relational.

The inspected implementation uses a patched **T5-encoder/MOMENT-style
backbone** with a forecasting head; the model builder references
`AutonLab/MOMENT-1-large`. The SDK supports multivariate inputs and optional
cross-channel attention configurations. The published forecast checkpoint
uses a typical 512-step context and native 72-step prediction window; longer
horizons use repeated rollout. Those settings are checkpoint-specific.

An additional **DARR** mode blends direct predictions with retrieved
nearest-neighbor historical continuations. The choice of context memory is
part of the forecasting system. A fair backtest must restrict that memory to
information available at each origin, including the retrieved continuation.
Retrieving examples whose continuations contain the test future would leak.

The SDK offers preprocessing, optional fine-tuning and interpretability
tools, as well as direct and hybrid point forecasts. Do not assume that
interpretability outputs are calibrated uncertainty intervals. Review
missing-data handling and standardizer provenance before using it on your data.

### Why it is not in the live notebooks

**It is not strictly NVIDIA-GPU-only.** Current Kumo-TS documentation describes
CUDA/MPS use and CPU fallback; its forecasting requirements recommend a GPU
but permit CPU. NVIDIA's separate Structured Data Models route is GPU-native.
These are different installation routes and should not be conflated.

The upstream forecasting guide recommends substantial disk capacity, and
states that combined DARR/interpretability examples can use far more memory
than baseline forecasting. Repository setup, extra assets, modes, and runtime
requirements make it a poor additional dependency for this 90-minute session.
Exclusion is a scope/reliability choice, not a claim that CPU is impossible.

### Getting started after the workshop

Use a separate environment and follow the official Kumo-Forecast guide.
Start with its basic DataFrame forecast and leave DARR, interpretability,
and fine-tuning off. Inspect the output schema and map it back to the exact
dates in this workshop before attempting the same three-fold protocol.
The official quickstart uses:

```python
# Tutorial only: requires the separately installed Kumo-Forecast environment.
from sdk.forecasting import ForecastingConfig, perform_forecasting

config = ForecastingConfig(
    timestamp_column="timestamp",
    target_column="target",
    forecast_horizon=14,
)
# training_frame must contain ONLY history available at the forecast origin.
result = perform_forecasting(df=training_frame, config=config)
```

This is a sketch of the official entry point, not a runnable workshop cell
or a validated drop-in sktime estimator.

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

**Kumo Relational** predicts from related tables rather than just a regularly
sampled vector. Example: from customers, orders, and products, predict which
customers will purchase in the next 30 days. Keys connect rows into a graph.
The entity, prediction timestamp, query, and relational context define the task.

The inspected NVIDIA `KumoRelational` implementation describes three stages:
**row embeddings within tables**, **schema-agnostic GNN message passing
between related rows**, and **dataset-level in-context learning** over labeled
context examples and query examples. Its source calls it an adaptation of
KumoRFM-2 and uses TabICLv2-style embedding/ICL components. This is a
different architecture and data problem from a univariate patch forecaster.
It can produce classification logits or regression quantiles in that interface.

The `kumo-relational-client` library calls a **NIM inference endpoint**.
The client itself downloads no model weights and requires no local GPU;
the serving endpoint owns inference hardware. Hosted endpoints can require
accounts/API keys and transmit supplied data; self-hosting has its own GPU,
deployment, and licensing requirements. Do not send private relational data
to a service merely to try a workshop example.

Keep separate:
**Kumo Engine** is infrastructure for relational context generation and can
train a GNN on your data; **Kumo Relational** is a pretrained model.
Neither is simply an alternate name for Kumo Forecast.

The current NVIDIA family also includes **Kumo Tabular** for tables and
**Kumo Anomaly** for time-series anomaly detection. More broadly, there are
other forecasting foundation models—Moirai, Toto, TiRex, Chronos-Bolt, and
newer TimesFM generations among them. Their tasks, covariate support, licenses,
and adaptation interfaces differ. A longer model list is not necessarily
a better workshop.

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

Mistral AI's **Timestral** is a family of pretrained forecasting models
(5M, 28M, 123M, and 209M parameters). It is a patch-based transformer with
time and variate attention, up to 32,768 steps of context, 21 quantile outputs,
and support for past-only and known-future covariates. Mistral reports
state-of-the-art zero-shot results on GIFT-Eval, TIME, and fev-bench; those are
vendor claims, not our benchmark, and its tech report is listed as
"coming soon".

**Many of us may not be allowed to use it, possibly including the workshop
instructor.** It is released under a "Modified MIT" license that says you are
not authorized to exercise any rights under it if the global consolidated
monthly revenue of your company, or of your employer, exceeded $20 million in
the preceding month. The cutoff is on the company's size, not on whether your
use is commercial, so it can apply to employees doing personal or learning
work. Read the license yourself and ask your legal team before using it.
Larger companies are directed to Mistral for a commercial license or its
hosted service.

**It is also not in sktime yet.** sktime 1.3.0, the version used here, has no
Timestral wrapper, so it cannot go through our shared `evaluate` benchmark.
A wrapper has been proposed (issue #11568, pull request #11569, both open when
this was written); check their status for current availability. For these two
reasons we neither run nor teach it in the workshop.

Model card: https://huggingface.co/mistralai/Timestral-209M

License text: https://huggingface.co/mistralai/Timestral-123M/blob/main/LICENSE

Announcement: https://mistral.ai/news/timestral/

Code: https://github.com/mistralai/timestral/

sktime proposal: https://github.com/sktime/sktime/issues/11568 and
https://github.com/sktime/sktime/pull/11569

## 10. Fair evaluation and honest conclusions

Use **temporal**, not random, splits. Any scaling, imputation, parameter
selection, feature construction, or retrieval memory must respect the
forecast origin. Do not use actual future covariates where only a forecast
would have been available in practice.

The notebooks use **rolling context** (fixed 168 days), not expanding context.
All three 14-day test windows are non-overlapping. `evaluate` uses refit
strategy; metrics are calculated on every horizon point. We average fold
MAE and fold-specific seasonal MASE, inspect each fold, and show forecasts.

For fold training values `y[0], ..., y[n-1]`, the MASE denominator is:

```text
scale = mean(abs(y[t] - y[t-7])) for t = 7,...,n-1
MASE = test_MAE / scale
```

MASE below one compares with an **in-sample seasonal-difference scale**, not
with the realized seasonal-naive test forecast. For a direct test comparison:

```text
improvement (%) = 100 * (1 - model_test_MAE / seasonal_naive_test_MAE)
```

Constant series can have zero MASE scale; a production benchmark needs an
explicit policy, not silent infinities. Our dataset has nonzero scales.
Equal-length folds make mean MAE directly comparable, but different hardware
and cache states make timing comparisons approximate.

Probabilistic evaluation is a useful extension: pinball loss, interval
coverage, and interval width ask different questions than point MAE.
An 80% interval should cover roughly 80% of relevant future outcomes over
many suitable cases; it is neither a guarantee on one point nor necessarily
a simultaneous interval for a whole forecast path.

Do not call a point-forecast MAE winner "best at uncertainty". Do not call
three folds a significance study. Do not tune on these folds and then report
the improved score as an untouched final test. And do not rank a failed
model as zero error: **not evaluated** is a meaningful status.

The final question is a decision: **is the improvement, if any, worth the
runtime, dependencies, operational burden, and uncertainty for this use case?**

Evaluation sources:
https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.model_evaluation.evaluate.html

Temporal cross-validation background:
https://otexts.com/fpp3/tscv.html

Accuracy measures:
https://otexts.com/fpp3/accuracy.html