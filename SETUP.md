# Setup and rehearsal

## What sktime does—and does not do

`sktime` supplies a shared estimator interface, temporal splitters, metrics,
and evaluation. Each wrapper delegates inference to a different library.
`fit()` estimates parameters for the classical models; for these zero-shot
wrappers it loads pretrained weights and stores the observed context, without
training new weights on our dataset.

There are three separate stages: **install Python libraries → download model
weights → run inference**. An import check verifies only the first.
Hugging Face normally caches downloaded weights under
`~/.cache/huggingface/hub` (or `HF_HOME`). Installing libraries alone is not a
warm-up. The rehearsal commands below verify an actual forecast and all folds.

## Google Colab (option 1: nothing to install locally)

Upload notebook 1, then run its bootstrap cell and select the workshop ZIP.
The archive must contain the top-level `tsfm-workshop/` folder. This puts the
data, helpers, requirements, and saved results next to the notebook runtime.
Use only the organizer's trusted ZIP.

Install core dependencies with the notebook's installation cell, or:

```python
%pip install -r requirements/core.txt
```

Before importing forecasting packages, optionally install the model extra:

```python
%pip install -r requirements/foundation.txt
```

Restart the runtime session after dependency changes if imports were already
loaded. Do not factory-reset/delete the runtime unless needed; that removes
files and caches. Rerun the bootstrap cell, leave installation flags false,
and check:

```python
!python scripts/check_environment.py --foundation
```

Use Colab's preinstalled CUDA-enabled PyTorch if it satisfies the requirement.
Do **not** install a CPU-only torch wheel in your Colab GPU runtime. If
`pip check` reports conflicts involving the workshop packages, start with a
fresh runtime and repeat the documented install, rather than randomly
downgrading transformers. Colab's unrelated preinstalled packages can also
report conflicts; record them and confirm the actual adapter checks/rehearsal.

The two notebooks are independent. A new notebook may have a new VM: upload
the ZIP and install again. Colab does not promise persistent disks, fixed GPU
types, sufficient free resources, or a particular Python version. Use the
instructor results when access fails. No Colab GPU subscription is required.

## Local uv

From the extracted workshop folder:

```bash
uv sync --locked --python 3.12
uv run python scripts/check_environment.py
uv run jupyter lab
```

For foundation models:

```bash
uv sync --locked --extra foundation
uv run --extra foundation python scripts/check_environment.py --foundation
uv run --extra foundation jupyter lab
```

Keep `--extra foundation` on subsequent `uv run` commands, or uv may resync
without that extra. The lock fixes the resolved local dependency graph;
Colab's requirements files pin the direct libraries but deliberately permit
its compatible preinstalled torch. The two environments are not bit-identical.
Record versions in the rehearsal artifacts rather than assuming equivalence.

On local Linux, the default torch distribution may also install large CUDA
libraries even when you use CPU. For CPU-only optional exploration, use a
separate venv and the official CPU wheels, then install the requirements:

```bash
uv venv --python 3.12 .venv-cpu
uv pip install --python .venv-cpu/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-cpu/bin/python -r requirements/core.txt -r requirements/foundation.txt
.venv-cpu/bin/python scripts/check_environment.py --foundation
```

On Windows use `.venv-cpu\Scripts\python.exe`. This CPU-wheel route does not
use `uv sync`, which could replace torch using the default lock's sources.
CPU foundation inference is an optional slow path, not a 90-minute guarantee.
For Apple silicon, baseline mode works; torch/accelerator support is model-
and version-specific. Do not make MPS debugging part of this workshop.

## Plain venv alternative

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements/core.txt
python scripts/check_environment.py
python -m jupyterlab
```

Optionally add `python -m pip install -r requirements/foundation.txt`.
For Prophet only: `python -m pip install -r requirements/prophet.txt`.
Prophet adds a Stan backend and possible platform friction, so it is bonus
material rather than a prerequisite. In uv use `--extra prophet`.

## Model downloads and instructor fallback

A 120M-parameter fp32 model has about 0.48 GB of raw parameters; 200M has about
0.8 GB; 256M about 1 GB. These are arithmetic estimates, **not measured peak
RAM or download sizes**. Frameworks, activations, caches and model shards
add overhead. Plan for 10–15 GB free disk for the full stack and cached models,
and preferably 16 GB host RAM. Run one model at a time.

The factories explicitly choose `amazon/chronos-2`,
`google/timesfm-2.5-200m-transformers`, and
`theforecastingcompany/t0-gamma`; wrapper defaults are not interchangeable.
The t0 adapter defaults to alpha, and gamma needs `tfc-t0>=0.6.0`.
TimesFM 2.5's transformers implementation needs a newer release than the
adapter's broad minimum: this workshop pins transformers 5.23.0.

In Colab, run the following in cells prefixed with `!`. Locally use
`uv run --extra foundation python` instead of `python`:

```bash
python scripts/rehearse.py --model chronos2 --device cuda --output artifacts/chronos2
python scripts/rehearse.py --model timesfm25 --device cuda --output artifacts/timesfm25
# Only after reading/accepting the tfc-t0 and t0-gamma license:
python scripts/rehearse.py --model t0-gamma --device cuda --accept-t0-license --output artifacts/t0-gamma
```

Each command is a new process, downloads/caches weights if needed, computes
the demonstration and three backtest folds, and writes `demo.csv`,
`scores.csv`, `predictions.csv`, and `metadata.json`. The metadata records
versions, device, data hash, protocol, and the cached checkpoint revision.
An existing output directory is rejected to prevent overwriting your evidence;
use a fresh directory for a new run.

**Offline after the first download.** Once a model's weights are cached,
`workshop.py` automatically switches Hugging Face to offline mode for that
model, so later runs make no Hub requests (and show no "unauthenticated
requests" warning). The first run of each model needs internet; so do models
not yet cached. To force this yourself, set `HF_HUB_OFFLINE=1` before starting
Python or Jupyter; an explicit setting is never overridden. Red text in a
notebook cell is only stderr (progress bars, warnings), not necessarily an error.

Do not substitute randomly initialized models, synthetic curves, or vendor
benchmark scores for a failed model. Failures are visible; no result means
"not evaluated". A model can be discussed without giving it a leaderboard rank.
The factories load Hub `main`; record the resolved revision and preserve that
cache/results for the session. For stronger future reproducibility, download
the recorded revision into a local snapshot and use its path in the factory.

Save the instructor's real result folders into the participant ZIP, then
verify the saved-results cells in notebook 2. To repackage after rehearsal:

```bash
python scripts/package_workshop.py --output ../tsfm-workshop-ready.zip
```

This packages only workshop files, not your environment, credentials,
downloaded weights, notebook outputs, or arbitrary local files. Distribute
checkpoint licenses/weights only if you have independently checked their terms.

## Troubleshooting and timeboxes

| Symptom | Action |
|---|---|
| `No module named workshop` | Run the bootstrap; confirm current folder contains `workshop.py`. |
| TimesFM class missing | Check transformers version, restart after installation, run adapter check. |
| CUDA out of memory | Stop the run; use a fresh process/runtime for one model, or saved results. `del` alone may not release adapter caches. |
| Download stalls / DNS failure | Check connectivity; use precomputed results after two minutes in class. |
| HTTP 401/403 | Check current model card/access terms. Authenticate privately only if actually required. |
| t0 license error | Read the upstream license and opt in explicitly; never bypass with `ignore_deps`. |
| ARIMA convergence warning | Keep it visible. Discuss the fixed specification; do not secretly retune on test folds. |
| ZIP is lost after reset | Upload again; local paths and HF cache are not persistent in Colab. |
| All models beat/lose to naive | That is a valid local result. Explain the protocol and its limitations. |

## Using the setup skill

The included skill is a text artifact, not an account setting. For an agent
that supports repository skills, keep
`.github/skills/prepare-tsfm-workshop/SKILL.md` in the workshop repository.
For an agent with user-level skills, copy that folder to the user skill
directory documented by that agent. Do not assume every chat client loads
skills or has terminal permissions.

Then say: **"prepare me for foundation models for time series workshop"**.
The skill checks the chosen environment, installs only the chosen track,
checks readiness, and asks before accepting terms or downloading weights.
It does not create a paid service, expose a notebook server, or change global
Python settings.