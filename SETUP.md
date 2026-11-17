# Setup and troubleshooting

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
warm-up: `scripts/check_environments.py` verifies imports, and each model's
weights download the first time it runs.

## Google Colab (option 1: nothing to install locally)

Upload `00_setup_check.ipynb`, then run its first cell and select the setup ZIP
(on the day, the same cell in notebook 1 and 2 takes the full ZIP).
The archive must contain the top-level `tsfm-workshop/` folder. This puts the
data, helpers, requirements, and saved results next to the notebook runtime.
Use only the organizer's trusted ZIP.

Install with the notebook's installation cell (set the flags to `True`), or:

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
!python scripts/check_environments.py --foundation
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
uv run python scripts/check_environments.py
uv run jupyter lab
```

For foundation models:

```bash
uv sync --locked --extra foundation
uv run python scripts/check_environments.py --foundation
uv run jupyter lab
```

After `uv sync --locked --extra foundation`, plain `uv run` keeps the foundation
packages (we checked that on one machine). A later plain `uv sync`, however,
removes them; rerun the sync with `--extra foundation` if that happens.

The lock fixes the resolved local dependency graph;
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
.venv-cpu/bin/python scripts/check_environments.py --foundation
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
python -m pip install -r requirements/core.txt -r requirements/jupyter.txt
python scripts/check_environments.py
python -m jupyterlab
```

`requirements/jupyter.txt` (JupyterLab and kernel tools) is only for local venvs;
on Colab it is deliberately not installed, because it would replace Colab's own
kernel packages and trigger the dependency-conflict errors.
Optionally add `python -m pip install -r requirements/foundation.txt`.
For Prophet only: `python -m pip install -r requirements/prophet.txt`.
Prophet adds a Stan backend and possible platform friction, so it is bonus
material rather than a prerequisite. In uv use `--extra prophet`.

## Model downloads

On our Linux test machine the foundation libraries (PyTorch and its CUDA
packages included) took about 5 GB, and the three models' weights are about
2.3 GB (Chronos-2 about 0.46 GB, TimesFM 2.5 about 0.88 GB, t0-gamma about
0.98 GB). Plan for **8–10 GB of free disk**, and preferably 16 GB RAM. Run one
model at a time.

The factories explicitly choose `amazon/chronos-2`,
`google/timesfm-2.5-200m-transformers`, and
`theforecastingcompany/t0-gamma`; wrapper defaults are not interchangeable.
The t0 adapter defaults to alpha, and gamma needs `tfc-t0>=0.6.0`.
TimesFM 2.5's transformers implementation needs a newer release than the
adapter's broad minimum: this workshop pins transformers 5.23.0.

**Offline after the first download.** Once a model's weights are cached,
`workshop.py` automatically switches Hugging Face to offline mode for that
model, so later runs make no Hub requests (and show no "unauthenticated
requests" warning). The first run of each model needs internet; so do models
not yet cached. To force offline mode yourself, set `HF_HUB_OFFLINE=1` before
starting Python or Jupyter; an explicit setting is never overridden. Red text in
a notebook cell is only stderr (progress bars, warnings), not necessarily an error.

If a model cannot run, the notebooks use the instructor's saved results and
label a missing model "not evaluated". That is a full way to participate.
t0-gamma needs you to read its license (Apache-2.0) and opt in explicitly.

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
