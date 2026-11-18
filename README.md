# Can AI Predict the Future?
## Exploring Time Series Foundation Models — a PyLadies workshop

**Two hours: 90 minutes hands-on + 30 minutes questions and open directions.**
Basic Python/pandas is assumed; no deep-learning or forecasting background is
required. Work in pairs if you like: one person drives, the other asks
"What would we actually know at forecast time?"

**Our question:** on one small daily dataset, does a zero-shot foundation
model beat a sensible seasonal baseline? We use
[sktime](https://github.com/sktime/sktime), a Python library with one common
interface for forecasting models, splitters and metrics, to compare them fairly.

## What's here

This repository holds everything you need **before** the workshop:

| File | Purpose |
|---|---|
| `PARTICIPANT_PREP.md` | **Start here**: setup instructions for local or Colab |
| `SETUP.md` | Reference: Colab, local uv/venv, model downloads, troubleshooting |
| `notebooks/00_setup_check.ipynb` | Install and readiness check (no exercises) |
| `workshop.py` | Shared model factories and the single `sktime.evaluate` benchmark |
| `data/` | The bundled public dataset (daily U.S. births) with attribution |
| `scripts/` | `check_environment.py` (readiness check), `download_weights.py` (fetch model weights) and `rehearse.py` (run one model) |

The lesson notebooks, slides, and a tutorial guide are shared on the day.
Exercise cells there are marked **YOUR TURN** and contain `None` placeholders
with helpful checks. **RUN THIS** cells are prewritten. Hints and expandable
solutions follow each exercise. Progress is private, rerunnable, and based on
finishing a learning step, not on beating another participant.

## Get started: follow [PARTICIPANT_PREP.md](PARTICIPANT_PREP.md)

Repository: https://github.com/irenkamk/tsfm-workshop-sktime

Choose **local** (if you already use an IDE or have Python set up) or **Colab**
(nothing to install; good for beginners). Both cover the whole workshop and no
GPU is needed.

- **Local:** clone the repository, then follow Option 1 in
  `PARTICIPANT_PREP.md`:

  ```bash
  git clone https://github.com/irenkamk/tsfm-workshop-sktime.git
  cd tsfm-workshop-sktime
  ```

  The setup there also downloads the model weights, so install everything
  before the workshop.
- **Colab:** on the repository page click the green **Code** button →
  **Download ZIP**, then follow Option 2 in `PARTICIPANT_PREP.md` just to get
  familiar with it; the requirements are installed on the day.

The local setup installs the foundation-model libraries too, which needs
8–10 GB of free disk; `PARTICIPANT_PREP.md` explains this and a lighter route.
`SETUP.md` has the details and troubleshooting.

## Licenses

Model/software licenses differ from the bundled data license. See
`data/README.md` and the upstream sources in `TUTORIAL.md`.
