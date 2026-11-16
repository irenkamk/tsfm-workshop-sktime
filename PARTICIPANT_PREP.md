Hi everyone! For our **Can AI Predict the Future?** PyLadies workshop,
please bring a laptop, charger, and a browser. We will use Python notebooks
to compare simple forecasts with pretrained AI models.

You only need basic Python: variables, functions, and a little pandas.
No prior forecasting knowledge, paid AI account, or personal GPU is required.
Pairing up is welcome. No one is expected to finish every optional challenge.

**There are two ways to join. Choose whichever you feel comfortable with.**
If you already work in an IDE such as VS Code or PyCharm, or have Python set up
on your laptop, stay **local** (option 2). If you have little coding experience
or nothing installed, go with **Google Colab** (option 1): it runs in your
browser and needs no installation on your machine. Both routes use the same
notebooks and cover the whole workshop.

Please allow **20–40 minutes before the event** for setup and downloads;
actual time depends on your internet speed.

## Option 1: Google Colab (nothing to install)

1. Open https://github.com/irenkamk/tsfm-workshop-sktime , click the green
   **Code** button → **Download ZIP**, and extract it. Keep the ZIP as well.
   Sign in to Google Colab: https://colab.research.google.com/
   (Please do not use Colab's "connect to GitHub" popup: it asks for access to
   all your repositories.)
2. Use **File → Upload notebook** to open `notebooks/00_setup_check.ipynb`
   from the extracted folder. It has no exercises; it only installs and checks.
3. **How to run a cell:** click inside it, then press **Shift+Enter** (or the
   ▶ button on its left). Run the first code cell. A **Choose Files** button
   appears under the cell: click it and select the ZIP you downloaded. Wait until it
   prints `Workshop files:`.
4. In the install cell, set `INSTALL_CORE = True` and
   `INSTALL_FOUNDATION = True`, then run it. This installs the libraries for
   running the pretrained models yourself (Colab already has PyTorch, so it is
   lighter than on a laptop). If Colab asks you to restart, use
   **Runtime → Restart session**, then rerun from the first cell with both
   flags `False`.
5. In the check cell, set `CHECK_FOUNDATION = True` and run it. It lists the
   installed versions. You should see **BASELINE READY**, then
   **FOUNDATION MODELS READY** (one line each for Chronos-2, TimesFM 2.5 and
   t0-gamma), then **✅ Setup OK**. That is enough to join. If you see **❌**, scroll up for the
   error and ask for help.
6. If the foundation install fails or is too slow, you can still join: leave
   `INSTALL_FOUNDATION` as `False`, and the notebooks will use the instructor's
   pre-computed forecasts. A GPU is not necessary (see below).

Colab sessions are temporary: files, installed packages, and downloaded weights
disappear when the runtime resets. On the day we will upload the ZIP and install
again together in the first minutes, so this step is only to make sure Colab
works for you. If you want to keep your work, use
**File → Download → Download .ipynb**.

## Option 2: On your own computer

You do **not** upload anything here: the notebook's upload step is only for
Colab.

**Please read this before you install.** The setup below installs everything,
including the foundation-model libraries (PyTorch and friends), so you can
**run the pretrained models yourself** in the workshop. You need
**8–10 GB of free disk space**, and the installation may take around 15 minutes
on a 100 Mbps connection (longer if yours is slower). The model weights, about
2.3 GB in total, download the first time each model runs. These are estimates
from one Linux machine, not guarantees; see `SETUP.md`. If you do not have that
much space, use the lighter route at the end of this section instead.

If you are fine with that, install uv using
https://docs.astral.sh/uv/getting-started/installation/ , then clone the
repository, open a terminal in it, and run:

```bash
git clone https://github.com/irenkamk/tsfm-workshop-sktime.git
cd tsfm-workshop-sktime
```

then:

```bash
uv sync --locked --extra foundation
uv run python scripts/check_environment.py --foundation
```

Run the first command once. A later plain `uv sync` would remove the
foundation libraries again; if that happens, run the first command again.

Seeing **BASELINE READY** and **FOUNDATION MODELS READY** means you are set;
there is nothing else to open before the workshop. On the day you will open
the notebooks with Jupyter Lab (`uv run jupyter lab`) or in VS Code or PyCharm,
selecting the project's `.venv` as the notebook kernel. Python 3.12 is
recommended; the project supports 3.11–3.13.

**Not enough disk space or bandwidth?** You can still do the whole workshop.
Install only the core packages (about 1–2 GB):

```bash
uv sync --locked
uv run python scripts/check_environment.py
```

You then run seasonal naive, ARIMA, and exponential smoothing yourself, and the
notebooks load the instructor's **pre-computed forecasts** for the pretrained
models. Nothing in the exercises is lost; you just don't run those models live.

## A GPU is not necessary

Neither route needs a GPU for the planned workshop. The dataset is small
(420 days), and in our tests all three pretrained models produced their
forecasts on an ordinary CPU in seconds once their weights were downloaded.
The weights (roughly 0.5–1 GB per model) are the main cost, and downloading them
needs internet the first time. You can run seasonal naive, ARIMA, and
exponential smoothing, and inspect the instructor's saved results from the
pretrained models. This is a full participation path, not a lesser exercise.
A Colab GPU runtime (**Runtime → Change runtime type → T4 GPU**) is only a
bonus, and free GPU access is not guaranteed.

## What not to worry about

Installing `sktime` does **not** download all foundation models. We use
separate optional dependencies, and each model's pretrained weights download
when it first loads. You do not need to train these models.

No API keys are needed for the core or currently public model checkpoints.
For t0, read the linked license before enabling its opt-in flag. If a service
later requests authentication, ask for help—never paste a token into a shared
notebook or send it to the organizer.

Please use only the bundled public dataset, not employer, customer, or
personal data. Bring one forecasting question you would like to explore!
