Hi everyone! For our **Can AI Predict the Future?** PyLadies workshop,
please bring a laptop, charger, and a browser. We will use Python notebooks
to compare simple forecasts with pretrained AI models.

You only need basic Python: variables, functions, and a little pandas.
No prior forecasting knowledge, paid AI account, or personal GPU is required.
Pairing up is welcome.

**There are two ways to join. Choose whichever you feel comfortable with.**
If you already work in an IDE such as VS Code or PyCharm, or have Python set up
on your laptop, stay **local** (option 1). If you have little coding experience
or nothing installed, go with **Google Colab** (option 2): it runs in your
browser and needs no installation on your machine. Both routes use the same
notebooks and cover the whole workshop.

**If you choose option 1 (local), please install all the needed libraries and
download the model weights before the workshop.** Please allow
**about 20–40 minutes** for this. With a fast internet connection it can take
much less, and most of that time is just waiting for downloads to finish.
If you choose option 2 (Colab), you only need a few minutes to get familiar
with it.

## Option 1: On your own computer

You do **not** upload anything here: the notebook's upload step is only for
Colab.

**Please read this before you install.** The setup below installs everything,
including the foundation-model libraries (PyTorch and friends), so you can
**run the pretrained models yourself** in the workshop. You need
**8–10 GB of free disk space**, and the installation may take around 15 minutes
on a 100 Mbps connection (longer if yours is slower). The model weights, about
2.3 GB in total, are downloaded in the step below, so that nothing is left to
download on the day. These are estimates from one Linux machine, not
guarantees; see `SETUP.md`. If you do not have that much space, use the
lighter route at the end of this section instead.

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
uv run python scripts/download_weights.py
```

Run the first command once. A later plain `uv sync` would remove the
foundation libraries again; if that happens, run the first command again.
The last command downloads the weights of the three pretrained models
(about 2.3 GB) into your Hugging Face cache and prints **WEIGHTS READY** when
it is done. Running t0-gamma means you accept its license terms (see
"What not to worry about" below). After that the models run without
downloading anything, even without internet.

Seeing **BASELINE READY**, **FOUNDATION MODELS READY** and **WEIGHTS READY**
means you are set; there is nothing else to open before the workshop. On the
day you will open the notebooks with Jupyter Lab (`uv run jupyter lab`) or in
VS Code or PyCharm, selecting the project's `.venv` as the notebook kernel.
Python 3.12 is recommended; the project supports 3.11–3.13.

**Not enough disk space or bandwidth?** You can still do the whole workshop.
Install only the core packages (about 1–2 GB, no weights to download):

```bash
uv sync --locked
uv run python scripts/check_environment.py
```

You then run seasonal naive, ARIMA, and exponential smoothing yourself, and the
notebooks load the instructor's **pre-computed forecasts** for the pretrained
models. Nothing in the exercises is lost; you just don't run those models live.

## Option 2: Google Colab (nothing to install)

Here you only need to get familiar with the tool and the setup. You can go
through the process below to see how it works, but you do **not** need to
install anything in advance: on the day, we will install the requirements
together in the workshop notebook.

1. Open https://github.com/irenkamk/tsfm-workshop-sktime , click the green
   **Code** button → **Download ZIP**, and extract it. Keep the ZIP as well.
   Sign in to Google Colab: https://colab.research.google.com/
   (Please do not use Colab's "connect to GitHub" popup: it asks for access to
   all your repositories.)
2. Use **File → Upload notebook** to open `notebooks/00_setup_check.ipynb`
   from the extracted folder. It has no exercises; it only installs and checks.
3. **How to run a cell:** click inside it, then press **Shift+Enter** (or the
   ▶ button on its left). Run the first code cell. A **Choose Files** button
   appears under the cell: click it and select the ZIP you downloaded. Wait
   until it prints `Workshop files:`.
4. Optional: if you want to try the install, set `INSTALL_CORE = True` and
   `INSTALL_FOUNDATION = True` in the install cell and run it, then run the
   check cell with `CHECK_FOUNDATION = True`. You should see **✅ Setup OK**.
   If you see **❌**, or it is too slow, don't worry: it is not needed now.

Colab sessions are temporary: files, installed packages, and downloaded weights
disappear when the runtime resets, so on the day we will upload the ZIP and
install again. If you want to keep your work, use
**File → Download → Download .ipynb**.

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
separate optional dependencies, and each model's pretrained weights are
downloaded by the `download_weights.py` step above (or, failing that, when the
model first loads). You do not need to train these models.

No API keys are needed for the core or currently public model checkpoints.
Running the t0-gamma model means you accept its license terms (the link is
next to the code that loads it). If a service
later requests authentication, ask for help—never paste a token into a shared
notebook or send it to the organizer.

Please use only the bundled public dataset, not employer, customer, or
personal data. Bring one forecasting question you would like to explore!

## API keys, tokens and credentials

This workshop does **not** use any API keys, access tokens, or other
credentials, and you should not need to enter any. As a general precaution:

- Never paste a token, password, or API key into a notebook, chat, or screenshot
  that others can see, and never send one to the organizer.
- If a download or service unexpectedly asks you to log in, stop and ask for
  help first.
- The organizer is not responsible for credentials you choose to share, or for
  any unauthorized access, usage, or charges that result from sharing them.
