# Instructor guide

Instructor-only notes. This file is **not** included in either ZIP
(`package_workshop.py` uses an allowlist, and this file is not on it). Do not
push it to a public participant repository.

Participant-facing files: `PARTICIPANT_PREP.md` (invitation and setup steps),
`SETUP.md` (reference and troubleshooting), and the notebooks.

## Two packages

```bash
rm -f ../tsfm-workshop-setup.zip ../tsfm-workshop-full.zip   # the script never overwrites
uv run python scripts/package_workshop.py --mode setup --output ../tsfm-workshop-setup.zip
uv run python scripts/package_workshop.py --mode full  --output ../tsfm-workshop-full.zip
```

- **setup:** sent or published before the event. Environment files, data,
  `workshop.py`, `scripts/check_environment.py` and `rehearse.py`, `README.md`,
  `PARTICIPANT_PREP.md`, `SETUP.md`, and only `notebooks/00_setup_check.ipynb`.
  No lesson notebooks, solutions, slides, tutorial, or saved model results, so
  there are no spoilers.
- **full:** given out on the day. Everything above plus both lesson notebooks,
  `artifacts/` (saved baseline and foundation-model results), TUTORIAL,
  slides, the scripts, tests, and the setup skill. It still excludes this file,
  your environment, credentials, downloaded weights, and notebook outputs.

The notebooks are generated. Edit `scripts/build_notebooks.py`, never the
`.ipynb` files: `uv run python scripts/build_notebooks.py` overwrites them and
clears outputs. Then run `uv run --extra foundation python scripts/check_notebooks.py`,
which executes copies with the solutions inserted and leaves the student
notebooks unchanged. Always rebuild the notebooks before packaging.

## Before the workshop: checklist

1. Rebuild notebooks, run `check_notebooks.py` and
   `uv run python -m unittest tests/test_workshop.py`.
2. Read both lesson notebooks once in Jupyter as a participant, on a copy
   (`cp notebooks/*.ipynb my-solved/`). `my-solved/`, `artifacts-live/` and
   `artifacts-local-test/` are not packaged.
3. Check that `artifacts/` holds the results you want to ship, and that
   `artifacts-live/` is empty or absent. A participant's own run in
   `artifacts-live/` takes priority over `artifacts/` in the notebooks.
4. Dry-run as a participant on Colab with the **setup** ZIP and
   `00_setup_check.ipynb` (Option 1 of `PARTICIPANT_PREP.md`), then the day-of
   flow with the **full** ZIP and notebook 1 and 2 in fresh runtimes. Note the
   install time, any Colab conflict messages, and whether the foundation
   install works there.
5. Check slide font size and projector contrast; set a 2-minute
   troubleshooting timebox for class.
6. Before minute 0, make sure participants can plot the bundled data. If a
   model download fails, use the saved-result path; do not spend teaching time
   debugging CUDA.

Update the verification status below after each rehearsal.

**Verification status.** All three foundation models (Chronos-2, TimesFM 2.5,
t0-gamma) were run for real on CPU, and their saved results are in
`artifacts/`. They have not yet been run on a Colab GPU, so Colab timings
and results on that hardware are unverified. Package versions and checkpoint
revisions are recorded in each `metadata.json`. The Colab install of the core
and foundation requirements has been tried once (the check cell reported setup
OK on a CPU runtime with Python 3.13); a full dry run is still to be done.

## Run of show

| Minutes | Activity | Participant milestone |
|---|---|---|
| 0–5 | Vote: who wins—last week, ARIMA, or the giant pretrained model? | Write a prediction and a reason |
| 5–12 | Read the daily series; what repeats? | Data detective |
| 12–22 | Draw a forecast cutoff; YOUR TURN split and horizon | Time traveller, no leakage |
| 22–32 | YOUR TURN seasonal naive; inspect its graph and error | Baseline builder |
| 32–45 | One live or saved foundation forecast; architecture sketch | Foundation explorer |
| 45–53 | Notebook 2: why one lucky split is not enough | Backtest designer |
| 53–65 | Three folds; run CPU baselines and start/read model results | Fair referee |
| 65–77 | YOUR TURN leaderboard and fold stability; compare to naive | Evidence investigator |
| 77–85 | Pair discussion: error, runtime, uncertainty, holidays | Decision maker |
| 85–90 | Exit ticket and "change your mind" vote | Forecasting finisher |
| 90–120 | Q&A, covariates, intervals, Kumo, own use cases | Open exploration |

The timings are a facilitation budget, not measured guarantees for all
hardware. All three foundation models are represented in the benchmark code,
but **do not make every participant run every model live**. Allocate
Chronos-2, TimesFM 2.5, and t0-gamma across tables or pairs, or use the saved
results. Use one model per fresh process; the sktime adapters cache weights.

## Make it interactive

Use a "forecast before you fit" vote, pair-driver swaps at minute 45, and a
one-minute explanation of the weakest fold. Invite participants to hold up
green/yellow/red for ready/question/help; badges never rank people.
Ask "Would you buy this error reduction at this runtime?" rather than
"Which AI wins?" Reward a well-justified baseline win equally.

Optional after the core: Prophet (separate extra), quantile coverage,
calendar covariates known in advance, or a longer context experiment with a
new evaluation plan. Do not add a second dataset during the 90-minute core.

## Producing the saved model results (rehearsal)

Each command is a new process, downloads and caches weights if needed,
computes the demonstration and three backtest folds, and writes `demo.csv`,
`scores.csv`, `predictions.csv`, and `metadata.json` (versions, device, data
hash, protocol, and the cached checkpoint revision). An existing output
directory is rejected so evidence is never overwritten; use a fresh directory
for each new run.

Locally (CPU is enough for this dataset; all three finished in seconds):

```bash
uv sync --locked --extra foundation
uv run python scripts/rehearse.py --model chronos2   --device cpu --output artifacts-local-test/chronos2
uv run python scripts/rehearse.py --model timesfm25  --device cpu --output artifacts-local-test/timesfm25
# Only after reading and accepting the tfc-t0 / t0-gamma license (Apache-2.0):
uv run python scripts/rehearse.py --model t0-gamma   --device cpu --accept-t0-license --output artifacts-local-test/t0-gamma
```

On a Colab GPU runtime, run the same commands in cells prefixed with `!`,
without `uv run` and with `--device cuda`. Then copy the folders you want to
ship into `artifacts/<model>` and baselines into `artifacts/baselines/<name>`,
and check them with `workshop.load_run`. A model that fails shows as "not
evaluated"; never substitute random initializations, synthetic curves, or
vendor benchmark scores. The factories load Hub `main`, so keep the recorded
`checkpoint_revision` and the cache for the session. For stronger
reproducibility, download that revision into a local snapshot and point the
factory at its path. `workshop.py` switches Hugging Face to offline mode
automatically for models whose weights are already cached.

Licenses: Chronos-2, TimesFM 2.5 and t0-gamma are separate checkpoints with
their own terms; read them and distribute weights only if you have checked.
Timestral is mentioned in `TUTORIAL.md` but not used (revenue-based license;
no sktime wrapper yet).

## Publishing the setup package on GitHub

Repository: https://github.com/irenkamk/tsfm-workshop-sktime (public). Only the
**setup package** goes to `main`; participants get the link, download the ZIP
with the green **Code** button (Colab) or clone it (local). The full material
(lesson notebooks, solutions, slides, saved results) goes out on the day.

1. Build the setup ZIP (see "Two packages") and unzip it to a fresh folder:
   `unzip ../tsfm-workshop-setup.zip -d /tmp/setup-publish`.
2. The files are inside `tsfm-workshop/`; the repository root must be that
   folder's contents. From `/tmp/setup-publish/tsfm-workshop`: `git init -b main`,
   `git remote add origin https://github.com/irenkamk/tsfm-workshop-sktime.git`,
   `git add .`, `git commit`, `git push -u origin main` (or copy the files into
   your clone and commit there).
3. Never push from the full project folder: a public repo shows everything
   pushed, including history, so notebooks, solutions, slides, this file and
   saved results would leak.
4. Check the repo page: the README renders, and the green Code → Download ZIP
   gives a folder named `tsfm-workshop-sktime-main/`. The notebooks' first cell
   accepts any single top-level folder, so that ZIP works on Colab.
5. Colab can also open a notebook straight from the public repo without any
   authorization: `https://colab.research.google.com/github/irenkamk/tsfm-workshop-sktime/blob/main/notebooks/00_setup_check.ipynb`.
   Tell participants **not** to use Colab's GitHub-linking popup: it asks for
   read/write access to all their repositories and gists.
6. The repository MIT license covers the code only. The data is CC BY 4.0
   (see `data/README.md`) and model weights keep their own licenses.
7. On the day, the full ZIP is uploaded to the same bootstrap cell. Rebuild
   the notebooks and both packages after any change, then push the new setup
   package.

## Using the setup skill

The skill at `.github/skills/prepare-tsfm-workshop/SKILL.md` is a text
artifact, not an account setting. For an agent that supports repository
skills, keep it in the repository. For an agent with user-level skills, copy
the folder to that agent's skill directory. Not every chat client loads skills
or has terminal permissions. Then say: **"prepare me for foundation models for
time series workshop"**. The skill checks the chosen environment, installs
only the chosen track, and asks before accepting terms or downloading weights.
It does not create a paid service, expose a notebook server, or change global
Python settings.

## Slides and licenses

Slides are plain Markdown (Marp); the VS Code Marp extension exports them. See
`slides.md` for export notes. Model and software licenses differ from the data
license; see `data/README.md` and the sources in `TUTORIAL.md`.
