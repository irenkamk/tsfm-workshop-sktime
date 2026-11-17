"""Generate the two student notebooks, with visible hints and opt-in solutions."""
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text, **metadata):
    return nbf.v4.new_code_cell(dedent(text).strip(), metadata=metadata)


def exercise(title, prompt, starter, solution, hint):
    return [
        md(f"### YOUR TURN — {title}\n\n{prompt}"),
        code(starter, tags=["exercise"], solution=dedent(solution).strip()),
        md(
            f"**Hint:** {hint}\n\n<details><summary>Show solution "
            "(copy into the exercise cell when you are ready)</summary>\n\n"
            f"```python\n{dedent(solution).strip()}\n```\n\n</details>"
        ),
    ]


def setup():
    return [
        md("""
        ## RUN THIS — get the workshop files
        **Colab:** upload the trusted workshop ZIP when asked (the one you get
        from the repository's green **Code → Download ZIP** button, or from the
        organizer). It contains one folder with the data and helpers. Each
        notebook may have its own runtime. **Local:** start Jupyter from the
        workshop folder.
        This cell does not install anything or download model weights.
        """),
        code("""
        from pathlib import Path
        import os
        import sys
        import zipfile

        here = Path.cwd()
        found = next(here.glob("*/workshop.py"), None)  # an already extracted ZIP
        if (here / "workshop.py").exists():
            ROOT = here
        elif (here.parent / "workshop.py").exists():
            ROOT = here.parent
        elif found:
            ROOT = found.parent
        else:
            if "google.colab" not in sys.modules and not os.environ.get("COLAB_RELEASE_TAG"):
                raise FileNotFoundError("Extract the ZIP and start Jupyter in the workshop folder.")
            from google.colab import files
            uploaded = files.upload()
            if len(uploaded) != 1:
                raise ValueError("Upload exactly one workshop ZIP.")
            archive_path = Path(next(iter(uploaded)))
            with zipfile.ZipFile(archive_path) as archive:
                tops = {name.split("/")[0] for name in archive.namelist() if name.strip("/")}
                if len(tops) != 1:
                    raise ValueError("The ZIP must contain exactly one top-level folder.")
                top = tops.pop()
                if top in ("", ".", "..") or Path(top).is_absolute():
                    raise ValueError("Unsafe folder name in the ZIP.")
                for member in archive.infolist():
                    destination = (here / member.filename).resolve()
                    if not destination.is_relative_to((here / top).resolve()):
                        raise ValueError("Unsafe path in the ZIP.")
                archive.extractall(here)
            ROOT = here / top
            if not (ROOT / "workshop.py").exists():
                raise ValueError("workshop.py not found: is this the workshop ZIP?")
        os.chdir(ROOT)
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        print("Workshop files:", ROOT)
        """, tags=["bootstrap"]),
        md("""
        ### SETUP ONLY — do this before the workshop
        Set `INSTALL_CORE = True` if the core packages are missing, and
        `INSTALL_FOUNDATION = True` to install the foundation-model packages.
        You need the foundation packages to run the pretrained models yourself;
        they are much larger than the core (see `SETUP.md`). Without them you
        can still use the instructor's pre-computed forecasts. After
        installation, restart the session if requested; rerun with both flags
        `False`. A GPU is not necessary.
        Read `SETUP.md` for downloads, CPU fallback, and troubleshooting.
        """),
        code("""
        import subprocess

        def run(command):
            # Run a command and show its output live, like a terminal.
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in process.stdout:
                print(line, end="")
            return process.wait()

        INSTALL_CORE = False
        INSTALL_FOUNDATION = False
        for wanted, name in ((INSTALL_CORE, "core"), (INSTALL_FOUNDATION, "foundation")):
            if wanted:
                print(f"Installing {name} packages: this can take several minutes...")
                if run([sys.executable, "-m", "pip", "install", "-r", f"requirements/{name}.txt"]):
                    raise RuntimeError(f"Installing {name} failed. Read the messages above.")
                print(f"Finished installing {name} packages.")
        """, tags=["setup"]),
        code("""
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        import plotly.io as pio
        from IPython.display import display, HTML
        from workshop import (
            load_series, demo_split, make_model, validate_prediction,
            benchmark_series, splitter, run_benchmark, leaderboard, load_run,
            CONTEXT, HORIZON, N_FOLDS, SEASON, BASELINES, FOUNDATIONS,
        )
        if run([sys.executable, "scripts/check_environment.py"]):
            raise RuntimeError("The environment check failed. Read the messages above.")

        # Plot helpers: one colour per entity, colour-blind-safe (Okabe-Ito).
        pio.templates.default = "plotly_white"
        BLUE, ORANGE, GREEN, INK = "#0072B2", "#D55E00", "#009E73", "#333333"
        MODEL_COLORS = {
            "seasonal-naive": ORANGE, "arima": "#E69F00", "ets": "#CC79A7",
            "chronos2": GREEN, "timesfm25": "#56B4E9", "t0-gamma": BLUE,
        }

        def new_figure(title, ylabel="Births per day"):
            fig = go.Figure()
            fig.update_layout(
                title=title, yaxis_title=ylabel, hovermode="x unified", height=380,
                legend=dict(orientation="h", y=-0.2), margin=dict(t=60, b=40),
            )
            return fig

        def add_line(fig, series, name, color, dash="solid", width=2, markers=False):
            fig.add_scatter(
                x=series.index, y=series.to_numpy(), name=name,
                mode="lines+markers" if markers else "lines",
                line=dict(color=color, dash=dash, width=width),
            )

        def mark_cutoff(fig, cutoff):
            fig.add_shape(
                type="line", x0=cutoff, x1=cutoff, y0=0, y1=1, yref="paper",
                line=dict(color=ORANGE, dash="dot"),
            )
            fig.add_annotation(
                x=cutoff, y=1, yref="paper", text="Forecast cutoff", showarrow=False,
                xanchor="right", yanchor="top", font=dict(color="#666666"),
            )

        y = load_series()
        milestones = {}

        def celebrate(key, message):
            milestones[key] = message
            display(HTML(
                f"<progress value='{len(milestones)}' max='4'></progress> "
                f"<b>{len(milestones)}/4 milestones:</b> {message}"
            ))
            print("Completed:", ", ".join(milestones))
        """),
        md("""
        > **Curious what a helper does?** Functions like `demo_split` or
        > `leaderboard` come from `workshop.py`. In any cell, run
        > `help(demo_split)` (or `demo_split?` in Jupyter) to read what it does.
        > You never have to edit that file.
        """),
    ]


# Spoiler-free pre-workshop notebook: install and check only, no exercises.
notebook0 = [
    md("""
    # 0. Setup check
    ## Can AI Predict the Future? Do this before the workshop

    This notebook installs the workshop packages and checks that everything
    works. There are no exercises and nothing to solve. It takes about
    5–10 minutes, depending on your internet speed.

    **How to run a cell:** click inside it, then press **Shift+Enter** (or the
    ▶ button on its left). Wait for the check mark or `[1]` before the next cell.

    1. Run the first code cell and upload the workshop ZIP when asked.
    2. Set `INSTALL_CORE = True` and `INSTALL_FOUNDATION = True` in the install
       cell and run it (this lets you run the pretrained models yourself). If Colab asks
       you to restart the session, do (**Runtime → Restart session**), then
       rerun from the first cell with `INSTALL_CORE = False`.
    3. Run the check cell. A line starting with **✅** means you are set.

    Colab sessions are temporary: you will repeat the install on the day.
    """),
    *setup()[:4],
    md("""
    ## RUN THIS — check the installation
    The list shows the installed package versions. At the end you should see
    **BASELINE READY**, then (with `CHECK_FOUNDATION = True`) **FOUNDATION MODELS
    READY**, then a line starting with **✅**. That is enough to join.
    If you see **❌** instead, scroll up for the error message and ask for help.
    A GPU is not necessary. Set `CHECK_FOUNDATION = True` (you installed the
    foundation-model packages) to check them too; it checks imports, not
    downloads or inference.
    """),
    code("""
    CHECK_FOUNDATION = False
    command = [sys.executable, "scripts/check_environment.py"]
    if CHECK_FOUNDATION:
        command.append("--foundation")
    if run(command) == 0:
        print("\\n✅ Setup OK: everything needed for the workshop works.")
    else:
        print("\\n❌ The check failed. Scroll up for the error message and ask for help.")
    """, tags=["check"]),
]


notebook1 = [
    md("""
    # 1. Forecast detectives
    ## Can AI Predict the Future?
    **45 minutes | pairs welcome | 4 milestones**

    Today we test a promise, not prove that AI can know the future.
    By the end you can prepare a time series, make a leak-free split,
    create a seasonal baseline, and interpret a pretrained-model forecast.

    **RUN THIS** = supplied code. **YOUR TURN** = replace `None` / write code.
    Hints and solutions are below each exercise. Getting help counts as progress.
    There are no points for speed.

    ### Before touching the keyboard (2 minutes)
    Predict who will win: repeat last week, a classical statistical model, or
    a pretrained transformer. Write **one reason**. Keep it for the exit vote.
    """),
    *setup(),
    md("""
    ## 1. Data detective (7 minutes)
    We have **420 daily aggregate U.S. birth counts**, from FiveThirtyEight/SSA,
    licensed CC BY 4.0. See `data/README.md` for attribution and the pinned source.
    This is not a medical decision tool.

    Frequency is **daily**; the seasonal period we will test is **7 days**.
    Frequency and seasonality are different concepts.

    We reserve the **last 42 days** for notebook 2's evaluation before looking
    at them. No smoothing or imputation uses the future. `sktime` accepts this
    regular pandas Series directly—datasets need not be built into sktime.
    """),
    code("""
    development = y.iloc[:-N_FOLDS * HORIZON]
    print("Development range:", development.index.min(), "to", development.index.max())
    print("Daily frequency:", development.index.freqstr)
    display(development.head().to_frame())
    """),
    *exercise(
        "Plot the series yourself (2 minutes)",
        "Draw the development data as a line chart with `px.line(...)`. "
        "Give it a title, then look: what repeats? What might surprise a model? "
        "Hover, zoom, and drag over the chart to explore.",
        """
        fig = None  # Replace None with px.line(development, title="...")
        assert fig is not None, "Hint: px.line(development, title=...)"
        assert len(fig.data) == 1 and len(fig.data[0].y) == len(development), "Plot the whole development series once."
        fig.update_layout(xaxis_title="Date", yaxis_title="Births per day")
        fig.show()
        """,
        """
        fig = px.line(development, title="What repeats? What might surprise a model?")
        fig.update_layout(xaxis_title="Date", yaxis_title="Births per day")
        fig.show()
        """,
        "`px.line` accepts a pandas Series directly: the index becomes the x axis.",
    ),
    code("""
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    frame = development.to_frame("births")
    frame["weekday"] = frame.index.dayofweek
    frame["week_start"] = frame.index - pd.to_timedelta(frame["weekday"].to_numpy(), unit="D")
    by_week = frame.pivot(index="weekday", columns="week_start", values="births").dropna(axis=1)
    fig = new_figure("Weeks look alike: one thin line per week (last 12), development data only")
    for i, week in enumerate(by_week.columns[-12:]):
        fig.add_scatter(
            x=weekday_names, y=by_week[week], mode="lines", hoverinfo="skip",
            name="Individual weeks", showlegend=(i == 0),
            line=dict(color=BLUE, width=1), opacity=0.45,
        )
    fig.add_scatter(
        x=weekday_names, y=by_week.mean(axis=1), name="Average week",
        line=dict(color=INK, width=3),
    )
    fig.update_layout(xaxis_title="Day of week")
    fig.show()
    """),
    *exercise(
        "Spot the weekly pattern (3 minutes)",
        "Calculate the mean count for each weekday using **development only**. "
        "Monday is 0. Discuss one possible explanation without treating it as proven causation.",
        """
        weekday_means = None  # Group development by development.index.dayofweek, then mean
        assert weekday_means is not None, "Hint: development.groupby(...).mean()"
        assert len(weekday_means) == 7, "We need one average per weekday."
        fig = px.bar(
            x=weekday_means.index, y=weekday_means.to_numpy(),
            title="Development data only: mean by weekday",
            labels={"x": "Weekday: Monday=0", "y": "Mean births per day"},
        )
        fig.show()
        celebrate("detective", "You found structure worth forecasting.")
        """,
        """
        weekday_means = development.groupby(development.index.dayofweek).mean()
        assert len(weekday_means) == 7
        fig = px.bar(
            x=weekday_means.index, y=weekday_means.to_numpy(),
            title="Development data only: mean by weekday",
            labels={"x": "Weekday: Monday=0", "y": "Mean births per day"},
        )
        fig.show()
        celebrate("detective", "You found structure worth forecasting.")
        """,
        "Dates have `.dayofweek`; grouping is descriptive, not a forecast.",
    ),
    md("""
    ## 2. A time machine with rules (10 minutes)
    A **forecast origin/cutoff** is the last date we are allowed to observe.
    **Context** is the historical data supplied to the model. A **horizon**
    specifies how far ahead to predict, here days 1 through 14.

    We use the last 14 **development** days as a demonstration holdout and
    the preceding 168 days as context. Notebook 2 has a separate evaluation
    period. Never use a random train/test split for this forecasting task.
    """),
    *exercise(
        "Build a leak-free demo split (4 minutes)",
        "Select the demo test window, the immediately preceding 168 training days, "
        "and relative horizon 1–14. Do not include any test value in training.",
        """
        y_train = None
        y_test = None
        fh = None
        assert y_train is not None and y_test is not None and fh is not None, "Fill all three."
        expected_train, expected_test = demo_split(y)
        pd.testing.assert_series_equal(y_train, expected_train)
        pd.testing.assert_series_equal(y_test, expected_test)
        assert list(fh) == list(range(1, 15)), "Forecast horizons start at 1, not 0."
        assert y_train.index.max() < y_test.index.min()
        celebrate("cutoff", "The future stayed out of training.")
        """,
        """
        y_train = development.iloc[-CONTEXT-HORIZON:-HORIZON]
        y_test = development.iloc[-HORIZON:]
        fh = np.arange(1, HORIZON + 1)
        expected_train, expected_test = demo_split(y)
        pd.testing.assert_series_equal(y_train, expected_train)
        pd.testing.assert_series_equal(y_test, expected_test)
        assert list(fh) == list(range(1, 15))
        assert y_train.index.max() < y_test.index.min()
        celebrate("cutoff", "The future stayed out of training.")
        """,
        "`iloc[-14:]` selects the final 14 observations; the training slice stops before them.",
    ),
    code("""
    fig = new_figure("Ask: could I have known this value at the cutoff?")
    add_line(fig, y_train.iloc[-42:], "Available history", BLUE)
    add_line(fig, y_test, "Hidden future (revealed for scoring)", INK, dash="dash")
    mark_cutoff(fig, y_train.index[-1])
    fig.show()
    """),
    md("""
    ## 3. The baseline is a contestant, not a straw person (10 minutes)
    **Seasonal naive:** repeat the last observed week. For horizons beyond
    seven days, repeat that same available week—not actual future observations.
    It has no learned neural weights, but often sets a high bar.

    `fit(y_train, fh=fh)` remembers the data and forecasting horizon.
    `predict()` returns timestamp-aligned forecasts.
    **MAE** = average absolute difference between actual and prediction;
    its unit here is births/day. Lower is better.
    """),
    *exercise(
        "Forecast with seasonal naive (4 minutes)",
        "Create a `NaiveForecaster` with `strategy='last'` and the right seasonal period; "
        "fit it on training only, then predict.",
        """
        from sktime.forecasting.naive import NaiveForecaster
        naive = None
        assert naive is not None, "Create NaiveForecaster(strategy='last', sp=...)."
        # YOUR TURN: call fit, then replace None with predict()
        naive_pred = None
        assert naive_pred is not None, "Fit on y_train, then call predict()."
        validate_prediction(naive_pred, y_test)
        np.testing.assert_allclose(naive_pred, np.tile(y_train.iloc[-7:], 2))
        celebrate("baseline", "You made a real forecast with a sensible baseline.")
        """,
        """
        from sktime.forecasting.naive import NaiveForecaster
        naive = NaiveForecaster(strategy="last", sp=7)
        naive.fit(y_train, fh=fh)
        naive_pred = naive.predict()
        validate_prediction(naive_pred, y_test)
        np.testing.assert_allclose(naive_pred, np.tile(y_train.iloc[-7:], 2))
        celebrate("baseline", "You made a real forecast with a sensible baseline.")
        """,
        "Seasonality is 7, not horizon 14. The same fit/predict interface is used for the AI models.",
    ),
    code("""
    from sktime.performance_metrics.forecasting import mean_absolute_error
    print(f"Demo seasonal-naive MAE: {mean_absolute_error(y_test, naive_pred):,.1f} births/day")
    fig = new_figure("Baseline forecast: does the shape make sense?")
    add_line(fig, y_train.iloc[-28:], "History", BLUE)
    add_line(fig, y_test, "Actual", INK, width=3)
    add_line(fig, naive_pred, "Seasonal naive", ORANGE, dash="dash", markers=True)
    mark_cutoff(fig, y_train.index[-1])
    fig.show()
    """),
    md("""
    ## 4. A pretrained model joins (13 minutes)
    Foundation models learn reusable patterns from many series. Here
    **zero-shot** means no weight training on our workshop series. `fit` still
    supplies observed history and can load a large checkpoint.

    - **Chronos-2:** encoder-only patched transformer; time/group attention.
    - **TimesFM 2.5:** decoder-only patched transformer.
    - **t0-gamma:** decoder-style patch transformer with time/covariate attention.

    This is a univariate task without exogenous variables. It does **not**
    exercise everything these models can do. See `TUTORIAL.md` for the important
    differences between native models and sktime wrappers.

    **Choose one**, not all three in this runtime. Default is saved-results
    mode. If its folder is absent, the code says so rather than inventing an
    AI curve. The instructor must supply real model results from rehearsal.
    On a prepared GPU runtime, set `LIVE = True`.

    Running t0-gamma means you accept its license terms (linked in `workshop.py`).
    Its public gamma checkpoint is selected explicitly; the adapter's default
    alpha is not gamma. Do not use `model_path=None`—that is an untrained model.
    """),
    code("""
    CHOSEN_MODEL = "chronos2"  # Or "timesfm25" or "t0-gamma"
    LIVE = False
    DEVICE = "cpu"  # CPU is fine for this small series; use "cuda" on a GPU runtime
    foundation_pred = None
    evidence_label = None

    if CHOSEN_MODEL not in FOUNDATIONS:
        raise ValueError("Choose chronos2, timesfm25, or t0-gamma.")
    if LIVE:
        foundation = make_model(CHOSEN_MODEL, DEVICE)
        foundation.fit(y_train, fh=fh)
        foundation_pred = foundation.predict()
        evidence_label = "live inference"
    else:
        # Prefer a run you made yourself (artifacts-live), else the instructor's.
        saved = ROOT / "artifacts-live" / CHOSEN_MODEL
        source = "your own run"
        if not saved.exists():
            saved = ROOT / "artifacts" / CHOSEN_MODEL
            source = "instructor run"
        if saved.exists():
            _, _, metadata = load_run(saved, y)
            if metadata["model"] != CHOSEN_MODEL:
                raise ValueError("Saved model does not match the selected model.")
            demo = pd.read_csv(saved / "demo.csv", parse_dates=["date"]).set_index("date")
            np.testing.assert_array_equal(demo["actual"], y_test)
            foundation_pred = demo["prediction"].asfreq("D")
            evidence_label = f"saved real inference ({source}); not run in this notebook"
            print("Provenance:", metadata)
        else:
            print(f"NOT EVALUATED: no saved {CHOSEN_MODEL} run. Use prepared live mode")
            print("or ask the instructor for rehearsal artifacts. No AI forecast is shown.")
    if foundation_pred is not None:
        validate_prediction(foundation_pred, y_test)
        fig = new_figure(f"{CHOSEN_MODEL}: {evidence_label}")
        add_line(fig, y_train.iloc[-28:], "History", BLUE)
        add_line(fig, y_test, "Actual", INK, width=3)
        add_line(fig, naive_pred, "Seasonal naive", ORANGE, dash="dash")
        add_line(fig, foundation_pred, CHOSEN_MODEL, GREEN, dash="dashdot", markers=True)
        mark_cutoff(fig, y_train.index[-1])
        fig.show()
        display(pd.Series({
            "seasonal-naive": mean_absolute_error(y_test, naive_pred),
            CHOSEN_MODEL: mean_absolute_error(y_test, foundation_pred),
        }, name="Demo MAE (births/day)").to_frame())
    """, tags=["optional-foundation"]),
    md("""
    ### YOUR TURN — explain, don't just rank (2 minutes)
    Write two sentences below. If an AI run is unavailable, explain what
    evidence would be needed; do not claim it won or lost.
    """),
    code("""
    reflection = ""  # Where did the forecast work/fail? Why is one split not enough?
    if reflection.strip():
        celebrate("explorer", "You distinguished a claim from the evidence behind it.")
    else:
        print("Write two sentences, then rerun. You can also share them verbally with a partner.")
    """, tags=["reflection"]),
    md("""
    ## Checkpoint: 45 minutes
    You can now forecast, plot, and ask whether the result is fair.
    Save this notebook and open **02_fair_forecast_arena.ipynb**.
    A new Colab notebook may need the ZIP/setup again. It does not need
    variables from this notebook.

    **Optional discussion:** a quantile interval is not a guarantee. The
    native Chronos-2 model produces quantiles, but this pinned sktime wrapper
    does not expose prediction intervals. We will not fabricate uncertainty
    bands from point predictions.
    """),
]

notebook2 = [
    md("""
    # 2. The fair forecast arena
    **45 minutes | 4 milestones | one dataset, one protocol**

    Goal: compare seasonal naive, seasonal ARIMA, exponential smoothing, and
    available Chronos-2 / TimesFM 2.5 / t0-gamma results on exactly the same
    three forecast origins. Prophet is optional after the main session.

    This notebook is self-contained. The last 42 days were reserved before
    exploratory analysis in notebook 1. We are now allowed to reveal them
    for evaluation, not use them to secretly tune model settings.
    """),
    *setup(),
    md("""
    ## 1. Backtesting: rehearse the future (8 minutes)
    One holdout can be lucky. A **rolling-window backtest** repeats:
    take the last 168 observed days, forecast 14 days, reveal those outcomes,
    move forward 14 days, and repeat. We use three non-overlapping test windows.

    ```
    fold 1: [          168 context days          ][14 test]
    fold 2:      [          168 context days          ][14 test]
    fold 3:           [          168 context days          ][14 test]
    ```

    Earlier test values can enter later training windows: they would have
    become known by that later cutoff. No fold sees its own future.
    `strategy="refit"` re-estimates classical models each fold; foundation
    models refresh context without training pretrained weights.
    """),
    *exercise(
        "Design the splitter (4 minutes)",
        "Use `SlidingWindowSplitter` with a 14-step horizon, 168-day window, "
        "and 14-day step. Inspect its three cutoffs before computing any score.",
        """
        from sktime.split import SlidingWindowSplitter
        cv = None
        assert cv is not None, "Set fh, window_length, and step_length."
        evaluation_y = benchmark_series(y)
        folds = list(cv.split(evaluation_y))
        expected_folds = list(splitter().split(evaluation_y))
        assert len(folds) == 3, "There should be exactly three backtest origins."
        for (train_idx, test_idx), (expected_train, expected_test) in zip(folds, expected_folds):
            np.testing.assert_array_equal(train_idx, expected_train)
            np.testing.assert_array_equal(test_idx, expected_test)
            assert train_idx.max() < test_idx.min()
        celebrate("designer", "Your backtest respects time.")
        """,
        """
        from sktime.split import SlidingWindowSplitter
        cv = SlidingWindowSplitter(
            fh=np.arange(1, 15), window_length=168, step_length=14
        )
        evaluation_y = benchmark_series(y)
        folds = list(cv.split(evaluation_y))
        expected_folds = list(splitter().split(evaluation_y))
        assert len(folds) == 3
        for (train_idx, test_idx), (expected_train, expected_test) in zip(folds, expected_folds):
            np.testing.assert_array_equal(train_idx, expected_train)
            np.testing.assert_array_equal(test_idx, expected_test)
            assert train_idx.max() < test_idx.min()
        celebrate("designer", "Your backtest respects time.")
        """,
        "The horizon is `np.arange(1, 15)`. Step length 14 avoids overlapping test windows.",
    ),
    md("""
    The workshop's own `splitter()` is what every model below is scored with.
    Read its description and compare it with the one you just built:
    """),
    code("""
    help(splitter)
    """),
    code("""
    schedule = pd.DataFrame([
        {"fold": number, "context_start": evaluation_y.index[tr[0]],
         "cutoff": evaluation_y.index[tr[-1]], "test_start": evaluation_y.index[te[0]],
         "test_end": evaluation_y.index[te[-1]]}
        for number, (tr, te) in enumerate(folds, start=1)
    ])
    display(schedule)
    """),
    md("""
    ## 2. Agree on the score before seeing the winner (4 minutes)
    **MAE:** `mean(abs(actual - forecast))`, in births/day; lower is better.

    **Seasonal MASE:** each fold's MAE divided by the mean absolute **7-day
    seasonal difference in that fold's training data**. Scale is learned from
    training only. MASE < 1 means error below that in-sample seasonal-difference
    scale; it does **not** prove a win over seasonal naive on the test window.
    Compare test MAEs directly to answer that question.

    We avoid MAPE because it is unstable with near-zero actual values and
    can distort comparisons. This dataset is nonzero, but MAE remains easier
    to explain. Absolute-error metrics favor medians; point-forecast semantics
    can differ across models. RMSE would answer a different, spike-sensitive question.

    With equal-size non-overlapping windows, mean fold MAE also equals MAE
    over all 42 forecast points. Mean fold MASE averages **fold-specific**
    scales. Timing here is a practical cost indicator, not a controlled
    hardware benchmark: first-load/download time and warm caches differ.
    """),
    md("""
    ## 3. One sktime benchmarking path (12 minutes)
    `workshop.run_benchmark` calls **`sktime.forecasting.model_evaluation.evaluate`**
    for every model, with our splitter, `strategy="refit"`, MAE and seasonal MASE.
    It retains actuals/predictions and raises on failure instead of ranking NaNs.
    The exact implementation is shown below so it is not a black box.

    Classical models:
    **seasonal naive** repeats last week;
    **ARIMA** here is a fixed SARIMA(1,0,1)(1,0,0)[7] with intercept;
    **ETS** has additive trend, damping, and additive 7-day seasonality.
    These are sensible teaching settings, not automatically tuned best baselines.
    """),
    code("""
    import inspect
    from IPython.display import Code
    display(Code(inspect.getsource(run_benchmark), language="python"))
    """),
    code("""
    USE_SAVED_BASELINES = False  # True is the explicit CPU/offline-results fallback
    score_parts, prediction_parts, evidence = [], [], []
    for name in BASELINES:
        if USE_SAVED_BASELINES:
            scores, predictions, metadata = load_run(ROOT / "artifacts/baselines" / name, y)
            origin = "saved CPU result"
        else:
            scores, predictions = run_benchmark(name, y, cv=cv)
            origin = "live in this runtime"
        score_parts.append(scores)
        prediction_parts.append(predictions)
        evidence.append({"model": name, "status": origin})
    print("Three classical models completed three folds each.")
    celebrate("referee", "All baseline contestants saw the same history.")
    """),
    md("""
    ### RUN THIS — add available foundation-model evidence
    The default loads instructor rehearsal results, validates the data hash
    and scores, and explicitly labels missing models. No missing model is ranked.

    Optional: run **one** assigned model below in a separate subprocess.
    That invokes the same `run_benchmark` path and releases model memory at exit.
    Your own run is saved in `artifacts-live/` and is preferred over the
    instructor's in `artifacts/`. Delete its folder to go back to the instructor
    result. Do not launch all three simultaneously.
    """),
    code("""
    RUN_ASSIGNED_MODEL = False
    ASSIGNED_MODEL = "chronos2"
    DEVICE = "cpu"  # "cuda" on a GPU runtime
    if RUN_ASSIGNED_MODEL:
        if ASSIGNED_MODEL not in FOUNDATIONS:
            raise ValueError("Choose a foundation model from FOUNDATIONS.")
        live_dir = ROOT / "artifacts-live" / ASSIGNED_MODEL
        if live_dir.exists():
            print(f"{live_dir} already exists; using it. Delete the folder to rerun.")
        else:
            command = [
                sys.executable, "scripts/rehearse.py", "--model", ASSIGNED_MODEL,
                "--device", DEVICE, "--output", str(live_dir),
            ]
            if run(command):
                raise RuntimeError("The model run failed. Read the messages above.")
    """, tags=["optional-foundation"]),
    code("""
    # Rebuild from the three baseline parts so rerunning does not duplicate folds.
    score_parts = score_parts[:3]
    prediction_parts = prediction_parts[:3]
    evidence = evidence[:3]
    for name in FOUNDATIONS:
        saved, source = ROOT / "artifacts-live" / name, "your own run"
        if not saved.exists():
            saved, source = ROOT / "artifacts" / name, "instructor run"
        if saved.exists():
            scores, predictions, metadata = load_run(saved, y)
            if metadata["model"] != name:
                raise ValueError("Saved model identity does not match its folder.")
            score_parts.append(scores)
            prediction_parts.append(predictions)
            evidence.append({"model": name, "status": f"saved real inference ({source})",
                             "device": metadata["device"],
                             "revision": metadata["checkpoint_revision"]})
        else:
            evidence.append({"model": name, "status": "NOT EVALUATED — no saved run"})
    all_scores = pd.concat(score_parts, ignore_index=True)
    all_predictions = pd.concat(prediction_parts, ignore_index=True)
    display(pd.DataFrame(evidence).fillna("not applicable"))
    """),
    *exercise(
        "Build an honest leaderboard (5 minutes)",
        "Group by model and average the fold MAEs; sort lowest first. "
        "Then calculate percentage improvement versus seasonal naive. A negative "
        "improvement is worse. Do not fill missing models with zero error.",
        """
        mean_mae = None
        assert mean_mae is not None, "Group all_scores by model, select mae, mean, sort_values."
        pd.testing.assert_series_equal(
            mean_mae, all_scores.groupby("model")["mae"].mean().sort_values()
        )
        improvement_percent = None
        assert improvement_percent is not None, "100 * (1 - model MAE / naive MAE)"
        np.testing.assert_allclose(
            improvement_percent, 100 * (1 - mean_mae / mean_mae["seasonal-naive"])
        )
        display(pd.DataFrame({"MAE": mean_mae, "improvement_vs_naive_%": improvement_percent}))
        celebrate("investigator", "You measured improvement, not just model size.")
        """,
        """
        mean_mae = all_scores.groupby("model")["mae"].mean().sort_values()
        improvement_percent = 100 * (1 - mean_mae / mean_mae["seasonal-naive"])
        display(pd.DataFrame({"MAE": mean_mae, "improvement_vs_naive_%": improvement_percent}))
        celebrate("investigator", "You measured improvement, not just model size.")
        """,
        "Use `all_scores.groupby('model')['mae'].mean().sort_values()`; normalize by naive.",
    ),
    code("""
    fig = go.Figure(go.Bar(
        x=improvement_percent.index, y=improvement_percent.to_numpy(),
        marker_color=[MODEL_COLORS.get(m, "#999999") for m in improvement_percent.index],
        text=[f"{v:+.1f}%" for v in improvement_percent], textposition="outside",
    ))
    fig.update_layout(
        title="Improvement over seasonal naive (above 0 = better, below 0 = worse)",
        yaxis_title="Improvement in mean MAE (%)", height=380, margin=dict(t=60),
    )
    fig.add_hline(y=0, line_color=INK)
    fig.show()
    """),
    md("""
    ## 4. A mean can hide a fragile winner (8 minutes)
    Look at each fold, not only the average. Three dependent time-series
    origins are not enough for a confident significance claim. The largest
    error may coincide with a calendar effect that no model was explicitly told.
    Future weather observations would leak; a holiday calendar known at the
    forecast date would not.
    """),
    code("""
    display(leaderboard(all_scores).round(2))
    fold_errors = all_scores.pivot(index="fold", columns="model", values="mae")
    display(fold_errors.round(1))
    fig = go.Figure([
        go.Bar(x=[f"Fold {f}" for f in fold_errors.index], y=fold_errors[m], name=m,
               marker_color=MODEL_COLORS.get(m, "#999999"))
        for m in fold_errors.columns
    ])
    fig.update_layout(
        barmode="group", title="Does the winner change by forecast origin?",
        yaxis_title="MAE (births/day)", height=380, margin=dict(t=60),
    )
    fig.show()

    selected_fold = 3  # Try 1 or 2 after inspecting the score table
    part = all_predictions.loc[all_predictions["fold"] == selected_fold].copy()
    part["date"] = pd.to_datetime(part["date"])
    actual = part.drop_duplicates("date").set_index("date")["actual"]
    fig = new_figure(f"Fold {selected_fold}: shape, timing, and misses")
    add_line(fig, actual, "Actual", INK, width=4)
    for name, group in part.groupby("model"):
        add_line(
            fig, group.set_index("date")["prediction"], name,
            MODEL_COLORS.get(name, "#999999"), dash="dash", markers=True,
        )
    fig.show()
    """),
    md("""
    Do errors grow the further ahead we forecast? Each line averages the
    absolute error at day 1, day 2, ... day 14 across the three folds.
    Look for a steady climb, then look for a **spike**: what could happen on
    one specific day that no model was told about?
    """),
    code("""
    errors = all_predictions.sort_values(["model", "fold", "date"]).copy()
    errors["abs_error"] = (errors["prediction"] - errors["actual"]).abs()
    errors["days_ahead"] = errors.groupby(["model", "fold"]).cumcount() + 1
    by_step = errors.groupby(["model", "days_ahead"])["abs_error"].mean().reset_index()
    fig = new_figure("Error by days ahead: a steady climb, or a spike?", ylabel="Mean absolute error (births/day)")
    for name, group in by_step.groupby("model"):
        add_line(fig, group.set_index("days_ahead")["abs_error"], name,
                 MODEL_COLORS.get(name, "#999999"), markers=True)
    fig.update_layout(xaxis_title="Days ahead")
    fig.show()
    """),
    md("""
    ### YOUR TURN — make a recommendation (3 minutes)
    Write: **"For this series, horizon, and three cutoffs, I would choose ___
    because ___. Before deployment I would test ___."**

    Consider accuracy, consistency across folds, setup effort, inference time,
    and missing evidence. If no foundation results were available, explicitly
    say the foundation-versus-baseline question remains unanswered.
    """),
    code("""
    recommendation = ""
    if recommendation.strip():
        celebrate("decision", "You made a qualified, evidence-based recommendation.")
    else:
        print("Write your recommendation or discuss it with a partner, then record one sentence.")
    """, tags=["reflection"]),
    md("""
    ## Finish line (90 minutes)
    Revisit your opening vote. Did you change your mind? What evidence did it?

    **Exit ticket:** name one leakage risk, one useful baseline, and one thing
    this experiment cannot establish.

    You now know the transferable workflow:
    **define decision → inspect history → establish baseline → forecast →
    backtest fairly → discuss limitations.**

    ### Optional after the core
    - Add Prophet with the separate dependency and the same benchmark below.
    - Evaluate 80% interval coverage and interval width with a wrapper that
      actually supports quantiles. Three folds are not a calibration study.
    - Add known-future calendar covariates in a separate experiment; compare
      models on equivalent available information, not targets disguised as X.
    - Change horizon/context and declare a **new** protocol; don't tune on
      these folds and call the resulting score an untouched test estimate.
    - Read the Kumo Forecast / Relational section in `TUTORIAL.md`.
    """),
    code("""
    RUN_PROPHET = False  # Optional. First install it: uv pip install -r requirements/prophet.txt
    if RUN_PROPHET:
        prophet_scores, prophet_predictions = run_benchmark("prophet", y, cv=cv)
        display(leaderboard(pd.concat([all_scores, prophet_scores], ignore_index=True)))
    """, tags=["optional-prophet"]),
]

for name, cells in [
    ("00_setup_check.ipynb", notebook0),
    ("01_forecast_detectives.ipynb", notebook1),
    ("02_fair_forecast_arena.ipynb", notebook2),
]:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    folder = ROOT / "notebooks"
    folder.mkdir(exist_ok=True)
    nbf.validate(notebook)
    nbf.write(notebook, folder / name)
    print(f"Created {name}: {len(cells)} cells")