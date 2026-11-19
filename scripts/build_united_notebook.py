"""Generate notebooks/01_forecasting_workshop.ipynb, the single lesson notebook.

One 90-minute flow (it replaced the earlier notebooks 01 and 02). The setup
notebook 00 has its own generator, build_notebooks.py.
"""
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NAME = "01_forecasting_workshop.ipynb"
_count = 0


def _cell_id():
    global _count
    _count += 1
    return f"cell-{_count:02d}"  # stable ids: rebuilding does not create diff noise


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip(), id=_cell_id())


def code(text, **metadata):
    return nbf.v4.new_code_cell(dedent(text).strip(), metadata=metadata, id=_cell_id())


def exercise(number, title, minutes, prompt, starter, answer, checks, hint):
    """Prompt, a starter cell (placeholders + checks), then a hint and solution.

    The solution shown to participants is only the lines to replace; the full
    solved cell (answer + checks) is kept in metadata for check_notebooks.py.
    """
    starter, answer, checks = (dedent(text).strip() for text in (starter, answer, checks))
    return [
        md(f"#### YOUR TURN {number} — {title} ({minutes} min)\n\n{dedent(prompt).strip()}"),
        code(
            f"{starter}\n\n# --- Checks: leave these lines as they are ---\n{checks}",
            tags=["exercise"], solution=f"{answer}\n\n{checks}",
        ),
        md(
            f"**Hint:** {dedent(hint).strip()}\n\n<details><summary>Show solution "
            "(replace the lines above the checks)</summary>\n\n"
            f"```python\n{answer}\n```\n\n</details>"
        ),
    ]


cells = [
    # ------------------------------------------------------------------ top
    md("""
    # Can AI Predict the Future?
    ### A hands-on forecasting workshop with sktime

    **90 minutes · pairs welcome · 8 short exercises**

    On one small daily dataset we ask: **does a pretrained "foundation" model
    beat a sensible seasonal baseline?** By the end you can prepare a time
    series, split it without leaking the future, build a baseline, compare
    models fairly, and say what the evidence does and does not show.

    ## Contents

    | Part | What you do | Time |
    |---|---|---|
    | **0. Setup** | Make your prediction, load the workshop files, install packages if needed, check the environment | 5 min |
    | **1. Look at the data** | Plot the series and spot the weekly pattern | 8 min |
    | **2. Make one forecast** | Split the data without leaking the future, build a baseline, try a pretrained model | 32 min |
    | **3. Compare the models** | Backtest, error measures, leaderboard, recommendation | 40 min |
    | **4. Wrap-up** | Revisit your vote, next steps | 5 min |

    ## How this notebook works

    - **RUN THIS** cells are ready to go: press Shift+Enter.
    - **YOUR TURN** cells have `None` placeholders. Replace them, leave the
      checks below as they are, and run the cell: the checks tell you whether
      you got it right. Each exercise has a **hint** and a collapsed **solution**.
    - The only settings you may want to change are `MODEL` and `RUN_LIVE` in
      Part 2.3. Everything else works as it is.
    - Curious what a helper does? Functions such as `demo_split` and
      `leaderboard` live in `workshop.py`. Run `help(demo_split)` (or
      `demo_split?`) to read about one. You never need to edit that file.
    - No points for speed. Getting help counts as progress. Feel free to also work in pairs.
    """),
    # ------------------------------------------------------------ part 0
    md("""
    ## Part 0 — Setup (5 min)

    ### Before we start (2 min)
    Predict who will win: **repeat last week**, a **classical statistical
    model**, or a **pretrained transformer**. Write **one reason** and keep it
    for the end.

    ### Load the workshop files (3 min)
    **Colab:** when asked, upload the workshop ZIP (repository page → green
    **Code → Download ZIP**, or from the organizer). **Local:** start Jupyter
    from the workshop folder.

    This cell does not install anything or download model weights. Packages
    missing? Use the install cell right after it.
    """),
    code("""
    import os
    import sys
    import zipfile
    from pathlib import Path

    here = Path.cwd()
    # Look for workshop.py here, one folder up, or in an already extracted ZIP folder.
    ROOT = next((p for p in (here, here.parent, *here.glob("*")) if (p / "workshop.py").exists()), None)
    if ROOT is None:
        if "google.colab" not in sys.modules:
            raise FileNotFoundError("Extract the ZIP and start Jupyter in the workshop folder.")
        from google.colab import files
        (zip_name,) = files.upload()  # upload exactly one file: the workshop ZIP
        zipfile.ZipFile(zip_name).extractall(here)
        ROOT = next(p for p in here.glob("*") if (p / "workshop.py").exists())
    os.chdir(ROOT)
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
    """),
    md("""
    ### Check the setup and load the tools
    """),
    code("""
    # Runs scripts/check_environment.py: a quick forecast on the bundled data.
    check = subprocess.run(
        [sys.executable, "scripts/check_environment.py"], capture_output=True, text=True
    )
    if check.returncode:
        raise RuntimeError(
            check.stdout + check.stderr
            + "\\nSet INSTALL_CORE = True in the install cell above and run it, or see SETUP.md."
        )
    print(check.stdout.strip().splitlines()[-1])

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    from IPython.display import display
    from workshop import (
        load_series, demo_split, make_model, validate_prediction, benchmark_series,
        splitter, run_benchmark, leaderboard, load_run,
        CONTEXT, HORIZON, N_FOLDS, SEASON, BASELINES, FOUNDATIONS,
    )

    y = load_series()
    print(f"Loaded {len(y)} daily observations: {y.index.min().date()} to {y.index.max().date()}")
    """),
    md("""
    *Plot helpers, a milestone counter and a loader for saved results. You can
    skip reading this cell.*
    """),
    code("""
    pio.templates.default = "plotly_white"
    BLUE, ORANGE, GREEN, INK = "#0072B2", "#D55E00", "#009E73", "#333333"  # colour-blind safe
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


    milestones = {}


    def celebrate(key, message):
        milestones[key] = message
        done = len(milestones)
        print(f"[{'■' * done}{'□' * (8 - done)}] {done}/8 milestones: {message}")


    def load_saved(name):
        # A saved real run: yours (artifacts-live) first, else the instructor's (artifacts).
        for folder, owner in (("artifacts-live", "your own"), ("artifacts", "instructor's")):
            path = ROOT / folder / name
            if path.exists():
                scores, predictions, metadata = load_run(path, y)
                assert metadata["model"] == name, "Saved model does not match its folder."
                demo = pd.read_csv(path / "demo.csv", parse_dates=["date"]).set_index("date")
                return {"scores": scores, "predictions": predictions, "demo": demo, "owner": owner}
        return None
    """),
    # ------------------------------------------------------------ part 1
    md("""
    ## Part 1 — Look at the data (8 min)

    ### 1.1 Look at the data (4 min)
    We have **420 daily counts of U.S. births** (FiveThirtyEight / SSA, licensed
    CC BY 4.0; see `data/README.md`). This is not a medical decision tool.

    - **Frequency** is how often we measure. Here: once a day.
    - **Seasonality** is how often a pattern repeats. Here we test 7 days.

    These are two different things.

    The **last 42 days are set aside** for Part 3, before we look at anything.
    Until then we only use `development`: everything before those 42 days.
    """),
    code("""
    development = y.iloc[:-N_FOLDS * HORIZON]  # everything except the last 42 days
    print(f"Development data: {development.index.min().date()} to {development.index.max().date()}")
    print("Frequency:", development.index.freqstr, "(D = daily)")
    display(development.head().to_frame())
    """),
    *exercise(
        1, "Plot the series", 2,
        """
        Draw the development data as a line chart. Give it a title, then look:
        what repeats? What might surprise a model? Hover, zoom and drag over the
        chart to explore.
        """,
        starter="""
        fig = None  # Replace None with px.line(development, title="...")
        """,
        answer="""
        fig = px.line(development, title="What repeats? What might surprise a model?")
        """,
        checks="""
        assert fig is not None, "Hint: px.line(development, title=...)"
        assert len(fig.data) == 1 and len(fig.data[0].y) == len(development), "Plot the whole development series once."
        fig.update_layout(xaxis_title="Date", yaxis_title="Births per day")
        fig.show()
        celebrate("plot", "You plotted the data.")
        """,
        hint="`px.line` accepts a pandas Series directly: the index becomes the x axis.",
    ),
    md("""
    ### 1.2 Spot the weekly pattern (4 min)
    Do weeks look alike? One thin line per week (the last 12), plus the average
    week. Development data only.
    """),
    code("""
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    frame = development.to_frame("births")
    frame["weekday"] = frame.index.dayofweek
    frame["week_start"] = frame.index - pd.to_timedelta(frame["weekday"].to_numpy(), unit="D")
    by_week = frame.pivot(index="weekday", columns="week_start", values="births").dropna(axis=1)

    fig = new_figure("Weeks look alike: one thin line per week (last 12)")
    for i, week in enumerate(by_week.columns[-12:]):
        fig.add_scatter(
            x=weekday_names, y=by_week[week], mode="lines", hoverinfo="skip",
            name="Individual weeks", showlegend=(i == 0),
            line=dict(color=BLUE, width=1), opacity=0.45,
        )
    fig.add_scatter(x=weekday_names, y=by_week.mean(axis=1), name="Average week", line=dict(color=INK, width=3))
    fig.update_layout(xaxis_title="Day of week")
    fig.show()
    """),
    *exercise(
        2, "Average by weekday", 3,
        """
        Calculate the mean count for each weekday using **development only**
        (Monday is 0). Then talk about one possible reason for the pattern.
        It is a guess, not proof.
        """,
        starter="""
        weekday_means = None  # Group development by development.index.dayofweek, then take the mean
        """,
        answer="""
        weekday_means = development.groupby(development.index.dayofweek).mean()
        """,
        checks="""
        assert weekday_means is not None, "Hint: development.groupby(...).mean()"
        assert len(weekday_means) == 7, "We need one average per weekday."
        fig = px.bar(
            x=weekday_means.index, y=weekday_means.to_numpy(),
            title="Development data only: mean by weekday",
            labels={"x": "Weekday (Monday = 0)", "y": "Mean births per day"},
        )
        fig.show()
        celebrate("weekday", "You found the weekly pattern.")
        """,
        hint="Dates have `.dayofweek`. Grouping describes the past; it is not a forecast.",
    ),
    # ------------------------------------------------------------ part 2
    md("""
    ## Part 2 — Make one forecast (32 min)

    ### 2.1 Split the data into past and future (10 min)
    A forecast starts at a **cutoff**: the last day we are allowed to know.
    Three terms:

    - **Cutoff**: the last date we are allowed to see.
    - **Context**: the history we give the model (here 168 days). This is our training data.
    - **Horizon**: how far ahead we predict (here days 1 to 14).

    For this demo we hide the last 14 days of `development` (the test days)
    and use the 168 days before them as training data. Part 3 uses a different
    period.

    Never split a time series at random. A random split puts days from the
    future into the training data. This is called **leakage**: the model has
    seen something it could not know in real life, so its score looks better
    than it really is.
    """),
    *exercise(
        3, "Build a leak-free split", 4,
        """
        `CONTEXT` (168) and `HORIZON` (14) are already defined. Using
        `development`, create three things. No test value may end up in the
        training data.
        """,
        starter="""
        # 1. The future we pretend not to know: the last HORIZON days of `development`
        y_test = None

        # 2. The history the model may use: the CONTEXT days right before y_test
        y_train = None

        # 3. The forecast horizon: days 1, 2, ... HORIZON ahead
        fh = None
        """,
        answer="""
        y_test = development.iloc[-HORIZON:]
        y_train = development.iloc[-(CONTEXT + HORIZON):-HORIZON]
        fh = np.arange(1, HORIZON + 1)
        """,
        checks="""
        assert y_test is not None and y_train is not None and fh is not None, "Fill in all three."
        assert len(y_test) == HORIZON, f"y_test should hold the last {HORIZON} days, not {len(y_test)}."
        assert len(y_train) == CONTEXT, f"y_train should hold {CONTEXT} days, not {len(y_train)}."
        assert y_train.index[-1] + pd.Timedelta(days=1) == y_test.index[0], (
            "y_train must end the day before y_test starts: no gap, no overlap, no leak."
        )
        assert list(fh) == list(range(1, HORIZON + 1)), "Forecast horizons start at 1, not 0."
        assert y_test.index[-1] == development.index[-1], (
            "Slice `development`, not `y`: the last 42 days stay set aside until Part 3."
        )
        expected_train, expected_test = demo_split(y)
        pd.testing.assert_series_equal(y_train, expected_train)
        pd.testing.assert_series_equal(y_test, expected_test)
        celebrate("cutoff", "Your split keeps the future out of the training data.")
        """,
        hint="""
        `iloc[-14:]` means "the last 14 rows", so `y_test` is
        `development.iloc[-HORIZON:]`. For `y_train`, remember that
        `iloc[start:stop]` stops *before* `stop` and that negative numbers count
        from the end: take the `CONTEXT` rows that end exactly where `y_test`
        begins, so `start` is `-(CONTEXT + HORIZON)` and `stop` is `-HORIZON`.
        For `fh`, `np.arange(1, HORIZON + 1)` counts 1, 2, ... 14.
        """,
    ),
    code("""
    fig = new_figure("Ask: could I have known this value at the cutoff?")
    add_line(fig, y_train.iloc[-42:], "Available history", BLUE)
    add_line(fig, y_test, "Test days (hidden from the model)", INK, dash="dash")
    mark_cutoff(fig, y_train.index[-1])
    fig.show()
    """),
    md("""
    ### 2.2 A simple baseline (10 min)
    A **baseline** is a simple forecast that every other model has to beat. If
    a complex model cannot beat it, the model is not worth using.

    Our baseline is **seasonal naive**: it repeats the last week. For days 8
    to 14 it repeats that same week again. It never looks at real future
    values. It learns nothing, and still it is often hard to beat.

    All sktime forecasters work the same way:

    - `fit(y_train, fh=fh)` gives the model the history and the horizon.
    - `predict()` returns the forecast, with dates.

    The pretrained models later use the same two calls.

    We measure the error with the **MAE**: the average size of the miss, here
    in births per day. Lower is better. Section 3.2 explains the MAE and other
    error measures.
    """),
    *exercise(
        4, "Forecast with seasonal naive", 4,
        """
        Create a `NaiveForecaster` with `strategy="last"` and the right seasonal
        period, fit it on the training data only, then predict.
        """,
        starter="""
        from sktime.forecasting.naive import NaiveForecaster

        # 1. Create the forecaster: repeat the last value of one season. How many days is a season?
        naive = None

        # 2. Fit it on the training data only, and pass the forecast horizon fh. Write the fit line here:


        # 3. Predict
        naive_pred = None
        """,
        answer="""
        from sktime.forecasting.naive import NaiveForecaster

        naive = NaiveForecaster(strategy="last", sp=SEASON)
        naive.fit(y_train, fh=fh)
        naive_pred = naive.predict()
        """,
        checks="""
        assert naive is not None, "Step 1: naive = NaiveForecaster(strategy='last', sp=...)"
        assert naive.is_fitted, "Step 2: call naive.fit(y_train, fh=fh)"
        assert naive_pred is not None, "Step 3: naive_pred = naive.predict()"
        validate_prediction(naive_pred, y_test)
        np.testing.assert_allclose(
            naive_pred, np.tile(y_train.iloc[-SEASON:], 2),
            err_msg="Fit on y_train only, and use sp=7 (one week), not the horizon 14.",
        )
        celebrate("baseline", "You made your first forecast.")
        """,
        hint="""
        A season is one week: 7 days (not the horizon, 14). The steps are
        `naive.fit(y_train, fh=fh)`, then `naive_pred = naive.predict()`.
        """,
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
    ### 2.3 A pretrained model (12 min)
    A **foundation model** is a large neural network that was already trained
    on a huge collection of time series from many areas. **Zero-shot** means
    we use it as it is, without training it on our births data. Calling `fit`
    only gives it the history to read.

    Three models to choose from:

    - **Chronos-2** (Amazon): predicts the next 14 days directly from the history.
    - **TimesFM 2.5** (Google): writes the forecast one block of days after another, like a language model writes text.
    - **t0-gamma** (The Forecasting Company): predicts many days ahead in parallel.

    `TUTORIAL.md` has a section on each of them.

    **Choose one** in the cell below. By default you see the instructor's saved
    forecast: nothing to download or run. Set `RUN_LIVE = True` to run the
    model yourself. That needs the foundation packages and the downloaded model
    weights (see `PARTICIPANT_PREP.md`). It also runs that model again in the
    Part 3 comparison, which takes a minute or two on a CPU. If there is no
    saved result and `RUN_LIVE` is `False`, the notebook says so and shows no
    AI forecast. It never makes one up.

    *t0-gamma has its own license, linked in `workshop.py`. Running it means
    you accept its terms.*
    """),
    code("""
    MODEL = "chronos2"  # or "timesfm25" or "t0-gamma"
    RUN_LIVE = False    # True: run the model here. False: use the saved forecast.

    assert MODEL in FOUNDATIONS, f"Choose one of {FOUNDATIONS}."
    foundation_pred = None
    saved = None if RUN_LIVE else load_saved(MODEL)
    if RUN_LIVE:
        model = make_model(MODEL)
        model.fit(y_train, fh=fh)
        foundation_pred = model.predict()
        source = "run live in this notebook"
    elif saved is not None:
        np.testing.assert_array_equal(saved["demo"]["actual"], y_test)
        foundation_pred = saved["demo"]["prediction"].asfreq("D")
        source = f"saved result, {saved['owner']} run"

    if foundation_pred is None:
        print(f"NOT EVALUATED: no saved {MODEL} run. Set RUN_LIVE = True or ask the instructor.")
    else:
        validate_prediction(foundation_pred, y_test)
        fig = new_figure(f"{MODEL}: {source}")
        add_line(fig, y_train.iloc[-28:], "History", BLUE)
        add_line(fig, y_test, "Actual", INK, width=3)
        add_line(fig, naive_pred, "Seasonal naive", ORANGE, dash="dash")
        add_line(fig, foundation_pred, MODEL, MODEL_COLORS[MODEL], dash="dashdot", markers=True)
        mark_cutoff(fig, y_train.index[-1])
        fig.show()
        display(pd.Series({
            "seasonal-naive": mean_absolute_error(y_test, naive_pred),
            MODEL: mean_absolute_error(y_test, foundation_pred),
        }, name="Demo MAE (births/day)").to_frame().round(1))
    """, tags=["optional-foundation"]),
    md("""
    #### YOUR TURN 5 — Describe what you see (2 min)
    Write two sentences: where did the forecast work, where did it miss, and
    why is one split not enough to judge a model? If there is no AI result,
    write what you would need to see first. Do not say it won or lost.
    """),
    code("""
    reflection = ""  # Where did the forecast work or miss? Why is one split not enough?
    if reflection.strip():
        celebrate("explorer", "You described what you saw.")
    else:
        print("Write two sentences, then rerun. You can also say them out loud to a partner.")
    """, tags=["reflection"]),
    # ------------------------------------------------------------ part 3
    md("""
    ## Part 3 — Compare the models (40 min)

    ### 3.1 Backtesting: test on several periods (8 min)
    One split can be lucky. A **backtest** repeats Part 2 three times. Each
    round is called a **fold**:

    1. Take the 168 days before the cutoff as training data.
    2. Forecast the next 14 days.
    3. Compare the forecast with what really happened.
    4. Move the cutoff forward by 14 days and start again.

    The three test windows together are the 42 days we set aside in Part 1.

    ```
    fold 1: [          168 context days          ][14 test]
    fold 2:      [          168 context days          ][14 test]
    fold 3:           [          168 context days          ][14 test]
    ```

    The test days of an earlier fold can be part of the training data of a later
    fold. That is fine: by the later cutoff those days are in the past. No fold
    sees its own test days.
    """),
    *exercise(
        6, "Design the splitter", 4,
        """
        Use `SlidingWindowSplitter` with a 14-step horizon, a 168-day window and
        a 14-day step. The workshop's own `splitter()` does the same thing, and
        the checks compare yours with it.
        """,
        starter="""
        from sktime.split import SlidingWindowSplitter

        # Rolling windows: forecast horizon, context length, and how far each fold moves forward
        cv = None  # SlidingWindowSplitter(fh=..., window_length=..., step_length=...)
        """,
        answer="""
        from sktime.split import SlidingWindowSplitter

        cv = SlidingWindowSplitter(
            fh=np.arange(1, HORIZON + 1), window_length=CONTEXT, step_length=HORIZON
        )
        """,
        checks="""
        assert cv is not None, "Set fh, window_length and step_length."
        evaluation_y = benchmark_series(y)
        folds = list(cv.split(evaluation_y))
        assert len(folds) == N_FOLDS, f"There should be {N_FOLDS} folds (cutoffs), not {len(folds)}."
        for (train_idx, test_idx), (expected_train, expected_test) in zip(
            folds, splitter().split(evaluation_y)
        ):
            np.testing.assert_array_equal(train_idx, expected_train, err_msg="Check window_length.")
            np.testing.assert_array_equal(test_idx, expected_test, err_msg="Check fh and step_length.")
            assert train_idx.max() < test_idx.min()
        celebrate("designer", "Your backtest only uses the past.")
        """,
        hint="""
        The horizon is `np.arange(1, HORIZON + 1)` (days 1 to 14). The window is
        `CONTEXT` days. A step of `HORIZON` days moves each fold forward by one
        test window, so the test windows do not overlap.
        """,
    ),
    code("""
    schedule = pd.DataFrame([
        {"fold": number,
         "context_start": evaluation_y.index[train[0]].date(),
         "cutoff": evaluation_y.index[train[-1]].date(),
         "test_start": evaluation_y.index[test[0]].date(),
         "test_end": evaluation_y.index[test[-1]].date()}
        for number, (train, test) in enumerate(folds, start=1)
    ])
    display(schedule)
    """),
    md(r"""
    ### 3.2 How we measure the error (4 min)
    We agree on the error measure **before** we see which model wins. These are
    the common ones. In the formulas, $y_t$ is the actual value on day $t$,
    $\hat{y}_t$ is the forecast, and $n$ is the number of forecast days.

    | Short | Spelled out | Formula | In plain words |
    |---|---|---|---|
    | **MAE** | Mean Absolute Error | $\frac{1}{n}\sum_t \lvert y_t-\hat{y}_t\rvert$ | The average size of the miss, in the unit of the data (here births per day). |
    | **RMSE** | Root Mean Squared Error | $\sqrt{\frac{1}{n}\sum_t (y_t-\hat{y}_t)^2}$ | Like MAE, but big misses count much more, because errors are squared. |
    | **MAPE** | Mean Absolute Percentage Error | $100\cdot\frac{1}{n}\sum_t \frac{\lvert y_t-\hat{y}_t\rvert}{\lvert y_t\rvert}$ | The average miss as a percent of the actual value. |
    | **WAPE** | Weighted Absolute Percentage Error | $100\cdot\frac{\sum_t \lvert y_t-\hat{y}_t\rvert}{\sum_t \lvert y_t\rvert}$ | The total miss as a percent of the total actual value. |
    | **MASE** | Mean Absolute Scaled Error | $\frac{\text{MAE}}{\frac{1}{T-m}\sum_{t=m+1}^{T}\lvert y_t-y_{t-m}\rvert}$ | MAE divided by the average miss of "repeat the value from one season ago" on the training data. It has no unit. |

    For MASE, $T$ is the number of training days and $m$ is the season length
    (7 days).

    **What we use.** MAE is our main score. It is easy to explain and its unit
    is births per day. MASE is a second check without a unit: below 1 means
    the model misses by less than the seasonal baseline does on the training
    data. That does **not** prove it beats seasonal naive on the test days. For
    that, compare the test MAEs directly.

    **Why not the others here?**

    - **MAPE** divides by each day's actual value. It blows up when an actual
      value is close to zero and is undefined at zero. It also punishes a
      forecast that is too high more than one that is too low.
    - **RMSE** is useful when big misses are much worse for you than small
      ones. A single spike can dominate it.
    - **WAPE** is for adding up many series, for example sales of many
      products. It divides the total error by the total actual value, so big
      series count more than small ones, and one percentage describes the whole
      group. It also works when some single values are zero, where MAPE breaks.
      We forecast one series that is never near zero, so MAE does the job.

    Run times in the tables below are only a rough guide: first loads,
    downloads and warm caches change them.
    """),
    md("""
    ### 3.3 Run the candidates (12 min)
    The models we compare are the **candidates**. The one with the lowest error
    will be the **champion**.

    `run_benchmark` in `workshop.py` tests every candidate in exactly the same
    way. In each fold it:

    1. fits the model on that fold's 168 training days,
    2. forecasts the next 14 days,
    3. scores the forecast with MAE and MASE.

    **Yes, every candidate is fitted again in every fold.** In sktime this is
    the setting `strategy="refit"`:

    ```python
    evaluate(forecaster=model, y=y, cv=cv, strategy="refit", scoring=[MAE, MASE])
    ```

    What "fitted again" means depends on the model. ARIMA and ETS estimate
    their numbers from the training days each time, so they are really
    retrained. Seasonal naive only remembers the last week. The foundation
    models are **not** trained at all: `fit` just gives them the new 168 days
    to read.

    `run_benchmark` also keeps the actual values and the forecasts, and stops
    with an error instead of ranking `NaN` values. You can read its source with
    `inspect.getsource(run_benchmark)`.

    **Classical candidates** run live here (about 30 seconds in total, ARIMA is
    the slowest):

    - **Seasonal naive**: repeats the last week.
    - **ARIMA**: predicts each day from the previous day, the previous error
      and the same weekday a week earlier (a fixed seasonal ARIMA(1,0,1)(1,0,0)[7]).
    - **ETS**: keeps a running estimate of the level, the trend and the weekly
      pattern, and gives recent days more weight (exponential smoothing).

    **Foundation model candidates** (Chronos-2, TimesFM 2.5 and t0-gamma, see
    2.3) come from the saved results, unless you set `RUN_LIVE = True` for your
    `MODEL`. A model with no result is labelled **NOT EVALUATED** and is not
    ranked.

    The classical settings are sensible choices for teaching. Nobody tuned them
    to be the best possible. `TUTORIAL.md` has more on each model.
    """),
    code("""
    score_parts, prediction_parts, evidence = [], [], []
    for name in BASELINES + FOUNDATIONS:
        saved = load_saved(name) if name in FOUNDATIONS else None
        if name in BASELINES or (RUN_LIVE and name == MODEL):
            print(f"Running {name} ...")
            scores, predictions = run_benchmark(name, y, cv=cv)
            status = "run live in this notebook"
        elif saved is not None:
            scores, predictions = saved["scores"], saved["predictions"]
            status = f"saved result, {saved['owner']} run"
        else:
            evidence.append({"model": name, "status": "NOT EVALUATED: no saved run"})
            continue
        score_parts.append(scores)
        prediction_parts.append(predictions)
        evidence.append({"model": name, "status": status})

    all_scores = pd.concat(score_parts, ignore_index=True)
    all_predictions = pd.concat(prediction_parts, ignore_index=True)
    display(pd.DataFrame(evidence))
    """),
    md("""
    ### 3.4 The leaderboard (5 min)
    Every candidate was tested on the same three cutoffs, so the scores can be
    compared. The candidate with the lowest MAE is the champion.
    """),
    *exercise(
        7, "Build the leaderboard", 5,
        """
        Average the three fold MAEs for each model, lowest first. Then work out
        how much better or worse each model is than seasonal naive, in percent
        (negative means worse). Models without a result stay out of the table;
        do not fill them with zeros.
        """,
        starter="""
        # 1. Mean MAE per model, best first: group all_scores by model, take mae, mean, sort
        mean_mae = None

        # 2. Improvement over seasonal naive, in percent: 100 * (1 - model MAE / naive MAE)
        improvement_percent = None
        """,
        answer="""
        mean_mae = all_scores.groupby("model")["mae"].mean().sort_values()
        improvement_percent = 100 * (1 - mean_mae / mean_mae["seasonal-naive"])
        """,
        checks="""
        assert mean_mae is not None, "Step 1: all_scores.groupby('model')['mae'].mean().sort_values()"
        pd.testing.assert_series_equal(
            mean_mae, all_scores.groupby("model")["mae"].mean().sort_values()
        )
        assert improvement_percent is not None, "Step 2: 100 * (1 - mean_mae / mean_mae['seasonal-naive'])"
        np.testing.assert_allclose(
            improvement_percent, 100 * (1 - mean_mae / mean_mae["seasonal-naive"])
        )
        display(pd.DataFrame({"MAE": mean_mae, "improvement_vs_naive_%": improvement_percent}).round(1))
        celebrate("investigator", "You compared every model with the baseline.")
        """,
        hint="""
        Step 1 is `all_scores.groupby("model")["mae"].mean().sort_values()`.
        For step 2, divide by the seasonal-naive entry of that same series:
        `mean_mae["seasonal-naive"]`.
        """,
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
    ### 3.5 Look at each fold (8 min)
    The average can hide differences between folds. Is the same model best in
    every fold? Three folds are too few to be sure of anything, and they are
    close in time, so they are not independent.

    The biggest error may fall on a day with a special event that no model knew
    about. A holiday calendar is known in advance, so using it would be fine.
    Weather during the forecast period is not known in advance, so using it
    would be leakage.
    """),
    code("""
    display(leaderboard(all_scores).round(2))  # means over the folds; lower MAE is better

    fold_errors = all_scores.pivot(index="fold", columns="model", values="mae")
    fig = go.Figure([
        go.Bar(x=[f"Fold {f}" for f in fold_errors.index], y=fold_errors[m], name=m,
               marker_color=MODEL_COLORS.get(m, "#999999"))
        for m in fold_errors.columns
    ])
    fig.update_layout(
        barmode="group", title="Is the same model best in every fold?",
        yaxis_title="MAE (births/day)", height=380, margin=dict(t=60),
    )
    fig.show()
    """),
    md("""
    Now look at the forecasts themselves. Is there a day where every model
    misses? What could have happened on that day?
    """),
    code("""
    selected_fold = 3  # try 1 or 2 after looking at the table above

    part = all_predictions.loc[all_predictions["fold"] == selected_fold].copy()
    part["date"] = pd.to_datetime(part["date"])
    actual = part.drop_duplicates("date").set_index("date")["actual"]
    fig = new_figure(f"Fold {selected_fold}: actual values and forecasts")
    add_line(fig, actual, "Actual", INK, width=4)
    for name, group in part.groupby("model"):
        add_line(
            fig, group.set_index("date")["prediction"], name,
            MODEL_COLORS.get(name, "#999999"), dash="dash", markers=True,
        )
    fig.show()
    """),
    md("""
    ### 3.6 Write a recommendation (3 min)
    #### YOUR TURN 8 — Your recommendation
    Complete this sentence: **"For this series, horizon and three cutoffs, I
    would choose ___ because ___. Before using it for real I would test ___."**

    Think about accuracy, how much the result changes between folds, setup
    effort, run time, and what evidence is missing. If there were no foundation
    model results, say that the question "does a foundation model beat the
    baseline?" is still open.
    """),
    code("""
    recommendation = ""
    if recommendation.strip():
        celebrate("decision", "You wrote a recommendation.")
    else:
        print("Write your recommendation (or discuss it with a partner), then record one sentence.")
    """, tags=["reflection"]),
    # ------------------------------------------------------------ part 4
    md("""
    ## Part 4 — Wrap-up (5 min)

    Look at your first prediction again. Did you change your mind? What changed it?

    **Talk it over with your partner:**

    - How can the future leak into a forecast?
    - Why do we compare a complex model with a simple baseline?
    - What can this experiment not tell us?

    The steps you practised: decide what to forecast → look at the history →
    set a baseline → forecast → backtest → discuss the results, the limits and
    the next steps.

    ### Optional: next steps
    - Add Prophet (first install it with `uv pip install -r requirements/prophet.txt`):

      ```python
      prophet_scores, _ = run_benchmark("prophet", y, cv=cv)
      display(leaderboard(pd.concat([all_scores, prophet_scores], ignore_index=True)))
      ```

    - Check how often an 80% prediction interval contains the actual value, and
      how wide it is. This needs a model wrapper that returns quantiles. The
      sktime Chronos-2 wrapper we use does not return intervals, and three folds
      are too few to judge it.
    - Add extra variables (covariates) that are known in advance, in a separate
      experiment. The dataset has only the date and the births, but you can
      compute the weekday from the date, and
      `pandas.tseries.holiday.USFederalHolidayCalendar` lists U.S. public
      holidays. Two of them fall in our test days: Thanksgiving (27 Nov 2014,
      fold 1) and Christmas Day (fold 3). Every model over-forecasts both days
      by about 5,000 births. ARIMA (`SARIMAX`), Chronos-2 and t0-gamma accept
      extra variables; seasonal naive, ETS and the TimesFM 2.5 adapter do not.
      Compare models only when they get the same information.
    - Change the horizon or the context length and treat it as a **new** setup.
      Do not tune on these folds and then call the score a clean test result.
    - Read the Kumo Forecast / Relational section in `TUTORIAL.md`.
    """),
]

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
nbf.write(notebook, folder / NAME)
print(f"Created {NAME}: {len(cells)} cells")
