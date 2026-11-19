"""Generate notebooks/00_setup_check.ipynb (the lesson notebook has its own generator,
build_united_notebook.py)."""
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text, **metadata):
    return nbf.v4.new_code_cell(dedent(text).strip(), metadata=metadata)


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


NAME = "00_setup_check.ipynb"
notebook = nbf.v4.new_notebook(
    cells=notebook0,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
)
folder = ROOT / "notebooks"
folder.mkdir(exist_ok=True)
nbf.validate(notebook)
nbf.write(notebook, folder / NAME)
print(f"Created {NAME}: {len(notebook0)} cells")
