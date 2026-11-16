"""Package an allowlist of workshop materials, excluding environments and secrets."""
import argparse
from pathlib import Path
import zipfile

import nbformat

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument(
    "--mode", choices=("full", "setup"), default="full",
    help="full: everything incl. notebooks and instructor files. "
    "setup: environment, data, helpers and the 00 setup-check notebook only; "
    "no lesson notebooks, saved results, slides, or solution scripts.",
)
args = parser.parse_args()
setup_only = args.mode == "setup"

# Files both packages need so participants can install, check, and later run.
common = [
    "README.md", "PARTICIPANT_PREP.md", "SETUP.md", "pyproject.toml", "uv.lock",
    "requirements/core.txt", "requirements/jupyter.txt", "requirements/foundation.txt",
    "requirements/prophet.txt", "workshop.py", ".gitignore",
    "data/README.md", "data/daily_births.csv",
    "scripts/check_environment.py", "scripts/rehearse.py",
]
instructor = [
    "TUTORIAL.md", "slides.md",
    ".github/skills/prepare-tsfm-workshop/SKILL.md",
]
files = [ROOT / name for name in common]
if not setup_only:
    files += [ROOT / name for name in instructor]
    files += sorted(
        path for path in (ROOT / "scripts").glob("*.py") if path not in files
    )
    files += sorted((ROOT / "tests").glob("test_*.py"))
    files += sorted((ROOT / "notebooks").glob("*.ipynb"))
else:
    # Only the spoiler-free setup check; the lesson notebooks come on the day.
    files.append(ROOT / "notebooks/00_setup_check.ipynb")
for relative in () if setup_only else (
    "baselines/seasonal-naive", "baselines/arima", "baselines/ets",
    "chronos2", "timesfm25", "t0-gamma", "prophet",
):
    folder = ROOT / "artifacts" / relative
    if folder.is_dir():
        files += [folder / name for name in ("demo.csv", "scores.csv", "predictions.csv", "metadata.json")]
assert not any(path.name == "INSTRUCTOR.md" for path in files), "instructor notes never ship"
args.output.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(args.output, mode="x", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f"Refusing a path outside the workshop: {path}")
        name = str(Path("tsfm-workshop") / path.relative_to(ROOT))
        if path.suffix == ".ipynb":
            notebook = nbformat.read(path, as_version=4)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.outputs = []
                    cell.execution_count = None
            archive.writestr(name, nbformat.writes(notebook))
        else:
            archive.write(path, name)
print(f"Created {args.output} ({args.mode}) with {len(files)} files.")