"""Readiness check; never downloads anything (use download_weights.py for that)."""
import argparse
import gc
from importlib.metadata import version
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workshop import (
    CHECKPOINTS, FOUNDATIONS, _is_cached, load_series, make_model, demo_split,
    validate_prediction,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--foundation", action="store_true", help="check the foundation-model imports")
parser.add_argument(
    "--weights", action="store_true",
    help="implies --foundation; also require the downloaded weights and run a real "
    "offline 14-day forecast with each foundation model",
)
args = parser.parse_args()
args.foundation = args.foundation or args.weights
print("Python:", sys.version)
for package in ("sktime", "numpy", "pandas", "statsmodels"):
    print(package, version(package))
train, truth = demo_split(load_series())
model = make_model("seasonal-naive")
model.fit(train, fh=list(range(1, 15)))
validate_prediction(model.predict(), truth)
print("BASELINE READY: data loaded and a 14-day forecast succeeded.")
if args.foundation:
    import importlib

    import torch

    # Each model needs its sktime wrapper plus the library that does inference.
    checks = {
        "chronos2": ("sktime.forecasting.chronos2", "chronos"),
        "timesfm25": ("sktime.forecasting.timesfm2", "transformers"),
        "t0-gamma": ("sktime.forecasting.t0", "t0"),
    }
    failed = []
    for model, modules in checks.items():
        try:
            for module in modules:
                importlib.import_module(module)
            if model == "timesfm25":
                from transformers import TimesFm2_5ModelForPrediction  # noqa: F401
            if model == "chronos2":
                from chronos import Chronos2Pipeline  # noqa: F401
            print(f"  {model}: imports OK")
        except Exception as error:
            failed.append(model)
            print(f"  {model}: FAILED ({type(error).__name__}: {error})")
    for package in ("torch", "transformers", "chronos-forecasting", "tfc-t0"):
        print(package, version(package))
    print("CUDA available:", torch.cuda.is_available(), "(a GPU is not necessary)")
    if failed:
        sys.exit(f"FOUNDATION MODELS NOT READY: {', '.join(failed)}")
    print("FOUNDATION MODELS READY: Chronos-2, TimesFM 2.5 and t0-gamma import fine.")
    if not args.weights:
        print("This checks imports only; each model's weights download on its first run.")
        print("After scripts/download_weights.py, run this with --weights to verify them.")
        sys.exit(0)

    # make_model switches Hugging Face to offline mode for cached weights, so a
    # missing or incomplete download fails here instead of silently downloading.
    for name in FOUNDATIONS:
        if not _is_cached(name):
            failed.append(name)
            print(f"  {name}: weights NOT DOWNLOADED ({CHECKPOINTS[name]})")
            continue
        forecaster = None
        try:
            start = time.perf_counter()
            forecaster = make_model(name)
            forecaster.fit(train, fh=list(range(1, 15)))
            validate_prediction(forecaster.predict(), truth)
            print(f"  {name}: weights OK, 14-day forecast in {time.perf_counter() - start:.1f}s")
        except Exception as error:
            failed.append(name)
            print(f"  {name}: FAILED ({type(error).__name__}: {error})")
        finally:
            del forecaster
            gc.collect()
    if failed:
        sys.exit(
            f"WEIGHTS NOT READY: {', '.join(failed)}. "
            "Run scripts/download_weights.py (re-run it if a download was interrupted)."
        )
    print("WEIGHTS READY: all three models ran a real forecast from the local cache.")
    print("Later runs need no download, even without internet.")
