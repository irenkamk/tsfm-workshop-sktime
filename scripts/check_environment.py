"""Lightweight readiness check; no weights are downloaded."""
import argparse
from importlib.metadata import version
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workshop import load_series, make_model, demo_split, validate_prediction

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--foundation", action="store_true")
args = parser.parse_args()
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
    print("This checks imports only; each model's weights download on its first run.")
    print("Run scripts/rehearse.py to test real inference for one model.")
