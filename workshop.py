"""Shared data, model factories, and one sktime evaluation path."""

from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import logging
import os
import platform
import time

import numpy as np
import pandas as pd
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsoluteScaledError,
)
from sktime.split import SlidingWindowSplitter

ROOT = Path(__file__).resolve().parent
HORIZON = 14
CONTEXT = 168
N_FOLDS = 3
SEASON = 7
BASELINES = ("seasonal-naive", "arima", "ets")
FOUNDATIONS = ("chronos2", "timesfm25", "t0-gamma")
CHECKPOINTS = {
    "chronos2": "amazon/chronos-2",
    "timesfm25": "google/timesfm-2.5-200m-transformers",
    "t0-gamma": "theforecastingcompany/t0-gamma",
}


def load_series():
    frame = pd.read_csv(ROOT / "data/daily_births.csv", parse_dates=["date"])
    y = frame.set_index("date")["births"].astype(float).asfreq("D")
    if len(y) != 420 or not y.index.is_unique or not np.isfinite(y).all():
        raise ValueError("Expected 420 unique, complete, finite daily observations.")
    y.name = "births"
    return y


def demo_split(y):
    development = y.iloc[: -N_FOLDS * HORIZON]
    return (
        development.iloc[-CONTEXT - HORIZON : -HORIZON],
        development.iloc[-HORIZON:],
    )


def benchmark_series(y):
    return y.iloc[-(CONTEXT + N_FOLDS * HORIZON) :]


def splitter():
    return SlidingWindowSplitter(
        fh=np.arange(1, HORIZON + 1), window_length=CONTEXT, step_length=HORIZON
    )


def _hub_cache():
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return Path(hub)
    home = os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface"
    return Path(home) / "hub"


def _is_cached(name):
    repo = _hub_cache() / ("models--" + CHECKPOINTS[name].replace("/", "--"))
    return (repo / "refs/main").exists() and any(repo.glob("snapshots/*/config.json"))


_AUTO_OFFLINE = False


def _prefer_cached_weights(name):
    """Skip Hub requests when this model's weights are already cached locally.

    A first run, or a model that is not cached, still goes online. An explicit
    HF_HUB_OFFLINE set by the user is left alone.
    """
    global _AUTO_OFFLINE
    if name not in CHECKPOINTS or (
        "HF_HUB_OFFLINE" in os.environ and not _AUTO_OFFLINE
    ):
        return
    from huggingface_hub import constants

    offline = _is_cached(name)
    constants.HF_HUB_OFFLINE = offline
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
    _AUTO_OFFLINE = offline


def _silence_dtype_notice():
    """sktime's Chronos-2 wrapper still passes `torch_dtype`; hide that one notice."""
    from transformers.utils import logging as hf_logging

    class _DtypeNotice(logging.Filter):
        def filter(self, record):
            return "`torch_dtype` is deprecated" not in record.getMessage()

    for handler in hf_logging._get_library_root_logger().handlers:
        if not any(isinstance(f, _DtypeNotice) for f in handler.filters):
            handler.addFilter(_DtypeNotice())


def make_model(name, device="cpu", accept_t0_license=False):
    if name == "seasonal-naive":
        return NaiveForecaster(strategy="last", sp=SEASON)
    if name == "arima":
        return SARIMAX(
            order=(1, 0, 1), seasonal_order=(1, 0, 0, SEASON),
            trend="c", maxiter=100,
        )
    if name == "ets":
        return ExponentialSmoothing(
            trend="add", damped_trend=True, seasonal="add", sp=SEASON
        )
    if name == "prophet":
        from sktime.forecasting.fbprophet import Prophet

        return Prophet(
            weekly_seasonality=True, yearly_seasonality=False,
            daily_seasonality=False,
        )
    _prefer_cached_weights(name)
    if name == "chronos2":
        import torch

        _silence_dtype_notice()
        from sktime.forecasting.chronos2 import Chronos2Forecaster

        return Chronos2Forecaster(
            model_path=CHECKPOINTS[name],
            config={
                "device_map": device, "torch_dtype": torch.float32,
                "context_length": CONTEXT, "batch_size": 1,
            },
            seed=42,
        )
    if name == "timesfm25":
        from sktime.forecasting.timesfm2 import TimesFM2Forecaster

        return TimesFM2Forecaster(
            model_path=CHECKPOINTS[name], device_map=device, dtype="float32"
        )
    if name == "t0-gamma":
        if not accept_t0_license:
            raise ValueError(
                "Read the tfc-t0 and t0-gamma licenses, then explicitly set "
                "accept_t0_license=True if you accept them."
            )
        from sktime.forecasting.t0 import T0Forecaster

        return T0Forecaster(
            model_path=CHECKPOINTS[name], context_length=CONTEXT,
            device=device, random_state=42, license_accepted=True,
        )
    raise ValueError(f"Unknown model: {name}")


def validate_prediction(prediction, truth):
    if not prediction.index.equals(truth.index):
        raise ValueError("Forecast timestamps do not match the evaluation timestamps.")
    if len(prediction) != HORIZON or not np.isfinite(prediction.to_numpy()).all():
        raise ValueError("Forecast must have 14 finite daily predictions.")


def run_benchmark(name, y, device="cpu", accept_t0_license=False, cv=None):
    """Run three non-overlapping horizons; errors stop this model's evaluation."""
    model = make_model(name, device, accept_t0_license)
    frame = evaluate(
        forecaster=model, y=benchmark_series(y), cv=splitter() if cv is None else cv,
        strategy="refit",
        scoring=[MeanAbsoluteError(), MeanAbsoluteScaledError(sp=SEASON)],
        return_data=True, error_score="raise",
    )
    if len(frame) != N_FOLDS:
        raise ValueError("Expected exactly three folds.")
    scores, predictions = [], []
    for fold, (_, row) in enumerate(frame.iterrows(), start=1):
        truth, pred = row["y_test"], row["y_pred"]
        validate_prediction(pred, truth)
        scores.append({
            "model": name, "fold": fold, "cutoff": str(row["cutoff"]),
            "mae": float(row["test_MeanAbsoluteError"]),
            "mase": float(row["test_MeanAbsoluteScaledError"]),
            "fit_seconds": float(row["fit_time"]),
            "predict_seconds": float(row["pred_time"]),
        })
        predictions.extend({
            "model": name, "fold": fold, "date": str(date.date()),
            "actual": float(actual), "prediction": float(estimate),
        } for date, actual, estimate in zip(truth.index, truth, pred))
    return pd.DataFrame(scores), pd.DataFrame(predictions)


def leaderboard(scores):
    """Only compare models with complete, matching folds."""
    required = {"model", "fold", "cutoff", "mae", "mase"}
    if not required.issubset(scores):
        raise ValueError(f"Scores require columns {required}.")
    reference = None
    for name, group in scores.groupby("model"):
        signature = list(group.sort_values("fold")[["fold", "cutoff"]].itertuples(
            index=False, name=None
        ))
        if len(group) != N_FOLDS or set(group["fold"]) != {1, 2, 3}:
            raise ValueError(f"{name} does not have exactly three distinct folds.")
        if reference is not None and signature != reference:
            raise ValueError("Models were scored on different cutoffs.")
        reference = signature
    if not np.isfinite(scores[["mae", "mase"]]).all().all():
        raise ValueError("Do not rank models with non-finite metrics.")
    return scores.groupby("model").agg(
        MAE=("mae", "mean"), MASE=("mase", "mean"),
        fit_seconds=("fit_seconds", "sum"),
        predict_seconds=("predict_seconds", "sum"),
    ).sort_values("MAE")


def save_run(name, y, output_dir, device="cpu", accept_t0_license=False):
    """Save real demo predictions, fold scores, and provenance; never fake results."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Choose a new run directory, not {output_dir}.")
    start = time.perf_counter()
    train, truth = demo_split(y)
    model = make_model(name, device, accept_t0_license)
    model.fit(train, fh=np.arange(1, HORIZON + 1))
    pred = model.predict()
    validate_prediction(pred, truth)
    scores, predictions = run_benchmark(name, y, device, accept_t0_license)
    packages = ["sktime", "numpy", "pandas", "statsmodels"]
    if name in FOUNDATIONS:
        packages += ["torch", "transformers", "chronos-forecasting", "tfc-t0"]
    if name == "prophet":
        packages += ["prophet"]
    # Record exact Hub revisions after a successful live download.
    revision = None
    if name in FOUNDATIONS:
        from huggingface_hub import scan_cache_dir

        revisions = {
            rev.commit_hash
            for repo in scan_cache_dir().repos
            if repo.repo_id == CHECKPOINTS[name]
            for rev in repo.revisions
            if "main" in rev.refs
        }
        if len(revisions) != 1:
            raise RuntimeError("Could not identify one cached main checkpoint revision.")
        revision = revisions.pop()
    output_dir.mkdir(parents=True)
    pd.DataFrame({
        "date": truth.index.strftime("%Y-%m-%d"),
        "actual": truth.to_numpy(), "prediction": pred.to_numpy(),
    }).to_csv(output_dir / "demo.csv", index=False)
    scores.to_csv(output_dir / "scores.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metadata = {
        "model": name, "checkpoint": CHECKPOINTS.get(name),
        "checkpoint_revision": revision,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_sha256": hashlib.sha256(
            (ROOT / "data/daily_births.csv").read_bytes()
        ).hexdigest(),
        "context": CONTEXT, "horizon": HORIZON, "folds": N_FOLDS,
        "device": device, "python": platform.python_version(),
        "versions": {package: version(package) for package in packages},
        "elapsed_seconds": time.perf_counter() - start,
        "source": "actual execution; not synthetic predictions",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return scores


def load_run(directory, y):
    """Validate saved evidence against this exact dataset and protocol."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    digest = hashlib.sha256((ROOT / "data/daily_births.csv").read_bytes()).hexdigest()
    if (
        metadata["data_sha256"] != digest
        or (metadata["context"], metadata["horizon"], metadata["folds"])
        != (CONTEXT, HORIZON, N_FOLDS)
    ):
        raise ValueError("Saved run belongs to a different dataset or protocol.")
    scores = pd.read_csv(directory / "scores.csv")
    predictions = pd.read_csv(directory / "predictions.csv", parse_dates=["date"])
    leaderboard(scores)
    if (
        set(scores["model"]) != {metadata["model"]}
        or set(predictions["model"]) != {metadata["model"]}
    ):
        raise ValueError("Model label does not match provenance.")
    if len(predictions) != N_FOLDS * HORIZON or set(predictions["fold"]) != {1, 2, 3}:
        raise ValueError("Saved predictions must contain exactly three complete folds.")
    expected = benchmark_series(y)
    for fold, (train_idx, test_idx) in enumerate(splitter().split(expected), start=1):
        part = predictions.loc[predictions["fold"] == fold]
        truth = expected.iloc[test_idx]
        if not pd.DatetimeIndex(part["date"]).equals(truth.index):
            raise ValueError("Saved forecast dates do not match the backtest.")
        if not np.array_equal(part["actual"], truth.to_numpy()):
            raise ValueError("Saved actual values do not match the dataset.")
        if not np.isfinite(part["prediction"]).all():
            raise ValueError("Saved predictions contain non-finite values.")
        train = expected.iloc[train_idx]
        scale = np.abs(train.to_numpy()[SEASON:] - train.to_numpy()[:-SEASON]).mean()
        mae = np.abs(part["prediction"].to_numpy() - truth.to_numpy()).mean()
        row = scores.loc[scores["fold"] == fold].iloc[0]
        if pd.Timestamp(row["cutoff"]) != train.index[-1]:
            raise ValueError("Saved cutoff does not match the backtest.")
        if not np.isclose(row["mae"], mae) or not np.isclose(row["mase"], mae / scale):
            raise ValueError("Saved scores disagree with saved predictions.")
    return scores, predictions, metadata