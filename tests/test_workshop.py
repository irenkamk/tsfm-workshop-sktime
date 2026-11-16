from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from workshop import (
    ROOT, HORIZON, CONTEXT, load_series, demo_split, benchmark_series,
    splitter, run_benchmark, load_run, leaderboard, validate_prediction,
)


class WorkshopTests(unittest.TestCase):
    def test_demo_is_separate_from_benchmark_and_folds_are_leak_free(self):
        y = load_series()
        train, test = demo_split(y)
        self.assertEqual((len(train), len(test)), (CONTEXT, HORIZON))
        self.assertLess(train.index[-1], test.index[0])
        evaluation_y = benchmark_series(y)
        folds = list(splitter().split(evaluation_y))
        all_test_dates = []
        for tr, te in folds:
            self.assertEqual((len(tr), len(te)), (168, 14))
            self.assertLess(tr[-1], te[0])
            all_test_dates.extend(evaluation_y.index[te])
        self.assertEqual(len(set(all_test_dates)), 42)
        self.assertEqual(all_test_dates, list(y.index[-42:]))
        self.assertLess(test.index[-1], all_test_dates[0])

    def test_naive_forecasts_and_train_only_mase_match_manual_calculation(self):
        y = load_series()
        scores, predictions = run_benchmark("seasonal-naive", y)
        evaluation_y = benchmark_series(y)
        for fold, (tr, te) in enumerate(splitter().split(evaluation_y), start=1):
            train, truth = evaluation_y.iloc[tr], evaluation_y.iloc[te]
            expected = np.tile(train.iloc[-7:].to_numpy(), 2)
            actual_predictions = predictions.loc[predictions["fold"] == fold, "prediction"]
            np.testing.assert_allclose(actual_predictions, expected)
            mae = np.abs(truth.to_numpy() - expected).mean()
            scale = np.abs(train.to_numpy()[7:] - train.to_numpy()[:-7]).mean()
            row = scores.loc[scores["fold"] == fold].iloc[0]
            self.assertAlmostEqual(row["mae"], mae)
            self.assertAlmostEqual(row["mase"], mae / scale)

    def test_saved_runs_and_reject_corrupted_evidence(self):
        y = load_series()
        for name in ("seasonal-naive", "arima", "ets"):
            scores, predictions, metadata = load_run(ROOT / "artifacts/baselines" / name, y)
            self.assertEqual((len(scores), len(predictions)), (3, 42))
            self.assertEqual(metadata["model"], name)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            shutil.copytree(ROOT / "artifacts/baselines/ets", directory)
            scores = pd.read_csv(directory / "scores.csv")
            scores.loc[0, "mae"] = 0
            scores.to_csv(directory / "scores.csv", index=False)
            with self.assertRaisesRegex(ValueError, "disagree"):
                load_run(directory, y)

    def test_reject_extra_prediction_folds_and_wrong_cutoffs(self):
        for corruption in ("extra-fold", "wrong-cutoff"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp) / "run"
                shutil.copytree(ROOT / "artifacts/baselines/ets", directory)
                if corruption == "extra-fold":
                    predictions = pd.read_csv(directory / "predictions.csv")
                    extra = predictions.iloc[[0]].copy()
                    extra["fold"] = 4
                    pd.concat([predictions, extra]).to_csv(
                        directory / "predictions.csv", index=False
                    )
                else:
                    scores = pd.read_csv(directory / "scores.csv")
                    scores.loc[0, "cutoff"] = "2099-01-01"
                    scores.to_csv(directory / "scores.csv", index=False)
                with self.assertRaises(ValueError):
                    load_run(directory, load_series())

    def test_refuse_incomplete_or_misaligned_comparisons(self):
        scores, _, _ = load_run(ROOT / "artifacts/baselines/ets", load_series())
        with self.assertRaises(ValueError):
            leaderboard(scores.iloc[:2])
        other = scores.copy()
        other["model"] = "different"
        other.loc[0, "cutoff"] = "2099-01-01"
        with self.assertRaisesRegex(ValueError, "different cutoffs"):
            leaderboard(pd.concat([scores, other]))
        _, truth = demo_split(load_series())
        shifted = truth.copy()
        shifted.index = shifted.index + pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "timestamps"):
            validate_prediction(shifted, truth)


if __name__ == "__main__":
    unittest.main()
