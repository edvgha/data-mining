"""Meaningful gates: leakage boundaries, probability arithmetic, bootstrap units, and saved models."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from importlib.util import find_spec
if find_spec("xgboost") is None or find_spec("optuna") is None:
    raise unittest.SkipTest("Install the tuning extra to run model-tuning tests.")
import numpy as np
import pandas as pd
import yaml
from model_tuning.config import configure, load_config
from model_tuning.data import Encoder, read_data, split_data
from model_tuning.metrics import (aggregate_units, bootstrap_metrics, evaluate,
                                 group_evaluation, resample_totals, row_statistics, unit_codes)
from model_tuning.predict import predict


def small_config(**extra):
    return configure({"target": "clicked", "time_column": "date", "features": {"x": "numerical", "category": "categorical"},
        "bootstrap": {"n_resamples": 20, "training_repeats": 1},
        "tuning": {"n_trials": 2, "num_boost_round": 12, "early_stopping_rounds": 3,
                   "search_space": {"max_depth": {"type": "int", "low": 2, "high": 3}}},
        "report": {"shap_rows": 20}, **extra})


def sample_frame():
    rng = np.random.default_rng(123)
    n = 800
    return pd.DataFrame({"x": rng.normal(size=n), "category": np.tile(["a", "b", None, "c"], n//4),
        "clicked": np.tile([0, 1, 0, 0], n//4), "date": np.repeat(pd.date_range("2025-01-01", periods=80, tz="UTC"), 10),
        "session": np.repeat(np.arange(n//2), 2), "publisher": np.tile(["p1", "p2"], n//2)})


class TuningUnitTests(unittest.TestCase):
    def test_metric_hand_calculation_and_undefined_cases(self):
        y, p = np.array([1, 0, 1, 0]), np.array([.8, .1, .7, .2])
        m = evaluate(y, p, .5)
        self.assertAlmostEqual(m["logloss"], -np.log([.8, .9, .7, .8]).mean())
        self.assertAlmostEqual(m["ne"], m["logloss"]/np.log(2))
        self.assertAlmostEqual(m["ne_train_baseline"], m["ne"])
        self.assertAlmostEqual(m["click_error_pct"], -10)
        self.assertAlmostEqual(m["z_score"], .2/np.sqrt(np.sum(p*(1-p))))
        z = evaluate([0, 0], [.1, .2], .3)
        self.assertTrue(np.isnan(z["ne"]))
        self.assertTrue(np.isnan(z["click_error_pct"]))
        self.assertTrue(np.isnan(z["roc_auc"]))
        self.assertTrue(np.isfinite(evaluate([1, 0], [0, 1], .5)["logloss"]))
        self.assertTrue(np.isnan(evaluate([1, 0], [1, 0], .5)["z_score"]))

    def test_time_ties_gap_and_explicit_boundaries(self):
        cfg = small_config(split={"gap": "1D"})
        parts, summary = split_data(sample_frame(), cfg)
        self.assertLess(parts["train"].date.max()+pd.Timedelta("1D"), parts["validation"].date.min())
        self.assertLess(parts["validation"].date.max()+pd.Timedelta("1D"), parts["test"].date.min())
        self.assertEqual(summary["excluded_gap_rows"], 20)
        for left, right in [("fit", "early_stop"), ("train", "validation"), ("validation", "test")]:
            self.assertFalse(set(parts[left].date) & set(parts[right].date))
        cfg["split"].update(train_end="2025-02-17", validation_end="2025-03-05")
        explicit, _ = split_data(sample_frame(), cfg)
        self.assertEqual(explicit["train"].date.max(), pd.Timestamp("2025-02-17", tz="UTC"))

    def test_entity_overlap_rejected_and_single_class_test_allowed(self):
        cfg = small_config(split={"entity_column": "session"})
        frame = sample_frame()
        split_data(frame, cfg)
        frame.loc[frame.index[-1], "session"] = frame.session.iloc[0]
        with self.assertRaisesRegex(ValueError, "crosses"):
            split_data(frame, cfg)
        frame = sample_frame()
        frame.loc[frame.date >= frame.date.unique()[64], "clicked"] = 0
        split_data(frame, small_config())

    def test_future_categories_never_enter_training_vocabulary(self):
        train = pd.DataFrame({"x": [1, np.inf], "category": ["a", "b"]})
        enc = Encoder({"x": "numerical", "category": "categorical"}).fit(train)
        result = enc.transform(pd.DataFrame({"x": [2, 3], "category": ["future", None]}))
        self.assertEqual(enc.categories["category"], ["a", "b"])
        self.assertTrue(result.f1.isna().all())
        self.assertTrue(np.isnan(enc.transform(train).f0.iloc[1]))
        self.assertEqual(Encoder(**enc.to_dict()).transform(train).f1.cat.categories.tolist(), ["a", "b"])

    def test_bootstrap_whole_clusters_and_sufficient_statistics(self):
        frame = sample_frame().iloc[:40]
        cfg = small_config(bootstrap={"method": "cluster", "cluster_column": "session", "n_resamples": 30})
        codes = unit_codes(frame, cfg)
        self.assertEqual(codes[0], codes[1])
        p = np.full(len(frame), .3)
        units = aggregate_units(row_statistics(frame.clicked, p, .3), codes)
        for draw in resample_totals(units, 10, 4):
            self.assertEqual(draw[0], 40)
        one = bootstrap_metrics(frame, frame.clicked, p, .3, cfg)
        two = bootstrap_metrics(frame, frame.clicked, p, .3, cfg)
        pd.testing.assert_frame_equal(one, two)
        self.assertTrue((one.units == 20).all())
        cfg["bootstrap"] = {**cfg["bootstrap"], "method": "time_block", "time_frequency": "7D"}
        intervals = bootstrap_metrics(frame.iloc[:10], frame.clicked.iloc[:10], p[:10], .3, cfg)
        self.assertTrue((intervals.status == "insufficient_units").all())

    def test_group_totals_include_null_and_zero_click_groups(self):
        frame = sample_frame().iloc[:40].reset_index(drop=True)
        frame.loc[:9, "publisher"] = None
        frame.loc[frame.publisher == "p1", "clicked"] = 0
        cfg = small_config(group_by=["publisher"])
        result = group_evaluation(frame, np.full(len(frame), .3), .3, cfg)[0][1]
        self.assertEqual(result.rows.sum(), len(frame))
        self.assertAlmostEqual(result.expected_clicks.sum(), len(frame)*.3)
        self.assertTrue(result.publisher.isna().any())
        self.assertTrue(result.loc[result.publisher == "p1", "click_error_pct"].isna().all())

    def test_invalid_configs_and_misspelled_target(self):
        for changes in [{"wat": 1}, {"split": {"gap_typo": 1}}, {"features": {"clicked": "numerical"}},
                        {"bootstrap": {"method": "cluster"}}, {"tuning": {"params": {"scale_pos_weight": 30}}},
                        {"tuning": {"search_space": {"max_depth": {"type": "float", "low": 1, "high": 5}}}}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                small_config(**changes)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"data.parquet"
            sample_frame().to_parquet(path)
            cfg = small_config(target="clickedd")
            with self.assertRaisesRegex(ValueError, "clicked"):
                read_data(path, cfg)
            config = Path(temp)/"config.yaml"
            config.write_text("target: a\ntarget: b\n")
            with self.assertRaisesRegex(ValueError, "Duplicate YAML key"):
                load_config(config)


class TuningIntegrationTests(unittest.TestCase):
    def test_cli_report_roundtrip_and_test_label_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = sample_frame()
            frame.loc[frame.index[-10:], "category"] = "test-only"
            data, config = root/"data.parquet", root/"config.yaml"
            cfg = small_config(group_by=[["publisher"], ["publisher", "category"]],
                               output_dir="reports", nthread=1,
                               report={"shap_rows": 20, "save_predictions": True})
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            runs = []
            for flipped in [False, True]:
                altered = frame.copy()
                if flipped:
                    altered.loc[altered.date >= altered.date.unique()[64], "clicked"] = 1-altered.clicked
                altered.to_parquet(data, index=False)
                proc = subprocess.run([sys.executable, "-m", "model_tuning", str(data), str(config)],
                                      text=True, capture_output=True, timeout=120)
                self.assertEqual(proc.returncode, 0, proc.stdout+proc.stderr)
                report = Path(next(x.removeprefix("Report: ") for x in proc.stdout.splitlines() if x.startswith("Report: ")))
                out = report.parent
                self.assertEqual(out.parent, root/"reports")
                self.assertIn("<!doctype html>", report.read_text())
                self.assertEqual(json.loads((out/"status.json").read_text())["status"], "complete")
                summary = json.loads((out/"summary.json").read_text())
                self.assertEqual(summary["unseen_test_category_rows"]["category"], 10)
                self.assertEqual(summary["training_bootstrap_completed"], 1)
                parts, _ = split_data(altered, cfg)
                pred = predict(parts["test"], out, nthread=1)
                saved = pd.read_parquet(out/"test_predictions.parquet")
                np.testing.assert_allclose(pred, saved["__predicted_probability__"], atol=1e-7)
                shap = pd.read_csv(out/"shap_values.csv")
                context = pd.read_csv(out/"shap_context.csv")
                np.testing.assert_allclose(shap.sum(axis=1), context.raw_margin, atol=1e-4)
                np.testing.assert_allclose(1/(1+np.exp(-shap.sum(axis=1))), context.prediction, atol=1e-5)
                self.assertEqual(len(pd.read_csv(out/"groups_0.csv")), 2)
                runs.append((summary, pred, out))
            self.assertEqual(runs[0][0]["best_params"], runs[1][0]["best_params"])
            self.assertEqual(runs[0][0]["best_num_boost_round"], runs[1][0]["best_num_boost_round"])
            self.assertEqual(runs[0][0]["best_objective"], runs[1][0]["best_objective"])
            np.testing.assert_array_equal(runs[0][1], runs[1][1])
            self.assertNotEqual(runs[0][0]["test_metrics"]["logloss"], runs[1][0]["test_metrics"]["logloss"])


if __name__ == "__main__":
    unittest.main()
