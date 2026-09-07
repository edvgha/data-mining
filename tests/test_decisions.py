import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_mining import audit
from data_mining.audit import DEFAULTS, standardized_mean_difference
from data_mining.decisions import decide
from data_mining.settings import configure_decisions


def fixture(names, **updates):
    cfg = configure_decisions({**DEFAULTS, "target": "y",
                              "features": {name: "numerical" for name in names}, **updates})
    profiles = pd.DataFrame([{"feature": name, "type": "numerical", "unique_nonmissing": 100,
        "missing_fraction": 0., "unusable_fraction": 0., "dominant_fraction_nonmissing": .01}
        for name in names])
    targets = pd.DataFrame([{"feature": name, "cohens_d": .5, "raw_numeric_auc": .7,
                             "mi_nats": .01} for name in names])
    pairs = pd.DataFrame(columns=["feature_a", "feature_b"])
    return cfg, profiles, targets, pairs


class DecisionTests(unittest.TestCase):
    def test_cohens_d_class_order_and_pooled_variance(self):
        x = np.array([1, 2, 3, 2, 3, 4], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        positive = standardized_mean_difference(x, y)
        self.assertAlmostEqual(positive["cohens_d"], 1.)
        self.assertAlmostEqual(positive["hedges_g"], .8)
        self.assertAlmostEqual(standardized_mean_difference(-x, y)["cohens_d"], -1.)
        self.assertTrue(np.isnan(standardized_mean_difference(np.ones(6), y)["cohens_d"]))

    def test_inverse_auc_is_kept_when_expected_direction_is_negative(self):
        cfg, p, t, pairs = fixture(["rank"], expected_direction={"rank": "negative"})
        t.loc[0, ["cohens_d", "raw_numeric_auc"]] = [-.7, .2]
        result, _ = decide(p, t, pairs, cfg)
        self.assertEqual(result.iloc[0]["decision"], "keep")
        self.assertIn("inverse ranking", result.iloc[0]["reason"])

    def test_mismatched_direction_and_disagreeing_metrics(self):
        cfg, p, t, pairs = fixture(["rank"], expected_direction={"rank": "negative"})
        result, logs = decide(p, t, pairs, cfg)
        self.assertEqual(result.iloc[0]["decision"], "review")
        self.assertIn("Expected negative", result.iloc[0]["reason"])
        t.loc[0, "raw_numeric_auc"] = .3  # Positive mean effect but inverse ranking.
        result, _ = decide(p, t, pairs, cfg)
        self.assertIn("direction disagree", result.iloc[0]["reason"])

    def test_redundancy_uses_retained_direct_anchor_and_explicit_preference(self):
        cfg, p, t, _ = fixture(["a", "b", "c"], prefer=["a"])
        pairs = pd.DataFrame([
            {"feature_a": "a", "feature_b": "b", "spearman_r": .99, "valid_rows": 5000, "sample_rows": 5000},
            {"feature_a": "b", "feature_b": "c", "spearman_r": .99, "valid_rows": 5000, "sample_rows": 5000},
            {"feature_a": "a", "feature_b": "c", "spearman_r": .7, "valid_rows": 5000, "sample_rows": 5000},
        ])
        result, _ = decide(p, t, pairs, cfg)
        decisions = result.set_index("feature")["decision"].to_dict()
        self.assertEqual(decisions, {"a": "keep", "b": "exclude", "c": "keep"})
        self.assertIn("retained 'a'", result[result.feature == "b"].iloc[0]["reason"])

    def test_signed_redundancy_and_pair_support(self):
        cfg, p, t, _ = fixture(["a", "b"], rules={"redundancy": {"sign": "positive"}})
        pairs = pd.DataFrame([{"feature_a": "a", "feature_b": "b", "spearman_r": -.99,
                               "valid_rows": 5000, "sample_rows": 5000}])
        result, _ = decide(p, t, pairs, cfg)
        self.assertNotIn("exclude", result.decision.tolist())
        cfg["rules"]["redundancy"]["sign"] = "absolute"
        result, _ = decide(p, t, pairs, cfg)
        self.assertEqual(result.decision.tolist().count("exclude"), 1)
        pairs["valid_rows"] = 5
        result, _ = decide(p, t, pairs, cfg)
        self.assertNotIn("exclude", result.decision.tolist())

    def test_weak_signal_does_not_drop_configured_interaction(self):
        cfg, p, t, pairs = fixture(["a", "b"], interactions=[["a", "b"]],
                                  rules={"weak_signal": {"action": "exclude"}})
        t["cohens_d"], t["raw_numeric_auc"], t["mi_nats"] = 0., .5, 0.
        result, _ = decide(p, t, pairs, cfg)
        self.assertEqual(result.decision.tolist(), ["review", "review"])

    def test_disabling_required_statistic_skips_rule(self):
        cfg, p, t, pairs = fixture(["a"], statistics={"include": "all", "exclude": ["cohens_d", "spearman"]})
        result, logs = decide(p, t.drop(columns="cohens_d"), pairs, cfg)
        weak = logs[logs.rule == "weak_signal"].iloc[0]
        self.assertEqual(weak["status"], "skipped")
        self.assertIn("cohens_d", weak["reason"])
        self.assertEqual(logs[logs.rule == "redundancy"].iloc[0]["status"], "skipped")

    def test_force_excluded_feature_cannot_be_anchor(self):
        cfg, p, t, _ = fixture(["a", "b"], force_exclude=["a"], prefer=["a"])
        pairs = pd.DataFrame([{"feature_a": "a", "feature_b": "b", "spearman_r": 1.,
                               "valid_rows": 5000, "sample_rows": 5000}])
        result, _ = decide(p, t, pairs, cfg)
        self.assertEqual(result.decision.tolist(), ["exclude", "keep"])

    def test_disabled_metrics_are_absent_from_end_to_end_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, conf = Path(tmp) / "d.parquet", Path(tmp) / "c.yaml"
            pd.DataFrame({"x": np.arange(100), "y": [0, 1] * 50}).to_parquet(path)
            conf.write_text(yaml.safe_dump({"target": "y", "features": {"x": "numerical"},
                 "statistics": {"include": ["auc", "cohens_d", "spearman"], "exclude": ["cohens_d"]}}))
            out = audit(path, conf)
            target = pd.read_csv(out / "feature_target.csv")
            self.assertIn("raw_numeric_auc", target)
            self.assertNotIn("cohens_d", target)
            self.assertNotIn("mi_nats", target)
            self.assertNotIn("mean", pd.read_csv(out / "numerical_univariate.csv"))
            self.assertEqual(list((out / "feature_positive_rate").glob("*.csv")), [])
            self.assertTrue((out / "selected_features.json").exists())
            # An empty feature type still has a readable CSV header.
            self.assertTrue(pd.read_csv(out / "categorical_univariate.csv").empty)


if __name__ == "__main__":
    unittest.main()
