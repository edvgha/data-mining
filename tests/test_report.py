import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_mining import audit


class CategoricalDiversityReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data_path = self.root / "data.parquet"
        self.features = {
            "<channel|name>": "categorical",
            "balanced": "categorical",
            "constant": "categorical",
            "empty": "categorical",
            "numeric": "numerical",
        }
        pd.DataFrame(
            {
                "<channel|name>": pd.Categorical(
                    ["a"] * 40 + ["b"] * 40 + ["c"] * 20 + [None] * 20,
                    categories=["a", "b", "c", "unused"],
                ),
                "balanced": ["a"] * 50 + ["b"] * 50 + [None] * 20,
                "constant": ["only"] * 100 + [None] * 20,
                "empty": [None] * 120,
                "numeric": np.arange(120),
                "y": [0, 1] * 60,
            }
        ).to_parquet(self.data_path)

    def run_audit(self, **updates):
        config = {
            "target": "y",
            "features": self.features,
            "statistics": {"include": ["entropy"]},
            "rare_min_count": 1,
            "max_categories": 1,
            "max_pairs": 0,
            **updates,
        }
        config["ignore"] = [name for name in self.features if name not in config["features"]]
        config_path = self.root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        return audit(self.data_path, config_path)

    def test_bounds_use_original_nonmissing_levels_in_every_report(self):
        out = self.run_audit()
        table = pd.read_csv(out / "categorical_diversity.csv")
        self.assertEqual(list(table.columns), ["feature", "metric", "value"])
        self.assertEqual(len(table), 8)
        values = table.set_index(["feature", "metric"])["value"]
        expected = {
            ("<channel|name>", "entropy_nats_nonmissing {0, 1.09861}"): 1.0549201679861442,
            ("<channel|name>", "effective_levels_nonmissing {1, 3}"): 2.871745887492588,
            ("balanced", "entropy_nats_nonmissing {0, 0.693147}"): np.log(2),
            ("balanced", "effective_levels_nonmissing {1, 2}"): 2,
            ("constant", "entropy_nats_nonmissing {0, 0}"): 0,
            ("constant", "effective_levels_nonmissing {1, 1}"): 1,
        }
        report_html = (out / "report.html").read_text()
        report_md = (out / "report.md").read_text()
        for key, expected_value in expected.items():
            with self.subTest(feature=key[0], metric=key[1]):
                self.assertAlmostEqual(values.loc[key], expected_value, places=12)
                self.assertIn(key[1], report_html)
                self.assertIn(f"`{key[1]}` | {expected_value:.6g} |", report_md)
        for metric in ["entropy_nats_nonmissing", "effective_levels_nonmissing"]:
            label = f"{metric} {{undefined, undefined}}"
            self.assertTrue(np.isnan(values.loc[("empty", label)]))
            self.assertIn(label, report_html)
            self.assertIn(f"`{label}` | undefined |", report_md)
        self.assertIn("&lt;channel|name&gt;", report_html)
        self.assertIn("| &lt;channel\\|name&gt; |", report_md)

        # Pooling, missing rows, and an unused declared category do not change K.
        profiles = pd.read_csv(out / "categorical_univariate.csv").set_index("feature")
        self.assertEqual(profiles.loc["<channel|name>", "unique_nonmissing"], 3)
        self.assertEqual(profiles.loc["<channel|name>", "retained_levels"], 1)
        self.assertEqual(profiles.loc["<channel|name>", "missing_count"], 20)
        self.assertIn("entropy_nats_nonmissing", profiles)
        self.assertIn("effective_levels_nonmissing", profiles)
        self.assertFalse(any("{" in column for column in profiles.columns))

    def test_disabling_entropy_omits_both_metrics_and_the_bounds_section(self):
        out = self.run_audit(statistics={"include": ["entropy"], "exclude": ["entropy"]})
        self.assertTrue(pd.read_csv(out / "categorical_diversity.csv").empty)
        profiles = pd.read_csv(out / "categorical_univariate.csv")
        for metric in ["entropy_nats_nonmissing", "effective_levels_nonmissing"]:
            self.assertNotIn(metric, profiles)
            self.assertNotIn(metric, (out / "report.html").read_text())
            self.assertNotIn(metric, (out / "report.md").read_text())
        self.assertNotIn(
            "Entropy and effective levels with bounds", (out / "report.md").read_text()
        )

    def test_numerical_only_report_has_a_readable_empty_diversity_csv(self):
        out = self.run_audit(features={"numeric": "numerical"})
        table = pd.read_csv(out / "categorical_diversity.csv")
        self.assertEqual(list(table.columns), ["feature", "metric", "value"])
        self.assertTrue(table.empty)
        for filename in ["report.html", "report.md"]:
            self.assertNotIn(
                "Entropy and effective levels with bounds", (out / filename).read_text()
            )


if __name__ == "__main__":
    unittest.main()
