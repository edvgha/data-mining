"""Focused regression tests for statistical correctness and unsafe schema cases."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_mining.audit import (DEFAULTS, analyze, bh_adjust, categorical_buckets,
                          flags_for, joint_information, load_config,
                          numerical_buckets, read_dataset, table_association, wilson)


class StatisticsTests(unittest.TestCase):
    def test_wilson_known_values_and_boundaries(self):
        lo, hi = wilson(np.array([0, 50, 100]), np.array([100, 100, 100]))
        self.assertAlmostEqual(lo[0], 0, places=12)
        self.assertAlmostEqual(hi[0], .0369934982, places=9)
        self.assertAlmostEqual(lo[1], .4038315304, places=9)
        self.assertAlmostEqual(hi[2], 1, places=12)

    def test_bh_family_and_missing_values(self):
        q = bh_adjust([.01, .04, .03, np.nan])
        np.testing.assert_allclose(q[:3], [.03, .04, .04])
        self.assertTrue(np.isnan(q[3]))

    def test_independent_and_perfect_tables(self):
        independent = table_association([[100, 100], [100, 100]])
        self.assertAlmostEqual(independent["mi_nats"], 0)
        self.assertAlmostEqual(independent["cramers_v_corrected"], 0)
        perfect = table_association([[100, 0], [0, 100]])
        self.assertAlmostEqual(perfect["mi_nats"], np.log(2))
        self.assertAlmostEqual(perfect["cramers_v_corrected"], 1)
        sparse = table_association([[1, 0], [0, 1]])
        self.assertTrue(np.isnan(sparse["chi2_p_iid"]))

    def test_xor_has_joint_signal_with_zero_marginal_signal(self):
        a = np.tile([0, 0, 1, 1], 100)
        b = np.tile([0, 1, 0, 1], 100)
        row = joint_information(pd.DataFrame({"a": a, "b": b}), a ^ b)
        self.assertAlmostEqual(row["best_single_mi_nats"], 0)
        self.assertAlmostEqual(row["joint_mi_nats"], np.log(2))
        self.assertAlmostEqual(row["conditional_mi_1_nats"], np.log(2))

    def test_category_sentinels_do_not_collide(self):
        s = pd.Series(["[missing]", None, "[other]", "x", "x"], name="c")
        b, labels, info = categorical_buckets(s, {**DEFAULTS, "rare_min_count": 1})
        self.assertEqual(b[1], 0)
        self.assertNotEqual(b[0], 0)
        self.assertEqual(len(set(b)), 4)
        self.assertEqual(info["unique_nonmissing"], 3)

    def test_constant_with_missingness_is_not_constant_feature(self):
        s = pd.Series([1, 1, 1, np.nan, np.inf], name="n")
        b, _, info, clean = numerical_buckets(s, DEFAULTS)
        self.assertEqual(len(set(b)), 2)
        self.assertEqual(info["nonfinite_nonmissing_count"], 1)
        flags = flags_for(info, DEFAULTS)
        self.assertNotIn("constant", flags)
        self.assertIn("constant_observed_values_with_missingness", flags)

    def test_all_null_and_tied_discrete_values(self):
        _, _, info, _ = numerical_buckets(pd.Series([np.nan, np.nan], name="a"), DEFAULTS)
        self.assertIn("all_missing_or_nonfinite", flags_for(info, DEFAULTS))
        b, _, _, _ = numerical_buckets(pd.Series([0] * 99 + [1], name="b"), DEFAULTS)
        self.assertEqual(len(set(b)), 2)

    def test_reject_duplicate_yaml_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yaml"
            path.write_text("target: y\ntarget: label\nfeatures: {x: numerical}\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_config(path)

    def test_reject_schema_and_label_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.parquet"
            cfg = {**DEFAULTS, "target": "y", "features": {"x": "numerical"}}
            pd.DataFrame({"x": [1, 2], "y": [0, 2]}).to_parquet(path)
            with self.assertRaisesRegex(ValueError, "only 0 and 1"):
                read_dataset(path, cfg)
            pd.DataFrame({"x": [1, 2], "y": [0, 1], "extra": [0, 0]}).to_parquet(path)
            with self.assertRaisesRegex(ValueError, "Undeclared"):
                read_dataset(path, cfg)

    def test_reader_preserves_values_dtypes_and_row_order_across_row_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.parquet"
            df = pd.DataFrame({
                "x": pd.array([1, None, 3, 4, 5, 6], dtype="Int64"),
                "category": pd.Categorical(["b", None, "a", "b", "a", "b"],
                                           categories=["b", "a"], ordered=True),
                "publisher.accountid": np.array([2, 1, 2, 1, 2, 1], dtype=np.int16),
                "y": [0, 1, 0, 1, 0, 1],
                "date": pd.date_range("2026-01-01", periods=6, tz="UTC"),
                "ignored": ["unused"] * 6,
            })
            df.index = pd.Index([9, 3, 8, 2, 7, 1], name="saved_index")
            df.to_parquet(path, row_group_size=2, compression="zstd")
            cfg = {**DEFAULTS, "target": "y", "features": {
                "x": "numerical", "category": "categorical", "publisher.accountid": "categorical"},
                "time_column": "date", "ignore": ["ignored", "saved_index"]}
            actual, extra, columns = read_dataset(path, cfg)
            pd.testing.assert_frame_equal(actual, df.drop(columns="ignored").reset_index(drop=True))
            self.assertEqual(extra, [])
            self.assertEqual(columns, list(df.columns) + ["saved_index"])

    def test_integration_all_null_nullable_and_escaped_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.parquet"
            conf = Path(tmp) / "config.yaml"
            df = pd.DataFrame({"<x>": pd.Series([1, None, 2, 3] * 10, dtype="Int64"),
                               "category": ["[missing]", None, "a", "b"] * 10,
                               "null_numeric": [np.nan] * 40, "null_cat": [None] * 40,
                               "clickoccured": [0, 1, 0, 1] * 10,
                               "session": np.repeat(np.arange(10), 4),
                               "date": pd.date_range("2026-01-01", periods=40)})
            df.to_parquet(path)
            conf.write_text(yaml.safe_dump({"target": "clickoccured", "features": {
                "<x>": "numerical", "category": "categorical", "null_numeric": "numerical", "null_cat": "categorical"},
                "group_column": "session", "time_column": "date", "rare_min_count": 1,
                "interactions": [["<x>", "category"]], "output_dir": "reports"}))
            out = analyze(path, conf)
            self.assertTrue((out / "report.html").exists())
            content = (out / "report.html").read_text()
            self.assertIn("&lt;x&gt;", content)
            target = pd.read_csv(out / "feature_target.csv")
            self.assertTrue(target["chi2_p_iid"].isna().all())
            for file in (out / "feature_positive_rate").glob("*.csv"):
                self.assertEqual(pd.read_csv(file)["rows"].sum(), len(df))
            self.assertEqual(pd.read_csv(out / "joint_positive_rate" / "joint_000.csv")["rows"].sum(), len(df))


if __name__ == "__main__":
    unittest.main()
