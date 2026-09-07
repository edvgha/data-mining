"""Exercise the module entry point and the report path a CLI user receives."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml


class CliTests(unittest.TestCase):
    def test_module_command_writes_report_at_printed_config_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            inputs = root / "inputs"
            inputs.mkdir()
            dataset = inputs / "data.parquet"
            config = inputs / "config.yaml"
            pd.DataFrame({
                "x": [1, 2, 3, 4] * 10,
                "category": ["a", "b", "a", "b"] * 10,
                "label": [0, 1, 1, 0] * 10,
                "date": pd.date_range("2026-01-01", periods=40),
            }).to_parquet(dataset)
            config.write_text(yaml.safe_dump({
                "target": "label",
                "features": {"x": "numerical", "category": "categorical"},
                "time_column": "date",
                "interactions": [["x", "category"]],
                "rare_min_count": 1,
                "output_dir": "../reports",
            }))

            result = subprocess.run(
                [sys.executable, "-m", "data_mining", str(dataset), str(config)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            paths = [line.removeprefix("Report: ") for line in result.stdout.splitlines()
                     if line.startswith("Report: ")]
            self.assertEqual(len(paths), 1, result.stdout)
            report = Path(paths[0])
            self.assertEqual(report, report.resolve())
            self.assertEqual(report.parent.parent, root / "reports")
            self.assertEqual(report.name, "report.html")
            self.assertIn("<!doctype html>", report.read_text())
            self.assertTrue((report.parent / "report.md").is_file())
            summary = json.loads((report.parent / "summary.json").read_text())
            self.assertEqual(summary["rows"], 40)
            self.assertEqual(summary["features"], 2)
            decisions = json.loads((report.parent / "feature_decisions.json").read_text())
            self.assertEqual({row["feature"] for row in decisions}, {"x", "category"})


if __name__ == "__main__":
    unittest.main()
