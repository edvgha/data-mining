"""Console/file filtering, error persistence, and logging lifecycle contracts."""

import io
import json
import logging
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from data_mining import audit
from data_mining.config import load_config as audit_config
from model_tuning.config import configure as tuning_config
from model_tuning.pipeline import run as tune
from run_logging import DEFAULT_LOGGING, logging_settings, run_logging


class LoggingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.data = self.root / "data.parquet"
        self.config = self.root / "config.yaml"
        pd.DataFrame({"x": list(range(40)), "y": [0, 1] * 20}).to_parquet(self.data)
        self.write_config()

    def write_config(self, **updates):
        self.config.write_text(
            yaml.safe_dump(
                {
                    "target": "y",
                    "features": {"x": "numerical"},
                    "statistics": {"include": []},
                    "output_dir": "reports",
                    **updates,
                }
            )
        )

    def test_both_configurations_validate_and_fill_logging_defaults(self):
        self.assertEqual(audit_config(self.config)["logging"], DEFAULT_LOGGING)
        self.assertEqual(
            logging_settings({"console_level": "DEBUG"}, log_level="warning"),
            {"console_level": "WARNING", "file_level": "DEBUG"},
        )
        minimum = {"target": "y", "time_column": "date", "features": {"x": "numerical"}}
        for settings in [
            None,
            "DEBUG",
            {"level": "INFO"},
            {"console_level": "TRACE"},
            {"file_level": 10},
        ]:
            with self.subTest(settings=settings):
                self.write_config(logging=settings)
                with self.assertRaisesRegex(ValueError, "logging"):
                    audit_config(self.config)
                with self.assertRaisesRegex(ValueError, "logging"):
                    tuning_config({**minimum, "logging": settings})
        self.write_config(logging={"console_level": "warning"})
        self.assertEqual(
            audit_config(self.config)["logging"],
            {"console_level": "WARNING", "file_level": "DEBUG"},
        )
        self.assertEqual(
            tuning_config({**minimum, "logging": {"console_level": "warning"}})["logging"],
            {"console_level": "WARNING", "file_level": "DEBUG"},
        )

    def test_console_and_file_thresholds_are_independent(self):
        console = io.StringIO()
        log = logging.getLogger("logging_test.thresholds")
        with (
            redirect_stderr(console),
            run_logging(
                self.root, {"console_level": "DEBUG", "file_level": "ERROR"}, "logging_test"
            ),
        ):
            log.debug("detail")
            log.info("progress")
            log.error("échec")
        self.assertIn("DEBUG logging_test.thresholds: detail", console.getvalue())
        self.assertIn("INFO logging_test.thresholds: progress", console.getvalue())
        saved = (self.root / "run.log").read_text(encoding="utf-8")
        self.assertNotIn("detail", saved)
        self.assertNotIn("progress", saved)
        self.assertIn("ERROR logging_test.thresholds: échec", saved)
        self.assertRegex(saved, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z ERROR")

    def test_handlers_and_levels_restore_after_failure(self):
        log = logging.getLogger("logging_test.restore")
        previous = (log.handlers[:], log.level, log.propagate, log.disabled)
        root_logger = logging.getLogger()
        root_previous = (root_logger.handlers[:], root_logger.level)
        try:
            log.setLevel(logging.CRITICAL)
            log.disabled = True
            with redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, "broken input"):
                with run_logging(self.root, DEFAULT_LOGGING, log.name):
                    raise ValueError("broken input")
            self.assertEqual(log.level, logging.CRITICAL)
            self.assertTrue(log.disabled)
            self.assertEqual(log.handlers, previous[0])
            self.assertEqual(log.propagate, previous[2])
            self.assertEqual((root_logger.handlers, root_logger.level), root_previous)
            saved = (self.root / "run.log").read_text()
            self.assertEqual(saved.count("Run failed."), 1)
            self.assertIn("Traceback (most recent call last)", saved)
            self.assertIn("ValueError: broken input", saved)
        finally:
            log.handlers, level, log.propagate, log.disabled = previous
            log.setLevel(level)

    def test_repeated_audits_have_separate_files_without_duplicate_messages(self):
        log = logging.getLogger("data_mining")
        before = (log.handlers[:], log.level, log.propagate)
        with redirect_stderr(io.StringIO()):
            first = audit(self.data, self.config, log_level="ERROR")
            first_log = (first / "run.log").read_text()
            second = audit(self.data, self.config, log_level="ERROR")
        self.assertNotEqual(first, second)
        self.assertEqual((first / "run.log").read_text(), first_log)
        for out in [first, second]:
            saved = (out / "run.log").read_text()
            self.assertEqual(saved.count("Report: "), 1)
            self.assertEqual(saved.count("Reading and validating Parquet:"), 1)
            self.assertIn("DEBUG data_mining.dataset: Converting column", saved)
            self.assertEqual(
                yaml.safe_load((out / "config_used.yaml").read_text())["logging"],
                {"console_level": "ERROR", "file_level": "DEBUG"},
            )
        self.assertEqual((log.handlers, log.level, log.propagate), before)

    def test_cli_overrides_yaml_with_quiet_console_and_info_file(self):
        self.write_config(logging={"console_level": "DEBUG", "file_level": "DEBUG"})
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "data_mining",
                str(self.data),
                str(self.config),
                "--log-level",
                "WARNING",
                "--file-log-level",
                "INFO",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout + result.stderr, "")
        (out,) = (self.root / "reports").glob("run_*")
        saved = (out / "run.log").read_text()
        self.assertIn("INFO data_mining.audit: Reading and validating Parquet", saved)
        self.assertNotIn("DEBUG", saved)
        self.assertEqual(
            yaml.safe_load((out / "config_used.yaml").read_text())["logging"],
            {"console_level": "WARNING", "file_level": "INFO"},
        )

    def test_cli_failure_persists_traceback_without_duplicate_console_error(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "data_mining",
                str(self.root / "missing.parquet"),
                str(self.config),
                "--log-level",
                "ERROR",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stderr.count("Run failed."), 1)
        (out,) = (self.root / "reports").glob("run_*")
        saved = (out / "run.log").read_text()
        self.assertIn("Traceback (most recent call last)", saved)
        self.assertIn("dataset must be an existing .parquet file", saved)
        self.assertFalse((out / "report.html").exists())

    def test_diagnostics_target_log_stream_and_cancel_on_read_failure(self):
        with (
            patch("data_mining.audit.faulthandler.dump_traceback_later") as dump,
            patch("data_mining.audit.faulthandler.cancel_dump_traceback_later") as cancel,
            patch("data_mining.audit.read_dataset", side_effect=ValueError("read failed")),
            redirect_stderr(io.StringIO()),
            self.assertRaisesRegex(ValueError, "read failed"),
        ):
            audit(self.data, self.config, diagnostics=True)
        dump.assert_called_once()
        stream = dump.call_args.kwargs["file"]
        self.assertEqual(Path(stream.name).name, "run.log")
        self.assertTrue(stream.closed)
        cancel.assert_called_once()

    def test_tuning_input_failure_has_log_and_failed_status(self):
        cfg = tuning_config(
            {
                "target": "y",
                "time_column": "date",
                "features": {"x": "numerical"},
                "output_dir": str(self.root / "reports"),
            }
        )
        with redirect_stderr(io.StringIO()), self.assertRaises(FileNotFoundError):
            tune(self.root / "missing.parquet", cfg)
        (out,) = (self.root / "reports").glob("run_*")
        self.assertEqual(json.loads((out / "status.json").read_text())["status"], "failed")
        self.assertIn("FileNotFoundError", (out / "run.log").read_text())
        self.assertIn("logging", yaml.safe_load((out / "config.yaml").read_text()))


if __name__ == "__main__":
    unittest.main()
