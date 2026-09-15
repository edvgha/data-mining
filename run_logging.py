"""Shared console and per-run file logging for the two command-line workflows."""

import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
DEFAULT_LOGGING = {"console_level": "INFO", "file_level": "DEBUG"}


def logging_settings(value, *, log_level=None, file_log_level=None):
    """Validate YAML settings and apply optional CLI/API overrides."""
    if not isinstance(value, dict) or set(value) - set(DEFAULT_LOGGING):
        raise ValueError("logging must be a mapping with console_level and/or file_level.")
    settings = {**DEFAULT_LOGGING, **value}
    for key, level in settings.items():
        if not isinstance(level, str) or level.upper() not in LOG_LEVELS:
            raise ValueError(f"logging.{key} must be one of {', '.join(LOG_LEVELS)}.")
        settings[key] = level.upper()
    for key, level in [("console_level", log_level), ("file_level", file_log_level)]:
        if level is not None:
            if not isinstance(level, str) or level.upper() not in LOG_LEVELS:
                raise ValueError(f"{key} override must be one of {', '.join(LOG_LEVELS)}.")
            settings[key] = level.upper()
    return settings


def add_logging_arguments(parser):
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVELS,
        help="Console logging level; overrides logging.console_level (default INFO)",
    )
    parser.add_argument(
        "--file-log-level",
        type=str.upper,
        choices=LOG_LEVELS,
        help="run.log logging level; overrides logging.file_level (default DEBUG)",
    )


def create_run_directory(root):
    """Allocate the output directory before loading data so failures leave a log."""
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    out = Path(root).resolve() / f"{stamp}_{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    return out


@contextmanager
def run_logging(out, settings, *logger_names):
    """Route named logger trees to stderr and run.log, then restore their state.

    Root logging is untouched. Restoring handlers and levels prevents duplicate
    messages or writes to earlier files when an API runs repeatedly in a process.
    The yielded stream can also receive explicit faulthandler stack dumps.
    """
    settings = logging_settings(settings)
    log_path = Path(out) / "run.log"
    file_handler = logging.FileHandler(log_path, mode="x", encoding="utf-8")
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ"
    )
    formatter.converter = time.gmtime
    file_handler.setLevel(settings["file_level"])
    console_handler.setLevel(settings["console_level"])
    handlers = [console_handler, file_handler]
    saved = []
    try:
        for handler in handlers:
            handler.setFormatter(formatter)
        for name in logger_names:
            logger = logging.getLogger(name)
            saved.append(
                (logger, logger.level, logger.handlers[:], logger.propagate, logger.disabled)
            )
            logger.handlers = []
            for handler in handlers:
                logger.addHandler(handler)
            logger.setLevel(min(handler.level for handler in handlers))
            logger.propagate = False
            logger.disabled = False
        logger = logging.getLogger(logger_names[0])
        logger.info("Log file: %s", log_path)
        yield file_handler.stream
    except BaseException as exc:
        logging.getLogger(logger_names[0]).exception("Run failed. Log file: %s", log_path)
        # The CLI can avoid printing an error that has already been logged.
        exc._run_log_path = log_path
        raise
    finally:
        for logger, level, original_handlers, propagate, disabled in reversed(saved):
            logger.handlers = original_handlers
            logger.setLevel(level)
            logger.propagate = propagate
            logger.disabled = disabled
        for handler in handlers:
            handler.close()


def exit_with_error(parser, exc, label):
    """Keep pre-run errors visible; errors during a run are already logged."""
    message = None if getattr(exc, "_run_log_path", None) else f"ERROR {label}: {exc}\n"
    parser.exit(2, message)
