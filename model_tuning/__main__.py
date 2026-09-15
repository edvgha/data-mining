"""Usage: uv run python -m model_tuning DATA.parquet CONFIG.yaml"""
import argparse
from pathlib import Path

from run_logging import add_logging_arguments, exit_with_error


def main():
    parser = argparse.ArgumentParser(description="Tune binary XGBoost with Optuna and chronological bootstrap evaluation.")
    parser.add_argument("data", type=Path, help="Local Parquet file; relative to the current directory")
    parser.add_argument("config", type=Path, help="YAML configuration; output_dir is relative to this file")
    add_logging_arguments(parser)
    args = parser.parse_args()
    try:
        from .config import load_config
        from .pipeline import run
        run(args.data.resolve(), load_config(args.config),
            log_level=args.log_level, file_log_level=args.file_log_level)
    except ImportError as exc:
        exit_with_error(parser, exc, "Missing dependency (run uv sync --locked from the repository root)")
    except KeyboardInterrupt:
        parser.exit(130)
    except Exception as exc:
        exit_with_error(parser, exc, "Model tuning failed")


if __name__ == "__main__":
    main()
