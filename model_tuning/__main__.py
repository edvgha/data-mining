"""Usage: uv run --extra tuning python -m model_tuning DATA.parquet CONFIG.yaml"""
import argparse
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description="Tune binary XGBoost with Optuna and chronological bootstrap evaluation.")
    parser.add_argument("data", type=Path, help="Local Parquet file; relative to the current directory")
    parser.add_argument("config", type=Path, help="YAML configuration; output_dir is relative to this file")
    args = parser.parse_args()
    try:
        from .config import load_config
        from .pipeline import run
        out = run(args.data.resolve(), load_config(args.config))
    except ImportError as exc:
        parser.exit(2, f"Missing dependency: {exc}. Run with uv run --extra tuning.\n")
    except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Report: {out / 'report.html'}", flush=True)
    print(f"Model: {out / 'model.ubj'}", flush=True)


if __name__ == "__main__":
    main()
