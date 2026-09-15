# Data analysis and model tuning

Two configurable workflows for binary classification datasets: inspect data and screen features with `data_mining`, then tune and evaluate a native XGBoost model with `model_tuning`. Both accept **a local Parquet file and a YAML configuration** and save reports.

## Repository layout

| Directory / file | Purpose |
|---|---|
| [`data_mining/`](data_mining/README.md) | Dataset validation, descriptive statistics, feature relationships, and configurable keep/review/exclude decisions. Runs an audit without fitting a model. |
| [`model_tuning/`](model_tuning/README.md) | Optuna tuning with native XGBoost, chronological splits, bootstrap diagnostics, group evaluation, model export, and SHAP/gain explanations. |
| [`demo/`](demo/) | All runnable demo inputs: one shared million-row synthetic Parquet and separate configs for analysis and tuning. |
| [`doc/`](doc/) | Detailed mathematical references: [data-mining statistics](doc/statistics-and-techniques.md), [Wilson interval proof](doc/wilson-confidence-interval-proof.md), and [model-tuning techniques](doc/model-tuning-and-techniques.md). |
| [`tests/`](tests/) | Formula, validation, leakage, model reload, and command-line integration checks. |
| [`run_logging.py`](run_logging.py) | Shared console/file logging, level overrides, and run-directory creation for both workflows. |
| `report/` (generated) | Run-specific HTML reports, tables, and fitted models. Ignored by Git. |
| [`pyproject.toml`](pyproject.toml), [`uv.lock`](uv.lock) | Python requirements, dependencies, and locked dependency versions. |
| [`VALIDATION.json`](VALIDATION.json), [`model_tuning/VALIDATION.json`](model_tuning/VALIDATION.json) | Recorded results from earlier validation runs, not live test status. |

## Setup

Use Python **3.12 or 3.13** and `uv`. Commands below select Python 3.12 explicitly and run from the **repository root**. This repository runs directly from source (`tool.uv.package = false`); there is no separate wheel or application build step.

Install the shared environment for data mining and model tuning:

```bash
uv sync --locked --python 3.12
```

Optuna and XGBoost are standard project dependencies, installed alongside the analysis libraries. Both workflows use the committed lockfile and require no extra dependency flags.

The supported tuning implementation uses CPU histogram training; GPU, multiclass, and regression modes are not exposed.

## Run the demos

All demo data and configurations live in the top-level `demo/` directory:

| Input | Used by |
|---|---|
| [`demo/data.parquet`](demo/data.parquet) | Both workflows: 1,000,000 synthetic rows, 27 columns, target `clickoccured` |
| [`demo/config.yaml`](demo/config.yaml) | Data-mining feature schema, statistic switches, and screening rules |
| [`demo/model_tuning.yaml`](demo/model_tuning.yaml) | Tuning features, chronological split, search budget, bootstrap, and group-report settings |

### Data-mining demo

```bash
uv run --locked --python 3.12 python -m data_mining \
  demo/data.parquet demo/config.yaml
```

Open the path logged after `Report:`. Outputs are saved under `report/run_*/`, including `report.html`, `report.md`, `run.log`, and CSV/JSON tables. Add `--log-level DEBUG --diagnostics` to the command if data loading needs investigation.

### Model-tuning demo

```bash
uv run --locked --python 3.12 python -m model_tuning \
  demo/data.parquet demo/model_tuning.yaml
```

This config runs 10 trials, 100 metric bootstrap resamples, 3 training bootstrap refits, and SHAP on up to 2,000 test rows. Outputs are saved under `report/model_tuning/run_*/`: open `report.html` and retain the directory for its linked tables, `model.ubj`, and `encoder.json`. The demo is a workflow check; its small search budget is not a thorough optimization.

Both demos read the committed data; no data-generation step is needed. Reports are written outside `demo/`. Neither command automatically invokes the other workflow or consumes the other's feature-selection output.

## Logging

Both workflows write progress to the console (stderr) and to `run.log` inside each run's output directory. Defaults show major stages and final paths at `INFO` on the console and save per-column, per-feature, per-plot, trial-parameter, and bootstrap-refit details at `DEBUG` in the file. Optuna's trial messages use the same handlers and levels.

Set independent thresholds in either workflow's YAML:

```yaml
logging:
  console_level: INFO
  file_level: DEBUG
```

Accepted levels are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`, in increasing severity. A threshold includes that level and all more severe levels. Names are case-insensitive. Missing fields use the defaults above; invalid levels and unknown logging keys are rejected.

Command-line flags override the corresponding YAML setting for that run:

```bash
# Detailed console progress as well as the default detailed file log.
uv run --locked -m data_mining demo/data.parquet demo/config.yaml --log-level DEBUG

# Quiet console; keep detailed progress and Optuna trials in run.log.
uv run --locked -m model_tuning demo/data.parquet demo/model_tuning.yaml --log-level WARNING

# Save only INFO and more severe records to the file.
uv run --locked -m data_mining demo/data.parquet demo/config.yaml --file-log-level INFO
```

Each record includes a UTC timestamp, level, logger name, and message. `Log file:`, `Report:`, and `Model:` are `INFO` messages, so console thresholds above `INFO` suppress those messages too. Console log lines now have a prefix and go to stderr; scripts reading the old unprefixed stdout lines should be updated. The saved config records the effective logging levels.

After configuration validation, a fresh run directory and log are created **before data loading**. Failures and interruptions during the run are logged with tracebacks at `ERROR`; their visibility follows the configured thresholds. Invalid configuration or an unwritable output directory is reported on the console before file logging can start. Normal log records are flushed as they are emitted. To follow a running job, substitute its actual directory:

```bash
tail -f /path/to/run_directory/run.log
```

The Python APIs accept the same optional overrides: `audit(data, config_path, log_level="WARNING", file_log_level="DEBUG")` and `model_tuning.pipeline.run(data, cfg, log_level="WARNING", file_log_level="DEBUG")`. Logging handlers are closed and previous logger settings restored when a run finishes. Native-library output and Python warnings written directly to stderr are separate from these application logs. The audit's explicit `--diagnostics` stack dumps are saved to `run.log` independently of severity filtering.

## Run with your data

```bash
uv run --locked --python 3.12 python -m data_mining \
  /path/to/data.parquet /path/to/audit.yaml

uv run --locked --python 3.12 python -m model_tuning \
  /path/to/data.parquet /path/to/tuning.yaml
```

Copy a matching demo config and edit it for your schema. [`model_tuning/config.yaml`](model_tuning/config.yaml) is an additional **custom-data template**, not a runnable demo against the supplied Parquet. Its target spelling is `clickoccurrred`; the supplied demo target is `clickoccured`. Always use the exact column names in your own data.

Input paths are resolved from the working directory. Each config's `output_dir` is resolved from that config's directory, so check it when copying a YAML file elsewhere. Audit and tuning configs have different schemas and are not interchangeable.

Define model candidates using pre-test information. An audit over the future test period can influence feature selection and undermine the held-out evaluation even when the tuner itself splits chronologically.

## Run checks

Run the full suite, including model-tuning tests, in the shared environment:

```bash
uv run --locked --python 3.12 python -m unittest discover -s tests -v
```

For workflow details, use the [data-mining README](data_mining/README.md) or [tuning README](model_tuning/README.md). The `doc/` guides explain formulas, configuration effects, examples, and interpretation limits.
