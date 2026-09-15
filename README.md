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

Open the path printed after `Report:`. Outputs are saved under `report/run_*/`, including `report.html`, `report.md`, and CSV/JSON tables. Add `--diagnostics` to the command if data loading needs investigation.

### Model-tuning demo

```bash
uv run --locked --python 3.12 python -m model_tuning \
  demo/data.parquet demo/model_tuning.yaml
```

This config runs 10 trials, 100 metric bootstrap resamples, 3 training bootstrap refits, and SHAP on up to 2,000 test rows. Outputs are saved under `report/model_tuning/run_*/`: open `report.html` and retain the directory for its linked tables, `model.ubj`, and `encoder.json`. The demo is a workflow check; its small search budget is not a thorough optimization.

Both demos read the committed data; no data-generation step is needed. Reports are written outside `demo/`. Neither command automatically invokes the other workflow or consumes the other's feature-selection output.

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
