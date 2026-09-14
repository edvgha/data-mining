# Binary classification model tuning

Tune a native `xgboost.Booster` with Optuna from a local Parquet file and YAML config. All fitting uses `xgboost.train` and `DMatrix`. The existing sklearn dependency supplies ranking metrics only; there is no sklearn model or preprocessing API in this workflow.

See [Model tuning and techniques: mathematics and interpretation](../doc/model-tuning-and-techniques.md) for the chronological selection procedure, XGBoost objective, search controls, bootstrap formulas, metric definitions, worked examples, and report interpretation. See the [project README](../README.md) for environment modes and both demos.

## Run the shared demo

Run from the repository root, where `pyproject.toml` lives:

```bash
uv sync --locked --extra tuning --python 3.12
uv run --locked --extra tuning --python 3.12 python -m model_tuning \
  demo/data.parquet demo/model_tuning.yaml
```

All demo inputs are in the top-level [`demo/`](../demo/) directory. [`model_tuning.yaml`](../demo/model_tuning.yaml) uses the same million-row [`data.parquet`](../demo/data.parquet) as the data-mining demo, with its own feature list and tuning settings.

Open the absolute path printed after `Report:`. This config saves a new run under `report/model_tuning/`. The report includes overall/group metrics, bootstrap diagnostics, optimization and learning curves, calibration/time plots, and SHAP/gain importance.

## Run with your data

```bash
uv run --locked --extra tuning --python 3.12 python -m model_tuning \
  /path/to/data.parquet model_tuning/config.yaml
```

Edit [`config.yaml`](config.yaml) first. It is a custom-data template derived from an analysis report, not a demo config for the committed Parquet. It uses the requested `clickoccurrred` spelling; the demo uses `clickoccured`. The report's `publisher_accountid` also differs from the demo's `publisher.accountid`. Configured names must match your schema exactly; missing columns produce suggestions, never automatic aliases.

The template's feature list is a starting point, not a validated selection. It omits possible identity/twin alternatives and quarantines `click_or_ref_url_host` until confirmed available before prediction. The tool does not automatically establish twin relationships or detect post-outcome leakage. Evaluate candidate lists with pre-test information.

Data paths are relative to the current working directory. `output_dir` is relative to the config file; update it when moving or copying a config. The input must be unweighted binary population data if expected clicks are to represent population counts.

## What the workflow does

1. Splits distinct timestamps into train, validation, and test periods, with an inner training tail for early stopping. Optional explicit cutoffs, embargo gaps, and strict session/entity-overlap checks are supported.
2. Uses Optuna to minimize validation logloss plus a configurable bootstrap standard-deviation penalty. The early-stop period chooses tree count; test is held out.
3. Refits the selected parameters and tree count on train + validation and saves the model and its fitted categorical encoder.
4. Evaluates test logloss, NE, z-score, click error, Brier, ROC AUC, and average precision, including configurable joint/marginal groups.
5. Computes row/cluster/time-block bootstrap intervals and separate bootstrap training-refit diagnostics, then exports native SHAP and gain explanations.

The z-score assumes independent Bernoulli outcomes. Metric intervals are conditional on fixed predictions; training refits are a separate sensitivity check. Neither guarantees robustness to future drift. The [detailed guide](../doc/model-tuning-and-techniques.md) explains these distinctions and undefined metric cases.

## Saved outputs and inference

Each successful run includes `report.html`, CSV/JSON tables, `study.sqlite3`, `model.ubj`, and `encoder.json`. Images are embedded in the HTML; keep the full run directory for linked tables and model artifacts. Predictions are optional via `report.save_predictions`. See the [artifact reference](../doc/model-tuning-and-techniques.md#10-reports-and-saved-artifacts) for the complete list.

Use the encoder saved with the model for future rows:

```python
import pandas as pd
from model_tuning.predict import predict

rows = pd.read_parquet("future_rows.parquet")
probabilities = predict(rows, "report/model_tuning/run_...")
expected_clicks = probabilities.sum()
```

This is in-memory CPU histogram training. Memory and runtime grow with dataset size, tree complexity, trials, and bootstrap refits; no silent training-data sampling is performed. Rolling-origin validation, GPU/external-memory training, and automatic study resume are not exposed.

## Source map and checks

| File | Responsibility |
|---|---|
| [`__main__.py`](__main__.py) | Two-input CLI and readable errors |
| [`config.py`](config.py) | Defaults, supported parameters, and config validation |
| [`data.py`](data.py) | Parquet loading, temporal partitions, and fitted encoding |
| [`pipeline.py`](pipeline.py) | Optuna trials, early stopping, final refit, and explanations |
| [`metrics.py`](metrics.py) | Probability metrics, bootstrap statistics, and group evaluation |
| [`report.py`](report.py) | HTML, plots, CSVs, and JSON serialization |
| [`predict.py`](predict.py) | Reload model plus encoder for inference |
| [`VALIDATION.json`](VALIDATION.json) | Historical validation and synthetic-demo results |

```bash
uv run --locked --extra tuning --python 3.12 python -m unittest discover -s tests -v
```

The tests cover chronological leakage, exact metric arithmetic, bootstrap units, null and zero-click groups, model reload, and SHAP additivity. Changing only test labels must leave chosen parameters and predictions unchanged.
