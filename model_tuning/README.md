# Binary classification model tuning

Tune a native `xgboost.Booster` with Optuna using a local Parquet file and YAML config. All fitting uses `xgboost.train` and `DMatrix`; there is no `XGBClassifier`, sklearn estimator, or sklearn preprocessing pipeline. The existing sklearn dependency supplies only ranking metrics here.

## Run

From the repository root:

```bash
uv sync --locked --extra tuning --python 3.12
uv run --locked --extra tuning --python 3.12 python -m model_tuning \
  demo/data.parquet model_tuning/config.demo.yaml
```

For your data:

```bash
uv run --locked --extra tuning --python 3.12 python -m model_tuning \
  /path/to/data.parquet model_tuning/config.yaml
```

The data path is relative to the current working directory. `output_dir` is relative to the YAML file. Each run creates a unique `run_*` directory and prints the absolute `Report:` and `Model:` paths. Open `report.html`; its images are embedded, so the HTML is viewable offline. Keep its directory for the linked CSVs and model artifacts.

`config.yaml` is a starting template based on the attached **analysis report**, not a claim that its feature selection is optimal. The attachment contains no training rows. The requested target spelling is `clickoccurrred`; the repository's existing demo actually uses `clickoccured`. The report shows `publisher_accountid`, while the demo uses `publisher.accountid`. Change the configuration to exactly match your own Parquet schema. Missing columns cause a readable error with close matches; the tool never guesses a replacement.

The example omits several possible identity/twin alternatives and quarantines `click_or_ref_url_host` pending confirmation that it exists before a prediction is made. These are editable starting choices. The tuner does **not** automatically prove that columns are twins or detect leakage. Compare alternative feature sets using pre-test data; reserve a new test window if it has already influenced your decisions.

## Time splitting and model selection

1. Parse the configured time column as UTC, sort chronologically, and keep identical timestamps together. Numeric epoch timestamps must be converted to datetimes before input.
2. The default fractions allocate 60% of distinct timestamps to training, 20% to validation, and the remainder to test. These are **fractions of timestamps**, not row fractions or elapsed duration. Alternatively provide both inclusive UTC cutoffs `split.train_end` and `split.validation_end`.
3. Reserve the last 15% of training timestamps for early stopping. The earlier training portion fits candidate trees. Learn categorical vocabularies only from this earlier fitting portion.
4. For each trial, choose the number of trees from the inner early-stopping window. Refit the trial on the full outer training window, with an encoding learned from that window. Evaluate on validation, which was not used for early stopping.
5. Minimize the robust validation objective described below. No test labels or test-fitted encoding are used in these steps.
6. Refit the winning parameters and selected tree count on train + validation. Do not early-stop on test. Save that model and its encoder, then evaluate the newest held-out test period.

With a positive `split.gap`, the early-stopping and evaluation windows begin strictly after the previous cutoff plus the gap. Outer gap rows are omitted even in the final refit; the inner gap is excluded for early stopping but belongs to full outer training. Boundary counts and time ranges are saved. Training, early-stopping and validation windows require both classes. A one-class test window is accepted, with undefined NE/AUC metrics explicitly reported.

If a search/session must not straddle partitions, configure `split.entity_column`. The tool rejects crossing or missing IDs so you can fix the boundaries or gap. Group reporting columns do not automatically impose entity isolation. Repeated publishers may validly appear in all periods when predicting future traffic for existing publishers; that is a different goal from generalizing to new publishers.

## Bootstrap and stability

Configure `bootstrap.method`:

| Method | Resampling unit | When to use |
|---|---|---|
| `time_block` (default) | All rows in a fixed non-overlapping interval, e.g. `1D` or `7D` | Shared temporal shocks within a period |
| `cluster` | All rows sharing `cluster_column`, e.g. `searchid` | Repeated observations within sessions/users |
| `row` | Individual observation | Approximately independent rows |

Each replicate draws the same number of units with replacement. Entire units receive a shared multinomial multiplicity. Unequal-size units therefore produce different row totals. Per-unit sufficient statistics keep metric bootstrap practical on large files without storing a resampled DataFrame for every replicate. The day/week blocks are sampled independently; dependence beyond the configured block duration remains unmodeled. This is not a moving-block or two-way cluster bootstrap.

For trial probabilities on validation, calculate the observed log loss `L` and standard deviation `s_B` of bootstrapped log losses:

```text
objective = L + tuning.stability_penalty * s_B
```

The default penalty is 1.0; 0.0 selects ordinary validation log loss. Identical resampling seeds across trials reduce comparison noise. This is a configurable stability preference, not a formal confidence bound or a guarantee of future robustness. It measures validation-sample uncertainty for a fixed fitted candidate. It does not retrain a candidate for every bootstrap draw.

After choosing the winner, `bootstrap.training_repeats` performs separate weighted multinomial bootstrap **refits** of train + validation at fixed hyperparameters, tree count, seed, and encoding. These runs report held-out metric variation and mean absolute changes in predictions. They are a sensitivity diagnostic, never a test-based model-selection step. The saved `model.ubj` is always the unresampled final refit. Weighting avoids duplicating millions of rows; histogram quantization and XGBoost's internal subsampling mean this is not bit-for-bit equivalent to physically duplicating rows. One-class training resamples are skipped and counted. Five repeats provide a coarse diagnostic, not a reliable training-uncertainty interval.

The held-out prediction bootstrap gives percentile intervals at `bootstrap.confidence` (default 95%). It also runs within every configured group, preserving the units present in that group. Groups with fewer than two units receive undefined intervals; the number of valid draws is saved for metrics that can become undefined. Intervals are marginal, not simultaneous multiple-comparison guarantees. Fixed-model intervals and training-refit diagnostics are separate; neither captures the entire hyperparameter-selection uncertainty. Very few blocks, temporal drift, and dependence between blocks can make intervals unreliable.

## Metrics

Let `y_i ∈ {0,1}`, predicted probability `p_i`, `A = Σy_i`, `E = Σp_i`, `n` be row count, and `r=A/n`.

| Metric | Definition | Interpretation |
|---|---|---|
| Logloss | `−mean(y ln p + (1−y) ln(1−p))` | Lower is better; natural logarithms |
| NE | `Logloss / [−r ln r − (1−r) ln(1−r)]` | Below 1 beats the constant empirical rate on this evaluation subset |
| NE against fitting-rate baseline | `Logloss / Logloss(y, constant fitting_rate)` | Baseline rate learned from train for validation, train+validation for final test |
| z-score | `(A−E) / sqrt(Σ p_i(1−p_i))` | Positive means underprediction; independent-Bernoulli approximation |
| Click error (%) | `100(E−A)/A` | Positive means overprediction; also exports absolute percentage error and raw count difference |
| Brier | `mean((y−p)^2)` | Squared probability error |
| ROC AUC | Probability that a positive ranks above a negative, with half credit for ties | Ranking quality, not calibration |
| Average precision | Precision averaged over recall increments | Ranking metric useful for rare positives; not trapezoidal PR AUC |

NE uses each evaluation group's own observed rate. This reference is retrospective and must not be mistaken for a deployable baseline; the second NE variant supplies the honest fitting-rate reference. NE is undefined for single-class groups. Percentage error is undefined when actual clicks are zero. z-score is undefined when its denominator is zero. JSON uses `null`, CSV uses empty values, and status columns explain these cases. Logs alone clip probabilities to `[1e-15, 1−1e-15]`; expected-click sums use original probabilities.

The z-score's denominator assumes independent Bernoulli outcomes with the predicted probabilities. Repeated search impressions and temporal dependence can invalidate that variance. Bootstrap intervals do not turn this z-score into a cluster-corrected hypothesis test. The report does not provide significance p-values or automatically reject groups based on `|z| > 1.96`.

All metrics are unweighted on the original input population. Do not use balanced/downsampled negatives and interpret the resulting click sums as population estimates. This version intentionally fixes `binary:logistic`, `logloss`, `gbtree`, and `hist`; class weights, `scale_pos_weight`, ranking objectives, and probability calibration fitting are out of scope.

## Grouping

A flat list creates a **joint** grouping:

```yaml
group_by: [publisher_account_id, service_code]
```

For separate marginals plus joint groups:

```yaml
group_by:
  - [publisher_account_id]
  - [service_code]
  - [publisher_account_id, service_code]
```

Use names in your actual file. Missing group values are retained. Every observed group gets metrics and bootstrap interval columns in `groups_0.csv`, etc. The HTML shows the 100 largest groups; CSVs contain all groups. `min_group_rows` and `min_group_clicks` only flag low support; they never drop groups or silently change the objective. Tuning optimizes overall validation log loss, not group error or a group-weighted objective.

## Features and tuning controls

`features` explicitly maps input names to `numerical` or `categorical`. Integer IDs should normally be categorical. Target/time and optional split entity ID cannot be model features. Grouping columns become model inputs only if also listed under `features`. Undeclared extra Parquet columns are not read.

Categorical values are represented as strings with a fitting-only vocabulary. Missing and unseen values become missing; XGBoost learns native categorical splits and missing-value branches. Numeric values become float32; infinities become missing. Numeric strings must be parseable. Keep categorical source types consistent across training and serving. Original names are saved alongside safe internal XGBoost names `f0`, `f1`, etc.

Supported fixed/search parameters: `eta`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `colsample_bylevel`, `colsample_bynode`, `lambda`, `alpha`, `gamma`, `max_delta_step`, `max_bin`, `max_cat_to_onehot`, `max_cat_threshold`.

| Control | Purpose |
|---|---|
| `n_trials`, `timeout_seconds` | Search budget; timeout is checked between trials, not a hard process deadline |
| `num_boost_round` | Maximum early-stopping search length |
| `early_stopping_rounds` | Patience on inner chronological early-stop log loss |
| `params` | Fixed XGBoost parameter values |
| `search_space` | Named int/float/categorical distributions; replaces the default space when supplied |
| `nthread` | XGBoost CPU threads; Optuna trials run sequentially |
| `seed` | Reproducible sampling and model RNG under the same software/environment |

Example custom space:

```yaml
tuning:
  params: {subsample: 1.0}
  search_space:
    max_depth: {type: int, low: 3, high: 7}
    eta: {type: float, low: 0.02, high: 0.15, log: true}
    min_child_weight: {type: categorical, choices: [5, 20, 50]}
```

A parameter cannot be both fixed and searched. Unknown keys and unsupported parameters fail validation. Each run stores an inspectable Optuna SQLite study; automatic cross-run resume is intentionally not exposed, to avoid accidentally mixing changed data/configurations.

## Report and saved artifacts

- `report.html`: offline report with optimization history, early-stopping learning curve, test calibration, daily actual/expected click rates, group click comparisons, native gain/total gain, SHAP importance and contribution distributions.
- `model.ubj`, `encoder.json`: final native model and matching preprocessing schema.
- `config.yaml`, `best_params.json`, `summary.json`, `status.json`: resolved settings, selected tree count, versions, data file metadata, split counts, unseen-category counts, and completion state.
- `study.sqlite3`, `trials.csv`: full Optuna history and validation objective components.
- `metrics.csv`, `bootstrap_intervals.csv`, `training_stability.csv`, `groups_*.csv`: overall/group evaluation and robustness diagnostics.
- `feature_importance.csv`: average/total gain, split count, average/total cover, mean absolute and signed SHAP contributions. Unused features have zero split importance.
- `shap_values.csv`, `shap_features.csv`, `shap_context.csv`: matching sampled test rows, contribution values plus bias, labels/predictions/raw margins. Context `test_row` indexes the chronologically sorted test partition.
- `test_predictions.parquet` if `report.save_predictions: true`: test time, target, grouping columns and probability.

SHAP uses native `Booster.predict(pred_contribs=True)`. Contributions plus bias sum to the raw margin (log-odds), and sigmoid gives the prediction. Positive contributions raise that row's modeled odds relative to its bias; they are not probability-point changes. Mean absolute SHAP measures average modeled impact on the fixed test sample. Average gain measures improvement at splits using a feature; total gain also reflects how often it is used. Correlated alternatives can redistribute both scores. Importance does not establish causality or validate a post-outcome input. The pipeline checks SHAP additivity before completing the report.

Load predictions through the saved encoder:

```python
import pandas as pd
from model_tuning.predict import predict

rows = pd.read_parquet("future_rows.parquet")
probabilities = predict(rows, "report/model_tuning/run_...")
expected_clicks = probabilities.sum()
```

This implementation is in-memory, CPU histogram training. Runtime and RAM scale with rows, categories, tree depth, trials, and bootstrap refits. Ten million rows can require substantial RAM because several training matrices and partitions are held during tuning. It does not silently sample your training data. For a first run reduce trials/tree cap/SHAP rows; production-scale streaming/external-memory and rolling-origin validation are future extensions.

## Validation and API references

Run all existing and new tests with:

```bash
uv run --locked --extra tuning python -m unittest discover -s tests -v
```

Tests cover manual metric arithmetic, undefined edge cases, tied timestamps, embargoes, entity overlap, train-only category encoding, whole-cluster sampling, null/zero-click groups, schema errors, a full CLI run, model reload and SHAP reconstruction. A second run flips only test labels and checks that selected parameters, rounds, validation objective and predictions are unchanged.

References: [XGBoost native Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html), [XGBoost categorical support](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html), [Optuna search spaces](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html).
