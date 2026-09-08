# Data Mining

A configurable Python project for data analysis and feature screening before
binary classification. Use it for clicks, conversions, churn, defects, or any
other target encoded as `0` and `1`.

Two inputs: **a local Parquet file and a YAML config**. Reports use generic
terms: rows, positives (`label=1`), negatives (`label=0`), and positive-class rate.
Regression and multiclass targets are not supported in this version.

See [Statistics and techniques: mathematics and interpretation](doc/statistics-and-techniques.md)
for every implemented metric, formulas, intuition, worked examples, sampling
details, configurable thresholds, and the exact keep/review/exclude rules.

The report answers:

1. What is the distribution and quality of each numerical/categorical column?
2. How does each feature relate to `label=1`, and how do features relate to one another?
3. Which features should be kept, reviewed, or excluded under your configured rules—and why?

The design is inspired by the report-oriented workflow in
[edvgha/click-dataset-audit](https://github.com/edvgha/click-dataset-audit).
This is a separate implementation using `uv`, Parquet read into pandas, explicit
feature types, analysis without dataset splitting, and one synthetic demo with
1,000,000 rows. It does not fit models, preprocess
training features, or claim that statistical screening establishes the best feature set.

## Run with your data

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
The committed `.python-version` selects Python 3.12 by default. The pinned
`pandas==2.2.3` does not support Python 3.14; pandas first added general Python
3.14 compatibility in [version 2.3.3](https://pandas.pydata.org/docs/whatsnew/v2.3.3.html).
The project's Python constraint excludes 3.14 and newer so uv cannot select that
unsupported combination. The lockfile pins dependencies; it does not by itself
pin the Python interpreter.
Unzip the project, change into its directory, and run:

```bash
uv sync --locked
uv run --locked -m data_mining /path/to/data.parquet config.yaml
```

Or use the same two-input Python interface:

```python
from data_mining import audit

report_directory = audit("/path/to/data.parquet", "config.yaml")
print(report_directory / "report.html")
```

Configure every input feature as `numerical` or `categorical`. Integer IDs should
usually be categorical. The target is configured separately and cannot be an input.
There are no primary/secondary categories or text features.

**Target spelling is exact.** Set `target` to any binary column in your file.
The included click-prediction demo uses `clickoccured`; that name is not hard-coded
in the audit engine.

```yaml
target: clickoccured
features:
  rank_pos: numerical
  apy: numerical
  device_type_id: categorical
  advertiser_account_id: categorical
ignore: []
strict_schema: true
output_dir: report

statistics:
  include: [cohens_d, auc, mutual_information, spearman, positive_rate]
  exclude: []

expected_direction:
  rank_pos: negative
  apy: positive

rules:
  weak_signal:
    enabled: true
    max_abs_d: 0.05
    max_auc_distance: 0.02
    max_mi: 0.0001
    action: review
  redundancy:
    enabled: true
    metric: spearman
    threshold: 0.95
    sign: absolute
    action: exclude
  direction:
    enabled: true
    action: review

prefer: [rank_pos, apy]
interactions:
  - [rank_pos, device_type_id]
```

This small config works for a dataset with exactly those four features and the
target. For your wider dataset, copy and edit the complete included
[`demo/config.yaml`](demo/config.yaml).
Unspecified settings and rule fields use documented defaults. Unknown config
keys, unknown metrics, and duplicate YAML keys produce readable errors.

## What to analyze

Strictly, univariate means one variable. A feature compared with the target is
bivariate, even when it is part of a per-feature report.

| Stage | Techniques and statistics | Why they matter for feature selection |
|---|---|---|
| Numerical univariate | Missing/nonfinite counts, unique values, dominance, mean, median, SD, quantiles, IQR, median absolute deviation, skewness, excess kurtosis, zeros, negatives, IQR outliers | Detect unusable columns, sentinels, heavy tails and rare values. A distribution need not be normal to work in a tree model. |
| Categorical univariate | Cardinality, dominant level, entropy, effective levels, singleton/rare levels, rare-row mass, pooled-row mass | Distinguish categories from identifiers and expose sparse/unseen-level concerns. |
| Numerical versus target | Class means, signed Cohen's d, optional Hedges' g, point-biserial correlation, Spearman correlation, raw-score AUC, KS statistic, binned mutual information | Cover mean, rank, distribution-shape and nonlinear relationships. |
| Any feature versus target | positive-class rate and lift by quantile/discrete/category bucket; counts, positives/negatives and Wilson intervals; binned/pooled MI; corrected Cramer's V; eligible chi-square tests and BH adjustment | Put effect size next to support. A million rows can make small effects look statistically significant. |
| Numerical feature pairs | Pearson and Spearman correlation; pairwise-complete sample counts | Identify linear/monotonic redundancy and give a named retained comparison feature. |
| Categorical/mixed pairs | Corrected Cramer's V for categorical pairs; correlation ratio eta-squared for categorical–numerical pairs; missingness phi | Detect redundancy and shared missingness beyond numeric correlations. |
| Selected pairs/triples plus target | Joint positive-class rate tables; joint and conditional MI; gain over the best single feature | Investigate nonlinear combinations and conditional associations. A marginally weak feature may matter jointly. |
| Optional time/group context | positive-class rate by period, bucket positive-class rate by period, Jensen–Shannon divergence of bucket distributions, group-size/positive-label summaries | Surface changing populations, changing positive-class rate and repeated sessions without creating training splits. |

During the later modeling stage, validate the candidate set with appropriate
time/group validation, log loss, average precision/PR-AUC and calibration. Use
permutation importance and feature/group ablation to measure incremental value.
Correlated columns can conceal each other's individual permutation importance;
consider removing or permuting a group together. See the
[scikit-learn permutation-importance guide](https://scikit-learn.org/stable/modules/permutation_importance.html).

Normality-test p-values, VIF cutoffs, and fixed WoE/IV thresholds are not prerequisites
for XGBoost feature selection. No automatic scaling or outlier removal is applied.
For later categorical training, consult the
[XGBoost categorical-data documentation](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html).

## Choose exactly which statistics to include

`statistics.include` accepts `all` or a list of names. `exclude` is a list and wins
when a name appears in both. An empty include list disables all optional metrics.

| Family | Accepted metric names |
|---|---|
| Numerical summaries | `mean`, `std`, `median`, `quantiles`, `iqr`, `mad`, `skewness`, `kurtosis`, `zero_fraction`, `negative_fraction`, `outlier_fraction` |
| Categorical summaries | `entropy`, `rare_categories` |
| Feature versus target | `positive_rate`, `cohens_d`, `hedges_g`, `auc`, `ks`, `point_biserial`, `spearman_target`, `mutual_information`, `cramers_v_target`, `chi_square` |
| Feature pairs | `pearson`, `spearman`, `cramers_v_pairs`, `eta_squared`, `missingness_phi` |
| Multivariate | `joint_positive_rate`, `joint_information` |
| Temporal | `time_positive_rate`, `time_feature_positive_rate`, `time_drift` |

For example:

```yaml
statistics:
  include: all
  exclude: [hedges_g, chi_square, kurtosis, time_drift]
```

Core bookkeeping—row/class counts, dtype, missingness, uniqueness, dominance,
analysis bucket support, and schema checks—always runs. Optional statistics
are omitted from reports and cannot drive rules when disabled. Shared profiling
or bucketing intermediates may still be calculated for other enabled diagnostics.
`positive_rate` controls per-feature positive-class rate tables/charts, not the mandatory dataset-level positive-class rate.
`joint_information` includes joint MI, conditional MI, and MI gains.

## Decisions and exact reasons

Each feature receives one final action:

- **keep**: retain as a candidate; no enabled review/exclusion condition fired.
- **review**: retain as a candidate, with a concern or weak/insufficient evidence.
- **exclude**: exclude from the suggested candidate list under a configured rule.

`selected_features.json` includes keep **and** review features. The original
Parquet file is never modified. These decisions are screening policy, not a
validated importance ranking.

Every evaluation is saved with `rule`, `status`, `requested_action`,
`effective_action`, `related_feature`, `reason`, and JSON `evidence` containing
values and thresholds. Disabled, skipped and nonapplicable rules are explicit.

| Rule | Default condition | Default action |
|---|---|---|
| `all_missing` | No usable observed values | exclude |
| `constant` | One value on every row, without varying missingness | exclude |
| `missingness` | Missing/nonfinite fraction >= 0.8 | review |
| `near_constant` | Dominant observed-value fraction >= 0.995 | review |
| `high_cardinality` | Categorical distinct count >= 1,000 | review |
| `weak_signal` | Numeric: all of `abs(d) < .05`, `abs(AUC-.5) < .02`, and MI < .0001. Categorical: pooled MI < .0001, unless pooling hides most rows. | review |
| `direction` | An appreciable d/AUC direction contradicts the configured expectation, or the two disagree | review |
| `redundancy` | Numeric absolute Spearman correlation >= .95 with a retained representative, with enough pair support | exclude |

Every rule accepts `enabled` and `action: keep|review|exclude`. `keep` records a
finding without escalating the action; it does not override another rule. Use
`force_keep` for an explicit override. Thresholds are tunable conventions.

The redundancy rule also accepts:

- `metric: pearson|spearman`.
- `sign: positive|negative|absolute`.
- `min_pair_rows`, default 1,000, and `min_pair_fraction`, default .5 of the audit sample.

Representative order is: forced keeps first, then the `prefer` list, lower
missingness, larger `abs(AUC-.5)` when enabled, then config order. Every excluded
column has a **direct** above-threshold correlation with a still-retained column;
transitive correlation chains do not silently delete whole groups. Categorical
and mixed redundancy measures are reported for review, but this rule only acts
on numeric pairs.

`force_exclude` acts before representative selection, so an explicitly excluded
column cannot be used as a retained anchor. `force_keep` wins over automatic
decisions and leaves all underlying findings in the rule log. Overlapping
force-keep/force-exclude lists are rejected.

`protect_configured_interactions: true` changes proposed weak-signal/redundancy
exclusions to reviews for features in configured interaction hypotheses. It
does not protect genuinely constant or all-missing features. This protection
does not prove an interaction exists; it avoids discarding a hypothesis solely
from marginal evidence. Unconfigured interactions can still be missed.

## Correctly interpret Cohen's d and AUC

For this project:

```text
pooled_SD = sqrt(((n1 - 1) * variance1 + (n0 - 1) * variance0) / (n1 + n0 - 2))
cohens_d  = (mean(feature | label=1) - mean(feature | label=0)) / pooled_SD
```

Positive d means the feature mean is higher among positives. Negative d means it is
lower. Zero pooled variance or fewer than two finite rows in either class makes
d undefined; the report gives a status instead of inventing a value.

Raw AUC uses the feature itself as a score for `label=1`. It measures ranking,
not the fitted XGBoost model:

- AUC .80: larger feature values generally rank positives above negatives.
- AUC .20: the ordering is inverse; `-feature` has AUC .80.
- AUC near .50 and small d: weak marginal mean/rank separation; U-shapes and interactions can still be useful.

The code reports `raw_numeric_auc`, `flipped_auc=1-AUC`, and
`direction_free_auc=max(AUC,1-AUC)`. The latter chooses direction on the same sample
and is descriptive. It does not estimate held-out performance. See
[scikit-learn's AUC definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).

Mean-based d and rank-based AUC can disagree when distributions are skewed or
have influential tails. A positive d with AUC below .5 is not necessarily an
implementation error. The rule records a review when both disagree beyond the
configured minimums (`min_abs_d=.05`, `min_auc_distance=.02`). Expected directions
are domain expectations about observed associations, not causal effects and
not XGBoost monotonic constraints.

## Binning, sampling and statistical limits

- Read the required Parquet columns into one pandas DataFrame; this is an in-memory tool, not an out-of-core engine. Available RAM must cover the frame and working copies.
- The local file reader uses PyArrow `ParquetFile` with background prefetching disabled and single-threaded decoding. It builds pandas columns without calling Arrow's `to_pandas()` converter, and prints each column being converted. Timestamps use exact integer epoch ticks viewed as NumPy datetime arrays; other columns use Python values. This bypasses the converter and Python timestamp-list construction implicated in the reported loading stalls. Numeric widths, pandas nullable dtypes, dictionary categories (including unused levels), timestamp units/timezones, and row order are preserved. Nullable integer/boolean columns without pandas metadata use pandas nullable dtypes to avoid losing values. A fresh row index is created; any stored index field follows the config's ordinary feature/context/ignore rules.
- Full data: profiles, bucket counts, positive-class rate, binned feature-target MI/Cramer's V/chi-square, temporal and group summaries.
- Fixed uniform sample without replacement (100,000 rows by default): numerical target metrics, feature pairs, joint/conditional MI. The sample is not class-balanced; the seed, row count and positive-label count are reported.
- Numerical buckets use observed values directly when cardinality <= `bins`; otherwise quantile bins with duplicate edges dropped. Missing/infinite values have an explicit bucket. Many ties may reduce the number of bins or hide a sparse tail, so inspect bucket counts and change `bins` when necessary.
- Categorical summaries describe all levels. Association tables retain up to `max_categories` frequent levels meeting `rare_min_count`; remaining levels become OTHER. Null has a separate collision-safe code. Category pooling can hide ID signal; no numeric rank/AUC or d is applied to unordered category codes.
- Missing and nonfinite numerical values are excluded from raw numerical class comparisons, with exact usable sample counts reported. No target-dependent imputation or encoding is performed.
- `max_pairs` caps pairs in config order (not target-score order). `max_joint_cells` caps joint/pair contingency-table size. Skips and coverage are reported. Only configured two-/three-feature combinations are examined.
- Empirical MI is in natural-log units (nats). It has finite-sample/cardinality bias. Joint MI gain can reflect additive information, nonlinear effects and bias; it is not a pure interaction score. Conditional MI is `I(feature; target | other configured features)`.
- Wilson confidence intervals assume independent unweighted rows. Repeated sessions/users can make them too narrow. With `group_column` configured, chi-square p-values are suppressed because row independence is not established; descriptive effect sizes and explicitly labeled IID intervals remain available.
- Without `group_column`, chi-square p-values are only emitted when every expected cell count is at least 5. BH q-values cover the finite feature-target chi-square p-values only and have the usual independence/positive-dependence assumptions. They do not validate every comparison in the audit. See [SciPy chi-square](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.chi2_contingency.html) and [false-discovery control](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).
- Temporal JS divergence compares fixed-bucket distributions in each period with the whole dataset. It is descriptive, bounded by ln(2), and is the square of SciPy's [Jensen–Shannon distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html). Missing dates are separate; unparseable nonmissing dates are rejected. `time_frequency` supports `D`, `W`, `M`, with at most `max_time_periods` periods.
- Counts, probabilities and effect sizes are unweighted. Use original rows to estimate population positive-class rate. Negative sampling/class balancing changes the observed population.

Review data lineage before choosing features. Measurements made after the
outcome and aggregates that include the current/future label must not become
predictors. In the click example, a logged position may be available after placement
but unavailable before ranking. No statistical test can certify prediction-time
availability. Analyze the development data and keep the eventual final test set
untouched by feature-selection decisions.

## Outputs

Every run creates a fresh directory under `output_dir`, resolved relative to the
config file. The absolute output directory is printed at startup, with progress
for feature profiles, feature pairs, interactions, temporal analysis, and report
writing. The final `Report:` line gives the exact HTML file to open in your
browser; the command does not open a browser automatically. Old reports cannot
be mixed with changed settings.

| File | Contents |
|---|---|
| `report.html` | Self-contained report with feature decisions, exact reasons, tables, and enabled positive-class rate plots |
| `report.md` | Readable per-feature decisions and interpretation notes |
| `feature_decisions.csv` / `.json` | One row/object per feature, final action, related columns, evidence values and reason |
| `rule_evaluations.csv` | Every evaluated, disabled or skipped rule; numeric thresholds in JSON evidence |
| `selected_features.json` | Keep + review candidates, excluded features and review list |
| `numerical_univariate.csv`, `categorical_univariate.csv` | Full-data profiles and enabled summaries |
| `feature_target.csv` | Enabled target-association statistics |
| `feature_pairs.csv` | Enabled numeric/categorical/mixed pair associations |
| `joint_information.csv`, `joint_positive_rate/` | Configured multivariate diagnostics |
| `feature_positive_rate/` | Enabled per-feature positive-class rate bucket tables |
| `time_positive_rate.csv`, `time_feature_positive_rate/`, `time_feature_drift.csv` | Enabled temporal diagnostics |
| `summary.json`, `manifest.json`, `bucket_dictionary.csv`, `config_used.yaml` | Row/sample/schema metadata, safe filename mappings, bucket labels and reproducible effective config |

## One million-row demo

```bash
uv sync --locked
uv run --locked -m data_mining demo/data.parquet demo/config.yaml
```

The `demo/` directory contains exactly two ready-to-use inputs:

- [`data.parquet`](demo/data.parquet): 1,000,000 synthetic rows, compressed with Zstandard.
- [`config.yaml`](demo/config.yaml): the corresponding feature schema, statistics, and decision rules.

The command reads the saved Parquet through the same two-input interface used
for your own datasets. The click-prediction example contains 27 columns:
24 declared candidate features, the `clickoccured` target, `searchid` and
`search_date`. The package itself accepts any binary target name and feature schema.

The data deliberately contains a duplicate `widget_pos`/`rank_pos`, inverse rank
signal, positive APY signal, sparse IDs, missing values, noise, and a joint
`listing_set`/`inventory_type` effect. These are synthetic demonstrations, not
findings about your actual data. The saved dataset was generated with seed 42.
APY is stored to two decimal places and synthetic asset values to two significant
digits. Dictionary/delta encoding and Zstandard compression keep the million-row
file compact without reducing the number of rows or columns.

The demo is already generated and committed; no generation script is needed.
Its configuration uses `output_dir: ../report`, resolved relative to `demo/`,
so every run writes a fresh report under the root `report/` directory.
Generated reports stay out of Git and out of `demo/`. The shipped Parquet is
synthetic demonstration data, not real customer data.

**Where is the report?** Wait for `Analysis complete`, then open the file shown
on the final `Report:` line. For this demo it is
`report/run_<UTC timestamp>/report.html` inside the project directory, with
`report.md` and the CSV/JSON files beside it. Reaching `[24/24] week_day` only
finishes the per-feature stage; pair, interaction, temporal, and report-writing
stages still follow. The million-row run can take a few minutes depending on
your machine and enabled statistics. If it exits with `Analysis failed: ...`,
the run did not complete; that error explains what needs correcting.

If the command stalls during **Reading and validating Parquet**, loading has not
finished and no report is expected yet. Separate progress messages identify
metadata loading, column decoding, pandas conversion, and validation. Stop the
stalled command with Ctrl+C, then run:

```bash
uv run --locked -m data_mining demo/data.parquet demo/config.yaml --diagnostics
```

This prints Python, OS/architecture, pandas, and PyArrow versions. If loading or
validation takes longer than 30 seconds, Python stack snapshots are printed to
stderr every 30 seconds until input validation finishes. A snapshot does not
terminate the process or by itself mean an error occurred. Include the last
progress message and the stack output when reporting a persistent stall.
If the shell prompt returns before `Analysis complete`, the process has exited;
it is no longer waiting for the 30-second stack snapshot. Capture its exit code
and enable Python's fatal-error traceback handler with:

```bash
uv run --locked --python 3.12 python -X faulthandler -m data_mining demo/data.parquet demo/config.yaml --diagnostics
echo "Exit code: $?"
```

This explicitly selects Python 3.12, including when an earlier run used 3.14.
uv manages the project environment and downloads the requested Python version
if needed. No manual deletion of `.venv` or source edits are required. Share the
full output and exit code if the command still ends before creating a report;
the exit code and any fatal traceback distinguish failure modes that progress
messages alone cannot identify.

If an older checkout stops at `table.to_pandas(use_threads=False)`, update it
with `git pull --ff-only`: the current reader no longer calls that conversion.
The same update handles stalls at `search_date (timestamp[ns])` while constructing
a Series: timestamp columns bypass `to_pylist()` and `to_pandas_dtype()`, use
their original timestamp unit, map nulls to `NaT` without floating-point
conversion, and restore any time zone from UTC instants.
The reader's APIs are documented in the
[PyArrow ParquetFile API](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html)
and [ChunkedArray.to_pylist API](https://arrow.apache.org/docs/python/generated/pyarrow.ChunkedArray.html#pyarrow.ChunkedArray.to_pylist).

## Project files and tests

- `data_mining/audit.py`: input validation, statistics, analysis pipeline and report writing.
- `data_mining/settings.py`: metric registry and validated rule configuration.
- `data_mining/decisions.py`: decision engine, precedence and evidence.
- `demo/data.parquet`: saved 1-million-row synthetic example dataset.
- `demo/config.yaml`: matching configuration for the example.
- `tests/`: formula checks, data edge cases, decision logic and end-to-end switches.
- `pyproject.toml`, `uv.lock`: pinned reproducible dependencies.
- `.python-version`: default Python 3.12 interpreter selection for uv.

```bash
uv run --locked -m unittest discover -s tests -v
```

There is no network lookup during an audit and no requirement for XGBoost,
SHAP or a database at this stage.
