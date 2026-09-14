# Model tuning and techniques: mathematics and interpretation

This is the mathematical and operational reference for `model_tuning`, complementing the [data-mining statistics guide](statistics-and-techniques.md). It describes the implementation inspected on 2026-09-14, following the tuning implementation merged in [94d4e46](https://github.com/edvgha/data-mining/commit/94d4e4620700c40761a33bd88891526e4c2ca5a6).

The code is the source of truth:

- [config.py](../model_tuning/config.py): supported settings and defaults.
- [data.py](../model_tuning/data.py): schema, time partitions, and categorical vocabularies.
- [pipeline.py](../model_tuning/pipeline.py): native training, trial selection, refit, and SHAP.
- [metrics.py](../model_tuning/metrics.py): metric and bootstrap calculations.
- [report.py](../model_tuning/report.py): plots and exported tables.
- [demo/model_tuning.yaml](../demo/model_tuning.yaml): runnable demo configuration.

All CLI examples run from the repository root. For setup and commands, start with the [project overview](../README.md) or [tuning README](../model_tuning/README.md).

## Contents

1. [Scope, notation, and calculation populations](#1-scope-notation-and-calculation-populations)
2. [Input validation and feature encoding](#2-input-validation-and-feature-encoding)
3. [Chronological selection and final refit](#3-chronological-selection-and-final-refit)
4. [Native XGBoost objective and regularization](#4-native-xgboost-objective-and-regularization)
5. [Optuna search and the robust objective](#5-optuna-search-and-the-robust-objective)
6. [Bootstrap uncertainty and training sensitivity](#6-bootstrap-uncertainty-and-training-sensitivity)
7. [Evaluation metrics and worked calculations](#7-evaluation-metrics-and-worked-calculations)
8. [Group evaluation and support](#8-group-evaluation-and-support)
9. [Gain and SHAP interpretation](#9-gain-and-shap-interpretation)
10. [Reports and saved artifacts](#10-reports-and-saved-artifacts)
11. [Complete configuration reference](#11-complete-configuration-reference)
12. [Practical review sequence and limitations](#12-practical-review-sequence-and-limitations)

## 1. Scope, notation, and calculation populations

The task is binary probability prediction. For observation $i$:

| Symbol | Meaning |
|---|---|
| $x_i$ | Configured numerical/categorical features |
| $y_i\in\{0,1\}$ | Observed binary target |
| $t_i$ | UTC timestamp used to order and split observations |
| $F(x_i)$ | Raw model margin, in log-odds |
| $p_i=\sigma(F(x_i))$ | Predicted positive-class probability, with $\sigma(z)=1/(1+e^{-z})$ |
| $n$ | Rows in the current evaluation population |
| $A=\sum_i y_i$ | Actual positives/clicks |
| $E=\sum_i p_i$ | Expected positives/clicks |
| $r=A/n$ | Observed positive rate |
| $q$ | Constant baseline probability estimated from fitting data |
| $\theta$ | Trial hyperparameters |
| $K$ | Selected boosting-round count |
| $B$ | Number of metric bootstrap resamples |
| $U$ | Number of resampling units in a population |

The word “click” is used in output names, but the formulas apply to any binary event. The code does not fit a ranking objective, multiclass classifier, or regression model.

### 1.1 Which data influence which result?

| Population | Purpose | May influence final parameters? |
|---|---|---|
| Inner `fit` | Fit trees during early stopping | Yes |
| Inner `early_stop` | Select a trial's boosting-round count | Yes |
| Outer `train` | Refit each candidate before validation | Yes |
| `validation` | Rank Optuna trials by the robust objective | Yes |
| `train` + `validation` | Fit the final saved model and encoder | Yes, after selection |
| `test` | Final evaluation and explanations | No |
| Uniform sample of test rows | SHAP export and SHAP importance | No; explanations only |

`fit` and `early_stop` are subsets of `train`, not additional independent datasets. With a gap, they do not necessarily cover every training row. Do not sum all five rows in `splits.csv` to get dataset size.

Reported validation scores are selected on and therefore optimistic. Test scores evaluate the final refit. Bootstrap training replicas are evaluated on the same test period but are never used to replace the saved model or select parameters.

## 2. Input validation and feature encoding

### 2.1 Schema and target

Optuna and XGBoost are standard project dependencies. Run `uv sync --locked --python 3.12` from the repository root to install the shared analysis and tuning environment.

The CLI takes exactly two positional inputs:

```bash
uv run --locked --python 3.12 python -m model_tuning \
  demo/data.parquet demo/model_tuning.yaml
```

The input must be a local `.parquet` file. The loader reads the target, time column, configured features, group columns, and configured entity/cluster identifiers. Other columns are ignored. Duplicate Parquet names, missing required columns, invalid/missing labels, and missing/unparseable timestamps are rejected. Labels must be numeric 0/1 (boolean values also satisfy the check); strings such as `"yes"` are not recoded.

Missing column errors include close-name suggestions without replacing names. The shared demo uses `clickoccured`; the separate custom-data template uses `clickoccurrred`. Neither spelling is hard-coded into the model.

YAML duplicate keys and unknown settings are rejected. Feature types are explicitly `numerical` or `categorical`. Target and time columns are excluded from model inputs; the optional split entity ID also cannot be a feature. The feature name `__bias__` is reserved for the SHAP output.

### 2.2 Numeric representation

Numeric values are converted to float32. Parseable numeric strings are accepted; nonnumeric strings fail. Positive/negative infinity becomes missing (`NaN`). There is no standardization, median imputation, winsorization, or target-dependent transformation. XGBoost handles missing branches natively.

Float32 conversion can reduce precision. Large numeric IDs should normally be declared categorical rather than relying on exact float32 integer representation.

### 2.3 Categorical vocabularies

For categorical column $j$ and fitting population $D$, the encoder stores:

$$
V_j(D)=\operatorname{sort}\{\operatorname{string}(x_{ij}):i\in D,\ x_{ij}\text{ nonmissing}\}.
$$

A later nonmissing value outside $V_j(D)$ becomes missing. The corresponding pandas categorical dtype lets XGBoost fit native categorical splits. This is not target encoding; no label average is assigned to a category.

Three fitting vocabularies have distinct roles:

1. Inner fitting vocabulary, reused for the early-stop window.
2. Outer training vocabulary, reused for validation.
3. Final train+validation vocabulary, reused for test and subsequent inference.

Future/test-only categories never enter the final fitted vocabulary. Unknown test counts are saved in `summary.json`. Missing and unknown categories follow the model's learned missing-value handling; an unknown category does not receive a newly trained category-specific effect.

Keep categorical source types consistent across training and serving: string representations such as `"123"` and `"123.0"` differ. Internal feature names are `f0`, `f1`, etc.; the encoder preserves original names and order. Always reload the saved encoder with the model.

## 3. Chronological selection and final refit

### 3.1 Fraction-based cutoffs

Let the ordered distinct timestamps be $u_1<\cdots<u_M$, training fraction $a$, and validation fraction $b$. The implementation uses:

$$
k_{\mathrm{train}}=\lfloor Ma\rfloor,
\qquad k_{\mathrm{val}}=\lfloor M(a+b)\rfloor,
$$

with cutoffs $c_{\mathrm{train}}=u_{k_{\mathrm{train}}}$ and $c_{\mathrm{val}}=u_{k_{\mathrm{val}}}$ for valid nonempty windows. These are timestamp fractions, not row-count fractions or elapsed-time fractions. Irregular traffic volumes can yield quite different row proportions.

The default $(a,b)=(0.6,0.2)$ reserves the remaining timestamps for test. Equal timestamps cannot straddle a boundary. At least five distinct timestamps are required, and actual partitions must also be nonempty. Very small windows can fail the inner-split or both-class checks despite satisfying the initial five-timestamp minimum.

### 3.2 Explicit cutoffs and embargo gaps

Alternatively set both `train_end` and `validation_end` to inclusive UTC cutoffs. With nonnegative duration $g$ from `split.gap`:

$$
D_{\mathrm{train}}=\{i:t_i\le c_{\mathrm{train}}\},
$$

$$
D_{\mathrm{val}}=\{i:c_{\mathrm{train}}+g<t_i\le c_{\mathrm{val}}\},
$$

$$
D_{\mathrm{test}}=\{i:t_i>c_{\mathrm{val}}+g\}.
$$

For example, a training cutoff of January 31 at midnight and `gap: 1D` excludes February 1 at midnight from validation: the comparison is strictly greater than cutoff plus gap. A date-only cutoff represents midnight, not the end of that date; include the intended time when input timestamps are intraday.

Both outer gaps remain excluded from the final refit. Counts appear as `excluded_gap_rows`. The code checks that the gap is nonnegative and that training cutoff precedes validation cutoff.

### 3.3 Inner early stopping

Let $M_T$ be the number of distinct training timestamps and $e$ be `early_stopping_fraction`. The inner fitting cutoff is at timestamp number:

$$
k_{\mathrm{fit}}=\lfloor M_T(1-e)\rfloor.
$$

Rows up to that cutoff form `fit`; later training rows strictly beyond the same configured gap form `early_stop`. The default $e=0.15$ reserves the training tail. The omitted inner gap is tracked as `inner_gap_rows`; these rows return when the entire outer training period is refitted.

For each trial, native `xgboost.train` monitors `early_stop` logloss. It stops after `early_stopping_rounds` without improvement, subject to `num_boost_round`. The selected count is:

$$
K_\theta=\texttt{best\_iteration}+1.
$$

The returned early-stopped booster can contain extra patience rounds. This pipeline reads the best count and trains a fresh candidate on full outer training with exactly $K_\theta$ rounds; it does not accidentally predict with the extra rounds. The final refit also uses that count. [Native training and early-stopping API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.train).

### 3.4 Why the separate validation period matters

Early-stop labels choose $K_\theta$; validation labels choose among $\theta$ values. Reusing one period for both creates more selection pressure on that period. The separate windows make those roles explicit, while the final test remains excluded from both decisions.

The final model fits $D_{\mathrm{train}}\cup D_{\mathrm{val}}$ at $(\theta^*,K_{\theta^*})$. It does not recompute optimal rounds on test, and it is not a continuation of a partially trained earlier booster. Its base intercept and feature vocabulary may change with the larger fitting population.

### 3.5 Entity overlap and class checks

If `split.entity_column` is configured, its IDs must be nonmissing and disjoint between outer train/validation/test, and between inner fit/early-stop. The code fails rather than reassigning rows. Use appropriate timestamp boundaries or a gap when a session crosses a cutoff.

Without that setting, repeated entities are not checked. Reporting groups do not impose isolation. Predicting future impressions for known publishers is different from predicting new publishers; choose the split definition that matches the intended use.

All non-test partitions require both classes. A single-class test is accepted; affected metrics are marked undefined. A gap alone cannot establish that labels have matured or that each input was available at prediction time—those properties require data definitions.

## 4. Native XGBoost objective and regularization

### 4.1 Logistic loss, gradients, and Hessians

For raw margin $F_i$:

$$
p_i=\sigma(F_i),\qquad
\ell_i=-y_i\ln p_i-(1-y_i)\ln(1-p_i).
$$

Differentiating with respect to the margin gives:

$$
g_i=\frac{\partial\ell_i}{\partial F_i}=p_i-y_i,
\qquad
h_i=\frac{\partial^2\ell_i}{\partial F_i^2}=p_i(1-p_i).
$$

For $y=1,p=0.2$, $g=-0.8$: a positive margin change initially reduces loss. For $y=0,p=0.8$, $g=0.8$: a negative change reduces loss. Both have Hessian $0.16$. These are derivatives in log-odds space, not direct probability updates.

A conceptual boosting update is:

$$
F^{(k)}(x)=F^{(k-1)}(x)+\eta f_k(x).
$$

The native model stores shrinkage in its learned tree contributions. Prediction does not multiply the saved model output by `eta` again.

### 4.2 Leaf updates

For one leaf containing observations $I$, define $G=\sum_{i\in I}g_i$ and $H=\sum_{i\in I}h_i$. Its second-order regularized objective, up to constants, is:

$$
Q(w)=Gw+\frac12(H+\lambda)w^2+\alpha|w|.
$$

Without an active step constraint, its minimizer is:

$$
w^*=-\frac{S(G,\alpha)}{H+\lambda},
\qquad
S(G,\alpha)=\operatorname{sign}(G)\max(|G|-\alpha,0).
$$

L2 regularization enlarges the denominator; L1 can set an update to zero. Neither is an identity-specific penalty. A theoretical split compares regularized child and parent objectives and pays a complexity cost. Actual split selection also depends on histogram candidates, child-support constraints, sampling, and native implementation conventions; exported gain is the native score, not a recomputation of this equation.

For $G=-8,H=4,\lambda=2,\alpha=0$, $w^*=8/6=1.3333$. With $\eta=0.1$, the contribution is about $0.1333$ log-odds. Starting from $p=0.2$ (margin $-1.3863$), the new probability is approximately $\sigma(-1.2530)=0.2222$, not $0.3333$.

### 4.3 Why `min_child_weight` is not a row count

For unweighted binary logistic training, a candidate child needs enough Hessian mass:

$$
H_{\mathrm{child}}=\sum_{i\in\mathrm{child}}p_i(1-p_i)
\ge \texttt{min\_child\_weight}.
$$

If probabilities in the child are approximately equal to $p$:

$$
n_{\mathrm{child}}\gtrsim
\frac{\texttt{min\_child\_weight}}{p(1-p)}.
$$

For a threshold of 56:

| Approximate current $p$ | Hessian per row | Approximate minimum rows |
|---|---:|---:|
| 0.50 | 0.2500 | 224 |
| 0.20 | 0.1600 | 350 |
| 0.03 | 0.0291 | 1,925 |
| 0.01 | 0.0099 | 5,657 |

This is an approximation for similarly predicted, unweighted rows. Hessians change over boosting rounds. Bootstrap multiplicity weights scale gradients and Hessians, so a raw row count is even less informative for those refits.

### 4.4 Exposed parameter meanings

| Parameter | Role |
|---|---|
| `eta` | Shrinks each tree's update |
| `max_depth` | Caps tree depth |
| `min_child_weight` | Requires child Hessian support |
| `lambda`, `alpha` | L2/L1 leaf regularization |
| `gamma` | Requires split improvement |
| `subsample` | Samples observations per tree |
| `colsample_bytree` | Samples features per tree |
| `colsample_bylevel` | Samples features per depth level |
| `colsample_bynode` | Samples features per split |
| `max_delta_step` | Bounds leaf-update magnitude when positive |
| `max_bin` | Controls histogram resolution |
| `max_cat_to_onehot` | Controls one-hot versus partition categorical splitting |
| `max_cat_threshold` | Limits categorical split search |

`nthread` is passed to the booster and matrices. `base_score` is not exposed or tuned; this implementation leaves intercept initialization to native XGBoost for each fit. Do not initialize it using future/test labels. See the [native parameter reference](https://xgboost.readthedocs.io/en/stable/parameter.html).

The wrapper fixes `objective=binary:logistic`, `eval_metric=logloss`, `booster=gbtree`, and `tree_method=hist`. It does not expose `scale_pos_weight`, a custom objective, a GPU device, or a manual probability calibration fit. Native aliases such as `learning_rate` are not accepted by this config; use the supported name `eta`.

## 5. Optuna search and the robust objective

### 5.1 Search distributions

Each `search_space` entry maps to `suggest_int`, `suggest_float`, or `suggest_categorical`. `log: true` searches on a logarithmic scale and requires positive bounds; it cannot be combined with `step` in this wrapper. A supplied search space replaces the entire default search space. A fixed parameter in `tuning.params` cannot also be searched.

The sampler is seeded `TPESampler`. It starts with random trials and then uses observed objective values to guide proposals. The locked Optuna 4.9 sampler defaults to 10 startup trials, so the shipped 10-trial demo primarily checks the workflow rather than demonstrating a long adaptive search. Production defaults request 30 trials. [Optuna 4.9 TPE reference](https://optuna.readthedocs.io/en/v4.9.0/reference/samplers/generated/optuna.samplers.TPESampler.html).

Trials run sequentially (`n_jobs=1`), while each XGBoost fit may use several threads. The pipeline does not install a pruning callback or resume a previous study automatically. An optional timeout is checked between trials and is not a hard wall-clock deadline for the current fit or post-search reporting.

### 5.2 What is minimized?

For trial $\theta$, first obtain $K_\theta$ by inner early stopping, refit outer train, and predict validation. Let $L_\theta$ be the observed validation logloss and $L_{\theta,1}^*,\ldots,L_{\theta,B}^*$ be metric-bootstrap losses for those fixed predictions. Define:

$$
\overline L_\theta^*=\frac1B\sum_{b=1}^{B}L_{\theta,b}^*,
\qquad
s_\theta^*=\sqrt{\frac{1}{B-1}\sum_{b=1}^{B}
(L_{\theta,b}^*-\overline L_\theta^*)^2}.
$$

The minimized objective is:

$$
J(\theta)=L_\theta+c\,s_\theta^*,
\qquad c=\texttt{tuning.stability\_penalty}.
$$

The center is the original observed loss, **not** the bootstrap mean. The bootstrap mean is saved for inspection. With $c=0$, selection is ordinary validation logloss. With $c=1$, the criterion adds one estimated bootstrap standard deviation. This is not a formal upper confidence bound and does not claim any fixed coverage probability.

Example with $c=1$:

| Trial | Observed loss | Bootstrap SD | Objective |
|---|---:|---:|---:|
| A | 0.1500 | 0.0040 | 0.1540 |
| B | 0.1510 | 0.0010 | 0.1520 |

B wins despite a slightly higher observed loss. It is less sensitive to the configured resampling units in this validation period. This does not establish that its parameters or predictions are stable under a different population.

### 5.3 Comparable resamples and trial records

Each candidate reuses the same resampling seed and unit ordering. This reduces noise in comparisons; it does not remove estimation uncertainty. Trial user attributes include selected rounds, validation logloss, bootstrap mean, and bootstrap SD. They are stored in `study.sqlite3` and `trials.csv`.

Optimization is overall row-weighted validation logloss with this penalty. It does not minimize NE, z-score, worst-group click error, or a publisher-balanced objective. Group metrics are diagnostic outputs.

## 6. Bootstrap uncertainty and training sensitivity

### 6.1 Resampling units

| Method | Unit | Dependence retained within each draw |
|---|---|---|
| `row` | One observation | None beyond the row |
| `cluster` | All observations with a shared `cluster_column` | Within-cluster dependence |
| `time_block` | All observations whose UTC timestamps fall in a fixed interval | Within-block temporal dependence |

`time_block` uses `timestamp.floor(time_frequency)`, with fixed durations such as `1D` or `7D`. Seven-day floor bins should not be assumed to start on a particular weekday. Calendar months are not fixed-duration blocks. The blocks are non-overlapping; this is not a moving-block, stationary, or two-way cluster bootstrap. Absent/empty time blocks are not inserted.

Choose a unit based on how observations share outcomes or conditions. Multiple impressions within a search may call for `cluster_column: searchid`; shared daily shocks may call for day blocks. Neither option automatically handles both cluster and temporal dependence simultaneously. Row resampling can understate uncertainty when rows are dependent.

### 6.2 Multinomial resampling

For $U$ units, each draw samples $U$ units with replacement, equivalently:

$$
(W_1^{(b)},\ldots,W_U^{(b)})
\sim \operatorname{Multinomial}\left(U;\frac1U,\ldots,\frac1U\right).
$$

All rows in unit $u$ receive the same multiplicity $W_u^{(b)}$. Unit count stays fixed but total row count can vary when units have different sizes.

For example, two units contain 100 and 300 rows. Their possible multinomial counts are $(2,0)$, $(1,1)$, and $(0,2)$, producing 200, 400, or 600 resampled rows. A loss average must divide by the resampled row count, not by the original 400.

### 6.3 Sufficient statistics

For fixed predictions and baseline $q$, the per-row statistics are:

$$
s_i=\left(1,\ y_i,\ p_i,\ p_i(1-p_i),\ \ell_i,\ (y_i-p_i)^2,\ \ell(y_i,q)\right).
$$

Aggregate within units, $S_u=\sum_{i\in u}s_i$, then form each draw's totals:

$$
S^{*(b)}=\sum_{u=1}^{U}W_u^{(b)}S_u.
$$

The metrics in Section 7 can be computed from these totals. The implementation avoids allocating $B$ resampled DataFrames or a $B\times n$ row-index matrix. It still stores per-row statistics initially; row bootstrap can therefore remain expensive on very large datasets. Trial selection only retains count/loss totals per unit for the bootstrap objective.

ROC AUC and average precision require ordering information, so this implementation computes their point estimates but does **not** give them bootstrap intervals.

### 6.4 Percentile intervals and undefined resamples

For confidence level $C$, let $a=(1-C)/2$. For metric $M$ the interval is:

$$
\left[Q_a(M^{*(1)},\ldots,M^{*(B)}),\;
Q_{1-a}(M^{*(1)},\ldots,M^{*(B)})\right].
$$

At $C=0.95$, these are the 2.5th and 97.5th percentiles. The code uses pandas quantiles with their default interpolation, and sample standard deviation (`ddof=1`).

Undefined/nonfinite draws are removed separately for each metric. The saved `valid_resamples` count identifies the remaining sample. At least two valid draws are required for bounds and SD. If some draws are removed, the resulting interval is conditional on the metric being defined and should be treated cautiously, particularly for sparse groups.

With fewer than two units, all intervals are undefined with `insufficient_units`. Train and validation require at least two units before tuning; test can have fewer, and its intervals then indicate the limitation. The report warns when the overall test has fewer than 20 units. This warning is not a guarantee that 20 or more units are sufficient.

The exported bootstrap metrics are logloss, NE, fitting-baseline NE, Brier, z-score, and signed click percentage error. Confidence level affects interval bounds, not the objective penalty or training-replica count.

### 6.5 Separate training-bootstrap refits

After selecting parameters and fitting the final model, `training_repeats` controls additional train+validation bootstrap fits. A row in unit $u$ receives weight $W_u^{(b)}$ in the native training matrix. The fitting loss becomes:

$$
\mathcal L_b(F)=\sum_i W_{u(i)}^{(b)}\ell(y_i,\sigma(F(x_i)))
+\text{tree regularization}.
$$

These refits hold hyperparameters, round count, encoding, and model RNG seed fixed. Their input weights change. The native fit can re-estimate its intercept using its weighted fitting population. One-class training resamples are skipped and counted.

Each valid replica predicts the unchanged test period. Besides point metrics, it reports:

$$
\operatorname{mean\_abs\_prediction\_change}_b
=\frac1{n_{\mathrm{test}}}\sum_i|p_i^{(b)}-p_i^{\mathrm{final}}|.
$$

The replica baseline rate for its fitting-baseline NE uses the weighted fitting mean. The saved `model.ubj` remains the original unresampled final model; no test score selects among replicas.

Weighted multiplicities avoid physical duplication, but histogram construction and XGBoost's internal subsampling mean the training algorithm need not be bit-for-bit identical to training on duplicated rows. This is a weighted bootstrap sensitivity diagnostic, not repeated hyperparameter tuning. Five default repeats are too few to characterize tail uncertainty reliably; the tool exports individual results without claiming a training-uncertainty confidence interval.

### 6.6 What the intervals do not cover

Fixed-model metric bootstrap conditions on learned predictions and the observed period. Training replicas measure sensitivity at fixed selected parameters. Neither includes the entire search/feature-selection process, new categories, future drift, label-definition changes, or dependence extending beyond the sampling units. More replicates reduce Monte Carlo noise, not structural bias from the wrong unit definition.

## 7. Evaluation metrics and worked calculations

### 7.1 Logloss — `logloss`

$$
L=-\frac1n\sum_i\left[y_i\ln p_i+(1-y_i)\ln(1-p_i)\right].
$$

Natural logarithms give units of nats per observation. Lower is better. Logloss evaluates predicted probabilities rather than labels at a fixed threshold and penalizes confident mistakes strongly.

The implementation clips probabilities to $[10^{-15},1-10^{-15}]$ **only for logarithms**. Original probabilities determine expected-click totals, z-score variance, and Brier loss. Probabilities outside $[0,1]$ or nonfinite probabilities are rejected.

### 7.2 Normalized entropy — `ne`

Define the binary entropy of the observed evaluation rate:

$$
H(r)=-r\ln r-(1-r)\ln(1-r),
\qquad\operatorname{NE}=\frac{L}{H(r)}.
$$

This denominator is the logloss of a constant predictor using the observed rate of the same evaluation subset. NE below 1 improves on that retrospective constant-rate reference. A group uses its own rate; two groups can have different NE denominators even when their logloss values match.

When $r=0$ or $r=1$, entropy is zero and NE is undefined. The code reports `ne_status=undefined_single_class`; it does not replace NE with zero or infinity.

NE is also not a causal fraction of “uncertainty explained.” It is a loss ratio. An NE of 0.90 means a 10% reduction relative to the specified entropy denominator, not 90% accuracy or 10% more clicks.

### 7.3 Fitting-rate baseline — `ne_train_baseline`

Let $q$ be the positive rate estimated from the fitting data, clipped for logs. Define:

$$
L_{\mathrm{baseline}}(q)
=-r\ln q-(1-r)\ln(1-q),
\qquad
\operatorname{NE}_{\mathrm{train}}=\frac{L}{L_{\mathrm{baseline}}(q)}.
$$

For validation, $q$ comes from outer train. For the final held-out test, it comes from train+validation. Every group uses that same global fitting baseline, not a separately fitted group rate. Metric bootstrap holds $q$ fixed; the evaluation subset changes.

This baseline is available before evaluating test. In contrast, the entropy denominator in ordinary NE uses the evaluation rate after labels are observed. Their relationship is:

$$
L_{\mathrm{baseline}}(q)=H(r)+
D_{\mathrm{KL}}(\operatorname{Bern}(r)\parallel\operatorname{Bern}(q)).
$$

Thus the fitting-baseline denominator is at least the entropy denominator. Drift between $q$ and $r$ can explain why the two NE values differ. The report also includes a separate test row for the constant fitting-rate predictor.

### 7.4 Expected clicks and percentage error

$$
A=\sum_i y_i,\qquad E=\sum_i p_i,
\qquad \Delta=E-A,
$$

$$
\operatorname{click\_error\_pct}=100\frac{E-A}{A},
\qquad
\operatorname{abs\_click\_error\_pct}
=\left|100\frac{E-A}{A}\right|.
$$

Positive signed error means **overprediction**; negative means underprediction. $E$ can be fractional. No 0.5 classification threshold is applied.

If $A=0$, percentage error is undefined even if $E=0$. Counts and raw error remain informative. Status is `undefined_zero_actual_clicks`. For sparse groups, one extra actual click can substantially change this ratio, so review support and its valid bootstrap draws.

Expected totals require representative population probabilities. Downsampling negatives or balancing classes changes the modeled rate unless explicitly corrected; no such population correction is implemented here.

### 7.5 z-score — `z_score`

Under independent Bernoulli outcomes with success probabilities $p_i$:

$$
\mathbb E[A]=E,\qquad
V=\operatorname{Var}(A)=\sum_i p_i(1-p_i),
$$

$$
z=\frac{A-E}{\sqrt V}.
$$

Positive z means **underprediction**, the opposite sign convention from signed click error. If $V=0$, z is undefined with `undefined_zero_variance`; otherwise its status explicitly says `iid_bernoulli_approximation`.

The denominator reflects an independence model, not an empirical cluster-adjusted standard error. Repeated sessions and common time shocks can invalidate it. Its normal approximation can also be poor for tiny expected counts. A bootstrap interval for this statistic does not turn it into a calibrated cluster-corrected hypothesis test. The tool emits no p-values, significance declarations, or multiple-testing corrections for group z-scores.

### 7.6 Brier — `brier`

$$
\operatorname{Brier}=\frac1n\sum_i(y_i-p_i)^2.
$$

Lower is better. This is a proper probability scoring rule with less extreme sensitivity to confident mistakes than logloss. In a rare-event dataset, a small raw Brier number can partly reflect low prevalence, so compare against a relevant baseline and inspect calibration as well.

### 7.7 ROC AUC and average precision

For a randomly selected positive-negative pair:

$$
\operatorname{AUC}=P(p_+>p_-)+\tfrac12P(p_+=p_-).
$$

AUC measures ranking, not agreement of expected and actual counts. Average precision is:

$$
\operatorname{AP}=\sum_k(R_k-R_{k-1})P_k,
$$

where $R_k$ and $P_k$ are recall and precision at successive score thresholds. It is not trapezoidal PR-curve area. The implementation delegates these point metrics to sklearn metric functions, without using an sklearn model API. Both outputs are undefined for single-class subsets under this tool's policy.

### 7.8 Four-row worked example

Take baseline $q=0.5$:

| Row | $y$ | $p$ | Correct-outcome probability | $p(1-p)$ | $(y-p)^2$ |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 0.8 | 0.8 | 0.16 | 0.04 |
| 2 | 0 | 0.1 | 0.9 | 0.09 | 0.01 |
| 3 | 1 | 0.7 | 0.7 | 0.21 | 0.09 |
| 4 | 0 | 0.2 | 0.8 | 0.16 | 0.04 |

Then $n=4$, $A=2$, $E=1.8$, $r=0.5$, and $V=0.62$.

$$
L=-\frac{\ln(0.8)+\ln(0.9)+\ln(0.7)+\ln(0.8)}4
\approx0.227081.
$$

$$
H(0.5)=\ln2\approx0.693147,
\qquad \operatorname{NE}=\operatorname{NE}_{\mathrm{train}}
\approx0.327608.
$$

$$
\operatorname{click\_error\_pct}=100(1.8-2)/2=-10\%,
\quad z=(2-1.8)/\sqrt{0.62}\approx0.254000,
$$

$$
\operatorname{Brier}=(0.04+0.01+0.09+0.04)/4=0.045.
$$

Both positives rank above both negatives, so AUC and AP equal 1. Nevertheless expected clicks are 10% low. Perfect ranking does not imply correct probability calibration.

### 7.9 Undefined-value policy

| Condition | Affected result |
|---|---|
| All labels identical | NE, ROC AUC, AP undefined |
| No actual clicks | Signed/absolute click percentage error undefined |
| Zero sum of Bernoulli variances | z-score undefined |
| Fewer than two bootstrap units | All metric intervals undefined |
| Fewer than two valid draws for a metric | That metric's interval and SD undefined |

JSON uses `null`, CSV uses empty cells, and available status columns state the cause. Undefined values should not be silently treated as zero when aggregating downstream.

## 8. Group evaluation and support

### 8.1 Joint versus marginal groups

A flat list specifies one joint grouping:

```yaml
group_by: [publisher.accountid, service_code]
```

For both marginals and their joint cells:

```yaml
group_by:
  - [publisher.accountid]
  - [service_code]
  - [publisher.accountid, service_code]
```

These names match the shared demo. Replace them with your own schema names for custom data. Missing group values are retained; only observed combinations are evaluated. Group columns are model inputs only when also listed under `features`.

### 8.2 Per-group calculations and aggregation

Every observed group receives the metrics in Section 7 on its rows, plus bootstrap bounds/status fields for the six supported bootstrap metrics. For one grouping, counts and expected clicks conserve totals:

$$
\sum_g A_g=A,\qquad\sum_g E_g=E,\qquad\sum_g n_g=n.
$$

Overall loss is the row-weighted mean of group losses:

$$
L=\sum_g\frac{n_g}{n}L_g.
$$

Do not average group NE or signed percentage errors to obtain overall NE/error. Ratios have different denominators. Opposing group errors can also cancel in the overall signed error.

### 8.3 Low support

A group is flagged if:

$$
\texttt{low\_support}
=(n_g<\texttt{min\_group\_rows})\lor
(A_g<\texttt{min\_group\_clicks}).
$$

Defaults are 100 rows and 10 clicks. This does not remove rows, omit a group, or alter the search objective. A group with many rows but only one time block can still lack a usable interval.

Bootstrap group intervals resample the units present within each group independently of the other group calculations. They are marginal intervals, not a joint confidence region or simultaneous family-wide guarantee. The HTML shows the first 100 groups sorted by row count; each CSV includes every group. Click-comparison plots show at most the 30 largest groups.

## 9. Gain and SHAP interpretation

### 9.1 Native split importance

Let $\mathcal S_j$ be the splits using feature $j$, with native gain $d_s$ and cover $c_s$. The exported metrics summarize these native values:

$$
\operatorname{weight}_j=|\mathcal S_j|,
\quad \operatorname{total\_gain}_j=\sum_{s\in\mathcal S_j}d_s,
\quad \operatorname{gain}_j=\frac{\operatorname{total\_gain}_j}{|\mathcal S_j|},
$$

$$
\operatorname{total\_cover}_j=\sum_{s\in\mathcal S_j}c_s,
\quad \operatorname{cover}_j=\frac{\operatorname{total\_cover}_j}{|\mathcal S_j|}.
$$

For tree models, cover reflects training Hessian mass rather than a unique-observation count. The tool takes native `get_score` results and fills missing feature entries with zero, ensuring unused configured features remain in the table. These are training-derived split measures. A feature can have high average gain from a few useful splits but low total gain. [Native importance API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.get_score).

### 9.2 Native TreeSHAP and additivity

For a sampled row $x_i$, the native contribution output has one value per feature and a bias term:

$$
F(x_i)=\phi_{i0}+\sum_{j=1}^{d}\phi_{ij},
\qquad
p_i=\sigma\left(\phi_{i0}+\sum_j\phi_{ij}\right).
$$

The pipeline calls `Booster.predict(pred_contribs=True)` and verifies this sum against `output_margin=True` with absolute and relative tolerances of $10^{-4}$. Contributions are on the **log-odds** scale for binary logistic prediction. The SHAP bias is a contribution baseline and need not equal the configured/estimated initial intercept. [Native prediction API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.predict).

The sample is uniform without replacement from test rows, seeded for reproducibility and capped by `report.shap_rows`. Raw contributions, corresponding original feature values, and row/prediction context are exported in matching order. SHAP exports explain the final unresampled refit, not bootstrap replica models.

### 9.3 A local explanation

Suppose a row has bias $-3.5$ and three contributions $0.4,-0.2,0.1$:

$$
F=-3.5+0.4-0.2+0.1=-3.2,
\qquad p=\sigma(-3.2)\approx0.03917.
$$

The bias probability is $\sigma(-3.5)\approx0.02931$. The net change is about **0.985 percentage points**, not 30 percentage points. Feature contributions add in margin space; probability changes are nonlinear and depend on the starting margin.

The saved rows can support a local waterfall explanation, but the current report implements a SHAP distribution plot and global importance bars; it does not automatically draw a per-row waterfall or SHAP interaction matrix.

### 9.4 Global importance

For a test sample of size $m$:

$$
\operatorname{mean\_abs\_shap}_j=\frac1m\sum_i|\phi_{ij}|,
\qquad
\operatorname{mean\_shap}_j=\frac1m\sum_i\phi_{ij}.
$$

Mean absolute SHAP measures average modeled impact in this sample; signed means can cancel. Neither yields a universal monotonic direction. Contributions can vary with context. The distribution plot uses color to distinguish feature rows, not to encode high/low feature values.

Gain and SHAP answer different questions: gain summarizes fitted splits; SHAP summarizes contributions on the sampled rows. Correlated features and identity proxies can substitute for one another and redistribute importance. Neither proves causality, confirms pre-prediction availability, or justifies dropping a feature based solely on this held-out test.

## 10. Reports and saved artifacts

### 10.1 Plot interpretation

| Plot/table | Population and calculation | How to read it |
|---|---|---|
| Optimization history | Validation objective per completed trial and best-so-far curve | Lower is better; improvements describe selection on validation |
| Learning curve | Inner fit/early-stop logloss for the selected candidate's parameters | Divergence can indicate overfitting; vertical marker is selected round count |
| Calibration curve | Up to 10 test prediction-quantile bins, dropping duplicate edges | Above the diagonal means underprediction in those bins |
| Daily click-rate curve | Actual and expected counts divided by daily rows on test | Highlights time-varying calibration gaps |
| Group click comparison | Actual versus expected clicks for up to 30 largest groups | Departures from the diagonal identify count mismatch; inspect group support |
| Gain / total gain bars | Native split importance of the final model | Compare average split utility with aggregate utility |
| Mean absolute SHAP bars | Uniform test sample | Larger values mean larger modeled margin contributions on that sample |
| SHAP distribution | Individual contributions in the test sample | Shows direction/spread; color distinguishes features |
| Training stability table | Bootstrap-refit predictions on unchanged test | Compares fixed-parameter training sensitivity |

If quantile binning produces no usable bins (for example, constant predictions), calibration falls back to a single all-row bin. The daily curve always uses `1D`; it is independent of the chosen bootstrap block duration. Calibration tables and curves do not fit or apply a probability calibrator.

### 10.2 Artifact reference

| Artifact | Contents |
|---|---|
| `report.html` | Offline report with embedded PNG images and links to companion files |
| `splits.csv` | Partition sizes, positives/rates, start/end timestamps |
| `metrics.csv` | Selected validation model, final held-out test model, and constant fitting-rate test baseline |
| `bootstrap_intervals.csv` | Six held-out test metric intervals, SDs, valid draws, units, and status |
| `groups_0.csv`, `groups_1.csv`, … | All observed groups in config order; metrics, low-support flag, and marginal intervals |
| `training_stability.csv` | Individual training-replica metrics and mean absolute prediction changes; skipped status where applicable |
| `trials.csv` | Optuna trials, parameters, objective, state, and user attributes |
| `study.sqlite3` | Persistent inspectable Optuna study for this run |
| `learning_curve.json` | Inner fit/early-stop losses for the selected parameters |
| `calibration.csv` | Prediction-quantile bin sizes and observed/predicted rates |
| `time_metrics.csv` | Daily rows and actual/expected test counts |
| `feature_importance.csv` | Native gain/cover/count metrics and sampled SHAP averages, with original/internal feature names |
| `shap_values.csv` | Feature contributions plus `__bias__`, one row per sampled test observation |
| `shap_features.csv` | Original feature values matching the SHAP rows |
| `shap_context.csv` | Matching sorted-test row positions, labels, probabilities, and raw margins |
| `model.ubj` | Final native XGBoost model fitted on unresampled train+validation |
| `encoder.json` | Feature order/types and final fitting vocabularies |
| `best_params.json` | Fixed plus selected native parameters and boosting-round count |
| `config.yaml` | Resolved config, including defaults and absolute output directory |
| `summary.json` | Selected trial, test metrics, data metadata, split metadata, versions, and run notes |
| `status.json` | `running`, `complete`, or `failed` after the run directory has been created |
| `test_predictions.parquet` (optional) | Test time, target, group columns, and `__predicted_probability__` |
| `*.png` | Separate copies of the plots embedded in HTML |

A unique `run_<UTC timestamp>_<suffix>` directory prevents overwriting earlier runs. Failures after directory creation can leave partial artifacts and a failed status; validation failures before creation do not produce a run directory. A model file alone does not mean the report completed.

`summary.json` records source path, file size, modification time, and row count, not a cryptographic content hash. Retain the input file/version separately for exact provenance. `study.sqlite3` is saved for inspection, but the CLI does not automatically resume it. Input/output data stay local during execution; dependency installation may require network access.

### 10.3 Reloading the model

Run from the repository environment:

```python
import pandas as pd
from model_tuning.predict import predict

frame = pd.read_parquet("future_rows.parquet")
p = predict(frame, "report/model_tuning/run_...", nthread=4)
print("Expected clicks:", p.sum())
```

Only configured feature columns are needed for this inference helper; target and split time are not required unless represented separately in the caller's workflow. The helper applies the saved vocabulary and names before native prediction. Do not reconstruct new categorical codes from future data and send them directly to the saved booster.

## 11. Complete configuration reference

### 11.1 Top-level keys

| Key | Default | Meaning / validation |
|---|---|---|
| `target` | Required | Exact binary target column |
| `time_column` | Required | Exact datetime/date/text timestamp column; UTC parsing |
| `features` | Required nonempty mapping | Explicit `numerical`/`categorical` model candidates |
| `group_by` | `[]` | Joint column list or list of column lists; no target grouping |
| `output_dir` | `../report/model_tuning` | Resolved relative to the YAML directory |
| `seed` | `42` | Integer in `[0, 2**32-1]`; model, search, and sampling reproducibility |
| `nthread` | `4` | Positive integer CPU threads for native fits/matrices |
| `split` | Defaults below | Chronological boundaries and identity isolation |
| `tuning` | Defaults below | Search settings and supported native parameters |
| `bootstrap` | Defaults below | Resampling units, intervals, and training refits |
| `report` | Defaults below | Explanation sample and support/output controls |

Nested `split`, `tuning`, `bootstrap`, and `report` mappings merge with their field defaults. `tuning.params` and `tuning.search_space` are supplied as complete mappings; the search space is not merged parameter by parameter with its default.

### 11.2 Time controls

| `split` field | Default | Behavior |
|---|---|---|
| `train_fraction` | `0.6` | Fraction of distinct timestamps in outer training |
| `validation_fraction` | `0.2` | Additional fraction of timestamps in validation |
| `early_stopping_fraction` | `0.15` | Tail fraction of training timestamps for inner early stopping |
| `train_end` | `null` | Optional inclusive UTC cutoff |
| `validation_end` | `null` | Optional inclusive UTC cutoff; both cutoff fields must be supplied together |
| `gap` | `0D` | Nonnegative duration omitted after each preceding cutoff |
| `entity_column` | `null` | Optional identifier whose overlap/missingness makes splitting fail |

All fractions must be in `[0.001, 0.999]`, and train plus validation must be below 1. The parser validates those fraction settings even when explicit cutoffs are supplied. Explicit cutoffs override their use for outer boundary calculation; the inner fraction still applies.

### 11.3 Search controls

| `tuning` field | Default | Meaning |
|---|---|---|
| `n_trials` | `30` | Positive integer number of requested trials |
| `timeout_seconds` | `null` | Optional finite search timeout of at least 1 second |
| `num_boost_round` | `1000` | Positive integer maximum inner training rounds |
| `early_stopping_rounds` | `50` | Positive integer patience on inner early-stop loss |
| `stability_penalty` | `1.0` | Finite nonnegative multiplier on bootstrap loss SD |
| `params` | `{}` | Fixed values for supported native parameters |
| `search_space` | Table below | Parameter distributions, replacing defaults if supplied |

Default distributions:

| Parameter | Type | Bounds | Sampling scale |
|---|---|---|---|
| `eta` | float | 0.02–0.2 | log |
| `max_depth` | int | 3–8 | linear |
| `min_child_weight` | float | 1–100 | log |
| `subsample` | float | 0.6–1 | linear |
| `colsample_bytree` | float | 0.6–1 | linear |
| `lambda` | float | 0.01–100 | log |
| `alpha` | float | 1e-8–10 | log |

`max_depth`, `max_bin`, `max_cat_to_onehot`, and `max_cat_threshold` must be integers. The wrapper requires `max_bin >= 2` and the other integer parameters at least 1; it does not expose native unlimited depth via `max_depth=0`. `eta`, `subsample`, and the three column-sampling controls must be in `(0,1]`. Other supported values are finite and nonnegative.

Example replacing the search space and fixing subsampling:

```yaml
tuning:
  n_trials: 40
  params: {subsample: 1.0}
  search_space:
    eta: {type: float, low: 0.03, high: 0.1, log: true}
    max_depth: {type: int, low: 3, high: 6}
    min_child_weight: {type: float, low: 30.0, high: 300.0, log: true}
    lambda: {type: float, low: 1.0, high: 100.0, log: true}
    colsample_bytree: {type: categorical, choices: [0.7, 0.85, 1.0]}
```

This example explores stronger leaf support than the default. It is a candidate search design, not a claim that those bounds are optimal. Unlisted native parameters use native defaults; because the example replaces the default space, `alpha` is no longer searched.

### 11.4 Bootstrap controls

| `bootstrap` field | Default | Meaning |
|---|---|---|
| `method` | `time_block` | `row`, `cluster`, or `time_block` |
| `time_frequency` | `1D` | Fixed-duration floor bins for time blocks |
| `cluster_column` | `null` | Required for cluster mode; IDs must be nonmissing |
| `n_resamples` | `200` | Integer at least 2; trial loss SD and metric intervals |
| `confidence` | `0.95` | Interval level in `[0.01, 0.999]` |
| `training_repeats` | `5` | Nonnegative integer additional fits; 0 disables training sensitivity refits |

`cluster_column` and `split.entity_column` are independent choices. One selects dependence units; the other enforces partition isolation. Neither may equal the target.

### 11.5 Reporting controls

| `report` field | Default | Meaning |
|---|---|---|
| `shap_rows` | `2000` | Positive integer cap on uniform test explanation sample |
| `plot_top_features` | `20` | Positive integer feature limit in importance/distribution plots |
| `min_group_rows` | `100` | Positive integer low-support threshold |
| `min_group_clicks` | `10` | Positive integer low-support threshold |
| `save_predictions` | `false` | Whether to save full test prediction Parquet |

These controls do not select training rows or change the objective. All modeled test rows receive overall/group evaluation; only SHAP is sampled.

### 11.6 Demo versus custom data

The shared [demo config](../demo/model_tuning.yaml) deliberately uses 10 trials, a 300-round ceiling, 30-round patience, 100 metric draws, and 3 training refits. Those are smaller than several application defaults. Both demo workflows read [the same committed synthetic Parquet](../demo/data.parquet).

The [custom-data template](../model_tuning/config.yaml) has a different feature/target schema and is not runnable against that demo unchanged. A config created for `data_mining` is also not directly accepted by `model_tuning`; copy the selected feature names/types into a tuning config and supply the time/search/evaluation settings.

## 12. Practical review sequence and limitations

1. Confirm that target and every candidate feature have the intended semantics and are available at the prediction time. An analysis report alone does not establish availability.
2. Check `splits.csv` for adequate positives, expected temporal coverage, and realistic embargoes. Fractions of distinct timestamps are not fractions of rows.
3. Check the best trial and its learning curve. A best round near the configured ceiling can indicate a constrained search budget, but any adjustment should be evaluated without reusing test for selection.
4. Compare held-out logloss/NE with the fitting-rate baseline, then inspect ranking metrics. AUC can remain good even when expected clicks drift.
5. Inspect daily and group calibration, signed error, zero-click groups, and low-support flags. Review bootstrap unit counts before trusting interval widths.
6. Compare the separate training-bootstrap results for sensitivity. Do not select the lowest-test-loss replica as the final model.
7. Use gain and SHAP to understand fitted behavior; validate any new feature decisions on separate pre-test data or a new future test period.
8. Keep the saved model, encoder, config, reports, and the exact dataset version together for reproducibility. A fixed random seed does not guarantee identical outputs across native library/hardware versions.

### 12.1 What is not implemented

- Rolling-origin cross-validation, a nested temporal search across multiple folds, or automated retraining schedules.
- Hyperparameter retuning inside each bootstrap replicate, simultaneous group intervals, a cluster-corrected z-test, or causal estimates.
- Negative-sampling correction, sample-weighted population evaluation, class balancing, or fitted probability calibration.
- Automatic identity/twin/leakage detection or automatic ingestion of the data-mining selected-feature artifact.
- GPU/external-memory/distributed training, streaming inference, or automatic study resume.
- Local waterfall plots, permutation importance, or SHAP interaction plots in the generated report.

The implementation is in-memory CPU training and can hold several partitions/matrices during search. Ten million rows may require substantial memory. Reduce the trial budget and tree ceiling for an initial workflow check; the code does not silently subsample training rows to fit a resource budget.

### 12.2 Recorded demo result

The earlier [validation record](../model_tuning/VALIDATION.json) documents a complete million-row synthetic run with 10 trials and 3 training refits. Its 202,060-row test period had logloss approximately 0.161523, NE 0.903474, ROC AUC 0.745666, and signed click error −10.577%. These are historical synthetic results, not measurements of the user's raw data and not a promise for a rerun on another environment.

That combination illustrates the intended report interpretation: ranking and normalized loss can improve while the model still underpredicts aggregate clicks. The artifact exposes those differences so the next data/model decision can be based on the intended probability-prediction use case.
