# Statistics and techniques: mathematics and interpretation

This guide explains every statistic and screening technique implemented in **Data Mining**: what it measures, its formula, why it is useful, what can go wrong, and how it affects feature decisions.

It describes the implementation at [commit e2238fb](https://github.com/edvgha/data-mining/tree/e2238fb214fed968bcf03f162fb31f39446d95b7), inspected on 2026-09-07. The implementation is the source of truth for calculation details:

- [audit.py](../data_mining/audit.py): validation, sampling, statistics, bucketing, and reports.
- [settings.py](../data_mining/settings.py): metric names and rule defaults.
- [decisions.py](../data_mining/decisions.py): decisions, precedence, and evidence.
- [config.yaml](../config.yaml): the supplied example configuration.

The project analyzes a supplied dataset with a binary target $Y\in\{0,1\}$. It does not train XGBoost, create train/test splits, encode model inputs, or measure performance on unseen data. The name of the target is configurable. Positive always means **label 1**, regardless of the business application.

## Contents

1. [Notation and calculation populations](#1-notation-and-calculation-populations)
2. [Data validation and quality](#2-data-validation-and-quality)
3. [Numerical univariate statistics](#3-numerical-univariate-statistics)
4. [Categorical univariate statistics](#4-categorical-univariate-statistics)
5. [Analysis buckets](#5-analysis-buckets)
6. [Feature versus target](#6-feature-versus-target)
7. [Feature pairs](#7-feature-pairs)
8. [Joint and conditional analysis](#8-joint-and-conditional-analysis)
9. [Temporal and group context](#9-temporal-and-group-context)
10. [Feature decisions](#10-feature-decisions)
11. [Configuration and output reference](#11-configuration-and-output-reference)
12. [Worked interpretation examples](#12-worked-interpretation-examples)
13. [Limits and a practical review sequence](#13-limits-and-a-practical-review-sequence)

## 1. Notation and calculation populations

| Symbol | Meaning |
|---|---|
| $N$ | Number of rows in the supplied dataset |
| $S$ | Fixed sample of rows used for more expensive calculations |
| $m$ | Sample size: $\min(N,\texttt{sample\_rows})$ |
| $X$ or $Z$ | A candidate feature |
| $Y$ | Binary target; 1 is positive and 0 is negative |
| $n$ | Number of usable observations in the calculation under discussion |
| $n_1,n_0$ | Usable positive and negative counts for a numerical class comparison |
| $\bar{x}_1,\bar{x}_0$ | Mean numerical value within each target class |
| $s_1^2,s_0^2$ | Within-class sample variances, with denominator $n_y-1$ |
| $B(X)$ | Discrete analysis bucket assigned to a feature value |
| $K$ | Number of distinct observed values or categories, as specified |
| $\mathbf{1}\{A\}$ | Indicator: 1 when condition $A$ is true, otherwise 0 |
| $\ln$ | Natural logarithm; information is measured in nats |

**Always check the denominator.** A missing fraction uses all $N$ rows; a numerical mean uses finite values; a numerical AUC uses only finite values in the fixed sample. These populations can differ substantially.

### 1.1 Full data versus sampled data

| Calculation | Rows used | Treatment of missing values |
|---|---|---|
| Numerical univariate summaries | All finite feature values in the full data | Nulls and infinities excluded from numerical summaries |
| Categorical univariate summaries | All observed categories in the full data | Nulls excluded from category probabilities; missingness reported separately |
| Bucket positive rates, lift, and Wilson intervals | Full data | Explicit missing bucket |
| Feature-target MI, corrected Cramér's V, chi-square | Full-data bucket-by-target table | Missing bucket included |
| Cohen's d, Hedges' g, AUC, KS, numerical target correlations | Fixed sample, then finite values for that feature | Nonfinite numerical rows omitted |
| Numerical pair correlations | Fixed sample, then rows finite in both features | Pairwise complete observations |
| Categorical pair association | Fixed sample of pooled bucket codes | Missing is a category |
| Mixed categorical-numerical eta squared | Fixed sample with finite numerical values | Categorical missing bucket retained |
| Missingness phi | Entire fixed sample | Measures missing/nonfinite indicators |
| Joint positive-rate tables | Full-data bucket tuples | Missing buckets included |
| Joint and conditional MI | Fixed sample of bucket tuples | Missing buckets included |
| Time and group summaries | Full data | Missing time/group has a separate group |

The sample is drawn uniformly **without replacement** once, using `numpy.random.default_rng(seed)`. All sampled calculations share its row indices. It is not stratified or class-balanced.

If the full data contains $N_1$ positives and $\pi=N_1/N$, the sample positive count $M_1$ follows a hypergeometric distribution:

$$
E[M_1]=m\pi,\qquad
\operatorname{Var}(M_1)=m\pi(1-\pi)\frac{N-m}{N-1}.
$$

Thus class balance is preserved in expectation, not exactly. A very rare class can be absent from the sample or from a feature's finite subset, even when both classes exist in the full data. The resulting undefined statistics are reported, not replaced with zero.

Bucketing uses the full dataset before sampling. Sampled MI therefore uses sample frequencies with full-data bucket definitions.

## 2. Data validation and quality

### 2.1 Schema and label validation

The input must be an existing local `.parquet` file. The engine checks the Parquet schema and reads the required columns into one pandas DataFrame.

It rejects duplicate column names, missing configured columns, empty data, null/nonbinary labels, and datasets containing only one target class. With `strict_schema: true`, every source column must be declared as a feature, target, context column, or ignored column.

Features have semantic type `numerical` or `categorical`. A numeric account ID can be categorical: its integer representation does not imply a meaningful distance or order. Numerical features require real numeric dtypes; categorical values must be supported scalar strings, numbers, or booleans.

The target, ignored columns, and optional time/group context are kept separate from candidate features. Duplicate YAML keys, unknown options, invalid thresholds, and unknown statistic names are rejected.

These checks establish that the input matches the requested analysis. They cannot establish that a feature existed before prediction or that the records represent the intended population.

### 2.2 Missingness, infinity, and unusable values

For a feature:

$$
\text{missing fraction}
=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\{X_i\text{ is null}\}.
$$

For numerical features:

$$
\text{unusable fraction}
=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\{X_i\text{ is not finite}\}.
$$

Numerical unusable values include nulls, NaN, $+\infty$, and $-\infty$. The report separately counts `nonfinite_nonmissing_count`, which ordinarily counts infinities. Decision rules use `unusable_fraction` for numerical features and `missing_fraction` for categorical features.

Categorical missingness follows pandas null detection. Numeric infinity declared as a category is an observed category, not automatically a missing value.

**Intuition:** availability itself can distinguish classes. A feature observed as 5 on half the rows and missing on the rest is not equivalent to a feature equal to 5 everywhere. The bucket analysis can reveal missingness signal even when finite-value AUC cannot.

### 2.3 Uniqueness and dominance

Let $n$ be the usable observed count and $n_k$ the count of distinct value $k$:

$$
K=\#\{k:n_k>0\},\qquad
\text{dominant fraction}=\max_k\frac{n_k}{n}.
$$

`unique_nonmissing` is $K$; for numerical columns it specifically counts distinct **finite** values. Categorical `unique_fraction_nonmissing` is $K/n$.

- $K=0$: no usable value.
- $K=1$ and no unusable rows: completely constant.
- Dominance near 1: almost all observed values are the same.
- Categorical $K/n$ near 1: possible identifier or very sparse category space.

These are structural diagnostics, not target-association statistics. A rare value can still mark an important positive subgroup.

### 2.4 Descriptive flags versus decision rules

Profile `flags` describe all-missing, constant, near-constant, high-missingness, infinity, high-cardinality, and heavily pooled columns. Two additional flag cutoffs are fixed in the implementation: categorical uniqueness fraction $\ge0.95$, and pooled row fraction $\ge0.5$.

Flags are generated even if a similarly named decision rule is disabled. Rule enablement controls decisions and rule logs, not whether a descriptive flag appears. A flag is not, by itself, an exclusion.

## 3. Numerical univariate statistics

These statistics use all finite observations of one numerical feature. They characterize its distribution without looking at the target.

### 3.1 Mean — `mean`

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i.
$$

The mean is the distribution's balance point. It minimizes $\sum_i(x_i-a)^2$ over a constant $a$. Large observations have large influence.

For $[1,2,3,4,100]$, the mean is 22. It is useful for scale and unit checks but does not describe a typical row well in this example. A mean by itself says nothing about predictive value.

### 3.2 Sample standard deviation — `std`

$$
s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2}.
$$

The code uses pandas' sample standard deviation, with `ddof=1`. Dividing the squared deviations by $n-1$ corrects the usual variance estimator for estimating the mean from the same data. The square root is not itself an exactly unbiased estimator of population standard deviation.

Standard deviation measures spread in the original units. It is sensitive to tails. It is undefined for fewer than two finite observations; a repeated finite constant has standard deviation zero.

### 3.3 Median and quantiles — `median`, `quantiles`

The median is $Q(0.5)$ and minimizes the sum of absolute deviations from a constant.

For sorted values $x_{(1)}\le\cdots\le x_{(n)}$, pandas' default linear quantile interpolation can be written as:

$$
h=(n-1)p,\quad j=\lfloor h\rfloor,\quad \gamma=h-j,
$$

$$
Q(p)=(1-\gamma)x_{(j+1)}+\gamma x_{(j+2)}.
$$

At $p=1$, use the last value directly. With a single finite observation, every quantile is that value.

The report provides `min`, `p01`, `p05`, `p25`, `p50`, `p75`, `p95`, `p99`, and `max`. The separately switchable `median` equals `p50` when both are emitted.

**Intuition:** quantiles answer “what value separates the bottom $p$ fraction of observations?” They reveal scale, tails, and concentrations that a mean hides. In $[1,2,3,4,100]$, the median is 3.

### 3.4 Interquartile range — `iqr`

$$
\operatorname{IQR}=Q(0.75)-Q(0.25).
$$

IQR measures the width of the middle half of the observations. It is less affected by extreme tails than standard deviation. In the running example, $Q(0.25)=2$, $Q(0.75)=4$, so IQR is 2.

An IQR of zero can coexist with meaningful rare values: it only says the middle half has no measured width under this quantile convention.

### 3.5 Median absolute deviation — `mad`

$$
\operatorname{MAD}
=\operatorname{median}_i\left|x_i-\operatorname{median}_j(x_j)\right|.
$$

The output is `mad_unscaled`. The code does **not** multiply by a normal-distribution scale factor.

MAD describes a typical absolute distance from the median. For $[1,2,3,4,100]$, the absolute deviations are $[2,1,0,1,97]$, so MAD is 1. Many repeated values can make MAD zero even when tails exist.

### 3.6 Skewness — `skewness`

Define the central moments with denominator $n$:

$$
m_r=\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^r.
$$

For $n>2$ and nonzero variance, the adjusted Fisher–Pearson statistic is:

$$
G_1=\frac{\sqrt{n(n-1)}}{n-2}\frac{m_3}{m_2^{3/2}}.
$$

Cubic deviations preserve sign and emphasize tails. Positive skewness often indicates an influential right tail; negative skewness often indicates a left tail. A value near zero does not establish symmetry or normality.

The project calls pandas `Series.skew()` and reports undefined values for constant features or inadequate support. Large skewness prompts inspection of units, sentinels, and tails; it does not automatically exclude a feature. See [pandas skewness](https://pandas.pydata.org/docs/reference/api/pandas.Series.skew.html) and the [adjusted coefficient formula](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html).

### 3.7 Excess kurtosis — `kurtosis`

Let $g_2=m_4/m_2^2-3$. The bias-adjusted excess kurtosis used by pandas is:

$$
G_2=\frac{n-1}{(n-2)(n-3)}\left[(n+1)g_2+6\right],
\qquad n>3.
$$

The output is `excess_kurtosis`. Subtracting 3 puts a normal population at zero excess kurtosis. Fourth powers emphasize extreme observations. Positive values often indicate heavier tails; negative values often indicate lighter tails relative to a normal distribution.

Kurtosis should not be read simply as “peak height.” It is unstable with little data or extreme observations. The code reports it as undefined for constant features or insufficient observations. See [pandas kurtosis](https://pandas.pydata.org/docs/reference/api/pandas.Series.kurt.html).

### 3.8 Zero and negative fractions — `zero_fraction`, `negative_fraction`

$$
f_0=\frac{1}{n}\sum_i\mathbf{1}\{x_i=0\},
\qquad
f_-=\frac{1}{n}\sum_i\mathbf{1}\{x_i<0\}.
$$

Outputs are `zero_fraction_finite` and `negative_fraction_finite`. Their denominator is the finite count, not $N$.

Zeros may represent a real value, nonparticipation, or a missing-value sentinel. Negative values may be valid for balances but suspicious for physical counts. The software does not infer these meanings or replace the values.

### 3.9 IQR outlier fraction — `outlier_fraction`

Define Tukey-style fences:

$$
L=Q(0.25)-1.5\operatorname{IQR},\qquad
U=Q(0.75)+1.5\operatorname{IQR}.
$$

$$
f_{\mathrm{outlier}}
=\frac{1}{n}\sum_i\mathbf{1}\{x_i<L\text{ or }x_i>U\}.
$$

The output is `iqr_outlier_fraction`. Values exactly on a fence are not counted. For $[1,2,3,4,100]$, the fences are $-1$ and 7; one of five observations is outside, giving 0.20.

This is a descriptive convention, not proof of data corruption. When IQR is zero, even small departures from the repeated central value can be flagged. The audit neither clips nor removes outliers.

## 4. Categorical univariate statistics

These summaries use the original observed categories **before pooling**. Let $n=N-\text{missing_count}$, with category counts $n_1,\ldots,n_K$ and $p_k=n_k/n$.

### 4.1 Entropy and effective number of levels — `entropy`

$$
H(X)=-\sum_{k=1}^{K}p_k\ln p_k,\qquad
K_{\mathrm{effective}}=\exp(H(X)).
$$

The outputs are `entropy_nats_nonmissing` and `effective_levels_nonmissing`. Terms with probability zero contribute zero.

Entropy measures uncertainty about the category of a randomly selected observed row. It is zero for a single category and $\ln K$ for $K$ equally frequent categories. Effective levels translate entropy into the number of equally frequent categories with the same uncertainty.

For two equally frequent categories, $H=\ln2\approx0.6931$ and effective levels = 2. For proportions 0.99 and 0.01, $H\approx0.0560$ and effective levels $\approx1.0576$: almost all rows occupy one category.

High entropy describes variety, not relevance to $Y$. A random identifier can have very high entropy.

### 4.2 Rare categories and singletons — `rare_categories`

With $r=\texttt{rare\_min\_count}$:

$$
\text{singleton levels}=\sum_k\mathbf{1}\{n_k=1\},
$$

$$
\text{rare levels}=\sum_k\mathbf{1}\{n_k<r\},
\qquad
\text{rare row fraction}=\frac{\sum_{k:n_k<r}n_k}{N}.
$$

The strict inequality matters: a level with exactly $r$ observations is not rare. The rare-row denominator includes all rows, including missing rows.

Numerous rare levels indicate thin support for category-specific conclusions. The level count and row fraction answer different questions: thousands of rare categories can represent either a tiny tail or most of the dataset.

`pooled_row_fraction` measures rows mapped to OTHER after both frequency filtering and the top-category cap. It can exceed the rare-row fraction because frequent categories beyond `max_categories` are also pooled. `retained_levels` counts categories that remain individually represented.

## 5. Analysis buckets

Bucketing makes tables and discrete association statistics manageable. It changes the representation used for analysis; it does not create encoded training features or overwrite the source data.

### 5.1 Numerical buckets

If the number of distinct finite values is at most `bins`, each distinct value gets its own bucket, sorted numerically.

Otherwise, the engine uses `pandas.qcut` with `q=bins` and `duplicates="drop"`. Cut points are empirical quantiles. These aim for similar row counts, not similar numerical widths. Repeated values can collapse edges and reduce the number of buckets. See [pandas qcut](https://pandas.pydata.org/docs/reference/api/pandas.qcut.html).

Bucket code 0 represents missing or nonfinite numerical values. Other codes start at 1. Intervals include the right endpoint; the first also includes the minimum.

**Tradeoff:** more bins can reveal a U-shape or narrow threshold, but create sparse cells. Fewer bins improve support but can hide local effects. Large concentrations of ties can make requested bin counts misleading, so inspect actual counts.

### 5.2 Categorical buckets

The algorithm counts original nonmissing categories, sorts by decreasing frequency, retains categories whose counts meet `rare_min_count`, and keeps at most `max_categories` of them. Ties follow first-observed category order.

- Code 0: missing.
- Code 1: OTHER, pooling every nonretained observed category.
- Codes 2 onward: individually retained categories.

Internal codes distinguish an actual text value such as `"[missing]"` from a null. Labels for real categories receive a `value:` prefix. Category codes have no numerical rank meaning and are not used for numerical AUC or Cohen's d.

Pooling is chosen by frequency, without target labels. It can conceal category-specific effects, especially for IDs. Changing the cap or threshold can change MI, Cramér's V, and joint results even on the same rows.

### 5.3 Fixed definitions and support

The full-data buckets are reused for sampled pair/joint analysis and for every time period. Recomputing bins independently by period would change what a bucket means and could hide drift; the implementation keeps them fixed within a run.

`bucket_dictionary.csv` records the feature/code/label mapping. It may include special buckets that happen to contain no rows. Rate tables contain observed buckets only.

## 6. Feature versus target

A feature compared with the target is technically **bivariate** analysis, even when reported one feature at a time. Different statistics answer different questions; they are not interchangeable measures on a common scale.

### 6.1 Positive-class rate, lift, and support — `positive_rate`

For a bucket $b$ with $n_b$ rows and $k_b$ positives:

$$
k_b=\sum_{i:B(X_i)=b}Y_i,\qquad
\hat{p}_b=\frac{k_b}{n_b},\qquad
\hat{p}=\frac{\sum_{i=1}^{N}Y_i}{N},
$$

$$
\operatorname{lift}_b=\frac{\hat{p}_b}{\hat{p}}.
$$

The table contains `rows`, `positives`, `negatives`, `positive_rate`, `lift`, interval bounds, and `low_support`.

At a dataset baseline of 4%, a bucket with 80 positives in 1,000 rows has positive rate 8% and lift 2. The absolute difference is 4 percentage points; lift is a ratio, not a percentage-point difference.

Positive-rate profiles reveal threshold effects, U-shapes, and missingness effects that a global mean or rank statistic can miss. Numerical profiles should be read in bucket order. Unordered category labels do not imply a slope.

The support flag is:

$$
\text{low support}_b=
(n_b<\texttt{min\_cell\_count})
\lor(k_b<\texttt{min\_cell\_events})
\lor(n_b-k_b<\texttt{min\_cell\_events}).
$$

Defaults are 200 rows, 10 positives, and 10 negatives. Equality meets the minimum. Low-support cells remain in tables and calculations; they are not filtered or smoothed.

The same helper produces joint and temporal rate tables. Their lift also uses the **whole-dataset** baseline, including time-by-feature tables; it is not normalized to each period.

### 6.2 Wilson 95% interval — included with rate tables

For $k$ positives out of $n>0$ independent unweighted rows, let $\hat p=k/n$ and $z=1.959963984540054$. The interval is:

$$
\frac{\hat p+\frac{z^2}{2n}
\ \pm\ z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}
{1+\frac{z^2}{n}}.
$$

The implementation clips the endpoints to $[0,1]$ and names them `positive_rate_low_iid` and `positive_rate_high_iid`.

**Where it comes from:** invert the binomial score-test inequality

$$
\frac{(\hat p-p)^2}{p(1-p)/n}\le z^2
$$

and solve the resulting quadratic for plausible values of $p$. Unlike a simple symmetric normal interval around $\hat p$, Wilson behaves sensibly when $k=0$ or $k=n$. For 0 positives in 100 rows, its interval is approximately $[0,0.0369935]$.

The interval is a repeated-sampling coverage statement, not a 95% posterior probability that the fixed population parameter lies inside it. Repeated sessions or users can make these row-IID intervals too narrow. No cluster adjustment or simultaneous interval correction is implemented. See the [Wilson method in statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html).

### 6.3 Cohen's d — `cohens_d`

The implementation uses the positive-minus-negative orientation:

$$
s_p=\sqrt{\frac{(n_1-1)s_1^2+(n_0-1)s_0^2}{n_1+n_0-2}},
\qquad
d=\frac{\bar{x}_1-\bar{x}_0}{s_p}.
$$

**Intuition:** express the difference between class means in units of typical within-class spread. Pooling weights each class variance by its degrees of freedom.

- $d>0$: the feature mean is higher among positives.
- $d<0$: the feature mean is lower among positives.
- $d=0$: equal class means; distributions may still differ strongly.
- $|d|$ is unbounded; it is not a probability or a correlation.

Example: means 7 and 5 with pooled SD 4 give $d=0.5$.

The calculation uses finite values in the fixed sample and requires at least two in each class. Zero or nonfinite pooled SD makes d undefined. This includes a feature constant *within* each class but taking a different value across classes: perfect separation can coexist with undefined d. Inspect AUC and buckets instead.

The report includes `sample_rows_y1`, `sample_rows_y0`, `sample_mean_y1`, `sample_mean_y0`, and `cohens_d_status` when d or g is requested. Status is `computed`, `insufficient_class_rows`, or `zero_or_nonfinite_pooled_sd`.

The formula is descriptive even with unequal class variances. Interpreting it as one standardized population shift is less straightforward when variances or shapes differ. Outliers can dominate the means. The project does not compute a d p-value or confidence interval and does not apply universal “small/medium/large” cutoffs.

### 6.4 Hedges' g — `hedges_g`

The project applies a small-sample correction to d:

$$
\nu=n_1+n_0-2,\qquad
J(\nu)\approx1-\frac{3}{4\nu-1},\qquad
g=J(\nu)d.
$$

It preserves d's sign and moves its magnitude slightly toward zero. For $n_1=n_0=10$, $\nu=18$, $J\approx0.95775$, so $d=0.5$ gives $g\approx0.47887$.

The code uses this conventional approximation, not an exact gamma-function correction. It inherits d's finite-value population and undefined cases. The supplied example config excludes `hedges_g`; removing it from `statistics.exclude` enables it. For background, see [bias-corrected standardized mean differences](https://www.statsmodels.org/stable/generated/statsmodels.stats.meta_analysis.effectsize_smd.html).

### 6.5 Raw-score ROC AUC — `auc`

Treat the feature itself as a score for label 1. The empirical AUC is:

$$
\operatorname{AUC}(X)=
\frac{1}{n_1n_0}
\sum_{i:Y_i=1}\sum_{j:Y_j=0}
\left[\mathbf{1}\{x_i>x_j\}+\frac12\mathbf{1}\{x_i=x_j\}\right].
$$

**Intuition:** choose a positive and a negative at random. AUC is the probability that the positive has the higher score, counting ties as half a win. This equals the area under the ROC curve across score thresholds.

The implementation uses [scikit-learn's ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) on finite sampled values and requires both classes. It does not fit a univariate model. AUC is undefined if a usable class is absent, but a finite constant feature with both classes has AUC 0.5.

Outputs:

$$
\texttt{raw\_numeric\_auc}=A,\qquad
\texttt{flipped\_auc}=1-A,\qquad
\texttt{direction\_free\_auc}=\max(A,1-A).
$$

Negating the score reverses every unequal comparison and preserves ties, explaining $A(-X)=1-A(X)$.

An AUC of 0.20 is strong inverse ranking: $-X$ has AUC 0.80 on the same rows. It is not a reason, by itself, to remove $X$. Direction-free AUC chooses orientation on the same sample and is therefore an exploratory summary, not validated performance.

AUC does not measure calibration, positive-class rate, or precision at an operating threshold. AUC near 0.5 can also occur for useful nonmonotonic features. For example, positives at both extremes can cancel in a global ranking.

### 6.6 Two-sample Kolmogorov–Smirnov statistic — `ks`

For the empirical cumulative distribution functions within each class:

$$
\hat F_y(t)=\frac{1}{n_y}\sum_{i:Y_i=y}\mathbf{1}\{x_i\le t\},
\qquad
D=\sup_t\left|\hat F_1(t)-\hat F_0(t)\right|.
$$

The output is `ks_statistic`, ranging from 0 to 1. It is the largest vertical separation between the two cumulative distributions and can detect changes in location, spread, or shape.

KS is unsigned: it does not say which class has larger values. It complements d, which compares means, and AUC, which averages pairwise rankings.

The code calls `scipy.stats.ks_2samp(..., method="asymp")` but keeps **only the statistic**. It does not report the KS p-value. Both finite sampled classes are required. Ties do not prevent computing the descriptive statistic, although they matter for classical continuous-distribution test calibration. See [SciPy KS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html).

### 6.7 Point-biserial correlation — `point_biserial`

This is Pearson correlation between numerical $X$ and binary $Y$:

$$
r_{pb}=
\frac{\bar{x}_1-\bar{x}_0}{s_X}
\sqrt{\frac{n_1n_0}{n(n-1)}},
\qquad n=n_1+n_0,
$$

where $s_X$ is the sample SD across all usable rows. This is equivalent to the centered Pearson formula in Section 7.1. See [SciPy's point-biserial definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pointbiserialr.html).

The output is `point_biserial_r`. Its sign follows the class-mean difference. Unlike d, it uses overall feature spread and has a factor depending on class balance. Severe imbalance can limit its magnitude.

The implementation calls a shared Pearson helper, which requires at least three complete rows and variation in both variables. No correlation p-value is reported.

### 6.8 Spearman correlation with the target — `spearman_target`

$$
\rho_s(X,Y)=r\big(R(X),R(Y)\big),
$$

where $R$ assigns average ranks to ties and $r$ is Pearson correlation.

The binary target has many ties; average ranks handle them. This compares the rank distribution of the feature across the two classes. It has range $[-1,1]$, preserves an association direction, and is less affected by the numerical size of an extreme value than a mean-based measure.

The calculation uses finite sampled values and the same minimum-support/variation checks as the Pearson helper. Its magnitude is not on the same scale as AUC or d, and it can miss U-shaped effects.

### 6.9 Empirical mutual information — `mutual_information`

Let $O_{by}$ be the full-data count in feature bucket $b$ and target class $y$. Define:

$$
p_{by}=\frac{O_{by}}{N},\qquad
p_b=\sum_y p_{by},\qquad p_y=\sum_b p_{by}.
$$

Then:

$$
I(B;Y)=\sum_{b,y:p_{by}>0}p_{by}
\ln\frac{p_{by}}{p_bp_y}.
$$

The output is `mi_nats`. This is the discrete contingency-table quantity also used by [scikit-learn mutual_info_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html), not a nearest-neighbor continuous MI estimator.

**Intuition:** if bucket and label are independent, the joint probability equals $p_bp_y$, every log ratio is zero, and MI is zero. MI grows when knowing the bucket reduces uncertainty about the label:

$$
I(B;Y)=H(Y)-H(Y\mid B),\qquad
0\le I(B;Y)\le H(Y).
$$

For a binary target with positive rate $\pi$:

$$
H(Y)=-\pi\ln\pi-(1-\pi)\ln(1-\pi)\le\ln2.
$$

At $\pi=0.04$, the ceiling is about 0.16794 nats, not 1. This entropy ceiling is an interpretation aid; normalized MI is not an emitted metric.

MI is unsigned and can detect nonlinear bucket-level associations. Numerical MI includes the missing/nonfinite bucket, unlike numerical d/AUC. Categorical MI describes pooled categories, not every original ID.

Empirical MI tends to be inflated by sparse or numerous cells. No smoothing, permutation null correction, or MI p-value is implemented. Changing bins or category pooling changes the question being measured. The report sorts target rows by MI when enabled; this is not a validated feature ranking.

### 6.10 Pearson chi-square test — `chi_square`

For a contingency table with $r$ observed rows and $c$ observed columns, remove empty margins and let the table total be $n$:

$$
E_{ij}=\frac{O_{i+}O_{+j}}{n},\qquad
\chi^2=\sum_{i,j}\frac{(O_{ij}-E_{ij})^2}{E_{ij}},
\qquad \nu=(r-1)(c-1).
$$

Under the independence null and its approximation conditions:

$$
p_{\mathrm{iid}}=P(\chi^2_\nu\ge\chi^2_{\mathrm{observed}}).
$$

**Intuition:** compare observed counts with counts expected if feature bucket and label were independent. Squaring deviations avoids cancellation; dividing by the expected count puts differently sized cells on a comparable scale.

The project calls [SciPy chi2_contingency](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.chi2_contingency.html) with `correction=False`: no Yates continuity correction. It emits `chi2_p_iid` and `min_expected_count`, not the raw chi-square statistic or degrees of freedom.

The p-value is suppressed when any expected count is below 5, when the table is degenerate, or whenever `group_column` is configured. Setting a group column does not fit a cluster-aware test; it disables this row-IID inference. A million correlated or highly imbalanced rows do not automatically justify the approximation.

Small p-values indicate incompatibility with the independence null under the assumptions. They do not establish a large effect, predictive usefulness, or the probability that the null is true. Chi-square p-values do not drive the current automatic feature-selection rules.

### 6.11 Benjamini–Hochberg adjustment — included with `chi_square`

Let $p_{(1)}\le\cdots\le p_{(M)}$ be the finite eligible feature-target chi-square p-values. The adjusted value is:

$$
q_{(i)}=\min\left(1,\min_{j\ge i}\frac{M}{j}p_{(j)}\right).
$$

The reverse cumulative minimum makes adjusted values monotone in sorted p-value order. They are restored to original feature positions and emitted as `chi2_q_bh_iid`. Suppressed p-values stay undefined and do not count toward $M$.

Example: input p-values $[0.01,0.04,0.03]$ produce adjusted values $[0.03,0.04,0.04]$.

BH addresses the expected proportion of false discoveries among rejected hypotheses under independence or suitable positive dependence. It is not a per-feature probability of being a false discovery. The family here contains only eligible feature-target chi-square tests; it does not include every pair, bucket, interaction, or later analyst choice. See [SciPy false-discovery control](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).

### 6.12 Bias-corrected Cramér's V — `cramers_v_target`

Using the same contingency table and chi-square statistic:

$$
\phi_c^2=\max\left(0,\frac{\chi^2}{n}
-\frac{(r-1)(c-1)}{n-1}\right),
$$

$$
r_c=r-\frac{(r-1)^2}{n-1},\qquad
c_c=c-\frac{(c-1)^2}{n-1},
$$

$$
V_c=\sqrt{\frac{\phi_c^2}{\min(r_c-1,c_c-1)}}.
$$

The output is `cramers_v_corrected`. The correction reduces finite-sample inflation in nominal association; see [Bergsma's original paper](https://stats.lse.ac.uk/bergsma/pdf/cramerV3.pdf).

It is an unsigned association measure on a 0-to-1 scale when defined. Zero can result from the correction truncating a small estimated association. It does not prove population independence.

The implementation returns undefined when $n\le1$, a dimension has fewer than two observed levels, or the corrected denominator is nonpositive. It can still report V when the chi-square p-value is suppressed: a descriptive effect-size calculation and a valid hypothesis test are different requirements.

Target V uses full-data numerical/category buckets. The same formula is reused for sampled categorical feature pairs.

## 7. Feature pairs

Pairs ask whether two predictors carry related information. They do not automatically establish that one adds nothing to a model.

The engine enumerates unordered pairs in configuration order, at most `max_pairs`. With $P$ features there are $P(P-1)/2$ possible pairs. A cap can leave important pairs unexamined; pair coverage is reported in `summary.json`. If all pair metrics are disabled, no pair rows are generated.

### 7.1 Pearson correlation — `pearson`

On rows finite in both numerical features:

$$
r_{XZ}=
\frac{\sum_i(x_i-\bar{x})(z_i-\bar{z})}
{\sqrt{\sum_i(x_i-\bar{x})^2\sum_i(z_i-\bar{z})^2}}.
$$

The output is `pearson_r`, with `valid_rows` giving pairwise-complete support. Values range from $-1$ to 1.

**Intuition:** centered features pointing in the same direction across rows have positive correlation. Exact affine copies $Z=a+bX$ have $r=1$ when $b>0$ and $r=-1$ when $b<0$.

Pearson detects linear association. Outliers can dominate it; nonlinear dependence can exist when $r=0$. A numerical pair needs at least three complete rows and variation in both columns. The code calls SciPy's Pearson statistic and discards its p-value.

Missingness can make different pair correlations describe different subsets. A correlation matrix assembled from pairwise-complete rows need not have the properties of a matrix computed from one common complete dataset.

### 7.2 Spearman correlation — `spearman`

$$
\rho_{XZ}=r\big(R(X),R(Z)\big).
$$

Average ranks handle ties. The output is `spearman_r`, using the same pairwise-complete support checks.

Spearman detects monotonic association: one feature tends to increase or decrease as the other does, without requiring a straight line. A strictly increasing transformation can preserve ranks while changing Pearson correlation.

For ties, use the rank-correlation formula above, not the simplified no-ties formula involving squared rank differences. High absolute Spearman correlation is the default redundancy criterion.

### 7.3 Categorical pair association — `cramers_v_pairs`

The engine creates a contingency table for two **pooled categorical** bucket codes on the fixed sample and applies the corrected V formula in Section 6.12.

Both missing categories remain in the table. `valid_rows` is the sample size. If the product of the two observed sample bucket counts exceeds `max_joint_cells`, the V calculation is skipped with `status: skipped_cell_limit`.

High V can reveal overlapping classifications or category mappings. There is no positive/negative direction for arbitrary categories. Pooling can conceal relationships between original categories.

This statistic is reported for review. The current redundancy rule does not automatically exclude categorical features using V.

### 7.4 Correlation ratio, eta squared — `eta_squared`

For pooled category groups $g$ and finite numerical values $x_i$, define group means $\bar{x}_g$ and group sizes $n_g$:

$$
SS_{\mathrm{between}}=\sum_g n_g(\bar{x}_g-\bar{x})^2,
\qquad
SS_{\mathrm{total}}=\sum_i(x_i-\bar{x})^2,
$$

$$
\eta^2=\frac{SS_{\mathrm{between}}}{SS_{\mathrm{total}}}.
$$

The output is `eta_squared`; the code clips numerical roundoff to $[0,1]$.

**Intuition:** decompose total squared deviation into variation between category means and variation remaining within categories. Eta squared is the fraction attributed to differences between those observed group means.

For category A with values $[0,2]$ and B with $[4,6]$, the overall mean is 3, between-group sum of squares is 16, and total sum of squares is 20. Thus $\eta^2=0.8$.

It is unsigned and asymmetric in role: the category groups the numerical feature. It is not ordinary correlation squared for arbitrary categories, and it is not an out-of-sample $R^2$.

The calculation uses sampled finite numerical values; categorical missing values remain a group. It is undefined for fewer than two numerical values or zero total variance. Many small groups can inflate the observed ratio. No ANOVA p-value or bias correction is implemented.

### 7.5 Missingness phi — `missingness_phi`

Define $M_X=\mathbf{1}\{B(X)=0\}$ and $M_Z=\mathbf{1}\{B(Z)=0\}$. For numerical features these indicate missing/nonfinite values; for categorical features they indicate nulls.

$$
\phi=r(M_X,M_Z).
$$

Equivalently, if $n_{ab}$ counts sampled rows with $(M_X,M_Z)=(a,b)$:

$$
\phi=
\frac{n_{11}n_{00}-n_{10}n_{01}}
{\sqrt{(n_{11}+n_{10})(n_{01}+n_{00})(n_{11}+n_{01})(n_{10}+n_{00})}}.
$$

Positive phi means missingness tends to occur together. Negative phi means one feature tends to be available when the other is missing. This can expose a shared source outage, optional workflow, or mutually exclusive collection paths.

It is undefined if either indicator is constant, including when a feature is never missing. The Pearson helper also requires at least three rows. Undefined does not mean independent missingness.

This diagnostic uses all sampled rows and does not compare missingness directly with the target. The feature-target bucket tables provide that latter view.

## 8. Joint and conditional analysis

Only pairs and triples explicitly listed in `interactions` are examined. Their presence expresses a hypothesis, not evidence that an interaction exists.

### 8.1 Joint positive-rate tables — `joint_positive_rate`

For a tuple of feature buckets $J=(B(X_1),\ldots,B(X_d))$, $d\in\{2,3\}$, compute counts, positive rates, lift, Wilson intervals, and low-support flags with the formulas in Section 6.

**Intuition:** a feature may distinguish labels only in a particular context. A position effect may differ by device; a rate effect may differ by term length. A joint table exposes those combinations.

These are observed conditional proportions, not causal effects. Differences can also reflect other unmeasured features or changes in which rows enter each cell.

### 8.2 Joint mutual information — `joint_information`

Treat the entire bucket tuple as one discrete variable:

$$
I(J;Y)=
\sum_{j,y:p_{jy}>0}p_{jy}\ln\frac{p_{jy}}{p_jp_y}.
$$

The implementation factorizes observed bucket tuples and calls `mutual_info_score` on the fixed sample. It emits `joint_mi_nats`, `sample_rows`, and `observed_sample_cells`.

Joint MI asks how much label uncertainty is reduced by knowing the combination. It can be positive even when every individual feature's MI is zero.

### 8.3 Gain over the best single feature

For the features in the configured group:

$$
\Delta_{\mathrm{best}}
=I(J;Y)-\max_{\ell}I(B(X_\ell);Y).
$$

Outputs are `best_single_mi_nats` and `gain_over_best_single_nats`. The code floors the gain at zero to handle floating-point noise.

All single-feature values in this calculation use the **same sample** as joint MI. Consequently, `best_single_mi_nats` can differ from the corresponding full-data MI values in `feature_target.csv`.

The gain measures information beyond the best one-feature description. It is not a pure interaction score: complementary additive information can also create a gain, and sparse joint cells can inflate the estimate.

### 8.4 Conditional mutual information

For a feature $A$ and the other configured bucket features grouped as $C$:

$$
I(A;Y\mid C)
=\sum_c p_c\sum_{a,y}p_{ay\mid c}
\ln\frac{p_{ay\mid c}}{p_{a\mid c}p_{y\mid c}}.
$$

The chain rule gives the equivalent calculation used by the code:

$$
I(A;Y\mid C)=I(A,C;Y)-I(C;Y).
$$

For each feature position $\ell$, the output includes `feature_1`, `feature_2`, and optionally `feature_3`, with matching `conditional_mi_1_nats` and subsequent columns.

**Intuition:** after knowing the other named features, how much label uncertainty remains that this feature helps reduce? An exact duplicate has zero additional empirical information once its copy is known.

“Conditional” refers only to the other features in this configured pair/triple, not every feature in the dataset. This is an empirical discrete calculation with a zero floor for numerical noise, not a significance test or a fitted estimate of incremental model value.

### 8.5 Cell limits and support

If feature $\ell$ has $K_\ell$ observed full-data buckets, then:

$$
\text{possible cells}=\prod_{\ell=1}^{d}K_\ell.
$$

A group is skipped when this exceeds `max_joint_cells`, even if only a few combinations actually occur. The default cap is 20,000.

`observed_full_cells` counts combinations in the full data; `observed_sample_cells` counts combinations in the sample. These are different from the product limit.

The row mass in poorly supported full-data cells is:

$$
\text{low-support row fraction}
=\frac{\sum_{j:\text{low support}_j}n_j}{N}.
$$

Large low-support mass weakens confidence in local patterns. The engine reports it but does not remove those cells from MI. It also computes full joint support for enabled joint MI even when the joint rate CSV export is disabled.

### 8.6 Why marginal screening can fail: XOR

Let $A,B$ be independent fair binary features and $Y=A\oplus B$:

| $A$ | $B$ | $Y$ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

With equal cell frequencies:

$$
I(A;Y)=I(B;Y)=0,\qquad
I(A,B;Y)=\ln2,
$$

$$
I(A;Y\mid B)=I(B;Y\mid A)=\ln2.
$$

Each feature alone leaves the label at 50/50. Together they determine it exactly. This is why low univariate d/AUC/MI cannot prove a feature useless and why configured interaction hypotheses can be protected from marginal exclusions.

## 9. Temporal and group context

### 9.1 Timestamp handling

Time analysis requires `time_column`. Timestamps or parseable date strings are converted to UTC, then assigned to daily (`D`), weekly (`W`), or monthly (`M`) calendar periods. Pandas `W` uses weeks ending Sunday. These are calendar groups, not rolling windows.

Missing dates form `[missing]`. Nonmissing unparseable dates raise an error. Numeric epoch columns are rejected because their time unit is ambiguous.

If the number of observed periods, including missing, exceeds `max_time_periods`, temporal analysis is skipped; a coarser frequency may be needed. The default cap is 120.

### 9.2 Overall positive rate over time — `time_positive_rate`

For period $t$:

$$
\hat p_t=\frac{\sum_{i:T_i=t}Y_i}{\#\{i:T_i=t\}}.
$$

The period table includes the usual counts, lift relative to the full-data positive rate, Wilson bounds, and support flags.

A changing positive rate can reflect seasonality, audience composition, policy changes, or changing outcomes. This table does not distinguish those explanations automatically.

### 9.3 Bucket positive rate over time — `time_feature_positive_rate`

For feature bucket $b$ and period $t$:

$$
\hat p_{b,t}
=\frac{\sum_iY_i\mathbf{1}\{B(X_i)=b,T_i=t\}}
{\sum_i\mathbf{1}\{B(X_i)=b,T_i=t\}}.
$$

Fixed full-data buckets make the same bucket comparable across periods. Compare support and positive rates together: a changed rate in a tiny cell may be noise, and an unchanged overall rate can conceal opposing subgroup changes.

The table's lift uses the global baseline $\hat p$, not $\hat p_t$. Missing dates remain an explicit period.

### 9.4 Jensen–Shannon divergence — `time_drift`

Let $P_t(b)$ be the feature's bucket distribution within period $t$ and $Q(b)$ its distribution across the entire supplied dataset. Let $M=(P_t+Q)/2$:

$$
D_{\mathrm{KL}}(P\|Q)=\sum_{b:P(b)>0}P(b)\ln\frac{P(b)}{Q(b)},
$$

$$
\operatorname{JS}(P_t,Q)
=\frac12D_{\mathrm{KL}}(P_t\|M)
+\frac12D_{\mathrm{KL}}(Q\|M).
$$

The output is `js_divergence_nats_vs_all`. With natural logs:

$$
0\le\operatorname{JS}(P_t,Q)\le\ln2.
$$

**Intuition:** measure how distinguishable the period's bucket mix is from the overall mix. Equal distributions give zero. Comparing each distribution with their average makes the measure symmetric and handles zero-probability buckets without an arbitrary smoothing constant.

SciPy returns the square-root **distance**; this project squares that result to report **divergence**. See [SciPy Jensen–Shannon distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html).

The reference is row-weighted across the full dataset and includes the period being compared. Therefore this is not a comparison with a separate historical baseline; large periods pull the reference toward themselves. The missing-date period is included too.

JS measures feature-distribution change, not directly a change in $P(Y\mid X)$, predictive degradation, or causation. There is no automatic drift threshold or drift-based exclusion rule.

### 9.5 Group summaries

With `group_column`, rows are grouped by that column, including a missing group. For $G$ groups, group sizes $n_g$, and positive counts $k_g$, the report includes:

$$
\text{median rows per group}=\operatorname{median}_g n_g,\qquad
\text{max rows per group}=\max_g n_g,
$$

$$
\text{fraction groups with positives}
=\frac{1}{G}\sum_g\mathbf{1}\{k_g>0\}.
$$

It also records group count and missing-group row count. The positive-group fraction weights each group equally and is not the row-level positive rate.

These statistics expose repeated observations such as multiple rows within a search. They do not estimate effective sample size, independence, cluster confidence intervals, or group-aware model performance. Configuring a group suppresses chi-square p-values but leaves explicitly labeled row-IID Wilson intervals available.

## 10. Feature decisions

Statistics describe the dataset. Rules translate selected descriptions into an explicit screening policy. Every feature starts as `keep`.

- **keep:** retain as a candidate; no stronger applicable action remains.
- **review:** retain as a candidate with a finding to investigate.
- **exclude:** omit from the recommended candidate list under the configured policy.

`selected_features.json` contains **keep plus review** features. The source dataset is never modified.

### 10.1 Basic rules and defaults

| Rule | Trigger | Default action |
|---|---|---|
| `all_missing` | No distinct usable observed values | `exclude` |
| `constant` | Exactly one distinct usable value and zero missing/unusable fraction | `exclude` |
| `missingness` | Missing/unusable fraction $\ge0.8$ | `review` |
| `near_constant` | Dominant fraction among usable rows $\ge0.995$ | `review` |
| `high_cardinality` | Categorical distinct observed count $\ge1000$ | `review` |

These cutoffs are configurable policy choices, not universal statistical constants.

The all-missing rule concerns variation in the supplied data; it does not claim that the feature could never be collected usefully. The constant rule deliberately excludes a constant everywhere but does not exclude a constant observed value accompanied by varying missingness.

### 10.2 Weak marginal signal — `weak_signal`

For a numerical feature, all three conditions must hold:

$$
|d|<d_{\max},
\qquad |A-0.5|<a_{\max},
\qquad I(B;Y)<I_{\max}.
$$

Defaults:

$$
d_{\max}=0.05,\qquad a_{\max}=0.02,\qquad I_{\max}=0.0001.
$$

For a categorical feature, only the MI comparison applies. The default action is `review`, acknowledging that weak marginal signal does not rule out interactions.

The rule is **skipped** if a required statistic is disabled or undefined. Numerical evaluation requires `cohens_d`, `auc`, and `mutual_information`; Hedges' g does not substitute for d. Categorical evaluation requires `mutual_information`.

For categorical features, if more than half the rows are pooled into OTHER, a would-be evaluation is skipped and a coverage review is recorded because pooled MI may hide original-level signal. This guard is reached after required-metric checks. The condition is strictly $>0.5$; the descriptive pooling flag uses $\ge0.5$.

A missing required metric never counts as evidence of weakness. At an exact weak-signal threshold, the strict less-than condition does not fire.

### 10.3 Expected direction and conflicting directions — `direction`

Expected directions are configured only for numerical features:

```yaml
expected_direction:
  rank_pos: negative
  apy: positive
```

The rule forms a direction signal from d if $|d|\ge0.05$ and $d\ne0$, and from AUC if $|A-0.5|\ge0.02$ and $A\ne0.5$. These minima are configurable.

$$
\operatorname{direction}_d=\operatorname{sign}(d),
\qquad
\operatorname{direction}_{A}=\operatorname{sign}(A-0.5).
$$

It triggers when either sufficiently strong signal disagrees with the configured expectation, or when the two signals disagree with each other. The second check can fire even if no expected direction is configured.

One enabled, defined, sufficiently strong metric can establish an expectation mismatch. If neither qualifies, the rule is skipped. Unordered categorical features are marked not applicable. The default action is `review`.

Mean-based d and rank-based AUC summarize different properties. A positive d with AUC below 0.5 can be mathematically correct when a small high-valued tail raises the positive mean while most positives lie below negatives.

Expected direction is an observed-association expectation. It does not encode a causal claim or create an XGBoost monotonicity constraint.

### 10.4 Numerical redundancy — `redundancy`

The rule chooses either `pearson` or `spearman`; the default is Spearman. For correlation $r$, the threshold score is:

$$
s(r)=
\begin{cases}
|r|,&\texttt{sign=absolute},\\
r,&\texttt{sign=positive},\\
-r,&\texttt{sign=negative}.
\end{cases}
$$

A pair can trigger only when:

$$
s(r)\ge0.95,\qquad
n_{\mathrm{pair}}\ge1000,\qquad
\frac{n_{\mathrm{pair}}}{m}\ge0.5.
$$

All three thresholds are configurable. Correlation must be finite, the pair must have been computed, and the corresponding statistic must be enabled.

The denominator of `min_pair_fraction` is the full fixed sample size, not the full dataset and not either feature's usable sample count.

**How the retained representative is selected:** features are ordered lexicographically by:

1. Membership in `force_keep`, first.
2. Position in `prefer`; listed features precede unlisted features.
3. Lower missing/unusable fraction.
4. Larger $|A-0.5|$ if raw AUC is available; unavailable AUC contributes zero.
5. Original feature order in the configuration.

The algorithm walks this order, comparing each eligible numerical feature with earlier retained representatives. It uses the first representative with an eligible above-threshold direct correlation, records that feature's name and signed correlation, and applies the configured action. Default action: `exclude`.

Features already excluded by another rule or `force_exclude` cannot be representatives. A `review` feature can be a representative because review features remain candidates.

This is a greedy deterministic procedure, not global optimization or clustering. Suppose A–B and B–C correlations exceed the threshold, but A–C does not. If B is excluded in favor of A, B cannot then justify excluding C. Each redundancy finding therefore refers to a directly correlated retained feature.

Changing configuration order, preferences, missingness, or pair coverage can change representatives. High correlation on complete rows also does not establish interchangeable missingness behavior or equal incremental model value.

### 10.5 Rule combinations and precedence

Ordinary triggered actions escalate by:

$$
\texttt{keep}<\texttt{review}<\texttt{exclude}.
$$

An ordinary rule with `action: keep` records a finding without downgrading an earlier review or exclusion.

The rules are independent. A numerical redundancy exclusion does **not** require low d, low MI, or a direction mismatch. A report may contain several reasons because several rules fired; that does not imply a combined cross-rule AND condition.

For example, setting weak signal and redundancy to `exclude` means either can exclude a feature. The current configuration cannot express “exclude only if both weak signal and redundancy trigger” as a new composite rule.

`force_exclude` is applied before representative selection. `force_keep` overrides automatic final actions while leaving the earlier findings in the audit log. The two lists cannot overlap.

With `protect_configured_interactions: true`, a feature named in an interaction hypothesis has a proposed **weak-signal or redundancy exclusion** changed to review. This does not protect it from all-missing, constant, direction, other rule exclusions, or `force_exclude`. It applies based on the configured hypothesis even if its joint analysis is disabled or skipped.

Forced keeps receive the same weak/redundancy protection during processing and end with `keep`. An explicit forced keep can therefore retain a structurally unusable feature; the recorded findings still explain the issue.

### 10.6 Coverage findings and the audit trail

The engine adds a coverage review if none of these target-strength switches is enabled: `mutual_information`, `auc`, `cohens_d`, `ks`, `cramers_v_target`.

This check is based on switches, not on whether each feature has a defined, applicable value. For example, enabling AUC globally does not assess categorical target strength. A keep decision must be read alongside skipped and not-applicable evaluations.

`rule_evaluations.csv` includes:

| Field | Meaning |
|---|---|
| `feature`, `rule` | Which feature and condition were examined |
| `status` | `triggered`, `not_triggered`, `disabled`, `skipped`, or `not_applicable` |
| `requested_action` | Configured action |
| `effective_action` | Action after protection, or `none` when not triggered |
| `related_feature` | Retained comparison feature for a redundancy finding |
| `reason` | Human-readable explanation |
| `evidence` | JSON containing measured values and relevant thresholds |

Final decisions summarize triggered findings; the rule log also preserves skipped and disabled work. A final keep does not mean every statistic was computed or every possible interaction was investigated.

Whenever finite raw AUC is below 0.5, the final reason appends its inverse-ranking interpretation. That annotation alone does not change the decision.

## 11. Configuration and output reference

### 11.1 Every supported statistic switch

The names below are accepted by `statistics.include` and `statistics.exclude`. Use `cohens_d`, not `cohence_d`. Unknown metric names fail validation.

| Family | Exact switches | Main explanation |
|---|---|---|
| Numerical center/spread | `mean`, `std`, `median`, `quantiles`, `iqr`, `mad` | Sections 3.1–3.5 |
| Numerical shape/value checks | `skewness`, `kurtosis`, `zero_fraction`, `negative_fraction`, `outlier_fraction` | Sections 3.6–3.9 |
| Categorical summaries | `entropy`, `rare_categories` | Section 4 |
| Rate profiles | `positive_rate` | Sections 6.1–6.2 |
| Numerical target comparisons | `cohens_d`, `hedges_g`, `auc`, `ks`, `point_biserial`, `spearman_target` | Sections 6.3–6.8 |
| Bucket-target association | `mutual_information`, `chi_square`, `cramers_v_target` | Sections 6.9–6.12 |
| Feature pairs | `pearson`, `spearman`, `cramers_v_pairs`, `eta_squared`, `missingness_phi` | Section 7 |
| Configured multivariate groups | `joint_positive_rate`, `joint_information` | Section 8 |
| Temporal diagnostics | `time_positive_rate`, `time_feature_positive_rate`, `time_drift` | Section 9 |

Derived values share their parent switch. For example, `auc` controls raw, flipped, and direction-free AUC; `entropy` also controls effective levels; `chi_square` also controls BH adjustment. There are no independent `wilson`, `lift`, or `bh` switches.

`include: all` requests every optional metric. A metric listed in `exclude` is disabled even if included. `include: []` disables all optional metrics.

```yaml
# Replace the statistics block in your complete dataset configuration.
statistics:
  include: all
  exclude: [hedges_g, chi_square]
```

A smaller selection might be:

```yaml
statistics:
  include:
    - median
    - quantiles
    - positive_rate
    - cohens_d
    - auc
    - mutual_information
    - spearman
    - joint_positive_rate
    - joint_information
  exclude: []
```

Row/class counts, schema checks, dtype, missingness, uniqueness, dominance, flags, and bucketing/support bookkeeping always run. Optional univariate results are removed from reports by `filter_profile`; shared intermediates may still be calculated. Disabling a metric is not a guarantee of zero computational work for its underlying formula.

`positive_rate` controls the per-feature rate CSVs and plots. It does not disable global class counts, target-association contingency tables, or separately enabled joint/time rate outputs.

### 11.2 Calculation controls

| Parameter | Default | Effect |
|---|---|---|
| `bins` | 10 | Numerical exact-value/quantile bucket control; at least 2 |
| `max_categories` | 20 | Maximum individually retained observed categories |
| `rare_min_count` | 100 | Category count below which a level is rare and cannot be individually retained |
| `min_cell_count` | 200 | Minimum rows for an unflagged rate cell |
| `min_cell_events` | 10 | Minimum count of each target class in an unflagged cell |
| `sample_rows` | 100,000 | Maximum fixed sample size; at least 10 |
| `seed` | 42 | Nonnegative random seed |
| `max_pairs` | 500 | First unordered feature pairs in config order; zero disables pair enumeration |
| `max_joint_cells` | 20,000 | Cell cap for joint hypotheses and categorical pair tables |
| `interactions` | Empty list | Distinct configured feature pairs/triples |
| `time_column` | null | Optional timestamp context |
| `time_frequency` | `W` | Calendar periods: `D`, `W`, or `M` |
| `max_time_periods` | 120 | Maximum observed periods, including missing |
| `group_column` | null | Optional repeated-observation context; suppresses row-IID chi-square p-values |
| `strict_schema` | true | Reject undeclared source columns |
| `ignore` | Empty list | Columns validated in the schema but not read for analysis |
| `output_dir` | `report` | Output root, relative to config location if not absolute |

The input file path and config path are the two command-line inputs:

```bash
uv run --locked -m data_mining /path/to/data.parquet config.yaml
```

All default decision thresholds are in Section 10. Rule names accept `enabled` and `action`; actions are `keep`, `review`, and `exclude`. Additional controls are `expected_direction`, `force_keep`, `force_exclude`, `prefer`, and `protect_configured_interactions`.

For positive-correlation redundancy only:

```yaml
# Merge this block into your complete config; it is not a standalone config.
rules:
  redundancy:
    enabled: true
    metric: spearman
    sign: positive
    threshold: 0.95
    min_pair_rows: 1000
    min_pair_fraction: 0.5
    action: exclude
  weak_signal:
    enabled: true
    max_abs_d: 0.05
    max_auc_distance: 0.02
    max_mi: 0.0001
    action: review
  direction:
    enabled: true
    min_abs_d: 0.05
    min_auc_distance: 0.02
    action: review
prefer: [rank_pos, apy]
protect_configured_interactions: true
```

Each named feature must exist in your own `features` mapping. Settings omitted from a rule inherit defaults. Changing a statistic switch does not silently disable its rule: missing prerequisites are logged as skipped.

### 11.3 Where to find the numbers

Every run creates a fresh timestamped folder under `output_dir`.

| Output | What it contains |
|---|---|
| `numerical_univariate.csv` | Numerical quality and enabled distribution summaries |
| `categorical_univariate.csv` | Original-category summaries, pooling metadata, and flags |
| `feature_target.csv` | Full-data bucket association plus sampled numerical target metrics |
| `feature_positive_rate/` | Per-feature bucket rates, lift, intervals, and support |
| `feature_pairs.csv` | Sampled pair statistics and pair support |
| `joint_information.csv` | Joint/conditional MI, gain, cell counts, support, and skip status |
| `joint_positive_rate/` | Full-data rates for configured pairs/triples |
| `time_positive_rate.csv` | Rates by period |
| `time_feature_positive_rate/` | Feature-bucket rates within periods |
| `time_feature_drift.csv` | Per-period feature JS divergence versus all rows |
| `feature_decisions.csv` / `.json` | Final decision, reasons, related features, and selected evidence values |
| `rule_evaluations.csv` | Complete rule audit trail |
| `selected_features.json` | Keep + review candidates and separate exclude/review lists |
| `summary.json` | Full/sample counts, class rate and interval, group context, versions, coverage, limitations |
| `bucket_dictionary.csv` | Feature bucket code/label mapping |
| `manifest.json` | Feature names mapped to safe output filenames; joint filename mapping |
| `config_used.yaml` | Resolved effective configuration, including inherited defaults |
| `report.html` | Self-contained decisions, tables, and enabled rate plots |
| `report.md` | Readable final decisions and interpretation notes |

Undefined numeric values are generally blank/NaN in CSV and `null` in JSON. Depending on the output, a disabled family may leave an empty table or an absent column rather than remove every related file.

The HTML is a viewing layer: generic tables normally show the first 100 rows, feature pairs up to 500, and feature-rate plots at most the first 32 buckets. Full CSVs remain the reference for complete results. Plot error bars are row-IID Wilson intervals, the dashed line is the global positive rate, orange markers indicate low support, and the lower panel shows row counts.

## 12. Worked interpretation examples

These small constructed examples explain the mathematics; they are not findings about your real dataset.

### 12.1 Positive d together with AUC below 0.5

Suppose the finite feature values are:

| Target class | Values | Mean |
|---|---|---|
| $Y=1$ | 0, 0, 0, 100 | 25 |
| $Y=0$ | 1, 1, 1, 1 | 1 |

The within-class sample variances are 2,500 and 0. Therefore:

$$
s_p=\sqrt{\frac{3(2500)+3(0)}{6}}=\sqrt{1250},
\qquad
d=\frac{25-1}{\sqrt{1250}}\approx0.67882.
$$

Only the positive value 100 beats the negatives: four winning comparisons out of 16.

$$
A=4/16=0.25,\qquad A(-X)=0.75.
$$

The positive mean is larger, but most positive rows have smaller feature values. The default direction rule records disagreement because both effects exceed its minimums. Inspect the distribution and provenance of the high-valued tail before making a decision.

### 12.2 Equal means and AUC 0.5 with perfect bucket information

Consider $X=[-2,-1,1,2]$ with $Y=[1,0,0,1]$.

The class means are both zero, so $d=0$. Two of four positive-negative comparisons are wins, so AUC is 0.5. Yet $|X|=2$ identifies positives exactly.

With each observed value retained as a bucket:

$$
I(B(X);Y)=H(Y)=\ln2,\qquad D_{\mathrm{KS}}=0.5.
$$

The feature has a U-shaped relationship. The default numerical weak-signal rule does not trigger because its MI condition is false. This illustrates why the project combines mean, rank, distribution, and bucket diagnostics.

### 12.3 A bucket with apparent lift but too little evidence

A bucket with 2 positives in 10 rows has rate 20%. Against a global 4% baseline, lift is 5.

Despite that large ratio, it has fewer than 200 rows and fewer than 10 observations in each class, so `low_support` is true. Its Wilson interval is approximately $[0.0567,0.5098]$ under the IID assumption.

The point estimate can motivate investigation; it is not a stable estimate of a fivefold effect. The code retains the cell and makes its support visible.

### 12.4 High positive correlation and a named exclusion

Suppose an eligible sampled pair has:

$$
\rho_s(\texttt{widget\_pos},\texttt{rank\_pos})=0.999,
\qquad n_{\mathrm{pair}}=100000,\qquad m=100000.
$$

With `prefer: [rank_pos]`, no conflicting override/protection, and the default redundancy threshold, `widget_pos` can be excluded in favor of retained `rank_pos`. The reason contains the signed correlation, threshold, support, and representative name.

If `widget_pos` also satisfies the weak-signal rule, that finding is logged separately. Weakness is not a prerequisite for the correlation exclusion.

If `widget_pos` is protected by a configured interaction, the proposed redundancy exclusion becomes review. If `rank_pos` is explicitly excluded, it cannot be used to justify removing `widget_pos`.

### 12.5 Small MI in an imbalanced target

At a 4% positive rate:

$$
H(Y)=-0.04\ln0.04-0.96\ln0.96\approx0.16794.
$$

An MI of 0.01 nats is about 5.95% of this entropy ceiling. The same absolute MI has a different relative scale when the target is balanced, whose ceiling is 0.69315 nats.

That ratio is for intuition, not an implemented normalized score or a direct estimate of model improvement. Check binning, support, missingness, and whether the supplied class balance represents deployment.

## 13. Limits and a practical review sequence

### 13.1 What the audit does not calculate

The current implementation does not calculate normality-test p-values, VIF, WoE/IV, PCA, partial Pearson correlations, cluster-adjusted inference, bootstrap confidence intervals, causal effects, or an effective training-set size.

It also does not calculate trained-model log loss, PR-AUC/average precision, calibration, feature importance, SHAP values, permutation importance, or feature-ablation performance. These may belong to a later modeling stage; they are not hidden inside the statistics described here.

There is no automatic monotonicity test, automatic search over every interaction, automatic transformation, outlier deletion, class rebalancing, target encoding, or missing-value imputation.

### 13.2 Common interpretation mistakes

| Observation | Appropriate interpretation |
|---|---|
| Raw AUC below 0.5 | Inverse ranking; inspect direction and nonlinear structure |
| Small d | Similar class means relative to spread; not proof of independence |
| d and AUC have conflicting signs | Means and typical ordering differ; inspect tails, ties, and mixture structure |
| Very small chi-square p-value | Evidence against independence under the test assumptions; not necessarily a useful effect |
| High categorical entropy | Many effectively used levels; no statement about the target |
| High pair correlation | Potential redundancy on the measured subset; incremental model value is still unknown |
| MI increases after adding bins | Could reflect revealed structure, sparse-cell bias, or both |
| Joint MI exceeds single MI | Additional empirical information; not necessarily pure interaction |
| High temporal JS | Feature mix differs from the full-data mix; not proof of model deterioration |
| Feature receives keep | No stronger final configured action; not validated predictive importance |

### 13.3 Review sequence

1. Verify target meaning, exact column names, units, and feature availability at the intended prediction moment.
2. Inspect class counts, missingness, constants, identifiers, rare levels, and the fraction pooled into OTHER.
3. Read numerical distributions before interpreting mean-based effects. Check finite sampled class counts.
4. Compare bucket rates, support, d, AUC, KS, and MI. Investigate disagreement instead of assuming one summary must be wrong.
5. Inspect direct feature-pair associations, overlap counts, missingness phi, and the representative selected by the redundancy rule.
6. Examine configured joint tables and conditional information, especially when marginal effects are weak.
7. Review time and group context before treating a precise-looking interval or small p-value as reliable.
8. Read both final decisions and the complete rule log. Adjust thresholds and preferences according to the documented screening policy.
9. In the later training stage, evaluate candidate sets with validation that matches deployment and an untouched final evaluation set. Measure incremental value with actual model comparisons.

All estimates describe the supplied rows and their chosen representations. Sampling rows uniformly does not repair selection bias, label leakage, repeated-observation dependence, or a nonrepresentative dataset. Negative sampling or class balancing changes the observed positive rate; all counts and probabilities here are unweighted.

The strongest use of this report is to make feature-selection hypotheses explicit and reviewable before model training.
