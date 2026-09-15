# Wilson confidence interval: step-by-step derivation, intuition, and proofs

This document explains the mathematics behind the Wilson interval used by `wilson()` and `positive_rate_table()` in [data_mining/statistics.py](../data_mining/statistics.py). It expands [section 6.2 of the statistics guide](statistics-and-techniques.md#62-wilson-95-interval--included-with-rate-tables).

The goal is to derive these three expressions:

$$
\boxed{D=1+\frac{z^2}{n}}
$$

$$
\boxed{C=\frac{\hat p+\frac{z^2}{2n}}{D}}
$$

$$
\boxed{
M=\frac{z}{D}
\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}
}
$$

They give the lower and upper endpoints:

$$
\boxed{L=C-M,\qquad U=C+M}.
$$

Here $C$ is the interval's center, $M$ is its half-width, and $D$ is their common denominator. All three expressions come from solving one quadratic inequality for the unknown probability $p$.

The algebra below is exact. The confidence interpretation starts from a normal approximation, so Wilson intervals do not guarantee exactly 95% coverage for every sample size and probability. [NIST describes this construction by inverting a hypothesis test](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).

## Notation and assumptions

| Symbol | Meaning |
|---|---|
| $X_i$ | Binary outcome for observation $i$: 1 for a click, 0 otherwise |
| $n$ | Number of observations in the bucket; the derivation requires $n>0$ |
| $k$ | Number of positive outcomes, with $0\le k\le n$ |
| $p$ | Unknown underlying positive probability |
| $\hat p=k/n$ | Observed positive proportion, pronounced "p-hat" |
| $z$ | Positive standard-normal critical value for the chosen confidence level |
| $D$ | Common denominator in the center and half-width formulas |
| $C$ | Center of the Wilson interval |
| $M$ | Nonnegative half-width of the Wilson interval |

Assume independent, unweighted binary observations with the same probability $p$ within the bucket. The report's `_iid` suffix identifies this assumption. Repeated, correlated observations from the same user or session can make the row-IID intervals too narrow; this helper does not adjust for that dependence.

For a two-sided 95% interval, $z\approx1.96$. The implementation uses `1.959963984540054`. Worked arithmetic below uses `1.96` and rounds displayed results.

## 1. Start with a model for the observations

Suppose each observation is binary:

$$
X_i=
\begin{cases}
1 & \text{a click occurs},\\
0 & \text{a click does not occur}.
\end{cases}
$$

The observed proportion is:

$$
\hat p=\frac{X_1+\cdots+X_n}{n}=\frac{k}{n}.
$$

Keep the distinction clear: $p$ is the underlying probability we want to estimate; $\hat p$ is the observed proportion we can calculate.

For one binary observation:

$$
E[X_i]=1\cdot p+0\cdot(1-p)=p.
$$

Because $X_i^2=X_i$:

$$
\operatorname{Var}(X_i)
=E[X_i^2]-E[X_i]^2
=p-p^2
=p(1-p).
$$

Independence lets us add the variances:

$$
\begin{aligned}
\operatorname{Var}(\hat p)
&=\operatorname{Var}\left(\frac{1}{n}\sum_{i=1}^{n}X_i\right)\\
&=\frac{1}{n^2}\sum_{i=1}^{n}\operatorname{Var}(X_i)\\
&=\frac{np(1-p)}{n^2}\\
&=\frac{p(1-p)}{n}.
\end{aligned}
$$

Therefore, the standard deviation of the observed proportion is:

$$
\boxed{\sqrt{\frac{p(1-p)}{n}}}.
$$

**Intuition:** the observed rate fluctuates across samples. More observations reduce those fluctuations, in proportion to $1/\sqrt n$ for fixed $p$.

## 2. Turn sampling uncertainty into an inequality

For a candidate probability $0<p<1$, the standardized difference between the observed rate and that candidate is:

$$
Z(p)=\frac{\hat p-p}{\sqrt{p(1-p)/n}}.
$$

If that candidate is the true probability, the central limit theorem makes this quantity approximately standard normal as $n$ grows. The approximation needs particular care when the probability is near 0 or 1 and the corresponding expected counts are small.

For a standard normal variable, approximately 95% of its probability lies between $-1.96$ and $+1.96$. A two-sided 95% score-test acceptance rule therefore retains candidate probabilities satisfying:

$$
|Z(p)|\le z.
$$

Equivalently:

$$
\boxed{
|\hat p-p|
\le
z\sqrt{\frac{p(1-p)}{n}}
}.
$$

**Intuition:** for each proposed $p$, ask whether the observed $\hat p$ is within the allowed sampling fluctuation for that candidate. The uncertainty depends on the candidate $p$ itself. This dependence produces the Wilson adjustments.

Both sides are nonnegative, so squaring preserves the inequality:

$$
(\hat p-p)^2
\le
\frac{z^2}{n}p(1-p).
$$

The resulting polynomial also gives the boundary cases $p=0$ and $p=1$ by continuity. We will calculate those cases explicitly below.

## 3. Derive D by collecting the quadratic terms

Introduce a temporary abbreviation:

$$
a=\frac{z^2}{n}.
$$

Our inequality becomes:

$$
(\hat p-p)^2\le ap(1-p).
$$

Expand both sides:

$$
\hat p^2-2\hat p p+p^2
\le
ap-ap^2.
$$

Move everything to the left:

$$
p^2+ap^2-2\hat p p-ap+\hat p^2\le0.
$$

Collect terms:

$$
\boxed{
(1+a)p^2-(2\hat p+a)p+\hat p^2\le0
}.
$$

The coefficient of $p^2$ is $1+a$. Call it $D$:

$$
\boxed{D=1+a=1+\frac{z^2}{n}}.
$$

The two contributions have specific origins:

| Contribution to D | Source |
|---|---|
| $1$ | The $p^2$ term from expanding $(\hat p-p)^2$ |
| $z^2/n$ | The additional $p^2$ term from expanding the sampling variance $p(1-p)$ and moving it to the left |

**Intuition:** $D$ is the quadratic coefficient that appears because the candidate probability affects both the distance from the observation and its sampling variance. At fixed confidence level, $z^2/n$ decreases as the sample size grows, so $D$ approaches 1.

## 4. Derive C by completing the square

The quadratic inequality is:

$$
Dp^2-(2\hat p+a)p+\hat p^2\le0.
$$

Divide by $D$, which is positive:

$$
p^2-\frac{2\hat p+a}{D}p+\frac{\hat p^2}{D}\le0.
$$

To complete the square, compare the first two terms with:

$$
(p-C)^2=p^2-2Cp+C^2.
$$

We need:

$$
2C=\frac{2\hat p+a}{D}.
$$

Therefore:

$$
C=\frac{2\hat p+a}{2D}
=\frac{\hat p+a/2}{D}.
$$

Substituting $a=z^2/n$:

$$
\boxed{C=\frac{\hat p+\frac{z^2}{2n}}{D}}.
$$

**This proves the center formula:** it is the value that converts the quadratic into a square centered at $C$.

Rewrite the inequality:

$$
(p-C)^2-C^2+\frac{\hat p^2}{D}\le0,
$$

or:

$$
\boxed{(p-C)^2\le C^2-\frac{\hat p^2}{D}}.
$$

This has the shape: squared distance from the center is at most the squared half-width.

## 5. Derive M by simplifying the squared half-width

Define:

$$
M^2=C^2-\frac{\hat p^2}{D}.
$$

Substitute $C=(\hat p+a/2)/D$:

$$
M^2
=\frac{(\hat p+a/2)^2}{D^2}
-\frac{\hat p^2}{D}.
$$

Put both terms over the same denominator:

$$
M^2
=\frac{(\hat p+a/2)^2-\hat p^2D}{D^2}.
$$

Expand the numerator, remembering that $D=1+a$:

$$
\begin{aligned}
(\hat p+a/2)^2-\hat p^2D
&=\hat p^2+a\hat p+\frac{a^2}{4}-\hat p^2(1+a)\\
&=\hat p^2+a\hat p+\frac{a^2}{4}-\hat p^2-a\hat p^2\\
&=a\hat p-a\hat p^2+\frac{a^2}{4}\\
&=a\hat p(1-\hat p)+\frac{a^2}{4}.
\end{aligned}
$$

Therefore:

$$
M^2=\frac{a\hat p(1-\hat p)+a^2/4}{D^2}.
$$

This is nonnegative because $a>0$ and $0\le\hat p\le1$. Taking the nonnegative square root gives:

$$
M=\frac{\sqrt{a\hat p(1-\hat p)+a^2/4}}{D}.
$$

Replace $a$ with $z^2/n$:

$$
M=\frac{
\sqrt{\frac{z^2}{n}\hat p(1-\hat p)+\frac{z^4}{4n^2}}
}{D}.
$$

Factor $z^2$ out of the square root. Since $z>0$:

$$
\boxed{
M=\frac{z}{D}
\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}
}.
$$

That proves the half-width formula. Finally:

$$
(p-C)^2\le M^2
\quad\Longleftrightarrow\quad
|p-C|\le M
\quad\Longleftrightarrow\quad
\boxed{C-M\le p\le C+M}.
$$

## 6. Understand why the center moves toward 50%

Multiply the numerator and denominator of $C$ by $n$:

$$
C=\frac{n\hat p+z^2/2}{n+z^2}.
$$

Separate the two contributions:

$$
\boxed{
C=\frac{n}{n+z^2}\hat p
+\frac{z^2}{n+z^2}\cdot\frac12
}.
$$

The weights are nonnegative and sum to 1. Therefore, **the Wilson center is a weighted average of the observed proportion and 50%.**

For $z=1.96$, so $z^2=3.8416$:

| Sample size n | Weight on observed rate | Weight on 50% |
|---:|---:|---:|
| 3 | 43.85% | 56.15% |
| 30 | 88.65% | 11.35% |
| 300 | 98.74% | 1.26% |

**Intuition:** with three observations, the interval center receives a substantial adjustment. With hundreds of observations, it stays close to the observed proportion. This weighting follows from solving the score inequality; no observations are added to the dataset.

The reported `positive_rate` remains $\hat p$. The adjusted center $C$ is used only to calculate the interval. The interval is symmetric around $C$ and can be asymmetric around $\hat p$.

## 7. Understand why the extra term inside M matters

The expression is:

$$
M=\frac{z}{D}\sqrt{
\underbrace{\frac{\hat p(1-\hat p)}{n}}_{\text{estimated sampling variance}}
+\underbrace{\frac{z^2}{4n^2}}_{\text{term from completing the square}}
}.
$$

Consider zero clicks, so $\hat p=0$. The first term becomes zero:

$$
\frac{\hat p(1-\hat p)}{n}=0.
$$

But observing zero clicks in a small sample still leaves uncertainty about the underlying probability. The additional term keeps the interval from collapsing.

Calculate the center:

$$
C=\frac{z^2}{2(n+z^2)}.
$$

Calculate the half-width:

$$
M=\frac{z}{D}\sqrt{\frac{z^2}{4n^2}}
=\frac{z^2}{2(n+z^2)}
=C.
$$

Consequently:

$$
\boxed{
\hat p=0
\quad\Longrightarrow\quad
[L,U]=\left[0,\frac{z^2}{n+z^2}\right]
}.
$$

For zero clicks from three observations, using $z=1.96$:

$$
[L,U]=\left[0,\frac{3.8416}{3+3.8416}\right]
\approx[0,\ 0.5615].
$$

**Intuition:** zero observed clicks from three rows leaves considerable uncertainty. As $n$ increases while the click count remains zero, the upper limit decreases.

The all-positive case is the reflected result:

$$
\boxed{
\hat p=1
\quad\Longrightarrow\quad
[L,U]=\left[\frac{n}{n+z^2},\ 1\right]
}.
$$

This follows by exchanging clicks and nonclicks, so their probabilities change from $p$ to $1-p$.

## 8. Prove that the interval stays between 0 and 1

We established:

$$
M^2=C^2-\frac{\hat p^2}{D}.
$$

Therefore:

$$
C^2-M^2=\frac{\hat p^2}{D}\ge0.
$$

Since $C\ge0$ and $M\ge0$, this implies $C\ge M$. Thus:

$$
\boxed{C-M\ge0}.
$$

For the upper endpoint:

$$
\begin{aligned}
(1-C)^2-M^2
&=1-2C+C^2-M^2\\
&=1-2C+\frac{\hat p^2}{D}\\
&=\frac{D-2\hat p-a+\hat p^2}{D}\\
&=\frac{1-2\hat p+\hat p^2}{D}\\
&=\frac{(1-\hat p)^2}{D}\\
&\ge0.
\end{aligned}
$$

The weighted-average representation in section 6 proves $0\le C\le1$. Thus $1-C\ge0$, and the last inequality implies $1-C\ge M$. Therefore:

$$
\boxed{C+M\le1}.
$$

Because $M\ge0$, the lower endpoint is at most the upper endpoint. In exact arithmetic:

$$
\boxed{0\le L\le U\le1}.
$$

The implementation's `np.clip(..., 0, 1)` also protects against tiny floating-point departures at the boundaries.

## 9. Check what happens with large samples

For fixed $z$, as $n$ grows:

$$
D=1+\frac{z^2}{n}\longrightarrow1.
$$

The difference between the center and the observed rate is:

$$
\begin{aligned}
C-\hat p
&=\frac{n\hat p+z^2/2}{n+z^2}-\hat p\\
&=\frac{z^2(1/2-\hat p)}{n+z^2}\\
&\longrightarrow0.
\end{aligned}
$$

When the observed proportion stays away from 0 and 1, the $1/n^2$ correction inside the square root becomes small relative to the $1/n$ term:

$$
M\approx z\sqrt{\frac{\hat p(1-\hat p)}{n}}.
$$

The Wilson interval then approaches:

$$
\hat p\pm z\sqrt{\frac{\hat p(1-\hat p)}{n}}.
$$

**Intuition:** the adjustments in $D$, $C$, and $M$ matter most when observations are few or the observed rate is near a boundary. For a fixed interior observed proportion, multiplying $n$ by four approximately halves the interval's width once the large-sample approximation is appropriate.

## 10. Work through two clicks from three rows

For a bucket with outcomes `[1, 0, 1]`:

$$
n=3,\qquad k=2,\qquad\hat p=\frac23,\qquad z=1.96.
$$

First square $z$:

$$
z^2=3.8416.
$$

Calculate the denominator:

$$
D=1+\frac{3.8416}{3}\approx2.280533.
$$

Calculate the center:

$$
C=\frac{\frac23+\frac{3.8416}{6}}{D}\approx0.573082.
$$

Calculate both terms inside the square root:

$$
\frac{\hat p(1-\hat p)}{n}
=\frac{\frac23\cdot\frac13}{3}
=\frac{2}{27}
\approx0.074074,
$$

$$
\frac{z^2}{4n^2}=\frac{3.8416}{36}\approx0.106711.
$$

Their sum and square root give:

$$
\sqrt{\frac{2}{27}+\frac{3.8416}{36}}\approx0.425188.
$$

Multiply by $z$ and divide by $D$:

$$
M=\frac{1.96}{D}\sqrt{\frac{2}{27}+\frac{3.8416}{36}}\approx0.365427.
$$

Finally:

$$
L=C-M\approx0.207655,\qquad
U=C+M\approx0.938510.
$$

The observed positive rate is **66.67%**, and the 95% Wilson interval is approximately **20.77% to 93.85%**. Intermediate values above are displayed with rounding; calculations use unrounded values.

The same observed rate can have much less uncertainty with more observations:

| Clicks | Rows | Observed rate | 95% Wilson interval, using z = 1.96 |
|---:|---:|---:|---|
| 2 | 3 | 66.67% | 20.77% to 93.85% |
| 200 | 300 | 66.67% | 61.15% to 71.76% |

The confidence level describes repeated-sampling performance: the procedure aims to produce intervals containing the underlying probability about 95% of the time. Actual Wilson coverage depends on $n$ and $p$. It does not assign a 95% posterior probability to the fixed parameter lying in this particular computed interval.

## 11. Match the derivation to the implementation

The calculation in [wilson()](../data_mining/statistics.py) follows these steps:

```python
# k and n can be arrays: one positive count and row count per bucket.
p = np.divide(k, n, out=np.full_like(n, np.nan), where=n > 0)

# D: common denominator
den = 1 + z * z / n

# C: interval center
center = (p + z * z / (2 * n)) / den

# M: interval half-width
half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den

lower = np.clip(center - half, 0, 1)
upper = np.clip(center + half, 0, 1)
```

| Mathematics | Code variable or report column |
|---|---|
| $k$ | `k`, from `positives` |
| $n$ | `n`, from `counts` / report `rows` |
| $\hat p$ | Local variable `p`; report `positive_rate` |
| $D$ | `den` |
| $C$ | `center` |
| $M$ | `half` |
| $L$ | `positive_rate_low_iid` |
| $U$ | `positive_rate_high_iid` |

The local code variable `p` represents the observed proportion $\hat p$, while the derivation uses $p$ for the unknown candidate probability. The function converts counts to floating-point arrays before this excerpt and suppresses expected divide/invalid warnings. With no observations ($n=0$), the helper returns undefined (`NaN`) endpoints; the proof assumes $n>0$.
