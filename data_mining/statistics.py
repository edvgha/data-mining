"""Exploratory association measures and support-aware positive-rate tables.

These functions do not fit models or estimate held-out performance.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mutual_info_score


def wilson(positives, counts, z=1.959963984540054):
    """Unweighted, row-IID Wilson 95% intervals. Cluster dependence is not corrected."""
    n = np.asarray(counts, dtype=float)
    k = np.asarray(positives, dtype=float)
    p = np.divide(k, n, out=np.full_like(n, np.nan), where=n > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        den = 1 + z * z / n
        center = (p + z * z / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)


def bh_adjust(pvalues):
    """Benjamini-Hochberg across finite p-values in one stated test family."""
    p = np.asarray(pvalues, dtype=float)
    q = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
        q[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return q


def table_association(table):
    """Empirical MI (nats), corrected Cramer's V, and an eligible chi-square p."""
    a = np.asarray(table, dtype=float)
    a = a[a.sum(axis=1) > 0]
    a = a[:, a.sum(axis=0) > 0]
    n = a.sum()
    r, c = a.shape
    if n <= 1 or min(r, c) < 2:
        return {
            "mi_nats": 0.0,
            "cramers_v_corrected": np.nan,
            "chi2_p_iid": np.nan,
            "min_expected_count": np.nan,
        }
    chi, p, _, expected = stats.chi2_contingency(a, correction=False)
    probs = a / n
    independent = np.outer(probs.sum(axis=1), probs.sum(axis=0))
    mask = probs > 0
    mi = float(np.sum(probs[mask] * np.log(probs[mask] / independent[mask])))
    phi = max(0.0, chi / n - (r - 1) * (c - 1) / (n - 1))
    den = min(r - (r - 1) ** 2 / (n - 1) - 1, c - (c - 1) ** 2 / (n - 1) - 1)
    v = math.sqrt(phi / den) if den > 0 else np.nan
    return {
        "mi_nats": max(0.0, mi),
        "cramers_v_corrected": v,
        "chi2_p_iid": float(p) if expected.min() >= 5 else np.nan,
        "min_expected_count": float(expected.min()),
    }


def positive_rate_table(bucket_frame, y, cfg):
    """Aggregate full-data counts, rates, Wilson intervals, and support flags."""
    frame = bucket_frame.copy()
    frame["__target__"] = np.asarray(y)
    keys = list(bucket_frame.columns)
    tab = frame.groupby(keys, observed=True, sort=True)["__target__"].agg(["size", "sum"])
    tab = tab.rename(columns={"size": "rows", "sum": "positives"}).reset_index()
    tab["negatives"] = tab["rows"] - tab["positives"]
    tab["positive_rate"] = tab["positives"] / tab["rows"]
    tab["lift"] = tab["positive_rate"] / float(np.mean(y))
    tab["positive_rate_low_iid"], tab["positive_rate_high_iid"] = wilson(
        tab["positives"], tab["rows"]
    )
    tab["low_support"] = (
        (tab["rows"] < cfg["min_cell_count"])
        | (tab["positives"] < cfg["min_cell_events"])
        | (tab["negatives"] < cfg["min_cell_events"])
    )
    return tab


def safe_corr(x, y, method="pearson"):
    """Return correlation and paired finite-row count; degenerate inputs yield NaN."""
    ok = np.isfinite(x) & np.isfinite(y)
    a, b = np.asarray(x)[ok], np.asarray(y)[ok]
    if len(a) < 3 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return np.nan, int(ok.sum())
    value = (
        stats.spearmanr(a, b).statistic if method == "spearman" else stats.pearsonr(a, b).statistic
    )
    return float(value), int(ok.sum())


def standardized_mean_difference(x, y):
    """d = (mean[label=1] - mean[label=0]) / pooled within-class sample SD.

    Undefined for fewer than 2 usable rows in either class or zero pooled SD.
    Hedges' g uses the conventional small-sample approximation J=1-3/(4df-1).
    """
    x, y = np.asarray(x), np.asarray(y)
    a, b = x[(y == 1) & np.isfinite(x)], x[(y == 0) & np.isfinite(x)]
    result = {
        "cohens_d": np.nan,
        "hedges_g": np.nan,
        "cohens_d_status": "insufficient_class_rows",
        "sample_rows_y1": len(a),
        "sample_rows_y0": len(b),
        "sample_mean_y1": float(a.mean()) if len(a) else np.nan,
        "sample_mean_y0": float(b.mean()) if len(b) else np.nan,
    }
    if min(len(a), len(b)) < 2:
        return result
    degrees = len(a) + len(b) - 2
    pooled = np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / degrees
    )
    if not np.isfinite(pooled) or pooled == 0:
        result["cohens_d_status"] = "zero_or_nonfinite_pooled_sd"
        return result
    d = float((a.mean() - b.mean()) / pooled)
    result.update(
        {"cohens_d": d, "hedges_g": d * (1 - 3 / (4 * degrees - 1)), "cohens_d_status": "computed"}
    )
    return result


def correlation_ratio(codes, values):
    """Eta squared: fraction of numeric variance between pooled category groups."""
    ok = np.isfinite(values)
    v = np.asarray(values)[ok]
    g = np.asarray(codes)[ok]
    if len(v) < 2 or np.var(v) == 0:
        return np.nan, len(v)
    means = pd.DataFrame({"g": g, "v": v}).groupby("g")["v"].agg(["mean", "size"])
    between = (means["size"] * (means["mean"] - v.mean()) ** 2).sum()
    return float(np.clip(between / ((v - v.mean()) ** 2).sum(), 0, 1)), len(v)


def joint_information(codes, y):
    """Descriptive discrete MI. More cells induce greater finite-sample bias."""
    names = list(codes)

    def mi_for(cols):
        joint_codes, _ = pd.factorize(pd.MultiIndex.from_frame(codes[cols]), sort=False)
        return float(mutual_info_score(joint_codes, y))

    joint = mi_for(names)
    single = {name: mi_for([name]) for name in names}
    row = {
        "features": " + ".join(names),
        "sample_rows": len(codes),
        "observed_sample_cells": len(codes.drop_duplicates()),
        "joint_mi_nats": joint,
        "best_single_mi_nats": max(single.values()),
        "gain_over_best_single_nats": max(0.0, joint - max(single.values())),
    }
    # I(feature; target | all other named features) = I(all;target) - I(others;target).
    for i, name in enumerate(names):
        row[f"feature_{i + 1}"] = name
        row[f"conditional_mi_{i + 1}_nats"] = max(
            0.0, joint - mi_for([c for c in names if c != name])
        )
    return row
