"""Full-data feature distributions, analysis buckets, and target associations."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from .settings import RULES, enabled, filter_profile
from .statistics import (
    bh_adjust,
    positive_rate_table,
    safe_corr,
    standardized_mean_difference,
    table_association,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureAnalysis:
    """Full-data profiles and buckets plus a fixed sample for associations.

    Bucket keys are internal IDs; display labels never become model inputs.
    Numeric sample arrays and sampled bucket rows share sample_indices order.
    """

    target: np.ndarray
    sample_indices: np.ndarray
    internal_names: dict[str, str]
    labels: dict[str, dict[int, str]]
    numeric_sample: dict[str, np.ndarray]
    bucket_codes: pd.DataFrame
    profiles: pd.DataFrame
    target_associations: pd.DataFrame
    positive_rate_tables: dict[str, pd.DataFrame]
    bucket_dictionary: list[dict]

    sampled_target: np.ndarray
    sampled_buckets: pd.DataFrame


def categorical_buckets(s, cfg):
    """Pool rare levels while keeping missing and literal category values distinct."""
    codes, values = pd.factorize(s, sort=False)
    counts = np.bincount(codes[codes >= 0], minlength=len(values))
    keep = np.argsort(-counts, kind="stable")
    keep = keep[counts[keep] >= cfg["rare_min_count"]][: cfg["max_categories"]]
    # 0 = missing, 1 = pooled other, 2+ = retained original values.
    mapping = np.ones(len(values), dtype=np.int32)
    mapping[keep] = np.arange(2, 2 + len(keep), dtype=np.int32)
    buckets = np.zeros(len(s), dtype=np.int32)
    valid = codes >= 0
    buckets[valid] = mapping[codes[valid]]
    labels = {0: "[missing]", 1: f"[other: {len(values) - len(keep)} pooled levels]"}
    labels.update({i + 2: f"value: {values[k]}" for i, k in enumerate(keep)})
    total = counts.sum()
    props = counts / total if total else np.array([])
    entropy = float(stats.entropy(props)) if total else np.nan
    info = {
        "feature": s.name,
        "type": "categorical",
        "rows": len(s),
        "missing_count": int(s.isna().sum()),
        "missing_fraction": float(s.isna().mean()),
        "unique_nonmissing": len(values),
        "unique_fraction_nonmissing": len(values) / total if total else np.nan,
        "dominant_fraction_nonmissing": float(props.max()) if total else np.nan,
        "entropy_nats_nonmissing": entropy,
        "effective_levels_nonmissing": float(np.exp(entropy)) if total else np.nan,
        "singleton_levels": int((counts == 1).sum()),
        "rare_levels": int((counts < cfg["rare_min_count"]).sum()),
        "rare_row_fraction": float(counts[counts < cfg["rare_min_count"]].sum() / len(s)),
        "pooled_row_fraction": float((buckets == 1).mean()),
        "retained_levels": len(keep),
    }
    return buckets, labels, info


def numerical_buckets(s, cfg):
    """Describe finite values and assign discrete or quantile analysis buckets."""
    x = s.to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(x)
    clean = pd.Series(np.where(finite, x, np.nan), name=s.name)
    v = clean.dropna()
    counts = v.value_counts()
    unique = len(counts)
    b = np.zeros(len(s), dtype=np.int32)
    labels = {0: "[missing or nonfinite]"}
    if unique and unique <= cfg["bins"]:
        vals = np.sort(v.unique())
        b[finite] = np.searchsorted(vals, x[finite]) + 1
        labels.update({i + 1: f"value: {value:g}" for i, value in enumerate(vals)})
    elif unique:
        q, edges = pd.qcut(v, q=cfg["bins"], labels=False, retbins=True, duplicates="drop")
        b[finite] = q.to_numpy(dtype=np.int32) + 1
        labels.update(
            {
                i + 1: f"{'[' if i == 0 else '('}{edges[i]:.7g}, {edges[i + 1]:.7g}]"
                for i in range(len(edges) - 1)
            }
        )
    pct = v.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
    iqr = pct.loc[0.75] - pct.loc[0.25]
    outliers = (v < pct.loc[0.25] - 1.5 * iqr) | (v > pct.loc[0.75] + 1.5 * iqr)
    info = {
        "feature": s.name,
        "type": "numerical",
        "rows": len(s),
        "finite_count": len(v),
        "missing_count": int(s.isna().sum()),
        "missing_fraction": float(s.isna().mean()),
        "nonfinite_nonmissing_count": int((~finite & ~np.isnan(x)).sum()),
        "unusable_fraction": float((~finite).mean()),
        "unique_nonmissing": unique,
        "dominant_fraction_nonmissing": float(counts.iloc[0] / len(v)) if len(v) else np.nan,
        "zero_fraction_finite": float((v == 0).mean()),
        "negative_fraction_finite": float((v < 0).mean()),
        "mean": v.mean(),
        "std": v.std(),
        "median": v.median(),
        "iqr": iqr,
        "mad_unscaled": (v - v.median()).abs().median(),
        "skewness": v.skew() if unique > 1 else np.nan,
        "excess_kurtosis": v.kurt() if unique > 1 else np.nan,
        "iqr_outlier_fraction": float(outliers.mean()) if len(v) else np.nan,
    }
    info.update(
        {
            name: pct.loc[p]
            for name, p in zip(
                ["min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"],
                [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1],
            )
        }
    )
    return b, labels, info, clean


def flags_for(info, cfg):
    """Label distribution concerns using the configured screening thresholds."""
    flags = []
    rules = {**RULES, **cfg.get("rules", {})}
    usable = info.get("finite_count", info["rows"] - info["missing_count"])
    missing = info.get("unusable_fraction", info["missing_fraction"])
    if usable == 0:
        flags.append("all_missing_or_nonfinite")
    elif info["unique_nonmissing"] == 1:
        flags.append("constant_observed_values_with_missingness" if missing > 0 else "constant")
    if info.get("dominant_fraction_nonmissing", 0) >= rules["near_constant"]["threshold"]:
        flags.append("near_constant_observed_values")
    if missing >= rules["missingness"]["threshold"]:
        flags.append("high_missingness")
    if info.get("nonfinite_nonmissing_count", 0):
        flags.append("contains_infinity")
    if (
        info["type"] == "categorical"
        and info["unique_nonmissing"] >= rules["high_cardinality"]["threshold"]
    ):
        flags.append("high_cardinality")
    if info["type"] == "categorical" and info.get("unique_fraction_nonmissing", 0) >= 0.95:
        flags.append("near_unique_review_identifier")
    if info.get("pooled_row_fraction", 0) >= 0.5:
        flags.append("most_categories_pooled_signal_may_be_hidden")
    return flags


def profile_features(df: pd.DataFrame, cfg: dict) -> FeatureAnalysis:
    """Profile all rows and use one reproducible sample for numeric target metrics."""
    n = len(df)
    y = df[cfg["target"]].to_numpy(dtype=np.int8)
    features = cfg["features"]
    names = list(features)
    # Numeric internal names prevent user column names colliding with report fields.
    internal = {name: f"f{i:04d}" for i, name in enumerate(names)}
    rng = np.random.default_rng(cfg["seed"])
    sample_idx = np.sort(rng.choice(n, min(n, cfg["sample_rows"]), replace=False))
    ys = y[sample_idx]
    logger.info(
        "Loaded %s rows; %s features; analysis sample: %s rows.",
        n,
        len(features),
        len(sample_idx),
    )
    logger.info("Profiling features and target associations...")
    buckets, labels, profiles, numeric = {}, {}, [], {}
    association_rows, feature_tables, bucket_dictionary = [], {}, []
    for i, (name, kind) in enumerate(features.items(), 1):
        logger.debug("Feature [%s/%s]: %s", i, len(features), name)
        if kind == "numerical":
            b, lab, info, clean = numerical_buckets(df[name], cfg)
            numeric[name] = clean.to_numpy()[sample_idx]
        else:
            b, lab, info = categorical_buckets(df[name], cfg)
        info["source_dtype"] = str(df[name].dtype)
        info["flags"] = "; ".join(flags_for(info, cfg))
        profiles.append(filter_profile(info, cfg))
        key = internal[name]
        buckets[key], labels[key] = b, lab
        for code, label in lab.items():
            bucket_dictionary.append({"feature": name, "bucket_code": code, "bucket_label": label})
        tab = positive_rate_table(pd.DataFrame({"bucket": b}), y, cfg)
        stats_row = {}
        if any(
            enabled(cfg, metric)
            for metric in ["mutual_information", "cramers_v_target", "chi_square"]
        ):
            assoc = table_association(tab[["negatives", "positives"]])
            for metric, columns in {
                "mutual_information": ["mi_nats"],
                "cramers_v_target": ["cramers_v_corrected"],
                "chi_square": ["chi2_p_iid", "min_expected_count"],
            }.items():
                if enabled(cfg, metric):
                    stats_row.update({col: assoc[col] for col in columns})
        # When a grouping unit is supplied, row-level independence is not established.
        if cfg["group_column"] and enabled(cfg, "chi_square"):
            stats_row["chi2_p_iid"] = np.nan
        stats_row.update(
            {
                "feature": name,
                "type": kind,
                "analysis_rows": n,
                "observed_buckets": len(tab),
                "flags": info["flags"],
                "sample_rows_numeric_target": len(sample_idx) if kind == "numerical" else np.nan,
            }
        )
        if kind == "numerical":
            x = numeric[name]
            good = np.isfinite(x)
            stats_row["sample_rows_numeric_target"] = int(good.sum())
            if enabled(cfg, "point_biserial"):
                stats_row["point_biserial_r"], _ = safe_corr(x, ys)
            if enabled(cfg, "spearman_target"):
                stats_row["spearman_target"], _ = safe_corr(x, ys, "spearman")
            both = good.sum() and np.unique(ys[good]).size == 2
            if enabled(cfg, "auc"):
                auc = float(roc_auc_score(ys[good], x[good])) if both else np.nan
                stats_row.update(
                    {
                        "raw_numeric_auc": auc,
                        "direction_free_auc": max(auc, 1 - auc),
                        "flipped_auc": 1 - auc,
                    }
                )
            if enabled(cfg, "ks"):
                stats_row["ks_statistic"] = (
                    float(
                        stats.ks_2samp(
                            x[good & (ys == 0)], x[good & (ys == 1)], method="asymp"
                        ).statistic
                    )
                    if both
                    else np.nan
                )
            if enabled(cfg, "cohens_d") or enabled(cfg, "hedges_g"):
                d_stats = standardized_mean_difference(x, ys)
                for metric in ["cohens_d", "hedges_g"]:
                    if not enabled(cfg, metric):
                        d_stats.pop(metric)
                stats_row.update(d_stats)
        association_rows.append(stats_row)
        tab["bucket"] = tab["bucket"].map(lab)
        feature_tables[name] = tab
    codes = pd.DataFrame(buckets)
    sampled = codes.iloc[sample_idx].reset_index(drop=True)
    profiles_df = pd.DataFrame(profiles)
    target_df = pd.DataFrame(association_rows)
    if enabled(cfg, "chi_square"):
        target_df["chi2_q_bh_iid"] = bh_adjust(target_df["chi2_p_iid"])
    if enabled(cfg, "mutual_information"):
        target_df = target_df.sort_values("mi_nats", ascending=False, kind="stable")

    return FeatureAnalysis(
        target=y,
        sample_indices=sample_idx,
        internal_names=internal,
        sampled_target=ys,
        sampled_buckets=sampled,
        labels=labels,
        numeric_sample=numeric,
        bucket_codes=codes,
        profiles=profiles_df,
        target_associations=target_df,
        positive_rate_tables=feature_tables,
        bucket_dictionary=bucket_dictionary,
    )
