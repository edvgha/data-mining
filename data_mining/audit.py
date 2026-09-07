"""Usage: uv run -m data_mining dataset.parquet config.yaml

All statistics are exploratory. No training, splitting, or feature encoding
for a downstream model is performed. Analysis buckets never become model inputs.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mutual_info_score, roc_auc_score
import yaml

from .settings import EXTRA_DEFAULTS, METRICS, RULES, configure_decisions, enabled, filter_profile
from .decisions import decide


DEFAULTS = {
    **EXTRA_DEFAULTS,
    "output_dir": "report",
    "ignore": [],
    "time_column": None,
    "group_column": None,
    "time_frequency": "W",
    "strict_schema": True,
    "bins": 10,
    "max_categories": 20,
    "rare_min_count": 100,
    "min_cell_count": 200,
    "min_cell_events": 10,
    "sample_rows": 100_000,
    "seed": 42,
    "max_pairs": 500,
    "max_joint_cells": 20_000,
    "interactions": [],
    "max_time_periods": 120,
}


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys, which otherwise silently overwrite settings."""


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_config(path):
    raw = yaml.load(Path(path).read_text(), Loader=UniqueKeyLoader)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping.")
    unknown = set(raw) - set(DEFAULTS) - {"target", "features"}
    if unknown:
        raise ValueError(f"Unknown config settings: {sorted(unknown)}")
    cfg = {**DEFAULTS, **raw}
    if not isinstance(cfg.get("target"), str) or not cfg["target"]:
        raise ValueError("target must be a nonempty column name.")
    features = cfg.get("features")
    if not isinstance(features, dict) or not features:
        raise ValueError("features must map column names to numerical or categorical.")
    for name, kind in features.items():
        if not isinstance(name, str) or not name or kind not in {"numerical", "categorical"}:
            raise ValueError(f"Invalid feature declaration: {name!r}: {kind!r}")
    if cfg["target"] in features:
        raise ValueError("The target cannot also be an input feature.")
    if not isinstance(cfg["ignore"], list) or not all(isinstance(x, str) for x in cfg["ignore"]):
        raise ValueError("ignore must be a list of column names.")
    if len(set(cfg["ignore"])) != len(cfg["ignore"]):
        raise ValueError("ignore contains duplicate names.")
    for key in ["time_column", "group_column"]:
        if cfg[key] is not None and (not isinstance(cfg[key], str) or not cfg[key]):
            raise ValueError(f"{key} must be null or a nonempty column name.")
    context = {cfg["time_column"], cfg["group_column"]} - {None}
    if context & ({cfg["target"]} | set(features)):
        raise ValueError("Context columns must be separate from the target and features.")
    if cfg["time_column"] and cfg["time_column"] == cfg["group_column"]:
        raise ValueError("time_column and group_column must differ.")
    if set(cfg["ignore"]) & (set(features) | {cfg["target"]} | context):
        raise ValueError("An ignored column cannot also be a feature, target, or context column.")
    for key in ["bins", "max_categories", "rare_min_count", "min_cell_count",
                "min_cell_events", "sample_rows", "max_joint_cells",
                "max_time_periods"]:
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if cfg["bins"] < 2 or cfg["sample_rows"] < 10:
        raise ValueError("bins must be >= 2 and sample_rows must be >= 10.")
    for key in ["max_pairs", "seed"]:
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer.")
    if type(cfg["strict_schema"]) is not bool:
        raise ValueError("strict_schema must be true or false.")
    if not isinstance(cfg["output_dir"], str) or not cfg["output_dir"]:
        raise ValueError("output_dir must be a nonempty path.")
    if cfg["time_frequency"] not in {"D", "W", "M"}:
        raise ValueError("time_frequency must be D, W, or M.")
    if not isinstance(cfg["interactions"], list):
        raise ValueError("interactions must be a list of feature-name lists.")
    seen = set()
    for group in cfg["interactions"]:
        if (not isinstance(group, list) or not 2 <= len(group) <= 3
                or not all(isinstance(x, str) for x in group)):
            raise ValueError("Each interaction must contain two or three feature names.")
        if len(set(group)) != len(group) or not set(group) <= set(features):
            raise ValueError(f"Invalid interaction: {group}")
        key = tuple(sorted(group))
        if key in seen:
            raise ValueError(f"Duplicate interaction: {group}")
        seen.add(key)
    return configure_decisions(cfg)


def read_dataset(path, cfg):
    path = Path(path)
    if path.suffix.lower() != ".parquet" or not path.is_file():
        raise ValueError("dataset must be an existing .parquet file.")
    schema = pq.read_schema(path)
    columns = schema.names
    if len(columns) != len(set(columns)):
        raise ValueError("Parquet contains duplicate column names.")
    declared = set(cfg["features"]) | set(cfg["ignore"]) | {cfg["target"]}
    declared |= {cfg["time_column"], cfg["group_column"]} - {None}
    missing = declared - set(columns)
    if missing:
        raise ValueError(f"Configured columns absent from Parquet: {sorted(missing)}. "
                         "Column names are case-sensitive and must match the schema exactly.")
    extra = set(columns) - declared
    if extra and cfg["strict_schema"]:
        raise ValueError(f"Undeclared columns: {sorted(extra)}. Add them to features or ignore.")
    # Read only needed columns; ignored columns are checked in the Parquet schema.
    selected = [x for x in columns if x in declared and x not in cfg["ignore"]]
    df = pd.read_parquet(path, columns=selected).reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset is empty.")
    y = df[cfg["target"]]
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1, with no missing values.")
    if y.nunique() != 2:
        raise ValueError("Both target classes must be present for association analysis.")
    for name, kind in cfg["features"].items():
        s = df[name]
        if kind == "numerical" and (not pd.api.types.is_numeric_dtype(s)
                                    or pd.api.types.is_complex_dtype(s)):
            raise ValueError(f"{name}: numerical features require a real numeric dtype, got {s.dtype}.")
        if kind == "categorical" and not (
                isinstance(s.dtype, pd.CategoricalDtype)
                or pd.api.types.infer_dtype(s, skipna=True) in {
                    "string", "unicode", "bytes", "integer", "floating",
                    "mixed-integer-float", "boolean", "empty"}):
            raise ValueError(f"{name}: categorical values must be scalar strings, numbers, or booleans.")
    return df, sorted(extra), columns


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
        return {"mi_nats": 0.0, "cramers_v_corrected": np.nan,
                "chi2_p_iid": np.nan, "min_expected_count": np.nan}
    chi, p, _, expected = stats.chi2_contingency(a, correction=False)
    probs = a / n
    independent = np.outer(probs.sum(axis=1), probs.sum(axis=0))
    mask = probs > 0
    mi = float(np.sum(probs[mask] * np.log(probs[mask] / independent[mask])))
    phi = max(0.0, chi / n - (r - 1) * (c - 1) / (n - 1))
    den = min(r - (r - 1) ** 2 / (n - 1) - 1,
              c - (c - 1) ** 2 / (n - 1) - 1)
    v = math.sqrt(phi / den) if den > 0 else np.nan
    return {"mi_nats": max(0.0, mi), "cramers_v_corrected": v,
            "chi2_p_iid": float(p) if expected.min() >= 5 else np.nan,
            "min_expected_count": float(expected.min())}


def categorical_buckets(s, cfg):
    codes, values = pd.factorize(s, sort=False)
    counts = np.bincount(codes[codes >= 0], minlength=len(values))
    keep = np.argsort(-counts, kind="stable")
    keep = keep[counts[keep] >= cfg["rare_min_count"]][:cfg["max_categories"]]
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
        "feature": s.name, "type": "categorical", "rows": len(s),
        "missing_count": int(s.isna().sum()), "missing_fraction": float(s.isna().mean()),
        "unique_nonmissing": len(values), "unique_fraction_nonmissing": len(values) / total if total else np.nan,
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
        labels.update({i + 1: f"{'[' if i == 0 else '('}{edges[i]:.7g}, {edges[i+1]:.7g}]"
                       for i in range(len(edges) - 1)})
    pct = v.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1])
    iqr = pct.loc[.75] - pct.loc[.25]
    outliers = (v < pct.loc[.25] - 1.5 * iqr) | (v > pct.loc[.75] + 1.5 * iqr)
    info = {
        "feature": s.name, "type": "numerical", "rows": len(s),
        "finite_count": len(v), "missing_count": int(s.isna().sum()),
        "missing_fraction": float(s.isna().mean()),
        "nonfinite_nonmissing_count": int((~finite & ~np.isnan(x)).sum()),
        "unusable_fraction": float((~finite).mean()), "unique_nonmissing": unique,
        "dominant_fraction_nonmissing": float(counts.iloc[0] / len(v)) if len(v) else np.nan,
        "zero_fraction_finite": float((v == 0).mean()),
        "negative_fraction_finite": float((v < 0).mean()),
        "mean": v.mean(), "std": v.std(), "median": v.median(),
        "iqr": iqr, "mad_unscaled": (v - v.median()).abs().median(),
        "skewness": v.skew() if unique > 1 else np.nan,
        "excess_kurtosis": v.kurt() if unique > 1 else np.nan,
        "iqr_outlier_fraction": float(outliers.mean()) if len(v) else np.nan,
    }
    info.update({name: pct.loc[p] for name, p in zip(
        ["min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"],
        [0, .01, .05, .25, .5, .75, .95, .99, 1])})
    return b, labels, info, clean


def positive_rate_table(bucket_frame, y, cfg):
    frame = bucket_frame.copy()
    frame["__target__"] = np.asarray(y)
    keys = list(bucket_frame.columns)
    tab = frame.groupby(keys, observed=True, sort=True)["__target__"].agg(["size", "sum"])
    tab = tab.rename(columns={"size": "rows", "sum": "positives"}).reset_index()
    tab["negatives"] = tab["rows"] - tab["positives"]
    tab["positive_rate"] = tab["positives"] / tab["rows"]
    tab["lift"] = tab["positive_rate"] / float(np.mean(y))
    tab["positive_rate_low_iid"], tab["positive_rate_high_iid"] = wilson(tab["positives"], tab["rows"])
    tab["low_support"] = ((tab["rows"] < cfg["min_cell_count"])
                          | (tab["positives"] < cfg["min_cell_events"])
                          | (tab["negatives"] < cfg["min_cell_events"]))
    return tab


def safe_corr(x, y, method="pearson"):
    ok = np.isfinite(x) & np.isfinite(y)
    a, b = np.asarray(x)[ok], np.asarray(y)[ok]
    if len(a) < 3 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return np.nan, int(ok.sum())
    value = stats.spearmanr(a, b).statistic if method == "spearman" else stats.pearsonr(a, b).statistic
    return float(value), int(ok.sum())


def standardized_mean_difference(x, y):
    """d = (mean[label=1] - mean[label=0]) / pooled within-class sample SD.

    Undefined for fewer than 2 usable rows in either class or zero pooled SD.
    Hedges' g uses the conventional small-sample approximation J=1-3/(4df-1).
    """
    x, y = np.asarray(x), np.asarray(y)
    a, b = x[(y == 1) & np.isfinite(x)], x[(y == 0) & np.isfinite(x)]
    result = {"cohens_d": np.nan, "hedges_g": np.nan, "cohens_d_status": "insufficient_class_rows",
              "sample_rows_y1": len(a), "sample_rows_y0": len(b),
              "sample_mean_y1": float(a.mean()) if len(a) else np.nan,
              "sample_mean_y0": float(b.mean()) if len(b) else np.nan}
    if min(len(a), len(b)) < 2:
        return result
    degrees = len(a) + len(b) - 2
    pooled = np.sqrt(((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1)) / degrees)
    if not np.isfinite(pooled) or pooled == 0:
        result["cohens_d_status"] = "zero_or_nonfinite_pooled_sd"
        return result
    d = float((a.mean() - b.mean()) / pooled)
    result.update({"cohens_d": d, "hedges_g": d * (1 - 3 / (4 * degrees - 1)), "cohens_d_status": "computed"})
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
    row = {"features": " + ".join(names), "sample_rows": len(codes),
           "observed_sample_cells": len(codes.drop_duplicates()),
           "joint_mi_nats": joint, "best_single_mi_nats": max(single.values()),
           "gain_over_best_single_nats": max(0.0, joint - max(single.values()))}
    # I(feature; target | all other named features) = I(all;target) - I(others;target).
    for i, name in enumerate(names):
        row[f"feature_{i+1}"] = name
        row[f"conditional_mi_{i+1}_nats"] = max(0.0, joint - mi_for([c for c in names if c != name]))
    return row


def flags_for(info, cfg):
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
    if info["type"] == "categorical" and info["unique_nonmissing"] >= rules["high_cardinality"]["threshold"]:
        flags.append("high_cardinality")
    if info["type"] == "categorical" and info.get("unique_fraction_nonmissing", 0) >= .95:
        flags.append("near_unique_review_identifier")
    if info.get("pooled_row_fraction", 0) >= .5:
        flags.append("most_categories_pooled_signal_may_be_hidden")
    return flags


def plot_image(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=125, bbox_inches="tight")
    plt.close(fig)
    return '<img alt="Analysis chart" src="data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode() + '">'


def plot_positive_rate(tab, title, baseline):
    view = tab.head(32)
    positions = np.arange(len(view))
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].errorbar(positions, view["positive_rate"].to_numpy(), yerr=np.array([np.maximum(0, view["positive_rate"] - view["positive_rate_low_iid"]),
                          np.maximum(0, view["positive_rate_high_iid"] - view["positive_rate"])]), fmt="o", color="#2563eb", capsize=2)
    sparse = view["low_support"].to_numpy()
    axes[0].scatter(positions[sparse], view.loc[sparse, "positive_rate"], color="#d97706", zorder=4, label="Low support")
    axes[0].axhline(baseline, color="#64748b", linestyle="--")
    axes[0].set(title=title, ylabel="positive-class rate; row-IID 95% CI")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].bar(positions, view["rows"], color="#94a3b8")
    axes[1].set(ylabel="Rows", xticks=positions)
    axes[1].set_xticklabels(view.iloc[:, 0].astype(str), rotation=55, ha="right", fontsize=8)
    fig.tight_layout()
    return plot_image(fig)


def html_table(df, limit=100):
    if df.empty:
        return "<p>No eligible results.</p>"
    return '<div class="table">' + df.head(limit).to_html(index=False, escape=True, float_format=lambda x: f"{x:.6g}") + "</div>"


def save_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, (np.bool_,)):
            return bool(x)
        return x
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def analyze(dataset, config):
    """Public API: analyze('data.parquet', 'config.yaml') -> report Path."""
    print(f"Loading config: {Path(config).resolve()}", flush=True)
    cfg = load_config(config)
    root = Path(cfg["output_dir"])
    if not root.is_absolute():
        root = Path(config).resolve().parent / root
    root = root.resolve()
    print(f"Output directory: {root} (a fresh run_* folder is created when writing reports)", flush=True)
    print(f"Reading and validating Parquet: {Path(dataset).resolve()}", flush=True)
    df, undeclared, schema_columns = read_dataset(dataset, cfg)
    n = len(df)
    y = df[cfg["target"]].to_numpy(dtype=np.int8)
    features = cfg["features"]
    names = list(features)
    # Numeric internal names prevent user column names colliding with report fields.
    internal = {name: f"f{i:04d}" for i, name in enumerate(names)}
    rng = np.random.default_rng(cfg["seed"])
    sample_idx = np.sort(rng.choice(n, min(n, cfg["sample_rows"]), replace=False))
    ys = y[sample_idx]
    print(f"Loaded {n:,} rows; {len(features)} features; analysis sample: {len(sample_idx):,} rows.", flush=True)
    print("Profiling features and target associations...", flush=True)
    buckets, labels, profiles, numeric = {}, {}, [], {}
    association_rows, feature_tables, time_tables, bucket_dictionary = [], {}, {}, []
    for i, (name, kind) in enumerate(features.items(), 1):
        print(f"[{i}/{len(features)}] {name}", flush=True)
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
        if any(enabled(cfg, metric) for metric in ["mutual_information", "cramers_v_target", "chi_square"]):
            assoc = table_association(tab[["negatives", "positives"]])
            for metric, columns in {"mutual_information": ["mi_nats"], "cramers_v_target": ["cramers_v_corrected"],
                                    "chi_square": ["chi2_p_iid", "min_expected_count"]}.items():
                if enabled(cfg, metric):
                    stats_row.update({col: assoc[col] for col in columns})
        # When a grouping unit is supplied, row-level independence is not established.
        if cfg["group_column"] and enabled(cfg, "chi_square"):
            stats_row["chi2_p_iid"] = np.nan
        stats_row.update({"feature": name, "type": kind, "analysis_rows": n,
                          "observed_buckets": len(tab), "flags": info["flags"],
                          "sample_rows_numeric_target": len(sample_idx) if kind == "numerical" else np.nan})
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
                stats_row.update({"raw_numeric_auc": auc, "direction_free_auc": max(auc, 1-auc), "flipped_auc": 1-auc})
            if enabled(cfg, "ks"):
                stats_row["ks_statistic"] = float(stats.ks_2samp(x[good & (ys == 0)], x[good & (ys == 1)], method="asymp").statistic) if both else np.nan
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

    pair_rows = []
    pair_count = len(names) * (len(names) - 1) // 2
    pairs_limit = cfg["max_pairs"] if any(enabled(cfg, metric) for metric in METRICS["feature_pairs"]) else 0
    pairs_to_analyze = min(pair_count, pairs_limit)
    print(f"Analyzing feature pairs: {pairs_to_analyze}/{pair_count} pairs...", flush=True)
    for a, b in itertools.islice(itertools.combinations(names, 2), pairs_limit):
        ka, kb = internal[a], internal[b]
        row = {"feature_a": a, "feature_b": b, "type_a": features[a], "type_b": features[b],
               "sample_rows": len(sample_idx)}
        if features[a] == features[b] == "numerical":
            if enabled(cfg, "pearson"):
                row["pearson_r"], row["valid_rows"] = safe_corr(numeric[a], numeric[b])
            if enabled(cfg, "spearman"):
                row["spearman_r"], row["valid_rows"] = safe_corr(numeric[a], numeric[b], "spearman")
        elif features[a] == features[b] == "categorical" and enabled(cfg, "cramers_v_pairs"):
            if sampled[ka].nunique() * sampled[kb].nunique() > cfg["max_joint_cells"]:
                row["status"] = "skipped_cell_limit"
                row["cramers_v_corrected"] = np.nan
            else:
                table = pd.crosstab(sampled[ka], sampled[kb])
                row["cramers_v_corrected"] = table_association(table)["cramers_v_corrected"]
            row["valid_rows"] = len(sample_idx)
        elif features[a] != features[b] and enabled(cfg, "eta_squared"):
            cat, num = (a, b) if features[a] == "categorical" else (b, a)
            row["eta_squared"], row["valid_rows"] = correlation_ratio(sampled[internal[cat]], numeric[num])
        if enabled(cfg, "missingness_phi"):
            row["missingness_phi"], _ = safe_corr((sampled[ka] == 0).to_numpy().astype(float),
                                                 (sampled[kb] == 0).to_numpy().astype(float))
        pair_rows.append(row)
        if len(pair_rows) % 25 == 0 or len(pair_rows) == pairs_to_analyze:
            print(f"  Pairs completed: {len(pair_rows)}/{pairs_to_analyze}", flush=True)
    pairs_df = pd.DataFrame(pair_rows, columns=None if pair_rows else ["feature_a", "feature_b", "sample_rows"])

    joint_rows, joint_tables = [], {}
    print("Checking configured interactions...", flush=True)
    for i, group in enumerate(cfg["interactions"], 1):
        print(f"  Interaction [{i}/{len(cfg['interactions'])}]: {' + '.join(group)}", flush=True)
        if not enabled(cfg, "joint_positive_rate") and not enabled(cfg, "joint_information"):
            joint_rows.append({"features": " + ".join(group), "status": "disabled"})
            continue
        keys = [internal[x] for x in group]
        possible_cells = math.prod(codes[key].nunique() for key in keys)
        if possible_cells > cfg["max_joint_cells"]:
            joint_rows.append({"features": " + ".join(group), "status": "skipped_cell_limit",
                               "possible_cells": possible_cells})
            continue
        tab = positive_rate_table(codes[keys], y, cfg)
        row = joint_information(sampled[keys].rename(columns=dict(zip(keys, group))), ys) if enabled(cfg, "joint_information") else {"features": " + ".join(group)}
        row.update({"status": "computed", "possible_cells": possible_cells,
                    "observed_full_cells": len(tab),
                    "low_support_row_fraction": float(tab.loc[tab["low_support"], "rows"].sum() / n)})
        joint_rows.append(row)
        for i, key in enumerate(keys, 1):
            tab[key] = tab[key].map(labels[key])
            tab = tab.rename(columns={key: f"bucket_{i}"})
        if enabled(cfg, "joint_positive_rate"):
            joint_tables[" + ".join(group)] = tab
    joints_df = pd.DataFrame(joint_rows, columns=None if joint_rows else ["features", "status"])

    temporal_positive_rate = pd.DataFrame(columns=["period", "rows", "positives", "positive_rate"])
    temporal_features = []
    temporal_status = "not_requested"
    time_missing = 0
    print("Checking temporal analysis...", flush=True)
    if cfg["time_column"] and not any(enabled(cfg, metric) for metric in METRICS["temporal"]):
        temporal_status = "disabled"
    if cfg["time_column"] and any(enabled(cfg, metric) for metric in METRICS["temporal"]):
        raw_time = df[cfg["time_column"]]
        if pd.api.types.is_numeric_dtype(raw_time):
            raise ValueError("time_column must be a timestamp or ISO date string; numeric epoch units are ambiguous.")
        time = pd.to_datetime(raw_time, errors="coerce", utc=True, format="mixed")
        bad = raw_time.notna() & time.isna()
        if bad.any():
            raise ValueError(f"time_column has {bad.sum()} unparseable nonmissing timestamps.")
        time_missing = int(time.isna().sum())
        periods = time.dt.tz_convert(None).dt.to_period(cfg["time_frequency"]).astype("string").fillna("[missing]")
        if periods.nunique() <= cfg["max_time_periods"]:
            temporal_status = "computed"
            if enabled(cfg, "time_positive_rate"):
                temporal_positive_rate = positive_rate_table(pd.DataFrame({"period": periods}), y, cfg)
            for i, name in enumerate(names, 1):
                print(f"  Temporal [{i}/{len(names)}]: {name}", flush=True)
                key = internal[name]
                if enabled(cfg, "time_drift"):
                    ct = pd.crosstab(periods, codes[key])
                    ref = ct.sum(axis=0).to_numpy(dtype=float)
                    ref /= ref.sum()
                    for period, counts in ct.iterrows():
                        p = counts.to_numpy(dtype=float) / counts.sum()
                        temporal_features.append({"feature": name, "period": str(period),
                            "rows": int(counts.sum()), "js_divergence_nats_vs_all": float(jensenshannon(p, ref) ** 2)})
                if enabled(cfg, "time_feature_positive_rate"):
                    by_period = positive_rate_table(pd.DataFrame({"period": periods, "bucket": codes[key]}), y, cfg)
                    by_period["bucket"] = by_period["bucket"].map(labels[key])
                    time_tables[key] = by_period
        else:
            temporal_status = "skipped_period_limit_choose_coarser_frequency"

    group_summary = None
    if cfg["group_column"]:
        print(f"Summarizing groups: {cfg['group_column']}...", flush=True)
        gs = df.groupby(cfg["group_column"], dropna=False, observed=True)[cfg["target"]].agg(["size", "sum"])
        group_summary = {"column": cfg["group_column"], "groups_including_missing": len(gs),
                         "missing_group_rows": int(df[cfg["group_column"]].isna().sum()),
                         "median_rows_per_group": float(gs["size"].median()),
                         "max_rows_per_group": int(gs["size"].max()),
                         "fraction_groups_with_positives": float((gs["sum"] > 0).mean())}
    print("Evaluating feature selection rules...", flush=True)
    decisions, rule_evaluations = decide(profiles_df, target_df, pairs_df, cfg)
    low, high = wilson(y.sum(), n)
    summary = {"dataset": str(Path(dataset).resolve()), "target": cfg["target"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "rows": n,
        "positives": int(y.sum()), "negatives": int(n - y.sum()), "positive_rate": float(y.mean()),
        "positive_rate_low_iid": float(low), "positive_rate_high_iid": float(high), "features": len(features),
        "sample_rows": len(sample_idx), "sample_positives": int(ys.sum()), "seed": cfg["seed"],
        "sample_method": "uniform without replacement; original class balance preserved in expectation",
        "schema_columns": schema_columns, "ignored_columns": cfg["ignore"],
        "undeclared_columns_not_analyzed": undeclared,
        "pairs_possible": pair_count, "pairs_computed": len(pair_rows),
        "pair_limit_note": "First max_pairs pairs in config order; not chosen by target score.",
        "time_status": temporal_status, "time_missing_rows": time_missing,
        "group_summary": group_summary,
        "enabled_statistics": [metric for group in METRICS.values() for metric in group if enabled(cfg, metric)],
        "decision_counts": decisions["decision"].value_counts().to_dict(),
        "pvalue_policy": "chi_square disabled" if not enabled(cfg, "chi_square") else
                          "suppressed because group_column is set" if cfg["group_column"] else
                          "exploratory row-IID chi-square; suppressed if minimum expected count < 5",
        "limitations": [
            "All observed feature-target relationships use the supplied data; none measure generalization.",
            "Wilson intervals assume independent, unweighted rows; repeated sessions/users may make them too narrow.",
            "Empirical binned MI and joint/conditional MI are biased upward with sparse or numerous cells.",
            "MI gain over the best single feature is not a pure interaction or causal effect.",
            "Low marginal association cannot rule out predictive interactions.",
            "Keep/review/exclude are configured screening recommendations, not proof of a feature's value to XGBoost.",
            "Cohen's d uses mean(label=1)-mean(label=0); raw AUC uses larger feature values as stronger evidence for label=1.",
            "AUC below 0.5 is inverse ranking, not absence of signal. Direction-free AUC=max(AUC,1-AUC) is descriptive and uses the same sample to choose orientation.",
            "Temporal divergence compares each period to the entire supplied dataset; this is descriptive drift, not validation.",
            "Counts and positive-class rate are unweighted; supply original rows, not class-balanced/negative-sampled data for population positive-class rate.",
        ], "versions": {"pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__}}
    # A fresh run directory prevents stale files from an earlier config being mixed in.
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    out = root / run_id
    print(f"Writing report tables: {out}", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    (out / "feature_positive_rate").mkdir()
    (out / "joint_positive_rate").mkdir()
    (out / "time_feature_positive_rate").mkdir()
    for kind in ["numerical", "categorical"]:
        subset = profiles_df[profiles_df["type"] == kind]
        if not subset.empty:
            subset = subset.dropna(axis=1, how="all")
        subset.to_csv(out / f"{kind}_univariate.csv", index=False)
    target_df.to_csv(out / "feature_target.csv", index=False)
    decisions.to_csv(out / "feature_decisions.csv", index=False)
    rule_evaluations.to_csv(out / "rule_evaluations.csv", index=False)
    save_json(out / "feature_decisions.json", decisions.to_dict("records"))
    save_json(out / "selected_features.json", {
        "selected_features": decisions.loc[decisions["decision"] != "exclude", "feature"].tolist(),
        "excluded_features": decisions.loc[decisions["decision"] == "exclude", "feature"].tolist(),
        "review_features": decisions.loc[decisions["decision"] == "review", "feature"].tolist(),
        "policy": "Review features are retained. These are screening recommendations for later validation; the input dataset is not modified."})
    pairs_df.to_csv(out / "feature_pairs.csv", index=False)
    joints_df.to_csv(out / "joint_information.csv", index=False)
    pd.DataFrame(bucket_dictionary).to_csv(out / "bucket_dictionary.csv", index=False)
    temporal_positive_rate.to_csv(out / "time_positive_rate.csv", index=False)
    pd.DataFrame(temporal_features, columns=["feature", "period", "rows", "js_divergence_nats_vs_all"]).to_csv(out / "time_feature_drift.csv", index=False)
    for name in names:
        key = internal[name]
        if enabled(cfg, "positive_rate"):
            feature_tables[name].to_csv(out / "feature_positive_rate" / f"{key}.csv", index=False)
        if key in time_tables:
            time_tables[key].to_csv(out / "time_feature_positive_rate" / f"{key}.csv", index=False)
    joint_files = {}
    for i, (name, tab) in enumerate(joint_tables.items()):
        filename = f"joint_{i:03d}.csv"
        joint_files[name] = filename
        tab.to_csv(out / "joint_positive_rate" / filename, index=False)
    save_json(out / "summary.json", summary)
    save_json(out / "manifest.json", {"feature_files": {name: internal[name] + ".csv" for name in names},
                                      "joint_files": joint_files})
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    print("Rendering HTML report...", flush=True)
    notes = "".join(f"<li>{html.escape(x)}</li>" for x in summary["limitations"])
    decision_html = html_table(decisions, len(decisions))
    sections = [f"<h1>Data Mining</h1><p class='lead'>{n:,} rows · {y.sum():,} positives · "
                f"{y.mean():.3%} positive-class rate · {len(features)} features</p>",
                "<h2>Feature decisions</h2><p>" + " · ".join(f"{name}: {int(count)}" for name, count in summary["decision_counts"].items()) +
                "</p><p>Review features remain in selected_features.json. Every rule evaluation, including disabled or skipped rules, is saved in rule_evaluations.csv. "
                "Rules are configurable screening policy; the dataset is preserved.</p><div class='decisions'>" + decision_html + "</div>",
                "<details><summary>Enabled statistics and policy</summary><p>" + html.escape(", ".join(summary["enabled_statistics"])) +
                "</p><pre>" + html.escape(yaml.safe_dump({"rules": cfg["rules"], "expected_direction": cfg["expected_direction"]}, sort_keys=False)) + "</pre></details>",
                "<p>Analysis only. Expand a feature to inspect its full-data distribution and positive-class rate profile. "
                "The downloadable CSVs contain the complete tables. Feature-target comparison belongs to stage 3.</p>",
                f"<details open><summary>Interpretation and sampling</summary><ul>{notes}</ul>"
                f"<p>Full-data profiles, bucket counts and bucket-target statistics. "
                f"Numeric target metrics, feature pairs and joint information use a fixed uniform sample of {len(sample_idx):,} rows "
                f"({ys.sum():,} positives). Pair coverage: {len(pair_rows)} / {pair_count}. "
                f"P-value policy: {html.escape(summary['pvalue_policy'])}. "
                "BH q-values cover only the eligible feature-vs-target bucket chi-square tests, "
                "with their usual independence/positive-dependence assumptions.</p></details>",
                "<h2>1. Numerical univariate</h2>" + html_table(profiles_df[profiles_df["type"] == "numerical"].dropna(axis=1, how="all")),
                "<h2>2. Categorical univariate</h2>" + html_table(profiles_df[profiles_df["type"] == "categorical"].dropna(axis=1, how="all")),
                "<h2>3. Feature–target associations</h2><p>Sorted by empirical binned/pooled MI in nats when enabled, otherwise config order. "
                "This is an exploratory association ordering, not a validated feature ranking. "
                "Raw numeric AUC treats the column itself as a score; it misses nonmonotonic effects and is not a fitted model score.</p>" + html_table(target_df)]
    for i, name in enumerate(names, 1):
        if enabled(cfg, "positive_rate"):
            print(f"  Plot [{i}/{len(names)}]: {name}", flush=True)
            sections.append(f"<details><summary>{html.escape(name)} — {features[name]}</summary>" +
                            plot_positive_rate(feature_tables[name], name, float(y.mean())) + html_table(feature_tables[name]) + "</details>")
    sections.append("<h2>Feature redundancy</h2><p>Pairwise-complete numeric rows; categorical comparisons use "
                    "pooled categories and an explicit missing bucket. Measures have different meanings and should not "
                    "be compared as one common scale. Missingness phi compares missing/nonfinite indicators.</p>" + html_table(pairs_df, 500))
    sections.append("<h2>Joint and conditional analysis</h2><p>Only configured pairs/triples are examined. "
                    "Conditional MI is I(feature; target | other named features). Joint MI gains can reflect additive effects, "
                    "interactions, and sampling bias. Cells with few rows, positives or negatives are flagged.</p>" + html_table(joints_df))
    for name, tab in joint_tables.items():
        sections.append(f"<details><summary>{html.escape(name)}</summary>" + html_table(tab.sort_values("rows", ascending=False)) + "</details>")
    sections.append(f"<h2>Temporal and group context</h2><p>Time analysis: {html.escape(temporal_status)}. "
                    "JS divergence uses fixed full-data buckets and natural logs; range 0 to ln(2). "
                    "Bucket-by-period positive-class rate tables are in time_feature_positive_rate/.</p>" + html_table(temporal_positive_rate))
    if group_summary:
        sections.append("<pre>" + html.escape(json.dumps(group_summary, indent=2)) + "</pre>")
    sections.append("<h2>Feature review workflow</h2><ol><li>Confirm every candidate exists at prediction time and inspect potential target proxies.</li>"
                    "<li>Review unusable/constant columns, identifiers, sparse categories and missingness. Constant observed values with varying missingness can still carry signal.</li>"
                    "<li>Inspect effect sizes, support, nonlinear positive-class rate profiles, redundant feature groups and configured interactions.</li>"
                    "<li>During the later training stage, confirm selections with validation appropriate to deployment, log loss, average precision/PR-AUC and calibration; use permutation importance and feature-group ablations.</li></ol>")
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data Mining</title><style>
body{font:15px/1.55 system-ui,sans-serif;color:#172033;background:#f8fafc;max-width:1300px;margin:40px auto;padding:0 24px}
h1{font-size:34px;margin-bottom:6px}h2{margin-top:36px} .lead{font-size:21px;color:#2563eb}
details{background:white;border:1px solid #dce3ed;border-radius:10px;padding:16px;margin:14px 0}summary{cursor:pointer;font-weight:650}
.table{overflow:auto;background:white;border:1px solid #dce3ed;border-radius:8px;margin:16px 0;max-height:650px}
table{border-collapse:collapse;font-size:12px;white-space:nowrap;width:100%}td,th{padding:8px 11px;border:0;border-bottom:1px solid #e5e7eb;text-align:left}
th{background:#eaf0f9;position:sticky;top:0}tr:nth-child(even){background:#f8fafc}img{display:block;max-width:100%;margin:20px auto}pre{white-space:pre-wrap}
.decisions td:nth-child(6){white-space:normal;min-width:440px}.decisions .table{max-height:850px}
</style>""" + "".join(sections) + "</html>"
    (out / "report.html").write_text(document)
    # Markdown includes full reasons without requiring the HTML viewer.
    lines = ["# Data Mining", f"\n{n:,} rows; {int(y.sum()):,} positives; positive-class rate {y.mean():.3%}.\n",
             "## Feature decisions\n", "Review features are retained as candidates; exclusions follow configured screening policy.\n"]
    for row in decisions.to_dict("records"):
        lines.append(f"- **{row['feature']} — {row['decision'].upper()}**: {row['reason']}")
    lines += ["\n## Interpretation\n"] + [f"- {note}" for note in summary["limitations"]]
    lines += ["\n## Report files\n", "See feature_target.csv, feature_pairs.csv, joint_information.csv, "
              "rule_evaluations.csv and the other CSV tables for exact values. The HTML report includes enabled positive-class rate plots.\n"]
    (out / "report.md").write_text("\n".join(lines).rstrip() + "\n")
    print(f"Report: {out / 'report.html'}", flush=True)
    print("Analysis complete. Open the report.html file above in your browser.", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Local Parquet file")
    parser.add_argument("config", help="YAML config file")
    args = parser.parse_args()
    try:
        analyze(args.dataset, args.config)
    except (ValueError, TypeError, KeyError, OSError, yaml.YAMLError) as exc:
        parser.exit(2, f"Analysis failed: {exc}\n")


if __name__ == "__main__":
    main()
