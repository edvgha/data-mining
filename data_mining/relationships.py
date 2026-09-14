"""Feature-pair, configured-interaction, temporal, and group analysis stages."""

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from .profiling import FeatureAnalysis
from .settings import METRICS, enabled
from .statistics import (
    correlation_ratio,
    joint_information,
    positive_rate_table,
    safe_corr,
    table_association,
)


@dataclass
class InteractionAnalysis:
    """Configured joint information and full-data bucket support tables."""

    information: pd.DataFrame
    positive_rate_tables: dict[str, pd.DataFrame]


@dataclass
class TemporalAnalysis:
    """Descriptive time context, including disabled and skipped statuses."""

    positive_rates: pd.DataFrame
    drift: pd.DataFrame
    feature_positive_rates: dict[str, pd.DataFrame]
    status: str
    missing_rows: int


def analyze_pairs(analysis: FeatureAnalysis, cfg: dict) -> pd.DataFrame:
    """Compare the first configured feature pairs using the shared sample."""
    names = list(cfg["features"])
    features = cfg["features"]
    internal = analysis.internal_names
    sample_idx = analysis.sample_indices
    sampled = analysis.sampled_buckets
    numeric = analysis.numeric_sample
    pair_rows = []
    pair_count = len(names) * (len(names) - 1) // 2
    pairs_limit = (
        cfg["max_pairs"] if any(enabled(cfg, metric) for metric in METRICS["feature_pairs"]) else 0
    )
    pairs_to_analyze = min(pair_count, pairs_limit)
    print(f"Analyzing feature pairs: {pairs_to_analyze}/{pair_count} pairs...", flush=True)
    for a, b in itertools.islice(itertools.combinations(names, 2), pairs_limit):
        ka, kb = internal[a], internal[b]
        row = {
            "feature_a": a,
            "feature_b": b,
            "type_a": features[a],
            "type_b": features[b],
            "sample_rows": len(sample_idx),
        }
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
            row["eta_squared"], row["valid_rows"] = correlation_ratio(
                sampled[internal[cat]], numeric[num]
            )
        if enabled(cfg, "missingness_phi"):
            row["missingness_phi"], _ = safe_corr(
                (sampled[ka] == 0).to_numpy().astype(float),
                (sampled[kb] == 0).to_numpy().astype(float),
            )
        pair_rows.append(row)
        if len(pair_rows) % 25 == 0 or len(pair_rows) == pairs_to_analyze:
            print(f"  Pairs completed: {len(pair_rows)}/{pairs_to_analyze}", flush=True)
    pairs_df = pd.DataFrame(
        pair_rows, columns=None if pair_rows else ["feature_a", "feature_b", "sample_rows"]
    )

    return pairs_df


def analyze_interactions(analysis: FeatureAnalysis, cfg: dict) -> InteractionAnalysis:
    """Measure only configured pairs/triples, subject to the joint-cell limit."""
    codes = analysis.bucket_codes
    internal = analysis.internal_names
    labels = analysis.labels
    sampled = analysis.sampled_buckets
    y, ys = analysis.target, analysis.sampled_target
    n = len(y)
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
            joint_rows.append(
                {
                    "features": " + ".join(group),
                    "status": "skipped_cell_limit",
                    "possible_cells": possible_cells,
                }
            )
            continue
        tab = positive_rate_table(codes[keys], y, cfg)
        row = (
            joint_information(sampled[keys].rename(columns=dict(zip(keys, group))), ys)
            if enabled(cfg, "joint_information")
            else {"features": " + ".join(group)}
        )
        row.update(
            {
                "status": "computed",
                "possible_cells": possible_cells,
                "observed_full_cells": len(tab),
                "low_support_row_fraction": float(tab.loc[tab["low_support"], "rows"].sum() / n),
            }
        )
        joint_rows.append(row)
        for i, key in enumerate(keys, 1):
            tab[key] = tab[key].map(labels[key])
            tab = tab.rename(columns={key: f"bucket_{i}"})
        if enabled(cfg, "joint_positive_rate"):
            joint_tables[" + ".join(group)] = tab
    joints_df = pd.DataFrame(joint_rows, columns=None if joint_rows else ["features", "status"])

    return InteractionAnalysis(information=joints_df, positive_rate_tables=joint_tables)


def analyze_temporal(df: pd.DataFrame, analysis: FeatureAnalysis, cfg: dict) -> TemporalAnalysis:
    """Compare periods using fixed full-data buckets; this is not validation."""
    names = list(cfg["features"])
    codes = analysis.bucket_codes
    internal = analysis.internal_names
    labels = analysis.labels
    y = analysis.target
    time_tables = {}
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
            raise ValueError(
                "time_column must be a timestamp or ISO date string; numeric epoch units are ambiguous."
            )
        time = pd.to_datetime(raw_time, errors="coerce", utc=True, format="mixed")
        bad = raw_time.notna() & time.isna()
        if bad.any():
            raise ValueError(f"time_column has {bad.sum()} unparseable nonmissing timestamps.")
        time_missing = int(time.isna().sum())
        periods = (
            time.dt.tz_convert(None)
            .dt.to_period(cfg["time_frequency"])
            .astype("string")
            .fillna("[missing]")
        )
        if periods.nunique() <= cfg["max_time_periods"]:
            temporal_status = "computed"
            if enabled(cfg, "time_positive_rate"):
                temporal_positive_rate = positive_rate_table(
                    pd.DataFrame({"period": periods}), y, cfg
                )
            for i, name in enumerate(names, 1):
                print(f"  Temporal [{i}/{len(names)}]: {name}", flush=True)
                key = internal[name]
                if enabled(cfg, "time_drift"):
                    ct = pd.crosstab(periods, codes[key])
                    ref = ct.sum(axis=0).to_numpy(dtype=float)
                    ref /= ref.sum()
                    for period, counts in ct.iterrows():
                        p = counts.to_numpy(dtype=float) / counts.sum()
                        temporal_features.append(
                            {
                                "feature": name,
                                "period": str(period),
                                "rows": int(counts.sum()),
                                "js_divergence_nats_vs_all": float(jensenshannon(p, ref) ** 2),
                            }
                        )
                if enabled(cfg, "time_feature_positive_rate"):
                    by_period = positive_rate_table(
                        pd.DataFrame({"period": periods, "bucket": codes[key]}), y, cfg
                    )
                    by_period["bucket"] = by_period["bucket"].map(labels[key])
                    time_tables[key] = by_period
        else:
            temporal_status = "skipped_period_limit_choose_coarser_frequency"

    return TemporalAnalysis(
        positive_rates=temporal_positive_rate,
        drift=pd.DataFrame(
            temporal_features, columns=["feature", "period", "rows", "js_divergence_nats_vs_all"]
        ),
        feature_positive_rates=time_tables,
        status=temporal_status,
        missing_rows=time_missing,
    )


def summarize_groups(df: pd.DataFrame, cfg: dict) -> dict | None:
    """Describe repeated-observation groups, including missing group values."""
    group_summary = None
    if cfg["group_column"]:
        print(f"Summarizing groups: {cfg['group_column']}...", flush=True)
        gs = df.groupby(cfg["group_column"], dropna=False, observed=True)[cfg["target"]].agg(
            ["size", "sum"]
        )
        group_summary = {
            "column": cfg["group_column"],
            "groups_including_missing": len(gs),
            "missing_group_rows": int(df[cfg["group_column"]].isna().sum()),
            "median_rows_per_group": float(gs["size"].median()),
            "max_rows_per_group": int(gs["size"].max()),
            "fraction_groups_with_positives": float((gs["sum"] > 0).mean()),
        }
    return group_summary
