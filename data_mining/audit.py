"""Usage: uv run -m data_mining dataset.parquet config.yaml

All statistics are exploratory. No training, splitting, or feature encoding
for a downstream model is performed. Analysis buckets never become model inputs.
"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy
import yaml

from run_logging import (
    add_logging_arguments,
    create_run_directory,
    exit_with_error,
    logging_settings,
    run_logging,
)

from .config import DEFAULTS, UniqueKeyLoader, load_config
from .dataset import dataframe_from_arrow, read_dataset
from .decisions import decide
from .profiling import (
    FeatureAnalysis,
    categorical_buckets,
    flags_for,
    numerical_buckets,
    profile_features,
)
from .relationships import (
    TemporalAnalysis,
    analyze_interactions,
    analyze_pairs,
    analyze_temporal,
    summarize_groups,
)
from .report import (
    AuditReport,
    html_table,
    plot_image,
    plot_positive_rate,
    save_json,
    write_report,
)
from .settings import METRICS, enabled
from .statistics import (
    bh_adjust,
    correlation_ratio,
    joint_information,
    positive_rate_table,
    safe_corr,
    standardized_mean_difference,
    table_association,
    wilson,
)

logger = logging.getLogger(__name__)

# Preserve helper imports used by existing callers; implementations live in the
# focused modules above. The package-level API remains `from data_mining import audit`.
__all__ = [
    "DEFAULTS",
    "UniqueKeyLoader",
    "load_config",
    "dataframe_from_arrow",
    "read_dataset",
    "categorical_buckets",
    "numerical_buckets",
    "flags_for",
    "wilson",
    "bh_adjust",
    "table_association",
    "positive_rate_table",
    "safe_corr",
    "standardized_mean_difference",
    "correlation_ratio",
    "joint_information",
    "plot_image",
    "plot_positive_rate",
    "html_table",
    "save_json",
    "analyze",
    "main",
]


def analyze(
    dataset: str | Path,
    config: str | Path,
    *,
    diagnostics: bool = False,
    log_level: str | None = None,
    file_log_level: str | None = None,
) -> Path:
    """Load inputs, run exploratory analysis stages, apply decisions, and write reports."""
    cfg = load_config(config)
    cfg["logging"] = logging_settings(
        cfg["logging"], log_level=log_level, file_log_level=file_log_level
    )
    root = Path(cfg["output_dir"])
    if not root.is_absolute():
        root = Path(config).resolve().parent / root
    root = root.resolve()
    out = create_run_directory(root)
    with run_logging(out, cfg["logging"], "data_mining") as log_stream:
        logger.info("Loaded config: %s", Path(config).resolve())
        logger.info("Output directory: %s", out)
        (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        logger.info("Reading and validating Parquet: %s", Path(dataset).resolve())
        versions = logger.info if diagnostics else logger.debug
        versions(
            "Runtime: Python %s; %s %s; pandas %s; PyArrow %s.",
            platform.python_version(),
            platform.system(),
            platform.machine(),
            pd.__version__,
            pa.__version__,
        )
        if diagnostics:
            logger.info(
                "Diagnostics: Python stacks will be saved to run.log every 30 seconds "
                "until input validation completes."
            )
            faulthandler.dump_traceback_later(30, repeat=True, file=log_stream)
        try:
            df, undeclared, schema_columns = read_dataset(dataset, cfg)
        finally:
            if diagnostics:
                faulthandler.cancel_dump_traceback_later()
        features = profile_features(df, cfg)
        pairs = analyze_pairs(features, cfg)
        interactions = analyze_interactions(features, cfg)
        temporal = analyze_temporal(df, features, cfg)
        group_summary = summarize_groups(df, cfg)

        logger.info("Evaluating feature selection rules...")
        decisions, rule_evaluations = decide(
            features.profiles, features.target_associations, pairs, cfg
        )
        summary = _build_summary(
            dataset=dataset,
            cfg=cfg,
            analysis=features,
            pairs=pairs,
            temporal=temporal,
            group_summary=group_summary,
            decisions=decisions,
            undeclared=undeclared,
            schema_columns=schema_columns,
        )
        return write_report(
            root,
            AuditReport(
                config=cfg,
                summary=summary,
                features=features,
                pairs=pairs,
                interactions=interactions,
                temporal=temporal,
                decisions=decisions,
                rule_evaluations=rule_evaluations,
            ),
            run_dir=out,
        )


def _build_summary(
    *,
    dataset,
    cfg: dict,
    analysis: FeatureAnalysis,
    pairs: pd.DataFrame,
    temporal: TemporalAnalysis,
    group_summary: dict | None,
    decisions: pd.DataFrame,
    undeclared: list[str],
    schema_columns: list[str],
) -> dict:
    """Collect provenance, coverage, interpretation limits, and decision counts."""
    low, high = wilson(analysis.target.sum(), len(analysis.target))
    summary = {
        "dataset": str(Path(dataset).resolve()),
        "target": cfg["target"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(analysis.target),
        "positives": int(analysis.target.sum()),
        "negatives": int(len(analysis.target) - analysis.target.sum()),
        "positive_rate": float(analysis.target.mean()),
        "positive_rate_low_iid": float(low),
        "positive_rate_high_iid": float(high),
        "features": len(cfg["features"]),
        "sample_rows": len(analysis.sample_indices),
        "sample_positives": int(analysis.sampled_target.sum()),
        "seed": cfg["seed"],
        "sample_method": "uniform without replacement; original class balance preserved in expectation",
        "schema_columns": schema_columns,
        "ignored_columns": cfg["ignore"],
        "undeclared_columns_not_analyzed": undeclared,
        "pairs_possible": len(cfg["features"]) * (len(cfg["features"]) - 1) // 2,
        "pairs_computed": len(pairs),
        "pair_limit_note": "First max_pairs pairs in config order; not chosen by target score.",
        "time_status": temporal.status,
        "time_missing_rows": temporal.missing_rows,
        "group_summary": group_summary,
        "enabled_statistics": [
            metric for group in METRICS.values() for metric in group if enabled(cfg, metric)
        ],
        "decision_counts": decisions["decision"].value_counts().to_dict(),
        "pvalue_policy": "chi_square disabled"
        if not enabled(cfg, "chi_square")
        else "suppressed because group_column is set"
        if cfg["group_column"]
        else "exploratory row-IID chi-square; suppressed if minimum expected count < 5",
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
        ],
        "versions": {"pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Local Parquet file")
    parser.add_argument("config", help="YAML config file")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Log runtime versions and save Python stacks to run.log every 30 seconds during input loading",
    )
    add_logging_arguments(parser)
    args = parser.parse_args()
    try:
        analyze(
            args.dataset,
            args.config,
            diagnostics=args.diagnostics,
            log_level=args.log_level,
            file_log_level=args.file_log_level,
        )
    except KeyboardInterrupt:
        parser.exit(130)
    except Exception as exc:
        exit_with_error(parser, exc, "Analysis failed")


if __name__ == "__main__":
    main()
