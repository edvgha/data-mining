"""Serialize audit results and render HTML/Markdown without recomputing statistics."""

import base64
import html
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .profiling import FeatureAnalysis
from .relationships import InteractionAnalysis, TemporalAnalysis
from .settings import enabled

HTML_HEADER = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data Mining</title><style>
body{font:15px/1.55 system-ui,sans-serif;color:#172033;background:#f8fafc;max-width:1300px;margin:40px auto;padding:0 24px}
h1{font-size:34px;margin-bottom:6px}h2{margin-top:36px} .lead{font-size:21px;color:#2563eb}
details{background:white;border:1px solid #dce3ed;border-radius:10px;padding:16px;margin:14px 0}summary{cursor:pointer;font-weight:650}
.table{overflow:auto;background:white;border:1px solid #dce3ed;border-radius:8px;margin:16px 0;max-height:650px}
table{border-collapse:collapse;font-size:12px;white-space:nowrap;width:100%}td,th{padding:8px 11px;border:0;border-bottom:1px solid #e5e7eb;text-align:left}
th{background:#eaf0f9;position:sticky;top:0}tr:nth-child(even){background:#f8fafc}img{display:block;max-width:100%;margin:20px auto}pre{white-space:pre-wrap}
.decisions td:nth-child(6){white-space:normal;min-width:440px}.decisions .table{max-height:850px}
</style>"""


@dataclass
class AuditReport:
    """Completed analysis results consumed by each report format."""

    config: dict
    summary: dict
    features: FeatureAnalysis
    pairs: pd.DataFrame
    interactions: InteractionAnalysis
    temporal: TemporalAnalysis
    decisions: pd.DataFrame
    rule_evaluations: pd.DataFrame


def plot_image(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=125, bbox_inches="tight")
    plt.close(fig)
    return (
        '<img alt="Analysis chart" src="data:image/png;base64,'
        + base64.b64encode(buffer.getvalue()).decode()
        + '">'
    )


def plot_positive_rate(tab, title, baseline):
    view = tab.head(32)
    positions = np.arange(len(view))
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    axes[0].errorbar(
        positions,
        view["positive_rate"].to_numpy(),
        yerr=np.array(
            [
                np.maximum(0, view["positive_rate"] - view["positive_rate_low_iid"]),
                np.maximum(0, view["positive_rate_high_iid"] - view["positive_rate"]),
            ]
        ),
        fmt="o",
        color="#2563eb",
        capsize=2,
    )
    sparse = view["low_support"].to_numpy()
    axes[0].scatter(
        positions[sparse],
        view.loc[sparse, "positive_rate"],
        color="#d97706",
        zorder=4,
        label="Low support",
    )
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
    return (
        '<div class="table">'
        + df.head(limit).to_html(index=False, escape=True, float_format=lambda x: f"{x:.6g}")
        + "</div>"
    )


def categorical_diversity_table(profiles: pd.DataFrame) -> pd.DataFrame:
    """Label each enabled diversity metric with its feature-specific bounds.

    K counts original nonmissing categories, before rare-category pooling.
    Keep the numeric profile keys stable: different features can have different K.
    """
    rows = []
    for profile in profiles.loc[profiles["type"] == "categorical"].to_dict("records"):
        levels = int(profile["unique_nonmissing"])
        bounds = {
            "entropy_nats_nonmissing": (0, np.log(levels)) if levels else None,
            "effective_levels_nonmissing": (1, levels) if levels else None,
        }
        for metric, limits in bounds.items():
            if metric not in profile:
                continue  # The entropy statistic may be disabled.
            interval = f"{limits[0]:.6g}, {limits[1]:.6g}" if limits else "undefined, undefined"
            rows.append(
                {
                    "feature": profile["feature"],
                    "metric": f"{metric} {{{interval}}}",
                    "value": profile[metric],
                }
            )
    return pd.DataFrame(rows, columns=["feature", "metric", "value"])


def _categorical_diversity_html(profiles: pd.DataFrame) -> str:
    table = categorical_diversity_table(profiles)
    if table.empty:
        return ""
    return (
        "<h3>Entropy and effective levels with bounds</h3>"
        "<p>Labels show {minimum, maximum} for each feature's original nonmissing "
        "category count K, before pooling. Entropy ranges from 0 to ln(K) nats; "
        "effective levels range from 1 to K. Bounds are displayed to six significant "
        "digits. Values and bounds are undefined when K = 0.</p>" + html_table(table, len(table))
    )


def _categorical_diversity_markdown(profiles: pd.DataFrame) -> list[str]:
    table = categorical_diversity_table(profiles)
    if table.empty:
        return []
    lines = [
        "\n## Entropy and effective levels with bounds\n",
        "Labels show {minimum, maximum} using the original nonmissing category count "
        "K, before pooling. Entropy ranges from 0 to ln(K) nats; effective levels "
        "range from 1 to K. Displayed values and bounds use six significant digits. "
        "Values and bounds are undefined when K = 0.\n",
        "| Feature | Metric {minimum, maximum} | Value |",
        "|---|---|---:|",
    ]
    for row in table.itertuples(index=False):
        name = html.escape(str(row.feature))
        for char in ["\\", "|", "`", "*", "_"]:
            name = name.replace(char, "\\" + char)
        name = name.replace("\n", " ").replace("\r", " ")
        value = f"{row.value:.6g}" if np.isfinite(row.value) else "undefined"
        lines.append(f"| {name} | `{row.metric}` | {value} |")
    return lines


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


def _write_tables(out: Path, result: AuditReport) -> None:
    """Write all CSV, JSON, and YAML artifacts with their existing filenames."""
    (out / "feature_positive_rate").mkdir()
    (out / "joint_positive_rate").mkdir()
    (out / "time_feature_positive_rate").mkdir()
    for kind in ["numerical", "categorical"]:
        subset = result.features.profiles[result.features.profiles["type"] == kind]
        if not subset.empty:
            subset = subset.dropna(axis=1, how="all")
        subset.to_csv(out / f"{kind}_univariate.csv", index=False)
    categorical_diversity_table(result.features.profiles).to_csv(
        out / "categorical_diversity.csv", index=False
    )
    result.features.target_associations.to_csv(out / "feature_target.csv", index=False)
    result.decisions.to_csv(out / "feature_decisions.csv", index=False)
    result.rule_evaluations.to_csv(out / "rule_evaluations.csv", index=False)
    save_json(out / "feature_decisions.json", result.decisions.to_dict("records"))
    save_json(
        out / "selected_features.json",
        {
            "selected_features": result.decisions.loc[
                result.decisions["decision"] != "exclude", "feature"
            ].tolist(),
            "excluded_features": result.decisions.loc[
                result.decisions["decision"] == "exclude", "feature"
            ].tolist(),
            "review_features": result.decisions.loc[
                result.decisions["decision"] == "review", "feature"
            ].tolist(),
            "policy": "Review features are retained. These are screening recommendations for later validation; the input dataset is not modified.",
        },
    )
    result.pairs.to_csv(out / "feature_pairs.csv", index=False)
    result.interactions.information.to_csv(out / "joint_information.csv", index=False)
    pd.DataFrame(result.features.bucket_dictionary).to_csv(
        out / "bucket_dictionary.csv", index=False
    )
    result.temporal.positive_rates.to_csv(out / "time_positive_rate.csv", index=False)
    result.temporal.drift.to_csv(out / "time_feature_drift.csv", index=False)
    for name in result.config["features"]:
        key = result.features.internal_names[name]
        if enabled(result.config, "positive_rate"):
            result.features.positive_rate_tables[name].to_csv(
                out / "feature_positive_rate" / f"{key}.csv", index=False
            )
        if key in result.temporal.feature_positive_rates:
            result.temporal.feature_positive_rates[key].to_csv(
                out / "time_feature_positive_rate" / f"{key}.csv", index=False
            )
    joint_files = {}
    for i, (name, tab) in enumerate(result.interactions.positive_rate_tables.items()):
        filename = f"joint_{i:03d}.csv"
        joint_files[name] = filename
        tab.to_csv(out / "joint_positive_rate" / filename, index=False)
    save_json(out / "summary.json", result.summary)
    save_json(
        out / "manifest.json",
        {
            "feature_files": {
                name: result.features.internal_names[name] + ".csv"
                for name in result.config["features"]
            },
            "joint_files": joint_files,
        },
    )
    (out / "config_used.yaml").write_text(yaml.safe_dump(result.config, sort_keys=False))


def _render_html(result: AuditReport) -> str:
    """Render the audit overview, detailed tables, and embedded charts."""
    notes = "".join(f"<li>{html.escape(x)}</li>" for x in result.summary["limitations"])
    decision_html = html_table(result.decisions, len(result.decisions))
    sections = [
        f"<h1>Data Mining</h1><p class='lead'>{result.summary['rows']:,} rows · {result.features.target.sum():,} positives · "
        f"{result.features.target.mean():.3%} positive-class rate · {len(result.config['features'])} features</p>",
        "<h2>Feature decisions</h2><p>"
        + " · ".join(
            f"{name}: {int(count)}" for name, count in result.summary["decision_counts"].items()
        )
        + "</p><p>Review features remain in selected_features.json. Every rule evaluation, including disabled or skipped rules, is saved in rule_evaluations.csv. "
        "Rules are configurable screening policy; the dataset is preserved.</p><div class='decisions'>"
        + decision_html
        + "</div>",
        "<details><summary>Enabled statistics and policy</summary><p>"
        + html.escape(", ".join(result.summary["enabled_statistics"]))
        + "</p><pre>"
        + html.escape(
            yaml.safe_dump(
                {
                    "rules": result.config["rules"],
                    "expected_direction": result.config["expected_direction"],
                },
                sort_keys=False,
            )
        )
        + "</pre></details>",
        "<p>Analysis only. Expand a feature to inspect its full-data distribution and positive-class rate profile. "
        "The downloadable CSVs contain the complete tables. Feature-target comparison belongs to stage 3.</p>",
        f"<details open><summary>Interpretation and sampling</summary><ul>{notes}</ul>"
        f"<p>Full-data profiles, bucket counts and bucket-target statistics. "
        f"Numeric target metrics, feature pairs and joint information use a fixed uniform sample of {len(result.features.sample_indices):,} rows "
        f"({result.features.sampled_target.sum():,} positives). Pair coverage: {len(result.pairs)} / {result.summary['pairs_possible']}. "
        f"P-value policy: {html.escape(result.summary['pvalue_policy'])}. "
        "BH q-values cover only the eligible feature-vs-target bucket chi-square tests, "
        "with their usual independence/positive-dependence assumptions.</p></details>",
        "<h2>1. Numerical univariate</h2>"
        + html_table(
            result.features.profiles[result.features.profiles["type"] == "numerical"].dropna(
                axis=1, how="all"
            )
        ),
        "<h2>2. Categorical univariate</h2>"
        + html_table(
            result.features.profiles[result.features.profiles["type"] == "categorical"]
            .drop(
                columns=["entropy_nats_nonmissing", "effective_levels_nonmissing"], errors="ignore"
            )
            .dropna(axis=1, how="all")
        )
        + _categorical_diversity_html(result.features.profiles),
        "<h2>3. Feature–target associations</h2><p>Sorted by empirical binned/pooled MI in nats when enabled, otherwise config order. "
        "This is an exploratory association ordering, not a validated feature ranking. "
        "Raw numeric AUC treats the column itself as a score; it misses nonmonotonic effects and is not a fitted model score.</p>"
        + html_table(result.features.target_associations),
    ]
    for i, name in enumerate(result.config["features"], 1):
        if enabled(result.config, "positive_rate"):
            print(f"  Plot [{i}/{len(result.config['features'])}]: {name}", flush=True)
            sections.append(
                f"<details><summary>{html.escape(name)} — {result.config['features'][name]}</summary>"
                + plot_positive_rate(
                    result.features.positive_rate_tables[name],
                    name,
                    float(result.features.target.mean()),
                )
                + html_table(result.features.positive_rate_tables[name])
                + "</details>"
            )
    sections.append(
        "<h2>Feature redundancy</h2><p>Pairwise-complete numeric rows; categorical comparisons use "
        "pooled categories and an explicit missing bucket. Measures have different meanings and should not "
        "be compared as one common scale. Missingness phi compares missing/nonfinite indicators.</p>"
        + html_table(result.pairs, 500)
    )
    sections.append(
        "<h2>Joint and conditional analysis</h2><p>Only configured pairs/triples are examined. "
        "Conditional MI is I(feature; target | other named features). Joint MI gains can reflect additive effects, "
        "interactions, and sampling bias. Cells with few rows, positives or negatives are flagged.</p>"
        + html_table(result.interactions.information)
    )
    for name, tab in result.interactions.positive_rate_tables.items():
        sections.append(
            f"<details><summary>{html.escape(name)}</summary>"
            + html_table(tab.sort_values("rows", ascending=False))
            + "</details>"
        )
    sections.append(
        f"<h2>Temporal and group context</h2><p>Time analysis: {html.escape(result.temporal.status)}. "
        "JS divergence uses fixed full-data buckets and natural logs; range 0 to ln(2). "
        "Bucket-by-period positive-class rate tables are in time_feature_positive_rate/.</p>"
        + html_table(result.temporal.positive_rates)
    )
    if result.summary["group_summary"]:
        sections.append(
            "<pre>" + html.escape(json.dumps(result.summary["group_summary"], indent=2)) + "</pre>"
        )
    sections.append(
        "<h2>Feature review workflow</h2><ol><li>Confirm every candidate exists at prediction time and inspect potential target proxies.</li>"
        "<li>Review unusable/constant columns, identifiers, sparse categories and missingness. Constant observed values with varying missingness can still carry signal.</li>"
        "<li>Inspect effect sizes, support, nonlinear positive-class rate profiles, redundant feature groups and configured interactions.</li>"
        "<li>During the later training stage, confirm selections with validation appropriate to deployment, log loss, average precision/PR-AUC and calibration; use permutation importance and feature-group ablations.</li></ol>"
    )
    document = HTML_HEADER + "".join(sections) + "</html>"
    return document


def _render_markdown(result: AuditReport) -> str:
    """Render decisions, categorical diversity, and interpretation as Markdown."""
    # Markdown includes full reasons without requiring the HTML viewer.
    lines = [
        "# Data Mining",
        f"\n{result.summary['rows']:,} rows; {int(result.features.target.sum()):,} positives; positive-class rate {result.features.target.mean():.3%}.\n",
        "## Feature decisions\n",
        "Review features are retained as candidates; exclusions follow configured screening policy.\n",
    ]
    for row in result.decisions.to_dict("records"):
        lines.append(f"- **{row['feature']} — {row['decision'].upper()}**: {row['reason']}")
    lines += _categorical_diversity_markdown(result.features.profiles)
    lines += ["\n## Interpretation\n"] + [f"- {note}" for note in result.summary["limitations"]]
    lines += [
        "\n## Report files\n",
        "See categorical_diversity.csv, feature_target.csv, feature_pairs.csv, joint_information.csv, "
        "rule_evaluations.csv and the other CSV tables for exact values. The HTML report includes enabled positive-class rate plots.\n",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_report(root: Path, result: AuditReport) -> Path:
    """Create a fresh run directory and save every report format."""
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    out = root / run_id
    print(f"Writing report tables: {out}", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    _write_tables(out, result)
    print("Rendering HTML report...", flush=True)
    (out / "report.html").write_text(_render_html(result))
    (out / "report.md").write_text(_render_markdown(result))
    print(f"Report: {out / 'report.html'}", flush=True)
    print("Analysis complete. Open the report.html file above in your browser.", flush=True)
    return out
