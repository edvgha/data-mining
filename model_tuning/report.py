"""Portable HTML report with embedded plots and complete CSV exports."""
import base64
import html
import io
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False, default=str)+"\n")


def write_report(out, summary, overall, intervals, groups, trials, importance, shap_values,
                 shap_features, history, test, prediction, cfg, stability):
    out = Path(out)
    sections = []
    def table(title, data, filename, limit=100):
        data.to_csv(out/filename, index=False)
        sections.append(f"<h2>{html.escape(title)}</h2><p><a href='{filename}'>Complete CSV</a></p>"+
                        data.head(limit).to_html(index=False, escape=True, float_format=lambda x: f"{x:.6g}"))
        if len(data) > limit:
            sections.append(f"<p>Showing {limit} of {len(data):,} rows; CSV contains every row.</p>")
    def plot(title, filename, draw):
        fig, ax = plt.subplots(figsize=(10, 5))
        draw(ax)
        ax.set_title(title)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=125)
        (out/filename).write_bytes(buffer.getvalue())
        plt.close(fig)
        sections.append(f"<h2>{html.escape(title)}</h2><img alt='{html.escape(title)}' src='data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}'>")
    table("Chronological partitions", pd.DataFrame(summary["split"]["partitions"]), "splits.csv")
    table("Evaluation — validation selected parameters; test is held out", overall, "metrics.csv")
    table("Held-out test bootstrap confidence intervals", intervals, "bootstrap_intervals.csv")
    table("Optuna trials", trials, "trials.csv")
    def trial_plot(ax):
        complete = trials.loc[trials.state == "COMPLETE"].sort_values("number")
        ax.scatter(complete.number, complete.value, label="Robust validation objective")
        ax.plot(complete.number, complete.value.cummin(), label="Best so far")
        ax.set(xlabel="Trial", ylabel="Log loss + penalty × bootstrap SD")
        ax.legend()
    plot("Optimization history", "optimization.png", trial_plot)
    def loss_plot(ax):
        for name, values in history.items():
            ax.plot(np.arange(1, len(values["logloss"])+1), values["logloss"], label=name)
        ax.axvline(summary["best_num_boost_round"], linestyle="--", color="black", label="Selected tree count")
        ax.set(xlabel="Boosting round", ylabel="Log loss")
        ax.legend()
    plot("Best candidate learning curve — inner chronological holdout", "learning_curve.png", loss_plot)
    cal = pd.DataFrame({"prediction": prediction, "target": test[cfg["target"]].to_numpy()})
    cal["bin"] = pd.qcut(cal.prediction, 10, duplicates="drop")
    calibration = cal.groupby("bin", observed=True).agg(rows=("target", "size"), actual_rate=("target", "mean"), predicted_rate=("prediction", "mean")).reset_index()
    if calibration.empty:
        calibration = pd.DataFrame([{"bin": "all", "rows": len(cal), "actual_rate": cal.target.mean(), "predicted_rate": cal.prediction.mean()}])
    table("Test calibration bins", calibration, "calibration.csv")
    def calibration_plot(ax):
        ax.plot(calibration.predicted_rate, calibration.actual_rate, "o-")
        maximum = max(calibration.predicted_rate.max(), calibration.actual_rate.max()) * 1.1
        ax.plot([0, maximum], [0, maximum], "--", color="gray")
        ax.set(xlabel="Mean predicted probability", ylabel="Observed click rate")
    plot("Test calibration", "calibration.png", calibration_plot)
    temporal = pd.DataFrame({"period": test[cfg["time_column"]].dt.floor("1D"), "actual": cal.target.to_numpy(), "expected": prediction})
    temporal = temporal.groupby("period").agg(rows=("actual", "size"), actual_clicks=("actual", "sum"), expected_clicks=("expected", "sum")).reset_index()
    table("Test performance over time", temporal, "time_metrics.csv")
    def time_plot(ax):
        for name in ["actual_clicks", "expected_clicks"]:
            ax.plot(temporal.period, temporal[name]/temporal.rows, label=name.replace("_clicks", " rate"))
        ax.tick_params(axis="x", rotation=25)
        ax.legend()
    plot("Test click rate over time", "time_calibration.png", time_plot)
    for index, (columns, group) in enumerate(groups):
        table("Test groups: " + " × ".join(columns), group, f"groups_{index}.csv")
        display = group.head(30)
        def group_plot(ax):
            ax.scatter(display.actual_clicks, display.expected_clicks)
            limit = max(display.actual_clicks.max(), display.expected_clicks.max()) * 1.05
            ax.plot([0, limit], [0, limit], "--", color="gray")
            ax.set(xlabel="Actual clicks", ylabel="Expected clicks (sum of probabilities)")
        plot("Actual vs expected clicks: " + " × ".join(columns) + " (30 largest groups)", f"groups_{index}.png", group_plot)
    table("Feature importance — native XGBoost gain and TreeSHAP", importance, "feature_importance.csv")
    top = cfg["report"]["plot_top_features"]
    for metric, title in [("mean_abs_shap", "Mean absolute SHAP (log-odds)"), ("gain", "XGBoost average split gain"), ("total_gain", "XGBoost total gain")]:
        values = importance.nlargest(top, metric).sort_values(metric)
        plot(title, f"{metric}.png", lambda ax, v=values, m=metric: ax.barh(v.feature, v[m]))
    ranking = np.argsort(np.abs(shap_values[:, :-1]).mean(axis=0))[-min(top, len(cfg["features"])):]
    def shap_plot(ax):
        rng = np.random.default_rng(cfg["seed"])
        names = list(cfg["features"])
        for level, index in enumerate(ranking):
            ax.scatter(shap_values[:, index], level+rng.uniform(-.25, .25, len(shap_values)), s=4, alpha=.25)
        ax.set_yticks(range(len(ranking)), [names[i] for i in ranking])
        ax.axvline(0, color="gray", linewidth=.6)
        ax.set_xlabel("SHAP contribution to raw margin (log-odds)")
    plot("SHAP distribution on a fixed test sample", "shap_distribution.png", shap_plot)
    pd.DataFrame(shap_values, columns=[*list(cfg["features"]), "__bias__"]).to_csv(out/"shap_values.csv", index=False)
    shap_features.to_csv(out/"shap_features.csv", index=False)
    table("Training bootstrap refits — descriptive test sensitivity, no model selection", stability, "training_stability.csv")
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in summary["warnings"])
    body = f"""<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>XGBoost binary tuning report</title><style>body{{font:15px system-ui;margin:32px;color:#17263a;background:#f7f9fc}}main{{max-width:1300px;margin:auto}}table{{border-collapse:collapse;display:block;overflow:auto;background:white;font-size:12px}}td,th{{padding:7px;border:1px solid #dce3ec;white-space:nowrap}}h1,h2{{color:#183f70}}img{{max-width:100%;background:white}}pre{{white-space:pre-wrap}}li{{margin:7px 0}}</style><main>
<h1>XGBoost binary classification tuning</h1><p>Best trial {summary['best_trial']} · {summary['best_num_boost_round']} boosting rounds · {len(test):,} held-out test rows.</p>
<p>Native xgboost.train / DMatrix. Final model refitted on train + validation only. Test labels were excluded from tuning, early stopping, feature encoding, and refitting.</p>
<h2>Definitions and interpretation</h2><ul>
<li>Logloss = mean(−y ln p − (1−y) ln(1−p)); lower is better. Predictions are clipped only for logarithms.</li>
<li>NE = logloss / entropy(observed evaluation click rate). NE &lt; 1 beats the evaluation-set constant-rate reference; undefined for a single-class group. ne_train_baseline compares with a constant probability learned from the fitting data.</li>
<li>Expected clicks E = Σp; actual clicks A = Σy. Signed error (%) = 100(E−A)/A. Positive means overprediction; undefined if A=0.</li>
<li>z_score = (A−E)/√Σp(1−p). Positive means underprediction. This is an independent-Bernoulli approximation, not a cluster-corrected significance test. No p-values or multiple-testing claims are made.</li>
<li>Bootstrap intervals use configured sampling units and percentile bounds. These are conditional on a fitted model and observed period, not protection against future drift. Group intervals are marginal, resampling the units present within each group.</li>
<li>SHAP sums plus bias reconstruct raw log-odds; apply sigmoid to obtain probability. Contributions are not percentage-point changes. Gain and SHAP explain fitted associations, not causal effects. Correlated substitutes can divide importance.</li>
<li>Average precision and ROC AUC measure ranking. Calibration, logloss and group click error measure probability quality. Undefined single-class ranking metrics remain blank.</li></ul>
<h2>Run notes</h2><ul>{warnings}</ul>
<details><summary>Best parameters</summary><pre>{html.escape(json.dumps(summary['best_params'], indent=2))}</pre></details>
{''.join(sections)}
<p>Additional artifacts: <a href='summary.json'>summary</a>, <a href='config.yaml'>resolved config</a>, <a href='model.ubj'>model</a>, <a href='encoder.json'>encoder</a>, <a href='shap_values.csv'>SHAP values</a>, <a href='shap_features.csv'>matching sample rows</a>.</p></main></html>"""
    (out/"report.html").write_text(body)
