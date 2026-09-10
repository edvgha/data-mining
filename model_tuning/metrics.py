"""Probability metrics and efficient sufficient-statistic bootstrap."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

METRICS = ["logloss", "ne", "ne_train_baseline", "brier", "z_score", "click_error_pct"]


def row_statistics(y, p, baseline_rate):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y) == 0 or len(y) != len(p) or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Metrics require nonempty binary labels and finite probabilities in [0, 1].")
    q = np.clip(p, 1e-15, 1-1e-15)
    base = np.clip(baseline_rate, 1e-15, 1-1e-15)
    return np.column_stack([np.ones(len(y)), y, p, p*(1-p),
        -(y*np.log(q)+(1-y)*np.log1p(-q)), (y-p)**2,
        -(y*np.log(base)+(1-y)*np.log1p(-base))])


def from_totals(totals):
    t = np.asarray(totals, dtype=float)
    n, actual, expected, variance, loss, brier, baseline = np.moveaxis(t, -1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = actual/n
        entropy = -(np.where(rate > 0, rate*np.log(rate), 0) +
                    np.where(rate < 1, (1-rate)*np.log1p(-rate), 0))
        ll = loss/n
        values = [ll, np.where(entropy > 0, ll/entropy, np.nan),
                  np.where(baseline > 0, loss/baseline, np.nan), brier/n,
                  np.where(variance > 0, (actual-expected)/np.sqrt(variance), np.nan),
                  np.where(actual > 0, 100*(expected-actual)/actual, np.nan)]
    return dict(zip(METRICS, values))


def evaluate(y, p, baseline_rate):
    totals = row_statistics(y, p, baseline_rate).sum(axis=0)
    result = {k: float(v) for k, v in from_totals(totals).items()}
    n, actual, expected = totals[:3]
    result.update(rows=int(n), actual_clicks=int(actual), expected_clicks=float(expected),
                  actual_rate=float(actual/n), predicted_rate=float(expected/n),
                  click_error=float(expected-actual), abs_click_error_pct=abs(result["click_error_pct"]),
                  baseline_logloss=float(totals[6]/n),
                  ne_status="ok" if 0 < actual < n else "undefined_single_class",
                  click_error_pct_status="ok" if actual > 0 else "undefined_zero_actual_clicks",
                  z_score_status="iid_bernoulli_approximation" if totals[3] > 0 else "undefined_zero_variance")
    both = 0 < actual < n
    result["roc_auc"] = float(roc_auc_score(y, p)) if both else np.nan
    result["average_precision"] = float(average_precision_score(y, p)) if both else np.nan
    return result


def unit_codes(frame, cfg):
    b = cfg["bootstrap"]
    if b["method"] == "row":
        return np.arange(len(frame))
    if b["method"] == "cluster":
        values = frame[b["cluster_column"]]
        if values.isna().any():
            raise ValueError("Bootstrap cluster IDs must not be missing.")
    else:
        try:
            values = frame[cfg["time_column"]].dt.floor(b["time_frequency"])
        except (ValueError, TypeError) as exc:
            raise ValueError("time_frequency must be a fixed duration, such as 1D or 7D.") from exc
    return pd.factorize(values, sort=False)[0]


def aggregate_units(stats, codes):
    codes = pd.factorize(codes, sort=False)[0]
    return np.column_stack([np.bincount(codes, weights=stats[:, j]) for j in range(stats.shape[1])])


def resample_totals(unit_totals, n_resamples, seed):
    # Never materialize B × N row indices. Sample whole units with replacement.
    rng = np.random.default_rng(seed)
    count = len(unit_totals)
    for _ in range(n_resamples):
        weights = rng.multinomial(count, np.full(count, 1/count))
        yield weights @ unit_totals


def bootstrap_metrics(frame, y, p, baseline_rate, cfg, seed=None):
    units = aggregate_units(row_statistics(y, p, baseline_rate), unit_codes(frame, cfg))
    b = cfg["bootstrap"]
    seed = cfg["seed"] if seed is None else seed
    if len(units) < 2:
        return pd.DataFrame([{"metric": m, "low": np.nan, "high": np.nan, "std": np.nan,
                              "valid_resamples": 0, "units": len(units), "status": "insufficient_units"} for m in METRICS])
    draws = pd.DataFrame([from_totals(t) for t in resample_totals(units, b["n_resamples"], seed)])
    alpha = (1-b["confidence"])/2
    rows = []
    for metric in METRICS:
        vals = draws[metric].replace([np.inf, -np.inf], np.nan).dropna()
        enough = len(vals) >= 2
        rows.append({"metric": metric, "low": vals.quantile(alpha) if enough else np.nan,
                     "high": vals.quantile(1-alpha) if enough else np.nan,
                     "std": vals.std(ddof=1) if enough else np.nan,
                     "valid_resamples": len(vals), "units": len(units),
                     "status": "ok" if len(vals) == b["n_resamples"] else "some_or_all_resamples_undefined"})
    return pd.DataFrame(rows)


def group_evaluation(frame, p, baseline_rate, cfg):
    tables = []
    for columns in cfg["group_by"]:
        rows = []
        for key, indices in frame.groupby(columns, dropna=False, observed=True, sort=False).indices.items():
            key = key if isinstance(key, tuple) else (key,)
            part = frame.iloc[indices]
            y, pred = part[cfg["target"]].to_numpy(), p[indices]
            row = dict(zip(columns, key))
            row.update(evaluate(y, pred, baseline_rate))
            row["low_support"] = len(part) < cfg["report"]["min_group_rows"] or y.sum() < cfg["report"]["min_group_clicks"]
            intervals = bootstrap_metrics(part, y, pred, baseline_rate, cfg)
            row["bootstrap_units"] = int(intervals["units"].iloc[0])
            for _, ci in intervals.iterrows():
                for suffix in ["low", "high", "valid_resamples", "status"]:
                    row[f"{ci['metric']}_bootstrap_{suffix}"] = ci[suffix]
            rows.append(row)
        tables.append((columns, pd.DataFrame(rows).sort_values("rows", ascending=False)))
    return tables
