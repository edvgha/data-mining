"""Metric switches and decision-rule configuration; no runtime dependencies."""
from copy import deepcopy


METRICS = {
    "numerical_univariate": ["mean", "std", "median", "quantiles", "iqr", "mad", "skewness",
                             "kurtosis", "zero_fraction", "negative_fraction", "outlier_fraction"],
    "categorical_univariate": ["entropy", "rare_categories"],
    "feature_target": ["positive_rate", "cohens_d", "hedges_g", "auc", "ks", "point_biserial",
                       "spearman_target", "mutual_information", "cramers_v_target", "chi_square"],
    "feature_pairs": ["pearson", "spearman", "cramers_v_pairs", "eta_squared", "missingness_phi"],
    "multivariate": ["joint_positive_rate", "joint_information"],
    "temporal": ["time_positive_rate", "time_feature_positive_rate", "time_drift"],
}
ALL_METRICS = {m for group in METRICS.values() for m in group}

RULES = {
    "all_missing": {"enabled": True, "action": "exclude"},
    "constant": {"enabled": True, "action": "exclude"},
    "missingness": {"enabled": True, "action": "review", "threshold": .8},
    "near_constant": {"enabled": True, "action": "review", "threshold": .995},
    "high_cardinality": {"enabled": True, "action": "review", "threshold": 1000},
    "weak_signal": {"enabled": True, "action": "review", "max_abs_d": .05,
                    "max_auc_distance": .02, "max_mi": .0001},
    "direction": {"enabled": True, "action": "review", "min_abs_d": .05,
                  "min_auc_distance": .02},
    "redundancy": {"enabled": True, "action": "exclude", "metric": "spearman",
                   "threshold": .95, "sign": "absolute", "min_pair_rows": 1000,
                   "min_pair_fraction": .5},
}

EXTRA_DEFAULTS = {
    "statistics": {"include": "all", "exclude": []},
    "rules": {},
    "expected_direction": {},
    "force_keep": [],
    "force_exclude": [],
    "prefer": [],
    "protect_configured_interactions": True,
}


def enabled(cfg, metric):
    selection = cfg.get("statistics", EXTRA_DEFAULTS["statistics"])
    return (selection["include"] == "all" or metric in selection["include"]) and metric not in selection["exclude"]


def configure_decisions(cfg):
    """Validate and fill all rule defaults without silently accepting typos."""
    cfg = deepcopy(cfg)
    selection = cfg["statistics"]
    if not isinstance(selection, dict) or set(selection) - {"include", "exclude"}:
        raise ValueError("statistics accepts only include and exclude.")
    selection = {"include": "all", "exclude": [], **selection}
    for key in ["include", "exclude"]:
        values = selection[key]
        if key == "include" and values == "all":
            continue
        if not isinstance(values, list) or not all(isinstance(x, str) for x in values):
            raise ValueError(f"statistics.{key} must be a metric-name list (include also accepts all).")
        unknown = set(values) - ALL_METRICS
        if unknown:
            raise ValueError(f"Unknown statistics: {sorted(unknown)}. Use cohens_d for Cohen's d.")
    cfg["statistics"] = selection
    supplied = cfg["rules"]
    if not isinstance(supplied, dict) or set(supplied) - set(RULES):
        raise ValueError(f"rules must use these names: {list(RULES)}")
    rules = deepcopy(RULES)
    for name, settings in supplied.items():
        if not isinstance(settings, dict) or set(settings) - set(rules[name]):
            raise ValueError(f"Unknown settings in rule {name}.")
        rules[name].update(settings)
    for name, rule in rules.items():
        if type(rule["enabled"]) is not bool or rule["action"] not in {"keep", "review", "exclude"}:
            raise ValueError(f"{name}: enabled must be boolean and action must be keep, review, or exclude.")
        for key, value in rule.items():
            if key in {"enabled", "action", "metric", "sign"}:
                continue
            if type(value) not in {float, int} or not value >= 0 or value == float("inf"):
                raise ValueError(f"rules.{name}.{key} must be a finite nonnegative number.")
    for name in ["missingness", "near_constant", "redundancy"]:
        if not 0 < rules[name]["threshold"] <= 1:
            raise ValueError(f"rules.{name}.threshold must be in (0, 1].")
    red = rules["redundancy"]
    if red["metric"] not in {"pearson", "spearman"} or red["sign"] not in {"absolute", "positive", "negative"}:
        raise ValueError("redundancy metric must be pearson/spearman and sign absolute/positive/negative.")
    if not 0 <= red["min_pair_fraction"] <= 1:
        raise ValueError("redundancy.min_pair_fraction must be in [0, 1].")
    if rules["weak_signal"]["max_auc_distance"] > .5 or rules["direction"]["min_auc_distance"] > .5:
        raise ValueError("AUC distances from 0.5 must be <= 0.5.")
    cfg["rules"] = rules
    names = set(cfg["features"])
    for key in ["force_keep", "force_exclude", "prefer"]:
        values = cfg[key]
        if (not isinstance(values, list) or not all(isinstance(x, str) for x in values)
                or not set(values) <= names or len(set(values)) != len(values)):
            raise ValueError(f"{key} must contain distinct configured feature names.")
    if set(cfg["force_keep"]) & set(cfg["force_exclude"]):
        raise ValueError("force_keep and force_exclude cannot overlap.")
    if type(cfg["protect_configured_interactions"]) is not bool:
        raise ValueError("protect_configured_interactions must be boolean.")
    directions = cfg["expected_direction"]
    if not isinstance(directions, dict):
        raise ValueError("expected_direction must be a mapping.")
    for name, direction in directions.items():
        if name not in names or cfg["features"][name] != "numerical" or direction not in {"positive", "negative"}:
            raise ValueError("expected_direction maps numerical features to positive or negative.")
    return cfg


PROFILE_COLUMNS = {
    "mean": ["mean"], "std": ["std"], "median": ["median"],
    "quantiles": ["min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"],
    "iqr": ["iqr"], "mad": ["mad_unscaled"], "skewness": ["skewness"],
    "kurtosis": ["excess_kurtosis"], "zero_fraction": ["zero_fraction_finite"],
    "negative_fraction": ["negative_fraction_finite"], "outlier_fraction": ["iqr_outlier_fraction"],
    "entropy": ["entropy_nats_nonmissing", "effective_levels_nonmissing"],
    "rare_categories": ["singleton_levels", "rare_levels", "rare_row_fraction"],
}


def filter_profile(info, cfg):
    for metric, columns in PROFILE_COLUMNS.items():
        if not enabled(cfg, metric):
            for col in columns:
                info.pop(col, None)
    return info
