"""Deterministic, inspectable screening rules. No claims of validated importance."""
import json

import numpy as np
import pandas as pd

from .settings import enabled


def finite(value):
    return isinstance(value, (int, float, np.number)) and np.isfinite(value)


def decide(profiles, targets, pairs, cfg):
    """Return per-feature decisions plus every evaluated/skipped rule and evidence."""
    names = list(cfg["features"])
    profile = profiles.set_index("feature").to_dict("index")
    target = targets.set_index("feature").to_dict("index")
    actions = {name: "keep" for name in names}
    reasons = {name: [] for name in names}
    related = {name: [] for name in names}
    audit = []
    priority = {"keep": 0, "review": 1, "exclude": 2}
    protected = set(cfg["force_keep"])
    if cfg["protect_configured_interactions"]:
        protected |= {name for group in cfg["interactions"] for name in group}

    def log(name, rule, status, reason, evidence=None, action=None, other=None):
        requested = action or cfg["rules"].get(rule, {}).get("action", "review")
        effective = requested
        if status == "triggered":
            if name in protected and rule in {"weak_signal", "redundancy"} and requested == "exclude":
                effective = "review"
                reason += " Exclusion changed to review because this feature is protected by force_keep or a configured interaction hypothesis."
            if priority[effective] > priority[actions[name]]:
                actions[name] = effective
            reasons[name].append(reason)
            if other:
                related[name].append(other)
        audit.append({"feature": name, "rule": rule, "status": status,
                      "requested_action": requested, "effective_action": effective if status == "triggered" else "none",
                      "related_feature": other, "reason": reason,
                      "evidence": json.dumps(evidence or {}, sort_keys=True, allow_nan=False)})

    def simple(name, rule, hit, reason, evidence):
        if not cfg["rules"][rule]["enabled"]:
            log(name, rule, "disabled", "Rule disabled in configuration.")
        else:
            log(name, rule, "triggered" if hit else "not_triggered", reason if hit else "Configured condition not met.", evidence)

    for name in names:
        p, t = profile[name], target[name]
        missing = p.get("unusable_fraction")
        if not finite(missing):
            missing = p["missing_fraction"]
        observed = p["unique_nonmissing"]
        simple(name, "all_missing", observed == 0,
               "No usable observed values: the feature cannot distinguish rows in the supplied data.",
               {"unique_usable_values": int(observed)})
        simple(name, "constant", observed == 1 and missing == 0,
               "One constant value on every row; no variation is available to a tree split.",
               {"unique_usable_values": int(observed), "missing_fraction": float(missing)})
        rule = cfg["rules"]["missingness"]
        simple(name, "missingness", missing >= rule["threshold"],
               f"Missing/nonfinite fraction {missing:.3%} >= {rule['threshold']:.3%}; inspect availability and missingness signal.",
               {"value": float(missing), "threshold": rule["threshold"], "operator": ">="})
        dominant = p.get("dominant_fraction_nonmissing")
        rule = cfg["rules"]["near_constant"]
        simple(name, "near_constant", finite(dominant) and dominant >= rule["threshold"],
               f"The dominant observed value occupies {dominant:.3%} of usable rows; threshold {rule['threshold']:.3%}. Missingness can still carry signal.",
               {"value": float(dominant) if finite(dominant) else None, "threshold": rule["threshold"], "operator": ">="})
        rule = cfg["rules"]["high_cardinality"]
        simple(name, "high_cardinality", p["type"] == "categorical" and observed >= rule["threshold"],
               f"{int(observed):,} observed categories >= {rule['threshold']:,}; review identifier semantics, sparsity and unseen values.",
               {"value": int(observed), "threshold": rule["threshold"], "operator": ">="})

        rule = cfg["rules"]["weak_signal"]
        need = ["mutual_information"] + (["cohens_d", "auc"] if p["type"] == "numerical" else [])
        missing_metrics = [metric for metric in need if not enabled(cfg, metric)]
        if not rule["enabled"]:
            log(name, "weak_signal", "disabled", "Rule disabled in configuration.")
        elif missing_metrics:
            log(name, "weak_signal", "skipped", f"Required statistics disabled: {', '.join(missing_metrics)}.")
        elif not finite(t.get("mi_nats")) or (p["type"] == "numerical" and
                not all(finite(t.get(k)) for k in ["cohens_d", "raw_numeric_auc"])):
            log(name, "weak_signal", "skipped", "Required statistics are undefined, for example because a class has too few usable rows or zero pooled variance.")
        elif p["type"] == "categorical" and p.get("pooled_row_fraction", 0) > .5:
            log(name, "weak_signal", "skipped", "More than half of rows fall in OTHER; pooled MI cannot assess the original categories reliably.")
            log(name, "coverage", "triggered", "Most categorical values were pooled for analysis; increase max_categories or review these IDs before deciding.", action="review")
        else:
            low = t["mi_nats"] < rule["max_mi"]
            evidence = {"mi_nats": t["mi_nats"], "max_mi": rule["max_mi"]}
            reason = f"Binned/pooled MI {t['mi_nats']:.6g} < {rule['max_mi']:.6g}"
            if p["type"] == "numerical":
                d, auc = t["cohens_d"], t["raw_numeric_auc"]
                low &= abs(d) < rule["max_abs_d"] and abs(auc - .5) < rule["max_auc_distance"]
                evidence.update({"cohens_d": d, "max_abs_d": rule["max_abs_d"], "raw_auc": auc,
                                 "auc_distance_from_half": abs(auc - .5), "max_auc_distance": rule["max_auc_distance"]})
                reason += (f", |Cohen's d| {abs(d):.6g} < {rule['max_abs_d']:.6g}, and "
                           f"|AUC - 0.5| {abs(auc-.5):.6g} < {rule['max_auc_distance']:.6g}")
            reason += "; weak marginal association under these thresholds. Interactions remain possible."
            log(name, "weak_signal", "triggered" if low else "not_triggered",
                reason if low else "At least one enabled marginal-strength criterion exceeds the weak-signal threshold.", evidence)

        rule = cfg["rules"]["direction"]
        expected = cfg["expected_direction"].get(name)
        if not rule["enabled"]:
            log(name, "direction", "disabled", "Rule disabled in configuration.")
        elif p["type"] != "numerical":
            log(name, "direction", "not_applicable", "Unordered categories have no global positive/negative direction.")
        else:
            d, auc = t.get("cohens_d"), t.get("raw_numeric_auc")
            signals, evidence = {}, {"expected_direction": expected}
            if enabled(cfg, "cohens_d") and finite(d):
                evidence["cohens_d"] = d
                if abs(d) >= rule["min_abs_d"] and d != 0:
                    signals["Cohen's d"] = "positive" if d > 0 else "negative"
            if enabled(cfg, "auc") and finite(auc):
                evidence["raw_auc"] = auc
                if abs(auc - .5) >= rule["min_auc_distance"] and auc != .5:
                    signals["raw-score AUC"] = "positive" if auc > .5 else "negative"
            evidence.update({"min_abs_d": rule["min_abs_d"], "min_auc_distance": rule["min_auc_distance"]})
            conflict = len(set(signals.values())) > 1
            wrong = {metric: sign for metric, sign in signals.items() if expected and sign != expected}
            descriptions = ", ".join(f"{metric}: {sign}" for metric, sign in signals.items())
            if wrong or conflict:
                why = (f"Expected {expected} association with label=1; observed {descriptions}. " if wrong else "")
                if conflict:
                    why += "Mean-difference and rank-based direction disagree; inspect skewness, outliers and nonmonotonicity. "
                why += "This is an association check, not a causal sign or monotonicity constraint."
                log(name, "direction", "triggered", why, evidence)
            else:
                status = "not_triggered" if signals else "skipped"
                reason = (f"Observed {descriptions}; no configured direction mismatch."
                          if signals else "Direction metrics are disabled, undefined, or below the configured minimum effect size.")
                log(name, "direction", status, reason, evidence)

    # Explicit user exclusions apply before anchor selection so an excluded column
    # can never justify excluding another feature.
    for name in cfg["force_exclude"]:
        actions[name] = "exclude"
        log(name, "override", "triggered", "Explicit force_exclude in configuration.", action="exclude")
    for name in cfg["force_keep"]:
        actions[name] = "keep"
        reasons[name].append("Explicit force_keep overrides automated decisions; earlier findings remain in rule_evaluations.csv.")
        log(name, "override", "triggered", "Explicit force_keep in configuration.", action="keep")

    rule = cfg["rules"]["redundancy"]
    metric_col = "spearman_r" if rule["metric"] == "spearman" else "pearson_r"
    pair_lookup = {frozenset([row["feature_a"], row["feature_b"]]): row for row in pairs.to_dict("records")}
    preference = {name: i for i, name in enumerate(cfg["prefer"])}
    def order(name):
        p, t = profile[name], target[name]
        missing = p.get("unusable_fraction")
        if not finite(missing):
            missing = p["missing_fraction"]
        auc = t.get("raw_numeric_auc")
        strength = abs(auc - .5) if finite(auc) else 0.0
        return (name not in cfg["force_keep"], preference.get(name, len(preference)),
                float(missing), -strength, names.index(name))

    anchors = []
    for name in sorted(names, key=order):
        if actions[name] == "exclude":
            log(name, "redundancy", "skipped", "Already excluded by another rule; cannot act as a retained comparison feature.")
            continue
        if not rule["enabled"]:
            log(name, "redundancy", "disabled", "Rule disabled in configuration.")
            continue
        if not enabled(cfg, rule["metric"]):
            log(name, "redundancy", "skipped", f"Required pair statistic {rule['metric']} is disabled.")
            continue
        if cfg["features"][name] != "numerical":
            log(name, "redundancy", "not_applicable", "Automatic redundancy decisions currently use numeric Pearson/Spearman pairs; inspect mixed/categorical associations separately.")
            continue
        match = None
        eligible = 0
        for anchor in anchors:
            row = pair_lookup.get(frozenset([name, anchor]), {})
            value = row.get(metric_col)
            if not finite(value) or row.get("valid_rows", 0) < rule["min_pair_rows"]:
                continue
            if row.get("valid_rows", 0) / max(1, row.get("sample_rows", 0)) < rule["min_pair_fraction"]:
                continue
            eligible += 1
            score = abs(value) if rule["sign"] == "absolute" else value if rule["sign"] == "positive" else -value
            if score >= rule["threshold"]:
                match = (anchor, value, row)
                break
        if match:
            anchor, value, row = match
            reason = (f"{rule['metric'].capitalize()} correlation with retained '{anchor}' is {value:+.6f}; "
                      f"{rule['sign']} correlation threshold {rule['threshold']:.3f}, "
                      f"based on {int(row['valid_rows']):,} pairwise-complete sampled rows. "
                      "Representative chosen by force_keep, prefer order, lower missingness, greater |AUC-0.5|, then config order.")
            log(name, "redundancy", "triggered", reason,
                {"metric": rule["metric"], "value": value, "threshold": rule["threshold"],
                 "sign": rule["sign"], "valid_rows": int(row["valid_rows"]), "retained_feature": anchor}, other=anchor)
        else:
            log(name, "redundancy", "not_triggered" if eligible or not anchors else "skipped",
                "No above-threshold direct correlation with an earlier retained representative." if eligible or not anchors else
                "No eligible computed pairs to retained representatives; check max_pairs and minimum pair support.")
        if actions[name] != "exclude":
            anchors.append(name)

    # A forced keep wins even if a later direction/redundancy review was recorded.
    for name in cfg["force_keep"]:
        actions[name] = "keep"
    rows = []
    for name in names:
        t, p = target[name], profile[name]
        if not any(enabled(cfg, metric) for metric in ["mutual_information", "auc", "cohens_d", "ks", "cramers_v_target"]):
            log(name, "coverage", "triggered", "No target-strength statistics enabled; predictive relevance was not assessed.", action="review")
        if name in cfg["force_keep"]:
            actions[name] = "keep"
        if not reasons[name]:
            reasons[name].append("No enabled exclusion or review rule fired; retain as a candidate for later model validation.")
        auc = t.get("raw_numeric_auc")
        if finite(auc) and auc < .5:
            reasons[name].append(f"Raw AUC {auc:.4f} indicates an inverse ranking; negating the score gives AUC {1-auc:.4f}. This alone is not an exclusion reason.")
        row = {"feature": name, "type": p["type"], "decision": actions[name],
               "expected_direction": cfg["expected_direction"].get(name, "unspecified"),
               "related_features": "; ".join(related[name]), "reason": " ".join(reasons[name])}
        for metric in ["cohens_d", "hedges_g", "raw_numeric_auc", "direction_free_auc", "mi_nats"]:
            if metric in t:
                row[metric] = t[metric]
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(audit)
