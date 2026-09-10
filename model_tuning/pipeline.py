"""Chronological native-API tuning and final refit."""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import gc
import platform
import uuid
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yaml
from .data import Encoder, read_data, split_data
from .metrics import (aggregate_units, bootstrap_metrics, evaluate, group_evaluation,
                      resample_totals, row_statistics, unit_codes)
from .report import save_json, write_report


def base_params(cfg):
    return {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
            "booster": "gbtree", "seed": cfg["seed"], "nthread": cfg["nthread"],
            "validate_parameters": True, **cfg["tuning"]["params"]}


def suggest(trial, space):
    params = {}
    for name, specification in space.items():
        spec = dict(specification)
        kind = spec.pop("type")
        params[name] = getattr(trial, "suggest_"+kind)(name, **spec)
    return params


def train_early(params, fit, early, cfg):
    history = {}
    model = xgb.train(params, fit, num_boost_round=cfg["tuning"]["num_boost_round"],
                      evals=[(fit, "fit"), (early, "early_stop")],
                      early_stopping_rounds=cfg["tuning"]["early_stopping_rounds"],
                      evals_result=history, verbose_eval=False)
    return model.best_iteration+1, history


def run(data_path, cfg):
    print("Reading and validating Parquet...", flush=True)
    frame = read_data(data_path, cfg)
    parts, split_summary = split_data(frame, cfg)
    for row in split_summary["partitions"]:
        print(f"  {row['split']}: {row['rows']:,} rows, {row['actual_clicks']:,} positives; {row['start']} to {row['end']}", flush=True)
    for name in ["train", "validation", "test"]:
        count = len(np.unique(unit_codes(parts[name], cfg)))
        if name != "test" and count < 2:
            raise ValueError(f"{name} needs at least two bootstrap units; adjust time_frequency or sampling method.")
    out = Path(cfg["output_dir"]) / (datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    (out/"config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    save_json(out/"status.json", {"status": "running"})
    try:
        return _fit(data_path, cfg, frame, parts, split_summary, out)
    except BaseException as exc:
        save_json(out/"status.json", {"status": "failed", "error": str(exc)})
        raise


def _fit(data_path, cfg, frame, parts, split_summary, out):
    target, threads = cfg["target"], cfg["nthread"]
    t, b = cfg["tuning"], cfg["bootstrap"]
    early_encoder = Encoder(cfg["features"]).fit(parts["fit"])
    fit = early_encoder.matrix(parts["fit"], target, threads)
    early = early_encoder.matrix(parts["early_stop"], target, threads)
    train_encoder = Encoder(cfg["features"]).fit(parts["train"])
    train = train_encoder.matrix(parts["train"], target, threads)
    validation = train_encoder.matrix(parts["validation"], target, threads)
    validation_y = parts["validation"][target].to_numpy()
    train_rate = float(parts["train"][target].mean())
    codes = unit_codes(parts["validation"], cfg)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
                                storage=f"sqlite:///{out / 'study.sqlite3'}", study_name="binary_xgboost")
    def objective(trial):
        params = {**base_params(cfg), **suggest(trial, t["search_space"])}
        rounds, _ = train_early(params, fit, early, cfg)
        candidate = xgb.train(params, train, num_boost_round=rounds, verbose_eval=False)
        pred = candidate.predict(validation)
        stats = row_statistics(validation_y, pred, train_rate)
        units = aggregate_units(stats[:, [0, 4]], codes)
        losses = np.array([total[1]/total[0] for total in resample_totals(units, b["n_resamples"], cfg["seed"])])
        loss, sd = float(stats[:, 4].mean()), float(losses.std(ddof=1))
        trial.set_user_attr("num_boost_round", rounds)
        trial.set_user_attr("validation_logloss", loss)
        trial.set_user_attr("bootstrap_logloss_mean", float(losses.mean()))
        trial.set_user_attr("bootstrap_logloss_std", sd)
        return loss + t["stability_penalty"]*sd
    print(f"Tuning {t['n_trials']} trials; final test period is held out...", flush=True)
    study.optimize(objective, n_trials=t["n_trials"], timeout=t["timeout_seconds"], n_jobs=1,
                   gc_after_trial=True, show_progress_bar=False)
    best = study.best_trial
    params = {**base_params(cfg), **best.params}
    rounds = best.user_attrs["num_boost_round"]
    _, history = train_early(params, fit, early, cfg)
    selected = xgb.train(params, train, num_boost_round=rounds, verbose_eval=False)
    validation_pred = selected.predict(validation)
    validation_metrics = {"dataset": "validation (selection)", **evaluate(validation_y, validation_pred, train_rate)}
    del fit, early, train, validation, selected, early_encoder, train_encoder
    gc.collect()
    print(f"Refitting trial {best.number}, {rounds} boosting rounds on train + validation...", flush=True)
    refit_frame = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    encoder = Encoder(cfg["features"]).fit(refit_frame)
    refit = encoder.matrix(refit_frame, target, threads)
    test = encoder.matrix(parts["test"], target, threads)
    refit_rate = float(refit_frame[target].mean())
    model = xgb.train(params, refit, num_boost_round=rounds, verbose_eval=False)
    model.save_model(out/"model.ubj")
    save_json(out/"encoder.json", encoder.to_dict())
    prediction = model.predict(test)
    y = parts["test"][target].to_numpy()
    overall = pd.DataFrame([validation_metrics, {"dataset": "test (held out)", **evaluate(y, prediction, refit_rate)},
                            {"dataset": "test constant refit-rate baseline", **evaluate(y, np.full(len(y), refit_rate), refit_rate)}])
    print("Computing bootstrap intervals and group calibration...", flush=True)
    intervals = bootstrap_metrics(parts["test"], y, prediction, refit_rate, cfg)
    groups = group_evaluation(parts["test"], prediction, refit_rate, cfg)
    stability = []
    train_codes = unit_codes(refit_frame, cfg)
    unit_count = int(train_codes.max())+1
    rng = np.random.default_rng(cfg["seed"]+1)
    refit_y = refit_frame[target].to_numpy()
    for repeat in range(b["training_repeats"]):
        print(f"Training bootstrap refit {repeat+1}/{b['training_repeats']}...", flush=True)
        counts = rng.multinomial(unit_count, np.full(unit_count, 1/unit_count))
        weights = counts[train_codes].astype(np.float32)
        weighted_rate = float(np.average(refit_y, weights=weights))
        if weighted_rate in (0., 1.):
            stability.append({"repeat": repeat, "status": "skipped_single_class_resample"})
            continue
        refit.set_weight(weights)
        replica = xgb.train(params, refit, num_boost_round=rounds, verbose_eval=False)
        replica_p = replica.predict(test)
        stability.append({"repeat": repeat, "status": "ok", **evaluate(y, replica_p, weighted_rate),
                          "mean_abs_prediction_change": float(np.abs(replica_p-prediction).mean())})
        del replica
    refit.set_weight(np.array([], dtype=np.float32))
    stability = pd.DataFrame(stability, columns=None if stability else ["repeat", "status"])
    print("Computing native TreeSHAP and feature importance...", flush=True)
    sample = np.sort(np.random.default_rng(cfg["seed"]).choice(len(y), min(len(y), cfg["report"]["shap_rows"]), replace=False))
    shap_frame = parts["test"].iloc[sample].copy()
    shap_matrix = encoder.matrix(shap_frame, nthread=threads)
    shap_values = model.predict(shap_matrix, pred_contribs=True)
    margins = model.predict(shap_matrix, output_margin=True)
    if not np.allclose(shap_values.sum(axis=1), margins, atol=1e-4, rtol=1e-4):
        raise RuntimeError("Native SHAP contributions failed the margin additivity check.")
    scores = {metric: model.get_score(importance_type=metric) for metric in ["gain", "total_gain", "weight", "cover", "total_cover"]}
    importance = pd.DataFrame([{"feature": name, "xgboost_name": f"f{i}",
        **{metric: value.get(f"f{i}", 0.) for metric, value in scores.items()},
        "mean_abs_shap": float(np.abs(shap_values[:, i]).mean()), "mean_shap": float(shap_values[:, i].mean())}
        for i, name in enumerate(cfg["features"])]).sort_values("mean_abs_shap", ascending=False)
    shap_features = shap_frame[list(cfg["features"])].reset_index(drop=True)
    pd.DataFrame({"test_row": sample, "target": y[sample], "prediction": prediction[sample], "raw_margin": margins}).to_csv(out/"shap_context.csv", index=False)
    warnings = ["Fractions partition distinct timestamps, not row counts; explicit inclusive cutoffs are also supported.",
                "Validation scores are selected on and optimistic. Test scores evaluate the final refit.",
                "The robust objective uses uncertainty of validation logloss for a fixed candidate, not repeated training per trial.",
                f"Sampling method: {b['method']}; {b['n_resamples']} metric resamples; {b['training_repeats']} training refits. Training refits hold hyperparameters and encoding fixed.",
                "Unweighted population metrics: negative sampling or class balancing changes click-rate calibration.",
                "Feature selection should be defined before viewing this test period; an audit over all dates can itself leak future information."]
    if b["method"] == "row":
        warnings.append("Row bootstrap assumes independent rows; prefer cluster/time_block for repeated sessions or temporal dependence.")
    if b["method"] == "time_block":
        warnings.append("Non-overlapping time blocks are resampled independently. Dependence across blocks and future drift are not captured; choose a duration matching the dependence horizon.")
    if not cfg["split"]["entity_column"]:
        warnings.append("Entity overlap across splits is not checked. Configure split.entity_column to require disjoint search/session IDs.")
    if int(intervals.units.iloc[0]) < 20:
        warnings.append("Fewer than 20 test bootstrap units: percentile intervals may be unstable; use a longer test window.")
    unknown = {c: int((parts["test"][c].notna() & ~parts["test"][c].astype("string").isin(levels)).sum())
               for c, levels in encoder.categories.items()}
    if any(unknown.values()):
        warnings.append("Unseen test categories are mapped to missing; counts are recorded in summary.json.")
    stat = Path(data_path).stat()
    summary = {"best_trial": best.number, "best_objective": best.value, "best_params": params,
               "best_num_boost_round": rounds, "split": split_summary, "test_metrics": overall.iloc[1].to_dict(),
               "train_baseline_rate": train_rate, "refit_baseline_rate": refit_rate,
               "unseen_test_category_rows": unknown, "shap_rows": len(sample),
               "training_bootstrap_completed": int((stability.status == "ok").sum()),
               "data": {"path": str(Path(data_path).resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "rows": len(frame)},
               "versions": {p: version(p) for p in ["xgboost", "optuna", "pandas", "numpy", "pyarrow"]},
               "python": platform.python_version(), "warnings": warnings}
    save_json(out/"summary.json", summary)
    save_json(out/"best_params.json", {"params": params, "num_boost_round": rounds})
    save_json(out/"learning_curve.json", history)
    if cfg["report"]["save_predictions"]:
        columns = list(dict.fromkeys([cfg["time_column"], target, *[c for g in cfg["group_by"] for c in g]]))
        output = parts["test"][columns].copy()
        if "__predicted_probability__" in output:
            raise ValueError("Reserved prediction column __predicted_probability__ already exists.")
        output["__predicted_probability__"] = prediction
        output.to_parquet(out/"test_predictions.parquet", index=False)
    write_report(out, summary, overall, intervals, groups, study.trials_dataframe(), importance,
                 shap_values, shap_features, history, parts["test"], prediction, cfg, stability)
    save_json(out/"status.json", {"status": "complete"})
    return out
