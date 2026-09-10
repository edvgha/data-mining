"""Strict, intentionally small configuration surface."""
from copy import deepcopy
from pathlib import Path
import math
import yaml

DEFAULTS = {
    "target": None, "time_column": None, "features": {}, "group_by": [],
    "output_dir": "../report/model_tuning", "seed": 42, "nthread": 4,
    "split": {"train_fraction": .6, "validation_fraction": .2,
              "early_stopping_fraction": .15, "train_end": None,
              "validation_end": None, "gap": "0D", "entity_column": None},
    "tuning": {"n_trials": 30, "timeout_seconds": None, "num_boost_round": 1000,
               "early_stopping_rounds": 50, "stability_penalty": 1.0,
               "params": {}, "search_space": {
                   "eta": {"type": "float", "low": .02, "high": .2, "log": True},
                   "max_depth": {"type": "int", "low": 3, "high": 8},
                   "min_child_weight": {"type": "float", "low": 1., "high": 100., "log": True},
                   "subsample": {"type": "float", "low": .6, "high": 1.},
                   "colsample_bytree": {"type": "float", "low": .6, "high": 1.},
                   "lambda": {"type": "float", "low": .01, "high": 100., "log": True},
                   "alpha": {"type": "float", "low": 1e-8, "high": 10., "log": True},
               }},
    "bootstrap": {"method": "time_block", "time_frequency": "1D", "cluster_column": None,
                  "n_resamples": 200, "confidence": .95, "training_repeats": 5},
    "report": {"shap_rows": 2000, "plot_top_features": 20, "min_group_rows": 100,
               "min_group_clicks": 10, "save_predictions": False},
}
# Only these controls are tunable: the loss/probability semantics cannot be overridden.
PARAMETERS = {"eta", "max_depth", "min_child_weight", "subsample", "colsample_bytree",
              "colsample_bylevel", "colsample_bynode", "lambda", "alpha", "gamma",
              "max_delta_step", "max_bin", "max_cat_to_onehot", "max_cat_threshold"}
INTEGER_PARAMS = {"max_depth", "max_bin", "max_cat_to_onehot", "max_cat_threshold"}


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate settings instead of silently replacing earlier values."""


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _number(value, label, low=0, high=math.inf, integer=False):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be a finite number in [{low}, {high}].")
    if integer and type(value) is not int:
        raise ValueError(f"{label} must be an integer.")


def _param(name, value):
    if name not in PARAMETERS:
        raise ValueError(f"Unsupported XGBoost parameter {name}; allowed: {sorted(PARAMETERS)}")
    _number(value, name, integer=name in INTEGER_PARAMS)
    if name in {"eta", "subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode"} and not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1].")
    if name in INTEGER_PARAMS and value < (2 if name == "max_bin" else 1):
        raise ValueError(f"{name} is too small.")


def configure(raw):
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError(f"Config must be a mapping with keys: {list(DEFAULTS)}")
    cfg = deepcopy(DEFAULTS)
    for key, value in raw.items():
        if key in {"split", "tuning", "bootstrap", "report"}:
            if not isinstance(value, dict) or set(value) - set(cfg[key]):
                raise ValueError(f"Unknown settings in {key}.")
            cfg[key].update(value)
        else:
            cfg[key] = value
    for key in ["target", "time_column", "output_dir"]:
        if not isinstance(cfg[key], str) or not cfg[key]:
            raise ValueError(f"{key} must be a nonempty string.")
    fs = cfg["features"]
    if not isinstance(fs, dict) or not fs or any(not isinstance(k, str) or not k or v not in {"numerical", "categorical"} for k, v in fs.items()):
        raise ValueError("features must map names to numerical/categorical.")
    if cfg["target"] == cfg["time_column"] or {cfg["target"], cfg["time_column"]} & set(fs):
        raise ValueError("Target and time column must be distinct and cannot be model features.")
    if "__bias__" in fs:
        raise ValueError("Feature name __bias__ is reserved for the SHAP bias column.")
    groups = cfg["group_by"]
    if not isinstance(groups, list):
        raise ValueError("group_by must be a list of columns, or a list of column lists.")
    if groups and all(isinstance(x, str) for x in groups):
        groups = [groups]
    if any(not isinstance(g, list) or not g or any(not isinstance(x, str) or not x for x in g)
           or len(set(g)) != len(g) or cfg["target"] in g for g in groups):
        raise ValueError("Each group_by entry must contain distinct non-target column names.")
    if len({tuple(g) for g in groups}) != len(groups):
        raise ValueError("Duplicate group_by definitions.")
    cfg["group_by"] = groups
    _number(cfg["seed"], "seed", high=2**32-1, integer=True)
    _number(cfg["nthread"], "nthread", low=1, integer=True)
    s, t, b, r = (cfg[k] for k in ["split", "tuning", "bootstrap", "report"])
    for key in ["train_fraction", "validation_fraction", "early_stopping_fraction"]:
        _number(s[key], f"split.{key}", low=.001, high=.999)
    if s["train_fraction"] + s["validation_fraction"] >= 1:
        raise ValueError("Train + validation fractions must be below 1 to reserve a test set.")
    if (s["train_end"] is None) != (s["validation_end"] is None):
        raise ValueError("Provide both train_end and validation_end, or neither.")
    for value in [s["entity_column"], b["cluster_column"]]:
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError("Entity and cluster columns must be nonempty names or null.")
    if s["entity_column"] in fs:
        raise ValueError("The split entity column is an identifier and cannot be a feature.")
    if b["method"] not in {"row", "cluster", "time_block"}:
        raise ValueError("bootstrap.method must be row, cluster, or time_block.")
    if b["method"] == "cluster" and not b["cluster_column"]:
        raise ValueError("cluster bootstrap requires cluster_column.")
    if b["cluster_column"] == cfg["target"] or s["entity_column"] == cfg["target"]:
        raise ValueError("Target cannot define sampling units or split entities.")
    for key in ["n_trials", "num_boost_round", "early_stopping_rounds"]:
        _number(t[key], f"tuning.{key}", low=1, integer=True)
    if t["timeout_seconds"] is not None:
        _number(t["timeout_seconds"], "timeout_seconds", low=1)
    _number(t["stability_penalty"], "stability_penalty")
    _number(b["n_resamples"], "n_resamples", low=2, integer=True)
    _number(b["training_repeats"], "training_repeats", integer=True)
    _number(b["confidence"], "confidence", low=.01, high=.999)
    for key in ["shap_rows", "plot_top_features", "min_group_rows", "min_group_clicks"]:
        _number(r[key], f"report.{key}", low=1, integer=True)
    if type(r["save_predictions"]) is not bool:
        raise ValueError("save_predictions must be boolean.")
    if not isinstance(t["params"], dict) or not isinstance(t["search_space"], dict):
        raise ValueError("params and search_space must be mappings.")
    for name, value in t["params"].items():
        _param(name, value)
    if set(t["params"]) & set(t["search_space"]):
        raise ValueError("Fixed params and search_space cannot overlap; remove fixed entries from search_space.")
    for name, spec in t["search_space"].items():
        if not isinstance(spec, dict) or spec.get("type") not in {"int", "float", "categorical"}:
            raise ValueError(f"Invalid search distribution for {name}.")
        if set(spec) - {"type", "low", "high", "log", "step", "choices"}:
            raise ValueError(f"Unknown search settings for {name}.")
        if spec["type"] == "categorical":
            if set(spec) != {"type", "choices"} or not isinstance(spec["choices"], list) or not spec["choices"]:
                raise ValueError(f"{name} categorical search requires choices only.")
            for value in spec["choices"]:
                _param(name, value)
        else:
            if "low" not in spec or "high" not in spec or "choices" in spec:
                raise ValueError(f"{name} requires low/high, without choices.")
            _param(name, spec["low"])
            _param(name, spec["high"])
            if name in INTEGER_PARAMS and spec["type"] != "int":
                raise ValueError(f"{name} needs an integer search distribution.")
            if spec["type"] == "int" and any(type(spec[k]) is not int for k in ["low", "high"]):
                raise ValueError(f"{name} integer bounds must be integers.")
            if spec["low"] > spec["high"] or type(spec.get("log", False)) is not bool:
                raise ValueError(f"Invalid bounds/log flag for {name}.")
            if "step" in spec:
                _number(spec["step"], f"{name}.step", low=1e-12, integer=spec["type"] == "int")
            if spec.get("log") and (spec["low"] <= 0 or "step" in spec):
                raise ValueError(f"{name}: log search requires positive bounds and no step.")
    return cfg


def load_config(path):
    path = Path(path).resolve()
    cfg = configure(yaml.load(path.read_text(), Loader=UniqueKeyLoader))
    cfg["output_dir"] = str((path.parent / cfg["output_dir"]).resolve())
    return cfg
