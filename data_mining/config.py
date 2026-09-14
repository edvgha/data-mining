"""Audit defaults and strict YAML configuration validation."""

from pathlib import Path

import yaml

from .settings import EXTRA_DEFAULTS, configure_decisions

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


def load_config(path: str | Path) -> dict:
    """Load YAML, reject ambiguous declarations, and fill validated defaults."""
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
    for key in [
        "bins",
        "max_categories",
        "rare_min_count",
        "min_cell_count",
        "min_cell_events",
        "sample_rows",
        "max_joint_cells",
        "max_time_periods",
    ]:
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
        if (
            not isinstance(group, list)
            or not 2 <= len(group) <= 3
            or not all(isinstance(x, str) for x in group)
        ):
            raise ValueError("Each interaction must contain two or three feature names.")
        if len(set(group)) != len(group) or not set(group) <= set(features):
            raise ValueError(f"Invalid interaction: {group}")
        key = tuple(sorted(group))
        if key in seen:
            raise ValueError(f"Duplicate interaction: {group}")
        seen.add(key)
    return configure_decisions(cfg)
