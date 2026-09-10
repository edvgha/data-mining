"""Time boundaries and train-only feature encoding."""
import difflib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb


def read_data(path, cfg):
    required = list(dict.fromkeys([cfg["target"], cfg["time_column"], *cfg["features"],
        *[c for g in cfg["group_by"] for c in g],
        *[c for c in [cfg["split"]["entity_column"], cfg["bootstrap"]["cluster_column"]] if c]]))
    if str(path).lower().endswith(".parquet") is False:
        raise ValueError("Provide a local .parquet file.")
    schema = pq.read_schema(path)
    missing = set(required) - set(schema.names)
    if missing:
        hints = {c: difflib.get_close_matches(c, schema.names, n=3) for c in sorted(missing)}
        raise ValueError(f"Missing configured columns (no automatic renaming): {hints}")
    if len(set(schema.names)) != len(schema.names):
        raise ValueError("Duplicate Parquet column names are not supported.")
    # Single-threaded read/conversion also supports the existing macOS workflow.
    table = pq.read_table(path, columns=required, use_threads=False)
    frame = pd.DataFrame({c: table[c].to_pandas(use_threads=False) for c in required})
    if frame.empty or frame[cfg["target"]].isna().any() or not frame[cfg["target"]].isin([0, 1]).all():
        raise ValueError("Target must contain only nonmissing numeric 0/1 labels.")
    col = cfg["time_column"]
    if pd.api.types.is_numeric_dtype(frame[col]):
        raise ValueError("Time column must be datetime/date/ISO text; convert numeric epoch timestamps explicitly.")
    frame[col] = pd.to_datetime(frame[col], errors="raise", utc=True, format="mixed")
    if frame[col].isna().any():
        raise ValueError("Time column contains missing values.")
    return frame.sort_values(col, kind="stable").reset_index(drop=True)


def split_data(frame, cfg):
    col, s = cfg["time_column"], cfg["split"]
    times = frame[col].drop_duplicates().sort_values().to_numpy()
    if len(times) < 5:
        raise ValueError("At least five distinct timestamps are required.")
    gap = pd.Timedelta(s["gap"])
    if pd.isna(gap) or gap < pd.Timedelta(0):
        raise ValueError("split.gap must be a nonnegative duration, e.g. 1D.")
    if s["train_end"] is not None:
        train_end, validation_end = [pd.to_datetime(s[k], utc=True) for k in ["train_end", "validation_end"]]
    else:
        # Fractions of DISTINCT timestamps; equal timestamps never cross a boundary.
        train_end = pd.Timestamp(times[int(len(times) * s["train_fraction"])-1])
        validation_end = pd.Timestamp(times[int(len(times) * (s["train_fraction"] + s["validation_fraction"]))-1])
    if not train_end < validation_end:
        raise ValueError("train_end must precede validation_end.")
    t = frame[col]
    masks = {"train": t <= train_end,
             "validation": (t > train_end + gap) & (t <= validation_end),
             "test": t > validation_end + gap}
    train_times = t[masks["train"]].drop_duplicates().to_numpy()
    cut = int(len(train_times) * (1 - s["early_stopping_fraction"]))
    if not 0 < cut < len(train_times):
        raise ValueError("Not enough training timestamps for the early-stopping tail.")
    early_end = pd.Timestamp(train_times[cut-1])
    masks["fit"] = t <= early_end
    masks["early_stop"] = (t > early_end + gap) & masks["train"]
    parts = {name: frame.loc[mask].reset_index(drop=True) for name, mask in masks.items()}
    for name, part in parts.items():
        if part.empty:
            raise ValueError(f"{name} split is empty; adjust boundaries or gap.")
        if name != "test" and part[cfg["target"]].nunique() != 2:
            raise ValueError(f"{name} needs both target classes; enlarge the time window.")
    # Optional strict entity isolation: fail explicitly, never silently reassign future rows.
    entity = s["entity_column"]
    if entity:
        for partition_names in [["train", "validation", "test"], ["fit", "early_stop"]]:
            sets = []
            for name in partition_names:
                if parts[name][entity].isna().any():
                    raise ValueError(f"Missing split entity in {name}.")
                values = set(parts[name][entity])
                if any(values & prev for prev in sets):
                    raise ValueError(f"{entity} crosses time partitions; choose boundaries/gap that keep entities intact.")
                sets.append(values)
    summary = [{"split": name, "rows": len(p), "actual_clicks": int(p[cfg["target"]].sum()),
                "positive_rate": float(p[cfg["target"]].mean()), "start": str(p[col].min()), "end": str(p[col].max())}
               for name, p in parts.items()]
    return parts, {"partitions": summary, "excluded_gap_rows": int(len(frame)-sum(len(parts[k]) for k in ["train", "validation", "test"])),
                   "inner_gap_rows": len(parts["train"])-len(parts["fit"])-len(parts["early_stop"]),
                   "train_end": str(train_end), "validation_end": str(validation_end)}


class Encoder:
    """Persisted categories; future/unseen categories become missing, without label encoding."""
    def __init__(self, features, categories=None):
        self.features = features
        self.categories = categories or {}

    def fit(self, frame):
        self.categories = {c: sorted(frame[c].dropna().astype(str).unique().tolist())
                           for c, kind in self.features.items() if kind == "categorical"}
        return self

    def transform(self, frame):
        out = pd.DataFrame(index=frame.index)
        for c, kind in self.features.items():
            if kind == "categorical":
                out[c] = pd.Categorical(frame[c].astype("string"), categories=self.categories[c])
            else:
                values = pd.to_numeric(frame[c], errors="raise").astype("float32")
                out[c] = values.replace([np.inf, -np.inf], np.nan)
        # Legal, stable native feature names; original names are kept in the encoder/importance tables.
        out.columns = [f"f{i}" for i in range(len(self.features))]
        return out

    def matrix(self, frame, target=None, nthread=4):
        return xgb.DMatrix(self.transform(frame), label=frame[target].to_numpy() if target else None,
                           enable_categorical=True, nthread=nthread)

    def to_dict(self):
        return {"features": self.features, "categories": self.categories}
