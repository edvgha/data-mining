"""Read and validate local Parquet files while preserving Arrow value precision."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def dataframe_from_arrow(table):
    """Build pandas columns without Arrow's pandas converter or timestamp lists."""
    metadata = {
        item["field_name"]: item for item in (table.schema.pandas_metadata or {}).get("columns", [])
    }
    data = {}
    for i, (field, column) in enumerate(zip(table.schema, table.columns), 1):
        logger.debug(
            "Converting column [%s/%s]: %s (%s)...",
            i,
            table.num_columns,
            field.name,
            field.type,
        )
        if pa.types.is_timestamp(field.type):
            # Keep exact epoch ticks; converting nullable integers to float64
            # would round nanoseconds. NumPy's NaT uses the minimum int64 value.
            ticks = column.cast(pa.int64()).fill_null(np.iinfo(np.int64).min)
            values = ticks.to_numpy(zero_copy_only=False).view(f"datetime64[{field.type.unit}]")
            series = pd.Series(values, copy=False)
            if field.type.tz is not None:
                # Arrow's stored ticks are UTC instants, including across DST.
                series = series.dt.tz_localize("UTC").dt.tz_convert(field.type.tz)
            data[field.name] = series
            continue
        values = column.to_pylist()
        if pa.types.is_dictionary(field.type):
            unified = column.unify_dictionaries()
            categories = unified.chunk(0).dictionary.to_pylist() if unified.num_chunks else []
            category_type = field.type.value_type
            category_dtype = (
                object
                if pa.types.is_date(category_type) or pa.types.is_time(category_type)
                else category_type.to_pandas_dtype()
            )
            dtype = pd.CategoricalDtype(
                pd.Index(categories, dtype=category_dtype), ordered=field.type.ordered
            )
        else:
            # Only extension dtypes need pandas metadata. Physical Arrow types
            # retain numeric widths and timestamp units/timezones on their own.
            stored = metadata.get(field.name, {}).get("numpy_type")
            dtype = pd.api.types.pandas_dtype(stored) if stored else None
            if not isinstance(dtype, pd.api.extensions.ExtensionDtype):
                if pa.types.is_integer(field.type) and column.null_count:
                    # Avoid rounding large IDs through float64 when nulls exist.
                    prefix = "UInt" if pa.types.is_unsigned_integer(field.type) else "Int"
                    dtype = f"{prefix}{field.type.bit_width}"
                elif pa.types.is_boolean(field.type) and column.null_count:
                    dtype = "boolean"
                elif pa.types.is_date(field.type) or pa.types.is_time(field.type):
                    dtype = object
                else:
                    dtype = field.type.to_pandas_dtype()
        if (
            pa.types.is_string(field.type)
            or pa.types.is_large_string(field.type)
            or pa.types.is_binary(field.type)
            or pa.types.is_large_binary(field.type)
        ):
            # Share repeated category strings, rather than retaining one Python
            # string object per row after converting a low-cardinality column.
            shared = {}
            values = [shared.setdefault(value, value) for value in values]
        data[field.name] = pd.Series(values, dtype=dtype)
    return pd.DataFrame(data, copy=False)


def read_dataset(path: str | Path, cfg: dict):
    """Return validated rows, undeclared columns, and the original schema names."""
    path = Path(path)
    if path.suffix.lower() != ".parquet" or not path.is_file():
        raise ValueError("dataset must be an existing .parquet file.")
    logger.debug("Opening Parquet metadata...")
    # This API accepts one local file. Avoid the dataset scanner, background
    # read-ahead, and threaded decoding.
    with pq.ParquetFile(path, pre_buffer=False) as parquet:
        columns = parquet.schema_arrow.names
        logger.debug(
            "Metadata: %s rows; %s columns; %s row groups.",
            parquet.metadata.num_rows,
            len(columns),
            parquet.num_row_groups,
        )
        if len(columns) != len(set(columns)):
            raise ValueError("Parquet contains duplicate column names.")
        declared = set(cfg["features"]) | set(cfg["ignore"]) | {cfg["target"]}
        declared |= {cfg["time_column"], cfg["group_column"]} - {None}
        missing = declared - set(columns)
        if missing:
            raise ValueError(
                f"Configured columns absent from Parquet: {sorted(missing)}. "
                "Column names are case-sensitive and must match the schema exactly."
            )
        extra = set(columns) - declared
        if extra and cfg["strict_schema"]:
            raise ValueError(
                f"Undeclared columns: {sorted(extra)}. Add them to features or ignore."
            )
        # Read configured physical columns. A stored pandas index is not restored;
        # it can be declared as an ordinary feature/context column or ignored.
        selected = [x for x in columns if x in declared and x not in cfg["ignore"]]
        logger.debug("Reading %s selected columns (single thread)...", len(selected))
        table = parquet.read(columns=selected, use_threads=False)
    logger.debug("Building pandas DataFrame column by column...")
    df = dataframe_from_arrow(table)
    del table
    if df.empty:
        raise ValueError("Dataset is empty.")
    logger.debug("Validating target: %s...", cfg["target"])
    y = df[cfg["target"]]
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1, with no missing values.")
    if y.nunique() != 2:
        raise ValueError("Both target classes must be present for association analysis.")
    logger.debug("Validating %s feature dtypes...", len(cfg["features"]))
    for name, kind in cfg["features"].items():
        s = df[name]
        if kind == "numerical" and (
            not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_complex_dtype(s)
        ):
            raise ValueError(
                f"{name}: numerical features require a real numeric dtype, got {s.dtype}."
            )
        if kind == "categorical" and not (
            isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.infer_dtype(s, skipna=True)
            in {
                "string",
                "unicode",
                "bytes",
                "integer",
                "floating",
                "mixed-integer-float",
                "boolean",
                "empty",
            }
        ):
            raise ValueError(
                f"{name}: categorical values must be scalar strings, numbers, or booleans."
            )
    return df, sorted(extra), columns
