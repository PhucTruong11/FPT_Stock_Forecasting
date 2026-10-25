"""Shared OHLCV validation and snapshot I/O. No price or volume imputation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
OHLCV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invalid_candle_mask(data: pd.DataFrame) -> pd.Series:
    tolerance = data["High"].abs() * 1e-6
    return ((data["High"] + tolerance < data[["Open", "Close", "Low"]].max(axis=1)) |
            (data["Low"] - tolerance > data[["Open", "Close", "High"]].min(axis=1)))


def normalize_ohlcv(
    frame: pd.DataFrame, *, allow_untraded_missing_prices: bool = False,
    allow_invalid_candles: bool = False,
) -> pd.DataFrame:
    """Normalize one ticker; reject ambiguous duplicates and invalid observations."""
    if frame is None or frame.empty:
        raise ValueError("OHLCV source is empty; existing files were not replaced.")
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        fields = set(OHLCV_COLUMNS[1:])
        levels = [i for i in range(data.columns.nlevels)
                  if fields.issubset(set(data.columns.get_level_values(i)))]
        if len(levels) != 1:
            raise ValueError("Cannot identify OHLCV fields in MultiIndex columns.")
        field_level = levels[0]
        for i in range(data.columns.nlevels):
            if i != field_level and data.columns.get_level_values(i).nunique() != 1:
                raise ValueError("Expected exactly one ticker, received multiple tickers.")
        data.columns = data.columns.get_level_values(field_level)
    if "Date" not in data.columns:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Expected a Date column or a DatetimeIndex.")
        data.index.name = "Date"
        data = data.reset_index()
    missing = set(OHLCV_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = data[OHLCV_COLUMNS].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    if data["Date"].isna().any():
        raise ValueError("Date contains missing values.")
    if data["Date"].dt.tz is not None:
        data["Date"] = data["Date"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None)
    if not data["Date"].eq(data["Date"].dt.normalize()).all():
        raise ValueError("Expected daily observations at midnight, not intraday data.")
    for column in OHLCV_COLUMNS[1:]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    prices = data[["Open", "High", "Low", "Close"]]
    allowed_missing = prices.isna().all(axis=1) & data["Volume"].eq(0)
    if not allow_untraded_missing_prices:
        allowed_missing[:] = False
    if (not np.isfinite(prices.loc[~allowed_missing].to_numpy(dtype=float)).all() or
        not np.isfinite(data["Volume"].to_numpy(dtype=float)).all()):
        raise ValueError("OHLCV contains missing/infinite values; no filling is allowed.")
    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be strictly positive.")
    if (data["Volume"] < 0).any() or not np.isclose(
        data["Volume"], np.round(data["Volume"]), rtol=0, atol=1e-6
    ).all():
        raise ValueError("Volume must be a nonnegative integer.")
    if not allow_invalid_candles and invalid_candle_mask(data).any():
        raise ValueError("Invalid candle: Low <= Open/Close <= High must hold.")
    if data.groupby("Date")[OHLCV_COLUMNS[1:]].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Conflicting OHLCV observations share the same Date.")
    data = data.drop_duplicates("Date").sort_values("Date", kind="stable").reset_index(drop=True)
    if ((data["Date"].dt.dayofweek >= 5) & (data["Volume"] > 0)).any():
        raise ValueError("Positive-volume weekend observation requires source review.")
    data["Volume"] = data["Volume"].astype("int64")
    return data


def write_snapshot(path: Path, value: pd.DataFrame | dict) -> None:
    """Stage before replacing; preserve each existing version in data/backups/."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            if isinstance(value, pd.DataFrame):
                value.to_csv(stream, index=False, date_format="%Y-%m-%d")
            else:
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        if path.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = BASE_DIR / "data" / "backups" / stamp / path.parent.name / path.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
