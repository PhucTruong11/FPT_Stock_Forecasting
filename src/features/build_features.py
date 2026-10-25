"""Build causal FPT features and explicit next-observation labels (schema v2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ta

BASE_DIR = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(BASE_DIR))

from src.data.contracts import file_sha256, invalid_candle_mask, normalize_ohlcv, utc_now, write_snapshot

TICKER = "FPT.VN"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
INTERIM_DATA_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
FEATURE_COLUMNS = [
    "Log_Return", "Return_Lag_1", "Return_Lag_2", "Return_Lag_5",
    "Return_5", "Return_20", "Volatility_5", "Volatility_20",
    "SMA_Distance_20", "SMA_Distance_50", "RSI_14", "ATR_Ratio",
    "Intraday_Range", "Volume_Ratio_20", "Volume_Log_Change", "Volume_ZScore_20",
]
TARGET_COLUMN = "Target_Return_1d"
TARGET_DATE_COLUMN = "Target_Date_1d"


def build_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Keep source intact; exclude untraded observations, never infer holiday dates.

    '1d' means next positive-volume observation in this snapshot, not an independently
    verified HOSE session. Explicit target dates make gaps visible to downstream code.
    """
    raw = normalize_ohlcv(frame, allow_untraded_missing_prices=True, allow_invalid_candles=True)
    invalid = invalid_candle_mask(raw)
    excluded = raw.loc[raw["Volume"].eq(0) | invalid].copy()
    excluded["Reason"] = "zero_volume_excluded_not_confirmed_holiday"
    excluded.loc[excluded["Close"].isna(), "Reason"] = "zero_volume_and_missing_prices"
    excluded.loc[invalid.loc[excluded.index], "Reason"] = "invalid_ohlc_candle"
    # Define label dates BEFORE removing invalid positive-volume candles, so the
    # preceding session does not silently turn into a two-session training target.
    traded = raw.loc[raw["Volume"].gt(0)].copy()
    traded[TARGET_DATE_COLUMN] = traded["Date"].shift(-1)
    traded[TARGET_COLUMN] = np.log(traded["Close"].shift(-1) / traded["Close"])
    next_invalid = invalid.loc[traded.index].shift(-1, fill_value=False)
    traded.loc[next_invalid, TARGET_COLUMN] = np.nan
    data = traded.loc[~invalid.loc[traded.index]].copy().reset_index(drop=True)
    if len(data) < 51:
        raise ValueError("Need at least 51 positive-volume observations after filtering.")
    close, volume = data["Close"], data["Volume"].astype(float)

    # Only past/current eligible candles contribute to features; targets stay separate.
    data["Log_Return"] = np.log(close / close.shift(1))
    for lag in (1, 2, 5):
        data[f"Return_Lag_{lag}"] = data["Log_Return"].shift(lag)
    for window in (5, 20):
        data[f"Return_{window}"] = np.log(close / close.shift(window))
        data[f"Volatility_{window}"] = data["Log_Return"].rolling(window).std(ddof=1)
    for window in (20, 50):
        data[f"SMA_Distance_{window}"] = close / close.rolling(window).mean() - 1
    data["RSI_14"] = ta.momentum.rsi(close, window=14) / 100.0
    data["ATR_Ratio"] = ta.volatility.average_true_range(
        data["High"], data["Low"], close, window=14
    ) / close
    data["Intraday_Range"] = (data["High"] - data["Low"]) / close
    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std(ddof=1)
    data["Volume_Ratio_20"] = volume / volume_mean
    data["Volume_Log_Change"] = np.log(volume).diff()
    data["Volume_ZScore_20"] = ((volume - volume_mean) / volume_std.where(volume_std.ne(0))).mask(
        volume_std.eq(0), 0.0
    )

    warmup_rows = 49  # SMA-50 is the longest required history.
    ready = data.iloc[warmup_rows:].copy().reset_index(drop=True)
    if not np.isfinite(ready[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite feature after warm-up; refusing to drop interior rows.")
    # Close is price context, deliberately absent from the model input allowlist.
    features = ready[["Date", "Close", *FEATURE_COLUMNS]].copy()
    supervised = ready.loc[ready[TARGET_COLUMN].notna(),
                           ["Date", TARGET_DATE_COLUMN, "Close", *FEATURE_COLUMNS, TARGET_COLUMN]].copy()
    if supervised.empty or not np.isfinite(supervised[TARGET_COLUMN]).all():
        raise ValueError("No valid labeled samples.")
    if not supervised[TARGET_DATE_COLUMN].gt(supervised["Date"]).all():
        raise ValueError("Each target date must be later than its forecast origin.")

    report = {
        "schema_version": "fpt-features-v2", "ticker": TICKER,
        "created_at_utc": utc_now(), "raw_rows": len(raw),
        "zero_volume_excluded": int(raw["Volume"].eq(0).sum()),
        "invalid_candle_excluded": int(invalid.sum()), "excluded_rows": len(excluded),
        "eligible_rows": len(data),
        "warmup_rows": warmup_rows, "feature_rows": len(features),
        "labeled_rows": len(supervised), "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS, "context_columns": ["Date", "Close"],
        "target_column": TARGET_COLUMN, "target_date_column": TARGET_DATE_COLUMN,
        "horizon_observations": 1,
        "target_definition": "log(next_positive_volume_close/current_close); reject labels touching invalid candles",
        "unlabeled_feature_rows": int(ready[TARGET_COLUMN].isna().sum()),
        "session_basis": "positive_volume_source_observations",
        "calendar_verified": False, "imputation": "none", "scaler_fitted": False,
        "market_context": "disabled; legacy market_index.csv is not consumed",
        "first_feature_date": str(features["Date"].iloc[0].date()),
        "last_feature_date": str(features["Date"].iloc[-1].date()),
        "last_labeled_origin": str(supervised["Date"].iloc[-1].date()),
        "last_target_date": str(supervised[TARGET_DATE_COLUMN].iloc[-1].date()),
        "target_calendar_gap_days": {
            str(int(days)): int(count) for days, count in
            (supervised[TARGET_DATE_COLUMN] - supervised["Date"]).dt.days.value_counts().sort_index().items()
        },
        "model_compatibility": "Existing v1 model/scaler artifacts require retraining.",
    }
    return features, supervised.reset_index(drop=True), excluded, report


def run_pipeline(
    input_path: Path = RAW_DATA_DIR / f"{TICKER}_raw.csv",
    interim_dir: Path = INTERIM_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> dict:
    input_path, interim_dir, processed_dir = map(Path, (input_path, interim_dir, processed_dir))
    metadata_path = input_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise ValueError(f"Missing source metadata: run download_dataset first ({metadata_path}).")
    raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (raw_metadata.get("schema_version") != "fpt-raw-v2" or
        raw_metadata.get("ticker") != TICKER or raw_metadata.get("auto_adjust") is not True or
        raw_metadata.get("raw_sha256") != file_sha256(input_path)):
        raise ValueError("Raw snapshot metadata/schema/hash mismatch; download a valid snapshot.")
    features, supervised, excluded, report = build_dataset(pd.read_csv(input_path))
    report["source_raw_sha256"] = file_sha256(input_path)
    report["source_metadata_sha256"] = file_sha256(metadata_path)
    report["source_downloaded_at_utc"] = raw_metadata["downloaded_at_utc"]
    report["price_basis"] = raw_metadata["price_basis"]
    paths = {
        "features": interim_dir / f"{TICKER}_features.csv",
        "supervised": processed_dir / f"{TICKER}_supervised.csv",
        "excluded": processed_dir / f"{TICKER}_excluded_rows.csv",
    }
    # Complete all transformations before replacing any output.
    for name, frame in (("features", features), ("supervised", supervised), ("excluded", excluded)):
        write_snapshot(paths[name], frame)
    report["output_sha256"] = {name: file_sha256(path) for name, path in paths.items()}
    write_snapshot(interim_dir / f"{TICKER}_features.metadata.json", report)
    print(f"Raw: {report['raw_rows']} | excluded zero volume: {report['zero_volume_excluded']}")
    print(f"Invalid candles excluded: {report['invalid_candle_excluded']}")
    print(f"Eligible: {report['eligible_rows']} | warm-up: {report['warmup_rows']}")
    print(f"Features: {len(features)} rows x {len(FEATURE_COLUMNS)} inputs -> {paths['features']}")
    print(f"Labeled: {len(supervised)} -> {paths['supervised']}")
    print("Previous models/scalers are incompatible with schema v2; retrain before inference.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DATA_DIR / f"{TICKER}_raw.csv")
    parser.add_argument("--interim-dir", type=Path, default=INTERIM_DATA_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()
    try:
        run_pipeline(args.input, args.interim_dir, args.processed_dir)
    except (ValueError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
