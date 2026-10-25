"""Download a validated, adjusted FPT daily snapshot from Yahoo Finance."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(BASE_DIR))

from src.data.contracts import file_sha256, invalid_candle_mask, normalize_ohlcv, utc_now, write_snapshot

TARGET_TICKER = "FPT.VN"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


def download_stock_data(
    ticker: str = TARGET_TICKER,
    start_date: str = "2018-01-01",
    end_date: str = "2026-01-01",
    *,
    output_dir: Path = RAW_DATA_DIR,
    retries: int = 3,
    timeout: int = 20,
) -> Path:
    """End is exclusive. Fail without replacing the snapshot on source errors."""
    if ticker != TARGET_TICKER:
        raise ValueError("This project currently supports FPT.VN only.")
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if start >= end or start.tz is not None or end.tz is not None:
        raise ValueError("Use timezone-naive dates with start < end.")
    if start != start.normalize() or end != end.normalize():
        raise ValueError("Use daily dates in YYYY-MM-DD format.")
    if retries < 1 or timeout < 1:
        raise ValueError("retries and timeout must be positive.")
    cache_dir = BASE_DIR / ".cache" / "yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))

    last_error = None
    for attempt in range(1, retries + 1):
        print(f"Downloading {ticker} [{start.date()}, {end.date()}) attempt {attempt}/{retries}")
        try:
            source = yf.download(
                ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                interval="1d", auto_adjust=True, back_adjust=False, repair=False,
                keepna=True, actions=False, progress=False, threads=False, timeout=timeout,
            )
            data = normalize_ohlcv(source, allow_untraded_missing_prices=True,
                                   allow_invalid_candles=True)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(attempt, 3))
            continue
        break
    else:
        raise RuntimeError(f"Download/validation failed; existing snapshot preserved: {last_error}")

    if not data["Date"].between(start, end, inclusive="left").all():
        raise ValueError("Provider returned rows outside the requested date window.")
    weekdays = len(pd.bdate_range(start, end, inclusive="left"))
    if weekdays >= 60 and len(data) < 0.75 * weekdays:
        raise ValueError("Provider returned less than 75% of requested weekdays; review coverage.")
    if (data["Volume"].gt(0) & ~invalid_candle_mask(data)).sum() < 51:
        raise ValueError("Need at least 51 positive-volume observations for features and target.")

    path = Path(output_dir) / f"{ticker}_raw.csv"
    write_snapshot(path, data)
    metadata = {
        "schema_version": "fpt-raw-v2", "ticker": ticker,
        "source": "Yahoo Finance via yfinance", "yfinance_version": yf.__version__,
        "downloaded_at_utc": utc_now(), "start_inclusive": str(start.date()),
        "end_exclusive": str(end.date()), "interval": "1d",
        "auto_adjust": True, "repair": False, "keepna": True,
        "price_basis": "provider_adjusted_ohlc_not_execution_prices",
        "rows": len(data), "first_date": str(data["Date"].min().date()),
        "last_date": str(data["Date"].max().date()),
        "zero_volume_rows": int(data["Volume"].eq(0).sum()),
        "untraded_missing_price_rows": int(data["Close"].isna().sum()),
        "invalid_candle_rows": int(invalid_candle_mask(data).sum()),
        "invalid_candle_dates": data.loc[invalid_candle_mask(data), "Date"].dt.strftime("%Y-%m-%d").tolist(),
        "calendar_verified": False, "raw_sha256": file_sha256(path),
        "imputation": "none", "market_proxy": "not_downloaded_or_used",
    }
    write_snapshot(path.with_suffix(".metadata.json"), metadata)
    print(f"Saved {len(data)} raw observations -> {path}")
    print(f"Zero-volume observations retained in raw: {metadata['zero_volume_rows']}")
    print(f"Invalid source candles retained for quarantine: {metadata['invalid_candle_rows']}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2018-01-01", help="Inclusive, YYYY-MM-DD")
    parser.add_argument("--end", default="2026-01-01", help="Exclusive, YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()
    try:
        download_stock_data(start_date=args.start, end_date=args.end,
                            output_dir=args.output_dir, retries=args.retries, timeout=args.timeout)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
