"""Optional Vietcap D1 observations for display; never update a training bundle."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.data.contracts import OHLCV_COLUMNS, normalize_ohlcv

SOURCE_URL = "https://trading.vietcap.com.vn/api/chart/OHLCChart/gap-chart"
COMPANY_URL = "https://trading.vietcap.com.vn/iq/company?ticker=FPT&tab=overview&isIndex=false"
CACHE_PATH = Path(".cache/observed-prices/vietcap_fpt.json")
SCHEMA = "fpt-observed-prices-v1"


def local_cutoff(now=None):
    stamp = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    if stamp.tzinfo is None:
        raise ValueError("Observation retrieval time must include a timezone.")
    return stamp.tz_convert("Asia/Ho_Chi_Minh").normalize()


def payload_hash(payload):
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def parse_observations(payload, *, now=None):
    if (not isinstance(payload, list) or len(payload) != 1
            or not isinstance(payload[0], dict) or payload[0].get("symbol") != "FPT"):
        raise ValueError("Expected exactly one Vietcap FPT response.")
    source = payload[0]
    if any(not isinstance(source.get(k), list) for k in "tohlcv"):
        raise ValueError("Missing Vietcap OHLCV arrays.")
    lengths = {len(source[k]) for k in "tohlcv"}
    if len(lengths) != 1 or not 1 <= next(iter(lengths)) <= 500:
        raise ValueError("Vietcap OHLCV arrays are empty, unequal or too large.")
    data = pd.DataFrame({col: source[k] for k, col in zip("tohlcv", OHLCV_COLUMNS)})
    data["Date"] = (pd.to_datetime(pd.to_numeric(data.Date, errors="raise"), unit="s", utc=True)
                    .dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None).dt.normalize())
    if data.Date.duplicated().any():
        raise ValueError("Duplicate Vietcap dates require review.")
    # Conservatively omit the local current day, even after market hours: no partial D1 bar.
    cutoff = local_cutoff(now).tz_localize(None)
    data = data.loc[data.Date < cutoff]
    data = normalize_ohlcv(data, allow_untraded_missing_prices=True)
    data = data.loc[data.Volume.gt(0)].reset_index(drop=True)
    if data.empty:
        raise ValueError("No completed positive-volume observations were returned.")
    return data


def _read_document(root):
    path = Path(root) / CACHE_PATH
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA or payload_hash(doc["payload"]) != doc.get("payload_sha256"):
        raise ValueError("Observed-price cache failed its integrity check.")
    return doc


def refresh_observations(root, *, now=None):
    cutoff = local_cutoff(now)
    request = {"timeFrame": "ONE_DAY", "symbols": ["FPT"],
               "to": int(cutoff.timestamp()), "countBack": 200}
    response = requests.post(SOURCE_URL, json=request, timeout=(5, 20),
                             headers={"User-Agent": "FPT-University-Demo/1.0", "Accept": "application/json"})
    response.raise_for_status()
    payload = response.json()
    parse_observations(payload, now=now)  # Validate before replacing a previously good cache.
    stamp = pd.Timestamp(now if now is not None else datetime.now(timezone.utc)).tz_convert("UTC")
    doc = {"schema": SCHEMA, "retrieved_at_utc": stamp.isoformat(), "request": request,
           "payload_sha256": payload_hash(payload), "payload": payload}
    path = Path(root) / CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".vietcap_fpt.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(doc, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def observation_response(root, frozen_raw, *, now=None):
    """Read only. Check overlap before comparing adjusted prices from two snapshots."""
    meta = {"ticker": "FPT.VN", "source": "Vietcap FPT", "source_url": SOURCE_URL,
            "company_url": COMPANY_URL, "currency": "VND", "price_basis": "provider_adjusted_ohlc",
            "current_day_excluded": True, "automatic_retraining": False}
    doc = _read_document(root)
    if doc is None:
        return {**meta, "status": "not_loaded", "observations": [], "data_as_of": None,
                "retrieved_at_utc": None, "basis_check": {"status": "not_checked", "allowed": False}}
    observed = parse_observations(doc["payload"], now=now)
    frozen = frozen_raw[["Date", "Close"]].copy()
    frozen["Date"] = pd.to_datetime(frozen.Date)
    common = observed[["Date", "Close"]].merge(frozen, on="Date", suffixes=("_new", "_frozen"),
                                                validate="one_to_one").sort_values("Date").tail(20)
    enough = len(common) >= 5
    same = bool(enough and np.isclose(common.Close_new, common.Close_frozen, rtol=1e-6, atol=.02).all())
    status = "matched" if same else ("changed" if enough else "insufficient_overlap")
    basis = {"status": status, "allowed": same, "overlap_n": len(common),
             "max_close_difference_vnd": float(abs(common.Close_new - common.Close_frozen).max())
             if len(common) else None}
    observed["Date"] = observed.Date.dt.strftime("%Y-%m-%d")
    return {**meta, "status": "ready", "data_as_of": observed.Date.iloc[-1],
            "retrieved_at_utc": doc["retrieved_at_utc"], "payload_sha256": doc["payload_sha256"],
            "cutoff_exclusive_local_date": local_cutoff(now).date().isoformat(),
            "basis_check": basis, "observations": observed.to_dict("records")}
