"""Read-only FPT source reconciliation. Never replace training data or serving models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from src.data.contracts import BASE_DIR, OHLCV_COLUMNS, file_sha256, invalid_candle_mask, normalize_ohlcv, utc_now
from src.data.download_dataset import download_stock_data

START, END = "2018-01-01", "2026-01-01"
VCI_URL = "https://trading.vietcap.com.vn/api/chart/OHLCChart/gap-chart"
KBS_URL = "https://kbbuddywts.kbsec.com.vn/iis-server/investment/stocks/FPT/data_day"
DOCS = {
    "vci_request_and_parser": "https://github.com/thinh-vu/vnstock/blob/main/vnstock/explorer/vci/quote.py",
    "vci_endpoint_and_fields": "https://github.com/thinh-vu/vnstock/blob/main/vnstock/explorer/vci/const.py",
    "price_basis_and_adapter_units": "https://www.vnstocks.com/docs/vnstock/thi-truong-market",
    "yahoo_auto_adjust": "https://github.com/ranaroussi/yfinance/blob/main/yfinance/utils.py",
}
GUARDED = [
    "data/raw/FPT.VN_raw.csv", "data/raw/FPT.VN_raw.metadata.json",
    "data/interim/FPT.VN_features.csv", "data/interim/FPT.VN_features.metadata.json",
    "data/processed/FPT.VN_supervised.csv", "data/processed/FPT.VN_excluded_rows.csv",
    "models/latest_experiment.json", "models/reference_experiment.json",
]


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def guarded_hashes() -> dict:
    return {name: file_sha256(BASE_DIR / name) for name in GUARDED}


def parse_vci(payload: list, start: str = START, end: str = END) -> pd.DataFrame:
    """The direct VCI response uses VND; do not apply vnstock's /1000 adapter."""
    if not isinstance(payload, list) or len(payload) != 1 or payload[0].get("symbol") != "FPT":
        raise ValueError("Expected exactly one VCI response for FPT.")
    raw = payload[0]
    fields = dict(zip("tohlcv", OHLCV_COLUMNS))
    lengths = {len(raw[k]) for k in fields}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("VCI arrays are empty or have unequal lengths.")
    data = pd.DataFrame({column: raw[k] for k, column in fields.items()})
    data["Date"] = (pd.to_datetime(pd.to_numeric(data.Date), unit="s", utc=True)
                    .dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None).dt.normalize())
    if data.Date.duplicated().any():
        raise ValueError("Duplicate VCI dates require review, not deduplication.")
    data = data.loc[data.Date.between(start, end, inclusive="left")]
    return normalize_ohlcv(data, allow_untraded_missing_prices=True, allow_invalid_candles=True)


def parse_kbs(payload: dict, start: str = START, end: str = END) -> pd.DataFrame:
    if payload.get("symbol") != "FPT" or not payload.get("data_day"):
        raise ValueError("Expected nonempty KBS daily response for FPT.")
    data = pd.DataFrame(payload["data_day"]).rename(columns=dict(zip("tohlcv", OHLCV_COLUMNS)))
    # KBS emits daily labels at 07:00, without an offset. The calendar date is retained.
    data["Date"] = pd.to_datetime(data.Date, format="%Y-%m-%d %H:%M").dt.normalize()
    if data.Date.duplicated().any():
        raise ValueError("Duplicate KBS dates require review.")
    data = data.loc[data.Date.between(start, end, inclusive="left")]
    return normalize_ohlcv(data, allow_untraded_missing_prices=True, allow_invalid_candles=True)


def confirm_kbs(run: Path) -> None:
    """Append an independent confirmation after Yahoo/VCI conflicts; keep evidence.zip intact."""
    if (run / "kbs_evidence.zip").exists():
        raise FileExistsError("KBS evidence already exists; it must not be overwritten.")
    before = guarded_hashes()
    params = {"sdate": "01-01-2018", "edate": "31-12-2025"}
    response = requests.get(KBS_URL, params=params, timeout=40,
                            headers={"User-Agent": "FPT-University-Source-Audit/1.0", "Accept": "application/json"})
    response.raise_for_status()
    kbs = parse_kbs(response.json())
    if len(kbs) < 1800 or kbs.Date.min() > pd.Timestamp("2018-01-08"):
        raise ValueError("KBS coverage is too short.")
    evidence = {"kbs_response.json": response.content,
                "kbs.csv": kbs.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")}
    with zipfile.ZipFile(run / "kbs_evidence.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in evidence.items():
            archive.writestr(name, content)
    after = guarded_hashes()
    save_json(run / "kbs_acquisition.json", {
        "stage": "secondary_confirmation_after_Yahoo_VCI_conflict", "retrieved_at_utc": utc_now(),
        "url": KBS_URL, "method": "GET", "params": params, "status_code": response.status_code,
        "currency": "VND", "price_multiplier": 1, "price_basis": "provider_corporate_action_adjusted_chart",
        "volume_basis": "provider_chart_volume_not_reconciled_to_exchange_total_volume",
        "docs": ["https://github.com/thinh-vu/vnstock/blob/main/vnstock/explorer/kbs/quote.py",
                 "https://github.com/thinh-vu/vnstock/blob/main/vnstock/explorer/kbs/const.py", DOCS["price_basis_and_adapter_units"]],
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in evidence.items()},
        "zip_sha256": file_sha256(run / "kbs_evidence.zip"), "protected_before": before, "protected_after": after,
    })
    if before != after:
        raise RuntimeError("Protected files changed during KBS acquisition.")


def collect(run: Path) -> None:
    before = guarded_hashes()
    save_json(run / "protected_before.json", before)
    work = BASE_DIR / ".cache" / "source-audits" / run.name
    work.mkdir(parents=True, exist_ok=False)
    (work / "frozen.csv").write_bytes((BASE_DIR / GUARDED[0]).read_bytes())
    (work / "frozen.metadata.json").write_bytes((BASE_DIR / GUARDED[1]).read_bytes())
    metadata = {"schema": "fpt-source-audit-v1", "started_at_utc": utc_now(),
                "start_inclusive": START, "end_exclusive": END,
                "sources": DOCS, "training_or_promotion": False,
                "python": platform.python_version(),
                "versions": {p: importlib.metadata.version(p) for p in ["pandas", "numpy", "yfinance", "requests"]},
                "thresholds": {"price_rtol": 1e-6, "price_atol_vnd": 0.02,
                               "material_price_difference_pct": 0.1, "return_difference_bps": 10},
                "notes": ["Thresholds flag diagnostics; they do not certify a provider as ground truth.",
                          "The direct VCI price is VND, not vnstock adapter thousand-VND.",
                          "Adjusted histories are retrieved now, not point-in-time histories.",
                          "No 2026 prices or model evaluation are requested in this audit."]}
    save_json(run / "protocol.json", metadata)
    try:
        download_stock_data(output_dir=work / "yahoo_refreshed")
        source = yf.download("FPT.VN", start=START, end=END, interval="1d",
                             auto_adjust=False, back_adjust=False, repair=False,
                             keepna=True, actions=True, progress=False, threads=False, timeout=30)
        if source.empty:
            raise ValueError("Yahoo unadjusted/action diagnostic request returned no rows.")
        if isinstance(source.columns, pd.MultiIndex):
            source.columns = source.columns.get_level_values(0)
        source.to_csv(work / "yahoo_unadjusted_actions.csv", date_format="%Y-%m-%d")
        metadata["yahoo_actions_retrieved_at_utc"] = utc_now()
        metadata["yahoo_actions_flags"] = {"auto_adjust": False, "actions": True, "repair": False, "keepna": True}
        request = {"timeFrame": "ONE_DAY", "symbols": ["FPT"],
                   "to": int(pd.Timestamp(END, tz="UTC").timestamp()),
                   "countBack": len(pd.bdate_range(START, END, inclusive="left")) + 10}
        response = requests.post(VCI_URL, json=request, timeout=40,
                                 headers={"User-Agent": "FPT-University-Source-Audit/1.0", "Accept": "application/json"})
        response.raise_for_status()
        (work / "vci_response.json").write_bytes(response.content)
        vci = parse_vci(response.json())
        if len(vci) < 1800 or vci.Date.min() > pd.Timestamp("2018-01-08"):
            raise ValueError("VCI coverage is too short; do not call this a full-period comparison.")
        vci.to_csv(work / "vci.csv", index=False, date_format="%Y-%m-%d")
        metadata["vci"] = {"url": VCI_URL, "method": "POST", "request": request,
                           "retrieved_at_utc": utc_now(), "status_code": response.status_code,
                           "response_sha256": hashlib.sha256(response.content).hexdigest(),
                           "currency": "VND", "price_multiplier": 1,
                           "price_basis": "provider_corporate_action_adjusted_chart",
                           "volume_basis": "chart_volume_definition_not_independently_verified",
                           "calendar": "provider_observation_dates_not_official_HOSE_calendar"}
        metadata["files"] = {str(p.relative_to(work)).replace("\\", "/"): file_sha256(p)
                             for p in sorted(work.rglob("*")) if p.is_file()}
        with zipfile.ZipFile(run / "evidence.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in metadata["files"]:
                archive.write(work / name, name)
        metadata["evidence_zip_sha256"] = file_sha256(run / "evidence.zip")
        metadata["finished_at_utc"] = utc_now()
        save_json(run / "acquisition.json", metadata)
    finally:
        after = guarded_hashes()
        save_json(run / "protected_after.json", after)
        if before != after:
            raise RuntimeError("Protected data/model hashes changed during audit.")


def read_evidence(run: Path) -> dict[str, pd.DataFrame]:
    meta = json.loads((run / "acquisition.json").read_text(encoding="utf-8"))
    if file_sha256(run / "evidence.zip") != meta["evidence_zip_sha256"]:
        raise ValueError("Evidence archive hash mismatch.")
    with zipfile.ZipFile(run / "evidence.zip") as archive:
        for name, sha in meta["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != sha:
                raise ValueError(f"Evidence hash mismatch: {name}")
        names = {"frozen": "frozen.csv", "refreshed": "yahoo_refreshed/FPT.VN_raw.csv",
                 "vci": "vci.csv", "yahoo_unadjusted": "yahoo_unadjusted_actions.csv"}
        result = {key: pd.read_csv(io.BytesIO(archive.read(name)), parse_dates=["Date"])
                  for key, name in names.items()}
    if (run / "kbs_acquisition.json").exists():
        kbs_meta = json.loads((run / "kbs_acquisition.json").read_text(encoding="utf-8"))
        if file_sha256(run / "kbs_evidence.zip") != kbs_meta["zip_sha256"]:
            raise ValueError("KBS archive hash mismatch.")
        with zipfile.ZipFile(run / "kbs_evidence.zip") as archive:
            for name, sha in kbs_meta["files"].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != sha:
                    raise ValueError(f"KBS evidence mismatch: {name}")
            result["kbs"] = pd.read_csv(io.BytesIO(archive.read("kbs.csv")), parse_dates=["Date"])
    return result


def profile(data: pd.DataFrame) -> dict:
    bad = invalid_candle_mask(data)
    return {"rows": len(data), "first_date": str(data.Date.min().date()), "last_date": str(data.Date.max().date()),
            "duplicate_dates": int(data.Date.duplicated().sum()), "positive_volume_rows": int(data.Volume.gt(0).sum()),
            "zero_volume_rows": int(data.Volume.eq(0).sum()), "missing_close_rows": int(data.Close.isna().sum()),
            "invalid_candle_rows": int(bad.sum()), "invalid_candle_dates": data.loc[bad, "Date"].dt.strftime("%Y-%m-%d").tolist()}


def compare_pair(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Outer join unique daily keys; missing values never count as agreement."""
    for data in [left, right]:
        if data.Date.duplicated().any():
            raise ValueError("Comparison requires unique dates.")
    joined = left[OHLCV_COLUMNS].merge(right[OHLCV_COLUMNS], on="Date", how="outer",
                                      suffixes=("_left", "_right"), indicator=True, validate="one_to_one")
    joined = joined.sort_values("Date").reset_index(drop=True)
    common = joined._merge.eq("both")
    changed_columns = []
    for col in OHLCV_COLUMNS[1:]:
        a, b = joined[f"{col}_left"], joined[f"{col}_right"]
        comparable = common & a.notna() & b.notna()
        same = np.isclose(a, b, rtol=0 if col == "Volume" else 1e-6,
                          atol=0 if col == "Volume" else 0.02, equal_nan=False)
        joined[f"{col}_comparable"] = comparable
        joined[f"{col}_changed"] = comparable & ~same
        joined[f"{col}_missing_mismatch"] = common & (a.isna() ^ b.isna())
        if col != "Volume":
            joined[f"{col}_difference_pct"] = np.where(comparable, (b / a - 1) * 100, np.nan)
        else:
            joined["Volume_ratio_right_over_left"] = b / a.where(a.gt(0))
        changed_columns += [f"{col}_changed", f"{col}_missing_mismatch"]
    joined["any_change"] = joined[changed_columns].any(axis=1)
    positive = common & joined.Volume_left.gt(0) & joined.Volume_right.gt(0)
    close_comparable = positive & joined.Close_left.notna() & joined.Close_right.notna()
    summary = {"left_rows": len(left), "right_rows": len(right), "common_dates": int(common.sum()),
               "left_only_dates": int(joined._merge.eq("left_only").sum()),
               "right_only_dates": int(joined._merge.eq("right_only").sum()),
               "changed_common_rows": int((common & joined.any_change).sum()),
               "common_positive_volume_dates": int(positive.sum()),
               "comparable_positive_close_rows": int(close_comparable.sum()),
               "close_material_difference_rows": int((close_comparable & joined.Close_difference_pct.abs().gt(.1)).sum()),
               "close_max_abs_difference_pct": float(joined.loc[close_comparable, "Close_difference_pct"].abs().max()),
               "volume_different_positive_rows": int((positive & joined.Volume_changed).sum())}
    for col in OHLCV_COLUMNS[1:]:
        summary[f"{col}_changed_rows"] = int(joined[f"{col}_changed"].sum())
        summary[f"{col}_missing_mismatch_rows"] = int(joined[f"{col}_missing_mismatch"].sum())
    if not close_comparable.any():
        summary["close_max_abs_difference_pct"] = None
    return joined, summary


def compare_returns(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    l, r = [data.loc[data.Volume.gt(0), ["Date", "Close"]] for data in [left, right]]
    returns = l.merge(r, on="Date", how="outer", suffixes=("_left", "_right"), validate="one_to_one").sort_values("Date")
    valid = returns[["Close_left", "Close_right"]].notna().all(axis=1)
    adjacent = valid & valid.shift(1, fill_value=False)
    for k in ["left", "right"]:
        returns[f"log_return_{k}"] = np.log(returns[f"Close_{k}"] / returns[f"Close_{k}"].shift(1)).where(adjacent)
    returns["difference_bps"] = (returns.log_return_left - returns.log_return_right) * 10000
    diffs = returns.difference_bps.dropna()
    stats = {"comparable_adjacent_pairs": len(diffs), "abs_difference_gt_10bps": int(diffs.abs().gt(10).sum()),
             "median_abs_difference_bps": float(diffs.abs().median()) if len(diffs) else None,
             "max_abs_difference_bps": float(diffs.abs().max()) if len(diffs) else None}
    return returns, stats


def analyze(run: Path) -> dict:
    sources = read_evidence(run)
    summary = {"run_id": run.name, "profiles": {k: profile(v) for k, v in sources.items()}, "pairs": {}}
    pairs = [("yahoo_revision", "frozen", "refreshed"), ("frozen_vs_vci", "frozen", "vci"),
             ("refreshed_vs_vci", "refreshed", "vci")]
    if "kbs" in sources:
        pairs += [("refreshed_vs_kbs", "refreshed", "kbs"), ("vci_vs_kbs", "vci", "kbs")]
    summary["return_pairs"] = {}
    for name, left, right in pairs:
        joined, stats = compare_pair(sources[left], sources[right])
        joined.to_csv(run / f"{name}.csv", index=False, date_format="%Y-%m-%d")
        summary["pairs"][name] = stats
        returns, stats = compare_returns(sources[left], sources[right])
        returns.to_csv(run / f"{name}_returns.csv", index=False, date_format="%Y-%m-%d")
        summary["return_pairs"][name] = stats
    # Return differences use consecutive dates in the union of positive-volume observations.
    # A missing session on either side invalidates both adjacent comparisons; never bridge it.
    returns, stats = compare_returns(sources["refreshed"], sources["vci"])
    summary["returns"] = stats
    yahoo = sources["yahoo_unadjusted"]
    bridge = sources["refreshed"][["Date", "Close"]].merge(yahoo, on="Date", suffixes=("_auto", "_unadjusted"), validate="one_to_one")
    bridge["adjustment_factor"] = bridge["Adj Close"] / bridge.Close_unadjusted
    bridge["auto_adjust_matches_adj_close"] = np.isclose(bridge.Close_auto, bridge["Adj Close"], rtol=1e-6, atol=.02)
    bridge.to_csv(run / "yahoo_adjustment_bridge.csv", index=False, date_format="%Y-%m-%d")
    actions = yahoo.loc[yahoo["Dividends"].ne(0) | yahoo["Stock Splits"].ne(0)]
    actions.to_csv(run / "yahoo_reported_actions.csv", index=False, date_format="%Y-%m-%d")
    summary["adjustment_bridge"] = {"comparable_close_rows": int(bridge.Close_auto.notna().sum()),
                                     "mismatch_rows": int((bridge.Close_auto.notna() & ~bridge.auto_adjust_matches_adj_close).sum()),
                                     "yahoo_reported_action_rows": len(actions),
                                     "corporate_actions_independently_certified": False}
    summary["protected_files_unchanged"] = (json.loads((run / "protected_before.json").read_text()) ==
                                            json.loads((run / "protected_after.json").read_text()))
    summary["decision"] = "audit_only_no_training_no_data_replacement_no_model_promotion"
    if "kbs" in sources:
        issues = sources["refreshed"].merge(sources["vci"], on="Date", suffixes=("_yahoo", "_vci"))
        issues = issues.merge(sources["kbs"].add_suffix("_kbs").rename(columns={"Date_kbs": "Date"}), on="Date", how="left")
        issues["yahoo_zero_two_providers_positive"] = issues.Volume_yahoo.eq(0) & issues.Volume_vci.gt(0) & issues.Volume_kbs.gt(0)
        bad_by_date = pd.Series(invalid_candle_mask(sources["refreshed"]).to_numpy(), index=sources["refreshed"].Date)
        issues["yahoo_invalid_candle"] = issues.Date.map(bad_by_date)
        issues["reason"] = np.select([issues.yahoo_zero_two_providers_positive, issues.yahoo_invalid_candle,
                                      issues.Volume_yahoo.eq(0) & issues.Volume_vci.eq(0)],
                                     ["yahoo_zero_but_two_providers_report_trades", "yahoo_invalid_OHLC", "VCI_zero_not_confirmed_session"], default="other_source_disagreement")
        issues.loc[issues.reason.ne("other_source_disagreement")].to_csv(run / "flagged_observations.csv", index=False, date_format="%Y-%m-%d")
        summary["confirmed_conflicts"] = {"yahoo_zero_but_vci_and_kbs_positive": int(issues.yahoo_zero_two_providers_positive.sum()),
                                          "official_calendar_verified": False, "action_ledger_verified": False}
        yearly = []
        for year in range(2018, 2026):
            row = {"year": year}
            for key in ["refreshed", "vci", "kbs"]:
                part = sources[key].loc[sources[key].Date.dt.year.eq(year)]
                row[f"{key}_positive_volume_days"] = int(part.Volume.gt(0).sum())
            part = issues.loc[issues.Date.dt.year.eq(year)]
            row["yahoo_zero_two_providers_positive"] = int(part.yahoo_zero_two_providers_positive.sum())
            for left, right in [("refreshed", "vci"), ("vci", "kbs")]:
                # Filter AFTER computing adjacent returns, retaining the prior year's last observation.
                ret, _ = compare_returns(sources[left], sources[right])
                ret = ret.loc[ret.Date.dt.year.eq(year), "difference_bps"].dropna()
                row[f"{left}_{right}_return_pairs"] = len(ret)
                row[f"{left}_{right}_abs_return_diff_gt_10bps"] = int(ret.abs().gt(10).sum())
            yearly.append(row)
        pd.DataFrame(yearly).to_csv(run / "yearly_quality.csv", index=False)
        # Exercise the existing pure feature builder, without writing a candidate or training a model.
        from src.features.build_features import build_dataset, TARGET_DATE_COLUMN
        summary["feature_builder_sha256"] = file_sha256(BASE_DIR / "src/features/build_features.py")
        summary["feature_build_dry_run"] = {}
        labeled = {}
        for name in ["frozen", "vci", "kbs"]:
            features, labeled[name], excluded, report = build_dataset(sources[name])
            summary["feature_build_dry_run"][name] = {
                "eligible_rows": report["eligible_rows"], "feature_rows": len(features),
                "supervised_rows": len(labeled[name]), "excluded_rows": len(excluded)}
        labels = labeled["frozen"][["Date", TARGET_DATE_COLUMN]].merge(
            labeled["vci"][["Date", TARGET_DATE_COLUMN]], on="Date", suffixes=("_yahoo", "_vci"), validate="one_to_one")
        labels["target_date_changed"] = labels[f"{TARGET_DATE_COLUMN}_yahoo"].ne(labels[f"{TARGET_DATE_COLUMN}_vci"])
        labels.loc[labels.target_date_changed].to_csv(run / "changed_target_dates.csv", index=False, date_format="%Y-%m-%d")
        summary["feature_build_dry_run"]["common_labeled_origins"] = len(labels)
        summary["feature_build_dry_run"]["common_origins_with_changed_target_date"] = int(labels.target_date_changed.sum())
    save_json(run / "summary.json", summary)
    # Keep row-level comparisons inspectable without committing repeated multi-MB CSVs.
    large = [run / f"{name}{suffix}.csv" for name, _, _ in pairs for suffix in ["", "_returns"]]
    large.append(run / "yahoo_adjustment_bridge.csv")
    with zipfile.ZipFile(run / "comparisons.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in large:
            archive.write(path, path.name)
    for path in large:
        if path.resolve().parent != run.resolve():
            raise ValueError("Generated output escaped run folder.")
        path.unlink()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", type=Path, help="Recompute diagnostics from a saved evidence archive, without network.")
    parser.add_argument("--confirm-kbs", type=Path, help="Append KBS confirmation to an existing run, then analyze.")
    args = parser.parse_args()
    if args.analyze and args.confirm_kbs:
        parser.error("Choose either --analyze or --confirm-kbs.")
    if args.confirm_kbs:
        run = args.confirm_kbs.resolve()
        confirm_kbs(run)
    elif args.analyze:
        run = args.analyze.resolve()
    else:
        run = BASE_DIR / "reports" / "data_audits" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run.mkdir(parents=True, exist_ok=False)
        collect(run)
    summary = analyze(run)
    print(json.dumps({"run": str(run), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
