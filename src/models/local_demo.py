"""Package and serve the approved FPT Ridge locally, without fitting or fetching data."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.contracts import BASE_DIR, file_sha256, normalize_ohlcv, utc_now
from src.features.build_features import build_dataset

RECENCY = "reports/recency_bias/20261030T022440928507Z"
EVALUATION = "reports/evaluation_2026/20261030T014012158951Z"
PERIOD = "reports/period_comparison/20261030T024048531444Z"
PACKAGE = "models/local_demo/ridge6_20261029"
POINTER = "models/local_demo.json"
MODEL_NAMES = ["Ridge+bias60", "Ridge", "Zero-Return"]
FEATURES = ["Log_Return", "Return_Lag_1", "Return_Lag_2", "Return_Lag_5", "Return_5", "Return_20"]


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def records(frame):
    """Convert optional CSV NaNs to actual JSON nulls."""
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=15))


def summarize(frame):
    rows = []
    for name, part in frame.groupby("strategy", sort=False):
        y = part.actual_log_return.to_numpy(float)
        p = part.predicted_log_return.to_numpy(float)
        nz = y != 0
        active = p != 0
        recalls = [float((np.sign(p[y * sign > 0]) == sign).mean()) for sign in (1, -1)
                   if (y * sign > 0).any()]
        mse = float(np.mean((p - y) ** 2))
        zero = float(np.mean(y ** 2))
        rows.append({"strategy": name, "n": len(part), "nonzero_actual_n": int(nz.sum()),
                     "neutral_actual_n": int((~nz).sum()), "rmse": float(np.sqrt(mse)),
                     "mae": float(np.mean(abs(p - y))),
                     "hit_rate_pct": float((np.sign(p[nz]) == np.sign(y[nz])).mean() * 100) if nz.any() and active.any() else None,
                     "balanced_accuracy_pct": float(np.mean(recalls) * 100) if len(recalls) == 2 and active.any() else None,
                     "direction_coverage_pct": float(active.mean() * 100),
                     "mse_skill_vs_zero": 1 - mse / zero if zero else None,
                     "price_mae_adjusted_vnd": float(np.mean(abs(part.predicted_close_adjusted - part.actual_close_adjusted)))})
    return rows


def package(root=BASE_DIR):
    """Copy a verified existing shadow artifact; this is not a training run."""
    root = Path(root)
    destination = root / PACKAGE
    if destination.exists():
        raise ValueError("Local package already exists; do not overwrite a versioned bundle.")
    bundle_path = root / RECENCY / "shadow_bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    model = root / bundle["model_path"]
    if file_sha256(model) != bundle["model_sha256"] or bundle["feature_columns"] != FEATURES:
        raise ValueError("Shadow model hash/feature mismatch.")
    acquisition = json.loads((root / EVALUATION / "acquisition.json").read_text())
    snapshot = root / EVALUATION / "data_snapshot.zip"
    if file_sha256(snapshot) != acquisition["zip_sha256"]:
        raise ValueError("Evaluation snapshot changed.")
    with zipfile.ZipFile(snapshot) as archive:
        raw_bytes = archive.read("vci.csv")
    if hashlib.sha256(raw_bytes).hexdigest() != acquisition["files"]["vci.csv"]:
        raise ValueError("Vietcap member changed.")
    raw = pd.read_csv(io.BytesIO(raw_bytes))
    features, _, _, quality = build_dataset(raw)
    if str(features.Date.max().date()) != bundle["data_completed_through"]:
        raise ValueError("Bundle and data cutoff mismatch.")
    # Historical controls remain explicitly separate from the newly packaged model.
    backtest = pd.read_csv(root / RECENCY / "predictions.csv.gz")
    backtest = backtest.loc[backtest.cohort.eq("vci_full") & backtest.strategy.isin(
        ["refit_2025_native", "refit_2025_native_bias60", "Zero-Return"])].copy()
    backtest = backtest.rename(columns={"Reference_Return": "actual_log_return"})
    backtest["evaluation_scope"] = "full_2026"
    june = pd.read_csv(root / PERIOD / "predictions.csv.gz")
    june = june.loc[june.strategy.isin(["LSTM_serving", "refit_2025_native", "refit_2025_native_bias60", "Zero-Return"])].copy()
    june["evaluation_scope"] = "jun_oct_2026"
    columns = ["Date", "Target_Date_1d", "strategy", "actual_log_return", "predicted_log_return",
               "origin_close_adjusted", "actual_close_adjusted", "predicted_close_adjusted", "evaluation_scope"]
    combined = pd.concat([backtest[columns], june[columns]], ignore_index=True)
    combined["strategy"] = combined.strategy.replace({"refit_2025_native": "Ridge_2025",
        "refit_2025_native_bias60": "Ridge_2025+bias60", "LSTM_serving": "LSTM_reference"})
    destination.mkdir(parents=True)
    shutil.copyfile(model, destination / "ridge.pkl")
    (destination / "raw.csv").write_bytes(raw_bytes)
    combined.to_csv(destination / "backtest.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    save_json(destination / "calibration.json", bundle)
    source_paths = [bundle_path, snapshot, root / RECENCY / "predictions.csv.gz", root / PERIOD / "predictions.csv.gz"]
    manifest = {"schema": "fpt-local-ridge-v1", "run_id": "ridge6_20261029",
        "packaged_at_utc": utc_now(), "new_training_fits": 0,
        "source_study": RECENCY, "source_hashes": {p.relative_to(root).as_posix(): file_sha256(p) for p in source_paths},
        "feature_columns": FEATURES, "training_target_end": bundle["training_target_end"],
        "data_as_of": bundle["data_completed_through"], "source": "Vietcap FPT",
        "price_basis": "provider_adjusted_ohlc", "point_in_time_certified": False,
        "default_model": MODEL_NAMES[0], "models": {name: {} for name in MODEL_NAMES},
        "selection": {"selected_model": MODEL_NAMES[0], "basis": "User approved local demonstration; no new untouched holdout or automatic promotion."},
        "splits": {"local_bundle": {"training_target_end": bundle["training_target_end"], "training_window_years": 3},
                   "backtest": {"year": 2026, "target_end": "2026-10-29", "separate_historical_models": True}},
        "quality": quality,
        "caveats": ["Fixed snapshot, not live market data.", "Historical backtest scores do not evaluate this October-trained bundle.",
                    "Bias and residual intervals are frozen; no automatic update or guaranteed coverage.",
                    "2026 was inspected during research; no demonstrated stable future edge."],
        "artifact_sha256": {name: file_sha256(destination / name) for name in ["ridge.pkl", "raw.csv", "calibration.json", "backtest.csv.gz"]}}
    save_json(destination / "manifest.json", manifest)
    save_json(root / POINTER, {"artifact_dir": PACKAGE, "manifest_sha256": file_sha256(destination / "manifest.json")})
    return manifest


class LocalRidgeService:
    def __init__(self, root=BASE_DIR):
        self.root = Path(root).resolve()
        pointer = json.loads((self.root / POINTER).read_text(encoding="utf-8"))
        folder = (self.root / pointer["artifact_dir"]).resolve()
        if not folder.is_relative_to(self.root / "models"):
            raise ValueError("Package outside models directory.")
        if file_sha256(folder / "manifest.json") != pointer["manifest_sha256"]:
            raise ValueError("Local manifest changed.")
        self.manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        for name, digest in self.manifest["artifact_sha256"].items():
            if Path(name).name != name or file_sha256(folder / name) != digest:
                raise ValueError(f"Local artifact changed: {name}")
        if self.manifest["feature_columns"] != FEATURES:
            raise ValueError("Local feature schema changed.")
        self.bundle = json.loads((folder / "calibration.json").read_text(encoding="utf-8"))
        self.raw = pd.read_csv(folder / "raw.csv")
        self.features, _, _, _ = build_dataset(self.raw)
        self.loaded = joblib.load(folder / "ridge.pkl")
        if list(self.loaded.feature_names_in_) != FEATURES:
            raise ValueError("Estimator feature order mismatch.")
        self.backtest = pd.read_csv(folder / "backtest.csv.gz")
        self.manifest["metrics"] = self.benchmark()["metrics"]

    @property
    def default_model(self):
        return self.manifest["default_model"]

    def predict(self, model_name=None, observations=None):
        name = model_name or self.default_model
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown local model: {name}")
        raw = normalize_ohlcv(self.raw if observations is None else observations,
                              allow_untraded_missing_prices=True, allow_invalid_candles=True)
        if raw.Date.max().date() >= pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").date():
            raise ValueError("Only completed prior-day observations are accepted; no current/future bars.")
        frame, _, _, quality = build_dataset(raw)
        latest = frame.iloc[-1]
        if latest.Date != raw.loc[raw.Volume.gt(0), "Date"].max():
            raise ValueError("Latest traded bar is invalid; refusing an older origin.")
        cutoff = max([self.bundle["training_target_end"], *self.bundle["calibration_targets"]])
        if latest.Date < pd.Timestamp(cutoff):
            raise ValueError("Origin precedes model training/calibration cutoff; use historical backtest instead.")
        raw_prediction = float(self.loaded.predict(frame[FEATURES].iloc[[-1]])[0])
        errors = np.asarray(self.bundle["raw_errors"], dtype=float)
        bias = float(errors[-self.bundle["bias_spec"]["window"]:].mean()) if len(errors) >= self.bundle["bias_spec"]["minimum_matured"] else 0.
        prediction = 0. if name == "Zero-Return" else raw_prediction + (bias if name == "Ridge+bias60" else 0.)
        close = float(latest.Close)
        with np.errstate(over="ignore", invalid="ignore"):
            predicted_close = float(close * np.exp(prediction))
            percent = float(np.expm1(prediction) * 100)
        if not np.isfinite([prediction, predicted_close, percent]).all():
            raise ValueError("Prediction is outside finite numeric bounds.")
        interval = None
        residuals = np.asarray(self.bundle["corrected_errors"], dtype=float)
        if name == "Ridge+bias60" and len(residuals) >= self.bundle["interval_spec"]["minimum_matured"]:
            lo, hi = np.quantile(residuals[-self.bundle["interval_spec"]["window"]:], self.bundle["interval_spec"]["quantiles"])
            interval = {"nominal_coverage": .8, "lower_close_adjusted": float(close * np.exp(prediction + lo)),
                        "upper_close_adjusted": float(close * np.exp(prediction + hi)),
                        "calibration_n": min(len(residuals), self.bundle["interval_spec"]["window"]),
                        "meaning": "Historical residual quantiles; coverage is not guaranteed."}
            if not np.isfinite([interval["lower_close_adjusted"], interval["upper_close_adjusted"]]).all():
                raise ValueError("Interval is outside finite numeric bounds.")
        return {"ticker": "FPT.VN", "model": name, "model_version": self.manifest["run_id"],
            "as_of": str(latest.Date.date()), "training_target_end": self.bundle["training_target_end"],
            "horizon_observations": 1, "target_date": None, "calendar_verified": False,
            "latest_close_adjusted": close, "predicted_close_adjusted": predicted_close,
            "predicted_log_return": prediction, "predicted_simple_return_pct": percent,
            "direction": "tang" if prediction > 0 else "giam" if prediction < 0 else "khong_doi",
            "bias_offset": bias if name == "Ridge+bias60" else 0., "interval": interval,
            "price_basis": "provider_adjusted_ohlc", "source": "Vietcap FPT" if observations is None else "caller_supplied_adjusted_ohlcv",
            "issued_at_utc": utc_now(), "excluded_input_rows": quality["excluded_rows"],
            "model_role": "local research demonstration", "calibration_policy": "frozen until explicit rebuild",
            "input_role": "frozen_snapshot" if observations is None else "caller_input",
            "forward_ledger_recorded": False, "caveats": self.manifest["caveats"]}

    def benchmark(self, scope="full_2026", month=None):
        if scope not in {"full_2026", "jun_oct_2026"}:
            raise ValueError("Unknown evaluation scope.")
        frame = self.backtest.loc[self.backtest.evaluation_scope.eq(scope)].copy()
        if month:
            frame = frame.loc[frame.Target_Date_1d.str.startswith(month)]
        if frame.empty:
            raise ValueError("No historical predictions in this period.")
        groups = list(frame.groupby("strategy"))
        keys = lambda p: set(zip(p.Date, p.Target_Date_1d))
        if any(keys(p) != keys(groups[0][1]) or p.duplicated(["Date", "Target_Date_1d"]).any() for _, p in groups):
            raise ValueError("Backtest models do not share the same unique date pairs.")
        return {"scope": scope, "month": month, "metrics": summarize(frame), "predictions": records(frame),
                "role": "retrospective backtest of historical models, not evaluation of the October-trained local bundle",
                "source_report": RECENCY if scope == "full_2026" else PERIOD,
                "bias_policy": "matured historical residuals updated causally in backtest; local demo calibration is frozen",
                "lstm_context": "LSTM reference trained on Yahoo; its stored forecasts use original source context and common Vietcap labels."}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(package(), ensure_ascii=False, indent=2))
