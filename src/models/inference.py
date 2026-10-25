"""Inference from one versioned v2 manifest; no training or alias fallback."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.contracts import BASE_DIR, file_sha256
from src.data.snapshot import restore_snapshot
from src.features.build_features import FEATURE_COLUMNS, build_dataset
from src.models.experiment import VAL_END, load_data, make_sequences


class ModelService:
    def __init__(self, root: Path = BASE_DIR, artifact_dir: Path | None = None):
        self.root = Path(root).resolve()
        restore_snapshot(self.root)
        pointer = json.loads((self.root / "models/latest_experiment.json").read_text(encoding="utf-8"))
        self.artifact_dir = Path(artifact_dir) if artifact_dir else self.root / pointer["artifact_dir"]
        self.manifest = json.loads((self.artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        self.features, _, meta = load_data(self.root)
        if self.manifest["data_sha256"] != meta["output_sha256"]:
            raise ValueError("Model and dataset snapshots differ.")
        if self.manifest["feature_columns"] != FEATURE_COLUMNS:
            raise ValueError("Model feature order does not match the pipeline.")
        for name, digest in self.manifest["artifact_sha256"].items():
            if Path(name).name != name or file_sha256(self.artifact_dir / name) != digest:
                raise ValueError(f"Invalid artifact: {name}")
        self.loaded = {}
        for name, spec in self.manifest["models"].items():
            if spec["kind"] == "sklearn":
                self.loaded[name] = joblib.load(self.artifact_dir / spec["file"])
            elif spec["kind"] == "xgboost":
                from xgboost import XGBRegressor
                self.loaded[name] = XGBRegressor()
                self.loaded[name].load_model(self.artifact_dir / spec["file"])
            elif spec["kind"] == "lstm":
                import tensorflow as tf
                self.loaded[name] = (
                    tf.keras.models.load_model(self.artifact_dir / spec["file"], compile=False),
                    joblib.load(self.artifact_dir / spec["scalers"]),
                )

    @property
    def default_model(self) -> str:
        return "LSTM" if "LSTM" in self.manifest["models"] else self.manifest["selection"]["selected_model"]

    def predict(self, model_name: str | None = None, observations: pd.DataFrame | None = None) -> dict:
        name = model_name or self.default_model
        if name not in self.manifest["models"]:
            raise ValueError(f"Unknown model: {name}")
        quality = None
        if observations is None:
            frame = self.features
        else:
            frame, _, _, quality = build_dataset(observations)
        spec = self.manifest["models"][name]
        as_of = frame["Date"].iloc[-1]
        if as_of <= VAL_END:
            raise ValueError("Inference origin must be after 2024-12-31, the configuration selection period.")
        if spec["kind"] == "constant":
            prediction = float(spec["value"])
        elif spec["kind"] == "persistence":
            prediction = float(frame["Log_Return"].iloc[-1])
        elif spec["kind"] in ("sklearn", "xgboost"):
            prediction = float(self.loaded[name].predict(frame[FEATURE_COLUMNS].iloc[[-1]])[0])
        else:
            if len(frame) < spec["lookback"]:
                raise ValueError(f"Need at least {49 + spec['lookback']} valid OHLCV observations for this LSTM.")
            model, scalers = self.loaded[name]
            scaled = scalers["features"].transform(frame[FEATURE_COLUMNS])
            tensor = make_sequences(scaled, np.array([len(frame) - 1]), spec["lookback"])
            prediction = float(scalers["target"].inverse_transform(model(tensor, training=False).numpy())[0, 0])
        if not np.isfinite(prediction):
            raise ValueError("Model returned a nonfinite prediction.")
        close = float(frame["Close"].iloc[-1])
        with np.errstate(over="ignore", invalid="ignore"):
            predicted_close = float(close * np.exp(prediction))
            predicted_pct = float(np.expm1(prediction) * 100)
        if not np.isfinite([predicted_close, predicted_pct]).all():
            raise ValueError("Prediction is outside finite numeric bounds.")
        return {
            "ticker": "FPT.VN", "as_of": str(as_of.date()), "horizon_observations": 1,
            "target_date": None, "model": name, "model_version": self.manifest["run_id"],
            "price_basis": "provider_adjusted_ohlc", "latest_close_adjusted": close,
            "predicted_log_return": prediction, "predicted_simple_return_pct": predicted_pct,
            "predicted_close_adjusted": predicted_close,
            "validation_selected_model": self.manifest["selection"]["selected_model"],
            "model_role": "ML demonstration" if name == "LSTM" else "comparison model",
            "training_target_end": self.manifest["splits"]["train"]["target_end"],
            "selection_cutoff": "2024-12-31", "calendar_verified": False,
            "excluded_input_rows": quality["excluded_rows"] if quality else None,
        }
