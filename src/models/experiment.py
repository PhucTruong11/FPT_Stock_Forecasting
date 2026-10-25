"""Reproducible FPT v2 experiments: common dates, validation selection, saved artifacts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.data.contracts import BASE_DIR, file_sha256
from src.features.build_features import FEATURE_COLUMNS, TARGET_COLUMN, TARGET_DATE_COLUMN

TRAIN_END = pd.Timestamp("2022-12-31")
VAL_END = pd.Timestamp("2024-12-31")
TEST_END = pd.Timestamp("2025-12-31")
COMMON_LOOKBACK = 60


def save_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def split_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    origin, target = frame["Date"], frame[TARGET_DATE_COLUMN]
    return {
        "train": ((origin <= TRAIN_END) & (target <= TRAIN_END)).to_numpy(),
        "validation": ((origin > TRAIN_END) & (origin <= VAL_END) & (target <= VAL_END)).to_numpy(),
        "test": ((origin > VAL_END) & (origin <= TEST_END) & (target <= TEST_END)).to_numpy(),
    }


def load_data(root: Path = BASE_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    root = Path(root)
    meta_path = root / "data/interim/FPT.VN_features.metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    paths = {
        "features": root / "data/interim/FPT.VN_features.csv",
        "supervised": root / "data/processed/FPT.VN_supervised.csv",
        "excluded": root / "data/processed/FPT.VN_excluded_rows.csv",
    }
    if meta.get("schema_version") != "fpt-features-v2" or meta.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Feature schema mismatch; rebuild the v2 dataset.")
    for name, path in paths.items():
        if file_sha256(path) != meta["output_sha256"][name]:
            raise ValueError(f"Dataset hash mismatch: {path}")
    if file_sha256(root / "data/raw/FPT.VN_raw.csv") != meta["source_raw_sha256"]:
        raise ValueError("Raw snapshot changed after features were built.")
    features = pd.read_csv(paths["features"], parse_dates=["Date"])
    labels = pd.read_csv(paths["supervised"], parse_dates=["Date", TARGET_DATE_COLUMN])
    for frame in (features, labels):
        if frame["Date"].duplicated().any() or not frame["Date"].is_monotonic_increasing:
            raise ValueError("Dates must be unique and sorted.")
        if not np.isfinite(frame[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite model inputs.")
    if not labels[TARGET_DATE_COLUMN].gt(labels["Date"]).all():
        raise ValueError("Target must follow forecast origin.")
    positions = pd.Index(features["Date"]).get_indexer(labels["Date"])
    if (positions < 0).any():
        raise ValueError("A label date is absent from the full feature history.")
    if not np.allclose(features.iloc[positions][FEATURE_COLUMNS], labels[FEATURE_COLUMNS], atol=1e-12):
        raise ValueError("Features and supervised inputs disagree.")
    # Identical training/evaluation dates for every model and every lookback.
    labels = labels.loc[positions >= COMMON_LOOKBACK - 1].copy().reset_index(drop=True)
    labels["Feature_Position"] = positions[positions >= COMMON_LOOKBACK - 1]
    for name, mask in split_masks(labels).items():
        if not mask.any():
            raise ValueError(f"Empty {name} partition.")
    return features, labels, meta


def make_sequences(values: np.ndarray, positions: np.ndarray, lookback: int) -> np.ndarray:
    if lookback < 1 or (positions < lookback - 1).any() or (positions >= len(values)).any():
        raise ValueError("Not enough context for the requested sequence.")
    return np.stack([values[i - lookback + 1:i + 1] for i in positions]).astype("float32")


def metrics(y_true, y_pred) -> dict:
    true, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape or not len(true) or not np.isfinite(pred).all():
        raise ValueError("Invalid predictions or alignment.")
    nonzero = true != 0
    issued = nonzero & (pred != 0)
    hit = None
    if issued.any():
        # Same nonzero-actual denominator; abstentions do not improve the score.
        hit = float(100 * (np.sign(true[nonzero]) == np.sign(pred[nonzero])).mean())
    mse = float(np.mean((true - pred) ** 2))
    zero_mse = float(np.mean(true ** 2))
    return {
        "n": len(true), "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(true - pred))), "hit_rate_pct": hit,
        "nonzero_actual_n": int(nonzero.sum()), "neutral_actual_n": int((~nonzero).sum()),
        "direction_coverage_pct": float(100 * issued.sum() / nonzero.sum()) if nonzero.any() else None,
        "mse_skill_vs_zero": float(1 - mse / zero_mse) if zero_mse else None,
    }


def _predict(spec: dict, features: pd.DataFrame, labels: pd.DataFrame, artifact_dir: Path) -> np.ndarray:
    if spec["kind"] == "constant":
        return np.full(len(labels), spec["value"])
    if spec["kind"] == "persistence":
        return labels["Log_Return"].to_numpy()
    if spec["kind"] == "sklearn":
        return joblib.load(artifact_dir / spec["file"]).predict(labels[FEATURE_COLUMNS])
    import tensorflow as tf
    model = tf.keras.models.load_model(artifact_dir / spec["file"], compile=False)
    scalers = joblib.load(artifact_dir / spec["scalers"])
    values = scalers["features"].transform(features[FEATURE_COLUMNS])
    tensor = make_sequences(values, labels["Feature_Position"].to_numpy(), spec["lookback"])
    return scalers["target"].inverse_transform(model.predict(tensor, verbose=0)).ravel()


def evaluate_saved(artifact_dir: Path, root: Path = BASE_DIR, verify_recorded: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifact_dir, root = Path(artifact_dir), Path(root)
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    features, labels, meta = load_data(root)
    if manifest["data_sha256"] != meta["output_sha256"]:
        raise ValueError("Artifact belongs to a different data snapshot.")
    for name, digest in manifest["artifact_sha256"].items():
        if file_sha256(artifact_dir / name) != digest:
            raise ValueError(f"Artifact hash mismatch: {name}")
    masks, rows, predictions = split_masks(labels), [], []
    for split in ("validation", "test"):
        frame = labels.loc[masks[split]].copy()
        for name, spec in manifest["models"].items():
            pred = _predict(spec, features, frame, artifact_dir)
            score = metrics(frame[TARGET_COLUMN].to_numpy(), pred)
            rows.append({"model": name, "split": split, **score})
            part = frame[["Date", TARGET_DATE_COLUMN, "Close", TARGET_COLUMN]].copy()
            part["model"], part["split"], part["predicted_log_return"] = name, split, pred
            part["predicted_close_adjusted"] = part["Close"] * np.exp(pred)
            predictions.append(part)
    scores, preds = pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)
    if verify_recorded and "metrics" in manifest:
        expected = pd.DataFrame(manifest["metrics"])
        for column in ("rmse", "mae"):
            if not np.allclose(scores[column], expected[column], rtol=1e-5, atol=1e-8):
                raise ValueError("Reloaded metrics disagree with training record.")
    return scores, preds


def run_experiment(root: Path = BASE_DIR, *, include_lstm: bool = True, epochs: int = 40, seed: int = 42) -> dict:
    root = Path(root)
    features, labels, data_meta = load_data(root)
    masks = split_masks(labels)
    train, val = labels.loc[masks["train"]], labels.loc[masks["validation"]]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_dir = root / "models/experiments" / run_id
    report_dir = root / "reports/experiments" / run_id
    artifact_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    ytr, yval = train[TARGET_COLUMN].to_numpy(), val[TARGET_COLUMN].to_numpy()
    specs = {
        "Zero-Return": {"kind": "constant", "value": 0.0},
        "Return Persistence": {"kind": "persistence"},
        "Train Mean": {"kind": "constant", "value": float(ytr.mean())},
        "Always Up": {"kind": "constant", "value": float(np.abs(ytr).mean())},
    }
    trials, validation_predictions, histories = [], {}, {}
    print(f"Run {run_id}; common samples: " + ", ".join(f"{k}={int(v.sum())}" for k, v in masks.items()), flush=True)

    def record(name, family, spec, pred, parameters):
        score = metrics(yval, pred)
        trials.append({"candidate": name, "family": family, "parameters": parameters, **score})
        validation_predictions[name] = pred
        print(f"{name}: validation RMSE={score['rmse']:.7f}", flush=True)
        return score["rmse"]

    for name, spec in specs.items():
        pred = np.full(len(val), spec["value"]) if spec["kind"] == "constant" else val["Log_Return"].to_numpy()
        record(name, name, spec, pred, {})
    for family, configurations in {
        "Linear Regression": [{}],
        "Ridge": [{"alpha": a} for a in (0.1, 1.0, 10.0)],
        "XGBoost": [{"max_depth": d} for d in (2, 3)],
    }.items():
        best = None
        for number, parameters in enumerate(configurations):
            if family == "Linear Regression":
                model = make_pipeline(StandardScaler(), LinearRegression())
            elif family == "Ridge":
                model = make_pipeline(StandardScaler(), Ridge(**parameters))
            else:
                model = XGBRegressor(**parameters, n_estimators=150, learning_rate=0.03,
                                     min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
                                     objective="reg:squarederror", n_jobs=2, random_state=seed)
            model.fit(train[FEATURE_COLUMNS], ytr)
            name, filename = f"{family}/{number}", f"{family.replace(' ', '_').lower()}_{number}.pkl"
            joblib.dump(model, artifact_dir / filename)
            spec = {"kind": "sklearn", "file": filename, "parameters": parameters}
            score = record(name, family, spec, model.predict(val[FEATURE_COLUMNS]), parameters)
            if best is None or score < best[0]:
                best = (score, spec, name)
        specs[family] = best[1]

    runtime = {"device": "CPU", "tensorflow": None}
    if include_lstm:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        try:
            tf.config.threading.set_intra_op_parallelism_threads(2)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except RuntimeError:
            pass
        tf.config.experimental.enable_op_determinism()
        runtime = {"device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU", "tensorflow": tf.__version__}
        fs = StandardScaler().fit(features.loc[features["Date"] <= TRAIN_END, FEATURE_COLUMNS])
        ts = StandardScaler().fit(ytr.reshape(-1, 1))
        values = fs.transform(features[FEATURE_COLUMNS])
        scaled_y = ts.transform(ytr.reshape(-1, 1)).ravel()
        scaled_yval = ts.transform(yval.reshape(-1, 1)).ravel()
        joblib.dump({"features": fs, "target": ts}, artifact_dir / "lstm_scalers.pkl")
        best = None
        for lb in (10, 30, 60):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            xtr = make_sequences(values, train["Feature_Position"].to_numpy(), lb)
            xval = make_sequences(values, val["Feature_Position"].to_numpy(), lb)
            model = tf.keras.Sequential([
                tf.keras.layers.Input((lb, len(FEATURE_COLUMNS))), tf.keras.layers.LSTM(32),
                tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(1),
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0), loss="huber")
            start = time.perf_counter()
            print(f"Training LSTM lookback={lb} on {runtime['device']}...", flush=True)
            history = model.fit(xtr, scaled_y, validation_data=(xval, scaled_yval),
                                epochs=epochs, batch_size=32, shuffle=False, verbose=0,
                                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
                                           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-5)])
            filename = f"lstm_lb{lb}.keras"
            model.save(artifact_dir / filename)
            spec = {"kind": "lstm", "file": filename, "scalers": "lstm_scalers.pkl", "lookback": lb,
                    "epochs_trained": len(history.history["loss"]), "seconds": time.perf_counter() - start}
            pred = ts.inverse_transform(model.predict(xval, verbose=0)).ravel()
            score = record(f"LSTM/{lb}", "LSTM", spec, pred, {"lookback": lb, "units": 32})
            histories[str(lb)] = {k: [float(v) for v in vals] for k, vals in history.history.items()}
            if best is None or score < best[0]:
                best = (score, spec, f"LSTM/{lb}")
        specs["LSTM"] = best[1]

    # Freeze family winners and selection using VALIDATION only, before test predictions.
    validation_scores = {}
    for name, spec in specs.items():
        pred = _predict(spec, features, val, artifact_dir)
        validation_scores[name] = metrics(yval, pred)["rmse"]
    selected = min(validation_scores, key=validation_scores.get)
    selection = {"criterion": "lowest validation RMSE", "selected_model": selected,
                 "validation_rmse": validation_scores, "test_used_for_selection": False}
    save_json(report_dir / "selection.json", selection)
    save_json(report_dir / "validation_trials.json", trials)
    save_json(report_dir / "lstm_history.json", histories)
    pd.DataFrame(trials).drop(columns="parameters").to_csv(report_dir / "validation_trials.csv", index=False)
    split_summary = {}
    for name, mask in masks.items():
        part = labels.loc[mask]
        split_summary[name] = {"n": len(part), "origin_start": str(part["Date"].min().date()),
                               "origin_end": str(part["Date"].max().date()),
                               "target_end": str(part[TARGET_DATE_COLUMN].max().date())}
    manifest = {
        "schema_version": "fpt-experiment-v1", "ticker": "FPT.VN", "run_id": run_id,
        "seed": seed, "feature_columns": FEATURE_COLUMNS, "target_column": TARGET_COLUMN,
        "target_date_column": TARGET_DATE_COLUMN, "common_context": COMMON_LOOKBACK,
        "splits": split_summary, "models": specs, "selection": selection,
        "data_sha256": data_meta["output_sha256"], "raw_sha256": data_meta["source_raw_sha256"],
        "source_code_sha256": file_sha256(Path(__file__)),
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")},
        "python": platform.python_version(), "runtime": runtime,
        "training_config": {
            "epochs_limit": epochs, "seed": seed, "common_context": COMMON_LOOKBACK,
            "train_end": str(TRAIN_END.date()), "validation_end": str(VAL_END.date()),
            "test_end": str(TEST_END.date()), "lookbacks": [10, 30, 60] if include_lstm else [],
            "lstm_units": 32, "dense_units": 8, "dropout": 0.2,
            "batch_size": 32, "shuffle": False, "loss": "huber",
            "learning_rate": 0.001, "clipnorm": 1.0, "early_stopping_patience": 7,
            "feature_scaler": "StandardScaler", "target_scaler": "StandardScaler",
        },
        "serving_policy": {"demo_model": "LSTM" if include_lstm else selected,
                           "validation_selected_model": selected,
                           "note": "LSTM demonstrates ML; the overall validation winner is reported separately."},
        "artifact_sha256": {p.name: file_sha256(p) for p in artifact_dir.iterdir() if p.is_file()},
        "caveats": ["2025 was inspected in earlier experiments; this is a reused historical evaluation, not a pristine holdout.",
                    "Snapshot-adjusted prices and observed sessions; HOSE calendar not independently verified.",
                    "No trading profitability or causal claims; preprocessing fit on training history only."],
    }
    save_json(artifact_dir / "manifest.json", manifest)
    scores, predictions = evaluate_saved(artifact_dir, root, verify_recorded=False)
    manifest["metrics"] = json.loads(scores.to_json(orient="records"))
    save_json(artifact_dir / "manifest.json", manifest)
    scores.to_csv(report_dir / "metrics.csv", index=False)
    predictions.to_csv(report_dir / "predictions.csv", index=False)
    pointer = {"run_id": run_id, "artifact_dir": artifact_dir.relative_to(root).as_posix(),
               "report_dir": report_dir.relative_to(root).as_posix()}
    save_json(root / "models/latest_experiment.json", pointer)
    (report_dir / "requirements-lock.txt").write_text(subprocess.check_output(
        [os.sys.executable, "-m", "pip", "freeze"], text=True, encoding="utf-8"), encoding="utf-8")
    print(f"Selected on validation: {selected}; reports -> {report_dir}", flush=True)
    return {**pointer, "manifest": manifest, "metrics": scores, "predictions": predictions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-lstm", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_experiment(include_lstm=not args.no_lstm, epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
