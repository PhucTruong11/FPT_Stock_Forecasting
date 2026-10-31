"""Versioned retrospective rolling-origin comparisons; never changes serving pointers."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.data.contracts import BASE_DIR, file_sha256, invalid_candle_mask
from src.data.snapshot import restore_snapshot
from src.features.build_features import FEATURE_COLUMNS, TARGET_COLUMN, TARGET_DATE_COLUMN
from src.models.experiment import load_data, make_sequences, metrics, save_json


def validate_protocol(config: dict) -> None:
    if config["selection"]["automatic_promotion"] is not False:
        raise ValueError("This retrospective study cannot promote serving models.")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seeds must be a nonempty unique list.")
    if not 1 <= config["lookback"] <= 60 or config["epochs_limit"] < 1:
        raise ValueError("Invalid training limits.")
    for fold in config["folds"]:
        tr, va, ev = (pd.Timestamp(fold[k]) for k in ("train_end", "validation_end", "evaluation_end"))
        if not tr < va < ev or ev > pd.Timestamp("2024-12-31"):
            raise ValueError("Ordered folds must end before the reused 2025 reference period.")
    for name, columns in config["feature_groups"].items():
        if not columns or len(set(columns)) != len(columns) or not set(columns).issubset(FEATURE_COLUMNS):
            raise ValueError(f"Invalid feature allowlist: {name}")
    if config["feature_groups"].get("full") != FEATURE_COLUMNS:
        raise ValueError("The full control must preserve all 16 original inputs and their order.")


def fold_masks(labels: pd.DataFrame, fold: dict) -> dict[str, np.ndarray]:
    origin, target = labels["Date"], labels[TARGET_DATE_COLUMN]
    train_end, val_end, eval_end = (pd.Timestamp(fold[k]) for k in
                                   ("train_end", "validation_end", "evaluation_end"))
    return {
        "train": ((origin <= train_end) & (target <= train_end)).to_numpy(),
        "validation": ((origin > train_end) & (origin <= val_end) & (target <= val_end)).to_numpy(),
        "evaluation": ((origin > val_end) & (origin <= eval_end) & (target <= eval_end)).to_numpy(),
    }


def quality_audit(root: Path, features: pd.DataFrame, labels: pd.DataFrame, meta: dict) -> dict:
    raw = pd.read_csv(root / "data/raw/FPT.VN_raw.csv", parse_dates=["Date"])
    finite_features = np.isfinite(features[FEATURE_COLUMNS].to_numpy(dtype=float)).all()
    invalid = invalid_candle_mask(raw)
    target_prices = raw.set_index("Date")["Close"].reindex(labels[TARGET_DATE_COLUMN]).to_numpy()
    expected = np.log(target_prices / labels["Close"].to_numpy())
    if not finite_features or not np.allclose(expected, labels[TARGET_COLUMN], atol=1e-12):
        raise ValueError("Feature finiteness or independently recomputed labels failed.")
    if raw["Date"].duplicated().any() or not raw["Date"].is_monotonic_increasing:
        raise ValueError("Raw dates are not unique and sorted.")
    return {
        "raw_rows": len(raw), "feature_rows": len(features), "common_context_label_rows": len(labels),
        "zero_volume_rows": int(raw.Volume.eq(0).sum()), "invalid_candles": int(invalid.sum()),
        "missing_close_rows": int(raw.Close.isna().sum()),
        "weekend_rows": int(raw.Date.dt.dayofweek.ge(5).sum()),
        "raw_dates_unique_and_sorted": True, "feature_inputs_finite": True,
        "labels_independently_recomputed_from_source_closes": True,
        "calendar_verified": meta["calendar_verified"],
        "unresolved": ["HOSE calendar not independently reconciled", "Corporate actions and point-in-time adjustments not independently reconciled"],
        "scope": "Internal validation of the frozen research snapshot, not independent source certification",
    }


def score_direction(actual, predicted) -> dict:
    result = metrics(actual, predicted)
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    recalls = [float((np.sign(predicted[actual > 0]) == 1).mean()) if (actual > 0).any() else None,
               float((np.sign(predicted[actual < 0]) == -1).mean()) if (actual < 0).any() else None]
    result["balanced_accuracy_pct"] = float(50 * sum(recalls)) if all(r is not None for r in recalls) and np.any(predicted != 0) else None
    return result


def run_study(config_path: Path, root: Path = BASE_DIR) -> Path:
    root = Path(root)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validate_protocol(config)
    restore_snapshot(root)
    features, labels, meta = load_data(root)
    pointers = {name: (root / "models" / name).read_bytes() for name in
                ("latest_experiment.json", "reference_experiment.json")}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    reports = root / "reports/robustness" / run_id
    artifacts = root / "models/research" / run_id
    reports.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    protocol = {
        "study_id": run_id, "registered_before_training_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "config_sha256": file_sha256(Path(config_path)),
        "source_sha256": {str(p.relative_to(root)).replace("\\", "/"): file_sha256(p) for p in
                          (Path(__file__), root / "src/models/experiment.py", root / "src/features/build_features.py",
                           root / "src/models/robustness_report.py", root / "src/models/research_notes.py")},
        "data_sha256": meta["output_sha256"], "raw_sha256": meta["source_raw_sha256"],
        "incumbent": json.loads(pointers["latest_experiment.json"]),
        "common_context": 60, "schema": "fpt-robustness-v1",
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")},
        "warning": "Retrospective diagnostic. Evaluation windows overlap historical development knowledge; not a new prospective holdout.",
    }
    save_json(reports / "protocol.json", protocol)
    save_json(reports / "data_quality.json", quality_audit(root, features, labels, meta))
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    tf.config.experimental.enable_op_determinism()
    records, predictions, manifests, splits, histories = [], [], [], [], {}
    save_json(reports / "status.json", {"status": "running", "completed_candidates": 0})

    def record(fold, group, model, seed, part, pred, phase, candidate):
        actual = part[TARGET_COLUMN].to_numpy()
        identity = {"fold": fold, "feature_group": group, "model": model, "seed": seed,
                    "phase": phase, "candidate": candidate}
        records.append({**identity, **score_direction(actual, pred)})
        output = part[["Date", TARGET_DATE_COLUMN, TARGET_COLUMN]].copy()
        output["predicted_log_return"] = pred
        for key, value in identity.items():
            output[key] = value
        predictions.append(output)

    for fold in config["folds"]:
        masks = fold_masks(labels, fold)
        parts = {name: labels.loc[mask] for name, mask in masks.items()}
        if any(part.empty for part in parts.values()):
            raise ValueError(f"Empty partition: {fold['id']}")
        for name, part in parts.items():
            splits.append({"fold": fold["id"], "phase": name, "n": len(part),
                           "origin_start": str(part.Date.min().date()), "origin_end": str(part.Date.max().date()),
                           "target_end": str(part[TARGET_DATE_COLUMN].max().date())})
        pd.DataFrame(splits).to_csv(reports / "splits.csv", index=False)
        train, val = parts["train"], parts["validation"]
        ytrain = train[TARGET_COLUMN].to_numpy()
        for baseline in ("Zero-Return", "Train Mean", "Return Persistence", "Always Up", "Always Down"):
            candidate = f"{fold['id']}_baseline_{baseline.replace(' ', '_')}"
            for phase in ("validation", "evaluation"):
                part = parts[phase]
                if baseline == "Return Persistence":
                    pred = part["Log_Return"].to_numpy()
                else:
                    value = {"Zero-Return": 0., "Train Mean": float(ytrain.mean()),
                             "Always Up": float(np.abs(ytrain).mean()), "Always Down": -float(np.abs(ytrain).mean())}[baseline]
                    pred = np.full(len(part), value)
                record(fold["id"], "baseline", baseline, None, part, pred, phase, candidate)
        for group, columns in config["feature_groups"].items():
            print(f"{fold['id']} / {group}: train={len(train)}, validation={len(val)}, evaluation={len(parts['evaluation'])}", flush=True)
            scaler = StandardScaler().fit(features.loc[features.Date.le(pd.Timestamp(fold["train_end"])), columns])
            target_scaler = StandardScaler().fit(ytrain.reshape(-1, 1))
            values = scaler.transform(features[columns])
            scalers_file = f"{fold['id']}_{group}_scalers.pkl"
            joblib.dump({"features": scaler, "target": target_scaler}, artifacts / scalers_file)
            for model_name in ("Ridge", "XGBoost", "LSTM"):
                for seed in ([None] if model_name == "Ridge" else config["seeds"]):
                    candidate = f"{fold['id']}_{group}_{model_name}_s{seed}"
                    start = time.perf_counter()
                    if model_name == "Ridge":
                        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                        model.fit(train[columns], ytrain)
                        filename = candidate + ".pkl"
                        joblib.dump(model, artifacts / filename)
                    elif model_name == "XGBoost":
                        model = XGBRegressor(max_depth=2, n_estimators=150, learning_rate=.03,
                                             min_child_weight=5, subsample=.9, colsample_bytree=.9,
                                             objective="reg:squarederror", n_jobs=2, random_state=seed)
                        model.fit(train[columns], ytrain)
                        filename = candidate + ".json"
                        model.save_model(artifacts / filename)
                    else:
                        tf.keras.backend.clear_session()
                        tf.keras.utils.set_random_seed(seed)
                        lb = config["lookback"]
                        model = tf.keras.Sequential([tf.keras.layers.Input((lb, len(columns))),
                            tf.keras.layers.LSTM(32), tf.keras.layers.Dropout(.2),
                            tf.keras.layers.Dense(8, activation="relu"), tf.keras.layers.Dense(1)])
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001, clipnorm=1.), loss="huber")
                        xtrain = make_sequences(values, train.Feature_Position.to_numpy(), lb)
                        xval = make_sequences(values, val.Feature_Position.to_numpy(), lb)
                        history = model.fit(xtrain, target_scaler.transform(ytrain.reshape(-1, 1)).ravel(),
                            validation_data=(xval, target_scaler.transform(val[[TARGET_COLUMN]].to_numpy()).ravel()),
                            epochs=config["epochs_limit"], batch_size=32, shuffle=False, verbose=0,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
                                       tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=.5, min_lr=1e-5)])
                        histories[candidate] = {k: [float(v) for v in vls] for k, vls in history.history.items()}
                        filename = candidate + ".keras"
                        model.save(artifacts / filename)
                    for phase in ("validation", "evaluation"):
                        part = parts[phase]
                        if model_name == "LSTM":
                            tensor = make_sequences(values, part.Feature_Position.to_numpy(), config["lookback"])
                            pred = target_scaler.inverse_transform(model(tensor, training=False).numpy()).ravel()
                        else:
                            pred = model.predict(part[columns])
                        record(fold["id"], group, model_name, seed, part, pred, phase, candidate)
                    manifests.append({"candidate": candidate, "fold": fold["id"], "feature_group": group,
                        "model": model_name, "seed": seed, "file": filename, "feature_columns": columns,
                        "scalers": scalers_file if model_name == "LSTM" else None,
                        "epochs_trained": len(histories[candidate]["loss"]) if model_name == "LSTM" else None,
                        "seconds": time.perf_counter() - start})
                    pd.DataFrame(records).to_csv(reports / "metrics.csv", index=False)
                    save_json(reports / "status.json", {"status": "running", "completed_candidates": len(manifests)})
                    print(f"  {candidate}: val RMSE={records[-2]['rmse']:.7f}, eval RMSE={records[-1]['rmse']:.7f}", flush=True)
    scores = pd.DataFrame(records)
    pd.concat(predictions, ignore_index=True).to_csv(reports / "predictions.csv.gz", index=False, compression="gzip")
    stability = scores.groupby(["fold", "feature_group", "model", "phase"], dropna=False).agg(
        runs=("rmse", "size"), rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        rmse_min=("rmse", "min"), rmse_max=("rmse", "max"), mae_mean=("mae", "mean"),
        hit_rate_mean=("hit_rate_pct", "mean"), balanced_accuracy_mean=("balanced_accuracy_pct", "mean"),
        mse_skill_vs_zero_mean=("mse_skill_vs_zero", "mean")).reset_index()
    stability.to_csv(reports / "seed_stability.csv", index=False)
    fold_summary = stability.loc[stability.phase.eq("evaluation")].groupby(["feature_group", "model"]).agg(
        folds=("fold", "size"), mean_fold_rmse=("rmse_mean", "mean"), worst_fold_rmse=("rmse_mean", "max"),
        mean_fold_mae=("mae_mean", "mean"), mean_seed_std=("rmse_std", "mean"),
        mean_mse_skill_vs_zero=("mse_skill_vs_zero_mean", "mean"),
        positive_skill_folds=("mse_skill_vs_zero_mean", lambda x: int((x > 0).sum()))).reset_index()
    fold_summary.to_csv(reports / "fold_summary.csv", index=False)
    save_json(reports / "lstm_history.json", histories)
    unchanged = all((root / "models" / name).read_bytes() == value for name, value in pointers.items())
    if not unchanged:
        raise RuntimeError("Serving pointers changed during the study; inspect concurrent work.")
    manifest = {"study_id": run_id, "protocol_sha256": file_sha256(reports / "protocol.json"),
        "tensorflow": tf.__version__, "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
        "candidates": manifests, "artifact_dir": artifacts.relative_to(root).as_posix(),
        "artifact_sha256": {p.name: file_sha256(p) for p in artifacts.iterdir() if p.is_file()},
        "serving_pointers_unchanged": unchanged, "automatic_promotion": False,
        "promotion_decision": "keep_incumbent_pending_unseen_forward_evaluation_and_source_verification"}
    save_json(reports / "manifest.json", manifest)
    save_json(reports / "status.json", {"status": "complete", "completed_candidates": len(manifests), "metric_rows": len(scores)})
    print(f"Study complete: {reports}; serving pointers unchanged.", flush=True)
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE_DIR / "configs/robustness_v1.json")
    args = parser.parse_args()
    run_study(args.config)
