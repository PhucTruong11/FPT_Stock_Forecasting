"""Controlled source/feature contrasts on common cohorts and one reference label."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.data.contracts import BASE_DIR, file_sha256, utc_now
from src.data.source_audit import guarded_hashes, read_evidence, save_json
from src.features.build_features import FEATURE_COLUMNS, TARGET_COLUMN, TARGET_DATE_COLUMN, build_dataset
from src.models.experiment import make_sequences
from src.models.robustness import fold_masks, score_direction

SOURCE_KEYS = {"yahoo_v2": "frozen", "vietcap_v3_candidate": "vci"}
REFERENCE = "Reference_Return"


def validate_config(config: dict) -> None:
    if config["automatic_promotion"] is not False or config["sources"] != list(SOURCE_KEYS):
        raise ValueError("This study supports two fixed sources and no serving promotion.")
    if config["reference_source"] != "vietcap_v3_candidate":
        raise ValueError("The reference label must be fixed before fitting.")
    if config["training_labels"] != "native_to_each_source" or config["validation_and_evaluation_labels"] != "one_fixed_vietcap_reference_for_all_candidates":
        raise ValueError("Registered label semantics do not match this implementation.")
    if config["feature_groups"]["full"] != FEATURE_COLUMNS:
        raise ValueError("Full control must preserve the original feature order.")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seeds must be nonempty and unique.")
    if not 1 <= config["lookback"] <= config["common_context"]:
        raise ValueError("Lookback exceeds the required common context.")
    if config["models"]["LSTM"]["epochs"] < 1:
        raise ValueError("Epoch limit must be positive.")
    for cols in config["feature_groups"].values():
        if not cols or len(set(cols)) != len(cols) or not set(cols).issubset(FEATURE_COLUMNS):
            raise ValueError("Invalid feature group.")
    for fold in config["folds"]:
        tr, va, ev = [pd.Timestamp(fold[k]) for k in ["train_end", "validation_end", "evaluation_end"]]
        if not tr < va < ev <= pd.Timestamp("2025-12-31"):
            raise ValueError("Ordered folds must not inspect 2026.")


def align_sources(datasets: dict, common_context: int) -> pd.DataFrame:
    cohort = None
    for source, (features, labels) in datasets.items():
        if features.Date.duplicated().any() or labels.Date.duplicated().any():
            raise ValueError("Unique dates are required before cohort alignment.")
        positions = pd.Index(features.Date).get_indexer(labels.Date)
        if (positions < 0).any():
            raise ValueError("A labeled origin is absent from the feature history.")
        native = labels[["Date", TARGET_DATE_COLUMN, TARGET_COLUMN]].copy()
        native[f"position_{source}"] = positions
        native = native.loc[positions >= common_context - 1].rename(columns={TARGET_COLUMN: f"return_{source}"})
        cohort = native if cohort is None else cohort.merge(native, on=["Date", TARGET_DATE_COLUMN], how="inner", validate="one_to_one")
    if cohort is None or cohort.empty:
        raise ValueError("No common origin/target pairs.")
    cohort = cohort.sort_values("Date").reset_index(drop=True)
    cohort[REFERENCE] = cohort["return_vietcap_v3_candidate"]
    return cohort


def source_part(cohort: pd.DataFrame, source: str, features: pd.DataFrame) -> pd.DataFrame:
    positions = cohort[f"position_{source}"].to_numpy(dtype=int)
    part = features.iloc[positions].reset_index(drop=True).copy()
    if not np.array_equal(part.Date, cohort.Date):
        raise ValueError("Feature positions do not match cohort origins.")
    part[TARGET_DATE_COLUMN] = cohort[TARGET_DATE_COLUMN].to_numpy()
    part[TARGET_COLUMN] = cohort[f"return_{source}"].to_numpy()
    part[REFERENCE] = cohort[REFERENCE].to_numpy()
    part["Feature_Position"] = positions
    return part


def context_positions(positions: np.ndarray, lookback: int) -> np.ndarray:
    positions = np.asarray(positions, dtype=int)
    if len(positions) == 0 or (positions < lookback - 1).any():
        raise ValueError("Insufficient training context.")
    return np.unique(np.concatenate([np.arange(i - lookback + 1, i + 1) for i in positions]))


def load_inputs(study: Path) -> tuple[dict, pd.DataFrame]:
    protocol = json.loads((study / "protocol.json").read_text(encoding="utf-8"))
    if file_sha256(study / "input_snapshot.zip") != protocol["input_zip_sha256"]:
        raise ValueError("Input archive changed after registration.")
    datasets = {}
    with zipfile.ZipFile(study / "input_snapshot.zip") as archive:
        for name, sha in protocol["input_files_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != sha:
                raise ValueError(f"Input changed: {name}")
        for source in SOURCE_KEYS:
            features = pd.read_csv(io.BytesIO(archive.read(f"{source}/features.csv")), parse_dates=["Date"])
            labels = pd.read_csv(io.BytesIO(archive.read(f"{source}/labels.csv")), parse_dates=["Date", TARGET_DATE_COLUMN])
            datasets[source] = (features, labels)
    cohort = align_sources(datasets, protocol["config"]["common_context"])
    return datasets, cohort


def prepare(config_path: Path, root: Path = BASE_DIR) -> Path:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validate_config(config)
    audit = root / "reports/data_audits" / config["audit_id"]
    evidence = read_evidence(audit)
    study = root / "reports/source_comparison" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    study.mkdir(parents=True, exist_ok=False)
    before = guarded_hashes()
    frames, quality = {}, {}
    for source, key in SOURCE_KEYS.items():
        raw = evidence[key]
        features, labels, excluded, info = build_dataset(raw)
        quality[source] = {**info, "source": key, "calendar_verified": False,
                           "point_in_time_verified": False, "known_provider_flags_not_removed_posthoc": True}
        for name, frame in [("raw", raw), ("features", features), ("labels", labels), ("excluded", excluded)]:
            frames[f"{source}/{name}.csv"] = frame.to_csv(index=False, date_format="%Y-%m-%d", lineterminator="\n").encode("utf-8")
    with zipfile.ZipFile(study / "input_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in frames.items():
            archive.writestr(name, content)
    code = ["src/models/source_comparison.py", "src/models/source_comparison_report.py", "src/models/robustness.py",
            "src/models/experiment.py", "src/features/build_features.py", "src/data/contracts.py", "src/data/source_audit.py"]
    protocol = {
        "schema": "fpt-source-comparison-v1", "study_id": study.name, "registered_before_training_utc": utc_now(),
        "config": config, "config_sha256": file_sha256(config_path), "protected_before": before,
        "audit_archive_sha256": file_sha256(audit / "evidence.zip"),
        "input_zip_sha256": file_sha256(study / "input_snapshot.zip"),
        "input_files_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in frames.items()},
        "source_sha256": {name: file_sha256(root / name) for name in code},
        "python": platform.python_version(), "versions": {p: importlib.metadata.version(p) for p in
            ["numpy", "pandas", "scikit-learn", "xgboost", "tensorflow", "joblib"]},
        "warning": "Retrospective source sensitivity; adjusted reference is not certified ground truth. No serving promotion.",
    }
    save_json(study / "protocol.json", protocol)
    with zipfile.ZipFile(study / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in code:
            archive.write(root / name, name)
        archive.write(config_path, "config.json")
    datasets, cohort = load_inputs(study)
    cohort.to_csv(study / "cohort.csv", index=False, date_format="%Y-%m-%d", lineterminator="\n")
    splits = []
    for fold in config["folds"]:
        for phase, mask in fold_masks(cohort, fold).items():
            part = cohort.loc[mask, ["Date", TARGET_DATE_COLUMN]].copy()
            if part.empty:
                raise ValueError(f"Empty cohort: {fold['id']}/{phase}")
            part["fold"], part["phase"] = fold["id"], phase
            splits.append(part)
    pd.concat(splits, ignore_index=True).to_csv(study / "splits.csv", index=False, date_format="%Y-%m-%d", lineterminator="\n")
    save_json(study / "data_quality.json", quality)
    save_json(study / "status.json", {"status": "prepared", "completed_candidates": 0})
    print(f"Prepared {study}; common cohort={len(cohort)}", flush=True)
    return study


def train(study: Path, root: Path = BASE_DIR) -> Path:
    protocol = json.loads((study / "protocol.json").read_text(encoding="utf-8"))
    if json.loads((study / "status.json").read_text())["status"] != "prepared":
        raise ValueError("Only a fresh prepared study may train; create a new ID for a rerun.")
    for name, sha in protocol["source_sha256"].items():
        if file_sha256(root / name) != sha:
            raise ValueError(f"Registered code changed: {name}")
    if guarded_hashes() != protocol["protected_before"]:
        raise ValueError("Canonical data/model changed after preparation.")
    config = protocol["config"]
    datasets, cohort = load_inputs(study)
    expected_count = len(config["folds"]) * len(SOURCE_KEYS) * len(config["feature_groups"]) * sum(
        1 if model == "Ridge" else len(config["seeds"]) for model in config["models"])
    artifact_dir = root / "models/research" / ("source-comparison_" + study.name)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    rows, exports, candidates, histories = [], [], [], {}
    save_json(study / "status.json", {"status": "running", "completed_candidates": 0})

    def record(identity: dict, phase: str, part: pd.DataFrame, prediction) -> None:
        actual = part[REFERENCE].to_numpy()
        rows.append({**identity, "phase": phase, **score_direction(actual, prediction)})
        frame = part[["Date", TARGET_DATE_COLUMN, REFERENCE, TARGET_COLUMN]].copy()
        frame["predicted_log_return"] = np.asarray(prediction).ravel()
        for key, value in {**identity, "phase": phase}.items():
            frame[key] = value
        exports.append(frame)

    for fold in config["folds"]:
        masks = fold_masks(cohort, fold)
        # Zero is a shared control, scored once per phase/fold on the same reference.
        ref_features = datasets[config["reference_source"]][0]
        for phase in ["validation", "evaluation"]:
            part = source_part(cohort.loc[masks[phase]].reset_index(drop=True), config["reference_source"], ref_features)
            record({"candidate": fold["id"] + "_zero", "fold": fold["id"], "source": "shared",
                    "feature_group": "baseline", "model": "Zero-Return", "seed": None}, phase, part, np.zeros(len(part)))
        for source in SOURCE_KEYS:
            features = datasets[source][0]
            parts = {phase: source_part(cohort.loc[mask].reset_index(drop=True), source, features) for phase, mask in masks.items()}
            training, validation = parts["train"], parts["validation"]
            y_train = training[TARGET_COLUMN].to_numpy()
            for baseline in ["Train Mean", "Return Persistence"]:
                identity = {"candidate": f"{fold['id']}_{source}_{baseline}", "fold": fold["id"],
                            "source": source, "feature_group": "baseline", "model": baseline, "seed": None}
                for phase in ["validation", "evaluation"]:
                    part = parts[phase]
                    pred = np.full(len(part), y_train.mean()) if baseline == "Train Mean" else part.Log_Return.to_numpy()
                    record(identity, phase, part, pred)
            print(f"{fold['id']}/{source}: train={len(training)} val={len(validation)} eval={len(parts['evaluation'])}", flush=True)
            for group, columns in config["feature_groups"].items():
                context = context_positions(training.Feature_Position.to_numpy(), config["lookback"])
                if not features.iloc[context].Date.le(pd.Timestamp(fold["train_end"])).all():
                    raise ValueError("Future context entered feature scaler.")
                scaler = StandardScaler().fit(features.iloc[context][columns])
                target_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
                values = scaler.transform(features[columns])
                scalers_file = f"{fold['id']}_{source}_{group}_scalers.pkl"
                joblib.dump({"features": scaler, "target": target_scaler, "context_positions": context}, artifact_dir / scalers_file)
                for model_name in ["Ridge", "XGBoost", "LSTM"]:
                    for seed in ([None] if model_name == "Ridge" else config["seeds"]):
                        started = time.perf_counter()
                        candidate = f"{fold['id']}_{source}_{group}_{model_name}_s{seed}"
                        identity = {"candidate": candidate, "fold": fold["id"], "source": source,
                                    "feature_group": group, "model": model_name, "seed": seed}
                        if model_name == "Ridge":
                            model = make_pipeline(StandardScaler(), Ridge(**config["models"]["Ridge"]))
                            model.fit(training[columns], y_train)
                            filename = candidate + ".pkl"
                            joblib.dump(model, artifact_dir / filename)
                        elif model_name == "XGBoost":
                            model = XGBRegressor(**config["models"]["XGBoost"], random_state=seed)
                            model.fit(training[columns], y_train)
                            filename = candidate + ".json"
                            model.save_model(artifact_dir / filename)
                        else:
                            tf.keras.backend.clear_session()
                            tf.keras.utils.set_random_seed(seed)
                            spec, lb = config["models"]["LSTM"], config["lookback"]
                            model = tf.keras.Sequential([tf.keras.layers.Input((lb, len(columns))),
                                tf.keras.layers.LSTM(spec["units"]), tf.keras.layers.Dropout(spec["dropout"]),
                                tf.keras.layers.Dense(spec["dense_units"], activation="relu"), tf.keras.layers.Dense(1)])
                            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=spec["learning_rate"], clipnorm=spec["clipnorm"]), loss=spec["loss"])
                            x_train = make_sequences(values, training.Feature_Position.to_numpy(), lb)
                            x_val = make_sequences(values, validation.Feature_Position.to_numpy(), lb)
                            history = model.fit(x_train, target_scaler.transform(y_train.reshape(-1, 1)),
                                validation_data=(x_val, target_scaler.transform(validation[REFERENCE].to_numpy().reshape(-1, 1))),
                                epochs=spec["epochs"], batch_size=spec["batch_size"], shuffle=False, verbose=0,
                                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=spec["early_stopping_patience"], restore_best_weights=True),
                                    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=spec["reduce_lr_patience"], factor=.5, min_lr=1e-5)])
                            histories[candidate] = {k: [float(v) for v in vals] for k, vals in history.history.items()}
                            filename = candidate + ".keras"
                            model.save(artifact_dir / filename)
                        for phase in ["validation", "evaluation"]:
                            part = parts[phase]
                            if model_name == "LSTM":
                                tensor = make_sequences(values, part.Feature_Position.to_numpy(), config["lookback"])
                                pred = target_scaler.inverse_transform(model(tensor, training=False).numpy()).ravel()
                            else:
                                pred = model.predict(part[columns])
                            record(identity, phase, part, pred)
                        candidates.append({**identity, "file": filename, "feature_columns": columns,
                            "scalers": scalers_file if model_name == "LSTM" else None,
                            "epochs_trained": len(histories[candidate]["loss"]) if model_name == "LSTM" else None,
                            "seconds": time.perf_counter() - started})
                        pd.DataFrame(rows).to_csv(study / "metrics.csv", index=False, lineterminator="\n")
                        save_json(study / "status.json", {"status": "running", "completed_candidates": len(candidates)})
                        print(f"{len(candidates):3d}/{expected_count} {candidate}: val={rows[-2]['rmse']:.7f} eval={rows[-1]['rmse']:.7f}", flush=True)
    pd.concat(exports, ignore_index=True).to_csv(study / "predictions.csv.gz", index=False, compression="gzip", date_format="%Y-%m-%d", lineterminator="\n")
    save_json(study / "lstm_history.json", histories)
    after = guarded_hashes()
    if after != protocol["protected_before"]:
        raise RuntimeError("Canonical data/model files changed during the study.")
    save_json(study / "manifest.json", {"schema": "fpt-source-comparison-artifacts-v1", "study_id": study.name,
        "protocol_sha256": file_sha256(study / "protocol.json"), "artifact_dir": artifact_dir.relative_to(root).as_posix(),
        "candidates": candidates, "artifact_sha256": {p.name: file_sha256(p) for p in artifact_dir.iterdir()},
        "protected_after": after, "automatic_promotion": False,
        "tensorflow": tf.__version__, "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU"})
    save_json(study / "status.json", {"status": "complete", "completed_candidates": len(candidates), "metric_rows": len(rows), "finished_at_utc": utc_now()})
    print(f"Complete: {study}", flush=True)
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE_DIR / "configs/source_comparison_v1.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-prepared", type=Path)
    args = parser.parse_args()
    if args.train_prepared:
        train(args.train_prepared.resolve())
    else:
        prepared = prepare(args.config.resolve())
        if not args.prepare_only:
            train(prepared)
