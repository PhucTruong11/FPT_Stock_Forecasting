"""Verify research evidence and write a comparable, reproducible Markdown record."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.contracts import BASE_DIR, file_sha256


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def verify_reloaded_models(study: Path, root: Path = BASE_DIR) -> None:
    import joblib
    import tensorflow as tf
    from xgboost import XGBRegressor
    from src.models.experiment import load_data, make_sequences
    from src.models.robustness import fold_masks

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    features, labels, _ = load_data(root)
    manifest = json.loads((study / "manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((study / "protocol.json").read_text(encoding="utf-8"))
    exported = pd.read_csv(study / "predictions.csv.gz")
    expected = {(candidate, phase): part for (candidate, phase), part in exported.groupby(["candidate", "phase"])}
    folds = {fold["id"]: fold for fold in protocol["config"]["folds"]}
    artifacts = Path(root) / manifest["artifact_dir"]
    checks = []
    for candidate in manifest["candidates"]:
        columns = candidate["feature_columns"]
        fold = folds[candidate["fold"]]
        masks = fold_masks(labels, fold)
        if candidate["model"] == "LSTM":
            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(artifacts / candidate["file"], compile=False)
            scalers = joblib.load(artifacts / candidate["scalers"])
            training_features = features.loc[features.Date.le(pd.Timestamp(fold["train_end"])), columns]
            training_targets = labels.loc[masks["train"], "Target_Return_1d"]
            if not np.allclose(scalers["features"].mean_, training_features.mean().to_numpy()) or not np.isclose(scalers["target"].mean_[0], training_targets.mean()):
                raise ValueError("Stored scaler does not match train-only history.")
            if scalers["features"].n_samples_seen_ != len(training_features) or scalers["target"].n_samples_seen_ != len(training_targets):
                raise ValueError("Stored scaler saw an incorrect number of training observations.")
            values = scalers["features"].transform(features[columns])
        elif candidate["model"] == "XGBoost":
            model = XGBRegressor()
            model.load_model(artifacts / candidate["file"])
        else:
            model = joblib.load(artifacts / candidate["file"])
        for phase in ("validation", "evaluation"):
            part = labels.loc[masks[phase]]
            reference = expected[(candidate["candidate"], phase)]
            if tuple(part.Date.dt.strftime("%Y-%m-%d")) != tuple(reference["Date"]):
                raise ValueError("Reloaded inference dates do not match prediction export.")
            if candidate["model"] == "LSTM":
                tensor = make_sequences(values, part.Feature_Position.to_numpy(), protocol["config"]["lookback"])
                pred = scalers["target"].inverse_transform(model(tensor, training=False).numpy()).ravel()
            else:
                pred = model.predict(part[columns])
            original = reference["predicted_log_return"].to_numpy()
            if not np.allclose(pred, original, atol=1e-8, rtol=1e-5):
                raise ValueError(f"Reloaded predictions differ: {candidate['candidate']}/{phase}")
            checks.append({"candidate": candidate["candidate"], "phase": phase,
                           "max_absolute_log_return_difference": float(np.max(np.abs(pred - original))), "verified": True})
        print("Reload verified:", candidate["candidate"], flush=True)
    write_json(study / "reload_verification.json", {"candidates_reloaded": len(manifest["candidates"]),
        "phase_checks": len(checks), "lstm_scalers_match_train_only_statistics": True,
        "absolute_tolerance": 1e-8, "relative_tolerance": 1e-5, "checks": checks})


def verify_and_report(study: Path, root: Path = BASE_DIR) -> Path:
    root, study = Path(root).resolve(), Path(study).resolve()
    status = json.loads((study / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete":
        raise ValueError("Do not report an incomplete study as complete.")
    manifest = json.loads((study / "manifest.json").read_text(encoding="utf-8"))
    protocol_path = study / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    raw_archive = study / "protocol.original.json.gz"
    if not raw_archive.exists():
        raw_archive.write_bytes(gzip.compress(protocol_path.read_bytes(), mtime=0))
    if hashlib.sha256(gzip.decompress(raw_archive.read_bytes())).hexdigest() != manifest["protocol_sha256"]:
        raise ValueError("The original preregistered protocol checksum changed.")
    source_zip = study / "source_snapshot.zip"
    if not source_zip.exists():
        for name, digest in protocol["source_sha256"].items():
            if file_sha256(root / name) != digest:
                raise ValueError(f"Source changed before capture: {name}; recover the exact source.")
        with zipfile.ZipFile(source_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in protocol["source_sha256"]:
                archive.write(root / name, name)
    with zipfile.ZipFile(source_zip) as archive:
        for name, digest in protocol["source_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Source snapshot mismatch: {name}")
    for name, digest in manifest["artifact_sha256"].items():
        if file_sha256(root / manifest["artifact_dir"] / name) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    scores = pd.read_csv(study / "metrics.csv")
    predictions = pd.read_csv(study / "predictions.csv.gz")
    recorded = scores.set_index(["candidate", "phase"])
    checked, date_sets = [], {}
    for (candidate, phase), frame in predictions.groupby(["candidate", "phase"]):
        actual, pred = frame["Target_Return_1d"].to_numpy(), frame["predicted_log_return"].to_numpy()
        expected = recorded.loc[(candidate, phase)]
        rmse = float(np.sqrt(np.square(actual - pred).mean()))
        mae = float(np.abs(actual - pred).mean())
        if not np.isclose(rmse, expected.rmse, rtol=1e-7, atol=1e-10) or not np.isclose(mae, expected.mae, rtol=1e-7, atol=1e-10):
            raise ValueError(f"Independently recomputed errors disagree: {candidate}/{phase}")
        nonzero = actual != 0
        hit = float(100 * (np.sign(actual[nonzero]) == np.sign(pred[nonzero])).mean()) if np.any(pred[nonzero] != 0) else None
        if (hit is None and not pd.isna(expected.hit_rate_pct)) or (hit is not None and not np.isclose(hit, expected.hit_rate_pct)):
            raise ValueError(f"Directional score disagreement: {candidate}/{phase}")
        origin_dates = tuple(frame["Date"])
        key = (frame["fold"].iloc[0], phase)
        if key in date_sets and date_sets[key] != origin_dates:
            raise ValueError("Models were scored on different eligible forecast origins.")
        date_sets[key] = origin_dates
        checked.append({"candidate": candidate, "phase": phase, "n": len(frame), "rmse": rmse,
                        "mae": mae, "hit_rate_pct": hit, "verified": True})
    if len(checked) != len(scores) or len(scores) != status["metric_rows"]:
        raise ValueError("A recorded candidate or phase is missing predictions.")
    write_json(study / "verification.json", {
        "metric_rows_independently_verified": len(checked), "same_origin_dates_per_fold_phase": True,
        "artifact_hashes_verified": len(manifest["artifact_sha256"]), "source_snapshot_verified": True,
        "original_protocol_verified": True, "checks": checked})
    # Exact original bytes stay in gzip; readable JSON follows repository LF attributes.
    protocol_path.write_bytes(protocol_path.read_bytes().replace(b"\r\n", b"\n"))
    manifest["protocol_original_archive"] = raw_archive.name
    manifest["protocol_sha256_lf"] = file_sha256(protocol_path)
    manifest["source_snapshot"] = source_zip.name
    manifest["source_snapshot_sha256"] = file_sha256(source_zip)
    manifest["postprocessing_note"] = "Original protocol bytes archived before LF normalization for Git; numeric results and pre-training configuration unchanged."
    write_json(study / "manifest.json", manifest)
    summary = pd.read_csv(study / "fold_summary.csv")
    seeds = pd.read_csv(study / "seed_stability.csv")
    fold_count = len(protocol["config"]["folds"])
    model_counts = {name: sum(c["model"] == name for c in manifest["candidates"]) for name in ("LSTM", "XGBoost", "Ridge")}
    table = []
    for row in summary.itertuples():
        deviation = "N/A (1 seed/run)" if pd.isna(row.mean_seed_std) else f"{row.mean_seed_std:.8f}"
        table.append(f"| {row.feature_group} | {row.model} | {row.mean_fold_rmse:.7f} | {row.worst_fold_rmse:.7f} | {deviation} | {row.positive_skill_folds}/{fold_count} |")
    lstm = seeds.loc[seeds.model.eq("LSTM") & seeds.phase.eq("evaluation")]
    fold_table = "\n".join(f"| {r.fold} | {r.feature_group} | {r.rmse_mean:.7f} | {r.rmse_std:.8f} | {r.balanced_accuracy_mean:.2f}% |" for r in lstm.itertuples())
    full = summary.loc[summary.feature_group.eq("full") & summary.model.eq("LSTM")].iloc[0]
    returns = summary.loc[summary.feature_group.eq("returns") & summary.model.eq("LSTM")].iloc[0]
    zero = summary.loc[summary.model.eq("Zero-Return")].iloc[0]
    mean = summary.loc[summary.model.eq("Train Mean")].iloc[0]
    delta = float(100 * (returns.mean_fold_rmse / full.mean_fold_rmse - 1))
    zero_delta = float(100 * (returns.mean_fold_rmse / zero.mean_fold_rmse - 1))
    report = f"""# Đối chứng độ ổn định và feature ablation FPT

Study `{protocol['study_id']}`; protocol đăng ký trước training lúc `{protocol['registered_before_training_utc']}` (UTC).

## Kết luận của lượt này

- Đã train **{status['completed_candidates']} candidate**: {model_counts['LSTM']} LSTM, {model_counts['XGBoost']} XGBoost, {model_counts['Ridge']} Ridge; thêm 5 baseline cho mỗi fold. Có **{len(scores)} hàng model/phase metrics**, cùng forecast origins trong từng fold/phase.
- LSTM chỉ dùng returns có mean-fold evaluation RMSE `{returns.mean_fold_rmse:.7f}`, so với full 16 features `{full.mean_fold_rmse:.7f}`: thay đổi **{delta:+.2f}%**. Đây là kết quả hồi cứu, không phải mức tăng chất lượng được bảo đảm cho tương lai.
- Returns-only LSTM vẫn cao hơn Zero-Return khoảng **{zero_delta:.2f}%** theo mean-fold RMSE. Train Mean đạt `{mean.mean_fold_rmse:.7f}`; chưa có bằng chứng để thay model đang phục vụ bằng một candidate mới.
- Độ lệch chuẩn RMSE theo seed của full LSTM trung bình `{full.mean_seed_std:.8f}`, returns-only `{returns.mean_seed_std:.8f}`. Ít feature giúp mean error trong mẫu nghiên cứu này nhưng không cải thiện đồng thời mọi chỉ số ổn định.
- **Quyết định: giữ model hiện tại; không tự promote.** Chưa xác minh độc lập lịch HOSE/corporate actions; chưa có acceptance period tương lai chưa dùng tuning.

## Giao thức đối chứng

| Fold | Train đến | Validation đến | Evaluation đến |
| --- | --- | --- | --- |
"""
    for fold in protocol["config"]["folds"]:
        report += f"| {fold['id']} | {fold['train_end']} | {fold['validation_end']} | {fold['evaluation_end']} |\n"
    report += f"""
Seed XGBoost/LSTM: `{protocol['config']['seeds']}`. Ridge xác định chạy một lần/fold/group. Feature groups gồm returns (6), returns+technical (13), full gồm volume (16). LSTM lookback {protocol['config']['lookback']}, 32 units, Dense 8, Huber, StandardScaler train-only, batch 32, shuffle=False, max {protocol['config']['epochs_limit']} epochs, early stopping patience 7. Hyperparameters được giữ cố định; không chọn seed bằng evaluation.

Mọi model dùng cùng origin/target dates và common context 60. Nhãn đi qua ranh giới tập bị loại. Baseline luôn tăng/giảm được khai báo trước study; không thêm sau khi nhìn tỷ lệ nhãn. `splits.csv` lưu ngày và số mẫu chính xác.

## Tổng hợp evaluation: trung bình seed trước, trung bình {fold_count} fold sau

Mean-fold RMSE là trung bình không trọng số các RMSE theo fold, **không phải pooled RMSE**. Cột std là trung bình sample standard deviation giữa các seed trong từng fold. Model chạy một lần không có bằng chứng để ước lượng seed variance. “Fold vượt zero” dùng mean MSE skill so với Zero-Return trong cùng fold, không phải kiểm định ý nghĩa thống kê.

| Feature group | Model | Mean-fold RMSE | Worst-fold RMSE | Mean seed std | Fold có MSE skill > 0 |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

## Chi tiết LSTM: lỗi thay đổi theo cả thời gian và seed

| Fold | Features | Mean seed RMSE | Seed std | Mean balanced accuracy |
| --- | --- | ---: | ---: | ---: |
{fold_table}

## Giới hạn và diễn giải

Các giai đoạn 2022–2024 đã nằm trong lịch sử phát triển; kiến trúc/features từng được chọn sau khi xem giai đoạn 2023–2025. Validation/evaluation giữa các fold có chồng lấn lịch sử. Đây là **retrospective diagnostic**, không phải pristine holdout hoặc một phép chọn model không thiên lệch. Không rank theo evaluation rồi công bố winner cho production.

Không kết luận drift hoặc catastrophic forgetting chỉ từ khác biệt các fold. Mức biến động có thể liên quan regime/data/model; study này chưa cô lập nguyên nhân. Pipeline train batch từ đầu, không thực hiện continual fine-tuning. Tối ưu tiếp cần calibration, nhiều bằng chứng ngoài mẫu và shadow forecasts trên dữ liệu tương lai.

Metadata chỉ kiểm chứng nội bộ snapshot: finite features, unique/sorted dates, nhãn được tính độc lập lại từ source closes; không chứng nhận calendar/point-in-time. Mọi pointer serving và reference giữ nguyên trong quá trình chạy.

## Bằng chứng và tái tạo

- [Protocol đã đăng ký](protocol.json), [protocol bytes gốc](protocol.original.json.gz), [source snapshot chính xác](source_snapshot.zip).
- [Data quality](data_quality.json), [split dates/counts](splits.csv), [từng candidate/phase](metrics.csv).
- [Prediction theo origin](predictions.csv.gz), [seed stability](seed_stability.csv), [fold summary](fold_summary.csv).
- [Manifest và hashes](manifest.json), [learning histories](lstm_history.json), [tính độc lập lại metrics](verification.json).

```powershell
python -m src.models.robustness --config configs/robustness_v1.json
python -m src.models.robustness_report
```

Mỗi lần train tạo thư mục mới và lưu đầy đủ kết quả, kể cả kém baseline. Trọng số nghiên cứu nằm ở `{manifest['artifact_dir']}`; không ghép với scaler/weights của model demo. Chỉ số deployment của ngày 25/10/2026 nằm trong `docs/TRAINING_HISTORY.md`, không so trực tiếp mean-fold RMSE ở đây với một test-year RMSE 2025.
"""
    if (study / "reload_verification.json").exists():
        reloaded = json.loads((study / "reload_verification.json").read_text(encoding="utf-8"))
        report += f"\n[Nạp lại artifact](reload_verification.json): {reloaded['candidates_reloaded']} model, {reloaded['phase_checks']} kiểm tra phase; scaler LSTM khớp thống kê train-only.\n"
    (study / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    studies = []
    for path in sorted((root / "reports/robustness").glob("*/manifest.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        studies.append({"study_id": item["study_id"], "report": str(path.with_name("REPORT.md").relative_to(root)).replace("\\", "/"),
                        "automatic_promotion": item["automatic_promotion"], "promotion_decision": item["promotion_decision"]})
    write_json(root / "reports/robustness/studies.json", studies)
    print(f"Verified {len(checked)} metric rows; report: {study / 'REPORT.md'}")
    return study / "REPORT.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--reload-artifacts", action="store_true")
    args = parser.parse_args()
    study = args.study
    if study is None:
        paths = sorted((BASE_DIR / "reports/robustness").glob("*/status.json"))
        complete = [p.parent for p in paths if json.loads(p.read_text())["status"] == "complete"]
        if not complete:
            raise ValueError("No complete robustness study found.")
        study = complete[-1]
    verify_and_report(study)
    if args.reload_artifacts:
        verify_reloaded_models(study)
