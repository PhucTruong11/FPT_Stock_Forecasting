"""Verify saved source contrasts, reload artifacts and write the research record."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.research_notes import write_note
from src.data.contracts import BASE_DIR, file_sha256
from src.data.source_audit import guarded_hashes, save_json
from src.features.build_features import TARGET_COLUMN, TARGET_DATE_COLUMN
from src.models.robustness import fold_masks
from src.models.source_comparison import REFERENCE, SOURCE_KEYS, context_positions, load_inputs, source_part


def verify(study: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    protocol = json.loads((study / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((study / "manifest.json").read_text(encoding="utf-8"))
    status = json.loads((study / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete" or manifest["automatic_promotion"] is not False:
        raise ValueError("Only completed research studies may be reported.")
    if file_sha256(study / "protocol.json") != manifest["protocol_sha256"]:
        raise ValueError("Preregistered protocol changed.")
    with zipfile.ZipFile(study / "source_snapshot.zip") as archive:
        for name, sha in protocol["source_sha256"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != sha:
                raise ValueError(f"Archived source changed: {name}")
        if hashlib.sha256(archive.read("config.json")).hexdigest() != protocol["config_sha256"]:
            raise ValueError("Archived config changed.")
    datasets, cohort = load_inputs(study)
    recorded = pd.read_csv(study / "cohort.csv", parse_dates=["Date", TARGET_DATE_COLUMN])
    pd.testing.assert_frame_equal(cohort, recorded, check_exact=False, atol=1e-12, rtol=1e-12)
    config = protocol["config"]
    expected_models = {(fold["id"], source, group, model, seed)
        for fold in config["folds"] for source in SOURCE_KEYS for group in config["feature_groups"]
        for model in config["models"] for seed in ([None] if model == "Ridge" else config["seeds"])}
    actual_models = {(x["fold"], x["source"], x["feature_group"], x["model"], x["seed"]) for x in manifest["candidates"]}
    if actual_models != expected_models or len(manifest["candidates"]) != len(expected_models):
        raise ValueError("Missing, duplicate or unexpected model candidate.")
    # Recompute native labels directly from the archived raw prices, not the feature builder.
    with zipfile.ZipFile(study / "input_snapshot.zip") as archive:
        for source, (_, labels) in datasets.items():
            raw = pd.read_csv(io.BytesIO(archive.read(f"{source}/raw.csv")), parse_dates=["Date"]).set_index("Date")
            expected = np.log(raw.Close.reindex(labels[TARGET_DATE_COLUMN]).to_numpy() / raw.Close.reindex(labels.Date).to_numpy())
            if not np.allclose(expected, labels[TARGET_COLUMN], atol=1e-12, rtol=1e-12):
                raise ValueError("Native labels do not match raw source prices.")
    scores = pd.read_csv(study / "metrics.csv")
    predictions = pd.read_csv(study / "predictions.csv.gz", parse_dates=["Date", TARGET_DATE_COLUMN])
    folds = {f["id"]: f for f in config["folds"]}
    checks, max_error = [], 0.
    for (candidate, phase), part in predictions.groupby(["candidate", "phase"], sort=False):
        identity = part.iloc[0]
        mask = fold_masks(cohort, folds[identity.fold])[phase]
        expected = cohort.loc[mask].reset_index(drop=True)
        if not np.array_equal(part.Date.to_numpy(), expected.Date.to_numpy()) or not np.array_equal(part[TARGET_DATE_COLUMN].to_numpy(), expected[TARGET_DATE_COLUMN].to_numpy()):
            raise ValueError("Forecast origin/target cohort differs across candidates.")
        truth, pred = part[REFERENCE].to_numpy(), part.predicted_log_return.to_numpy()
        if not np.isfinite(pred).all() or not np.allclose(truth, expected[REFERENCE], atol=1e-12, rtol=1e-12):
            raise ValueError("Predictions do not share the fixed reference labels.")
        original = scores.loc[scores.candidate.eq(candidate) & scores.phase.eq(phase)]
        if len(original) != 1:
            raise ValueError("Metric identity is not unique.")
        row = original.iloc[0]
        rmse, mae = float(np.sqrt(np.mean(np.square(truth - pred)))), float(np.mean(np.abs(truth - pred)))
        errors = [abs(rmse - row.rmse), abs(mae - row.mae)]
        if max(errors) > 1e-12:
            raise ValueError("Saved metrics differ from independent recomputation.")
        nonzero = truth != 0
        if nonzero.any() and (pred[nonzero] != 0).any():
            hit = float(100 * np.mean(np.sign(truth[nonzero]) == np.sign(pred[nonzero])))
            if not np.isclose(hit, row.hit_rate_pct, atol=1e-10):
                raise ValueError("Directional denominator differs.")
        source = identity.source if identity.source != "shared" else config["reference_source"]
        if not np.allclose(part[TARGET_COLUMN], expected[f"return_{source}"], atol=1e-12, rtol=1e-12):
            raise ValueError("Exported native labels do not match source.")
        if identity.feature_group == "baseline":
            native_train = cohort.loc[fold_masks(cohort, folds[identity.fold])["train"], f"return_{source}"]
            if identity.model == "Zero-Return":
                baseline = np.zeros(len(part))
            elif identity.model == "Train Mean":
                baseline = np.full(len(part), native_train.mean())
            else:
                native = source_part(expected, source, datasets[source][0])
                baseline = native.Log_Return.to_numpy()
            if not np.allclose(pred, baseline, atol=1e-12, rtol=1e-12):
                raise ValueError("Baseline differs from its registered training-only rule.")
        checks.append({"candidate": candidate, "phase": phase, "n": len(part), "verified": True})
        max_error = max(max_error, *errors)
    if len(checks) != len(scores) or len(scores) != 2 * len(expected_models) + 10 * len(config["folds"]):
        raise ValueError("Incomplete prediction/metric export.")
    if protocol["protected_before"] != manifest["protected_after"]:
        raise ValueError("Serving/data guards changed during training.")
    if all((BASE_DIR / n).exists() for n in protocol["protected_before"]):
        if guarded_hashes() != protocol["protected_before"]:
            raise ValueError("Current canonical data/model differ from registered guards.")
    save_json(study / "verification.json", {"protocol_and_input_and_source_hashes_verified": True,
        "trained_candidates": len(expected_models), "metric_rows_verified": len(checks),
        "same_origin_target_and_reference_for_every_candidate": True,
        "raw_native_labels_recomputed": True, "baselines_train_only_verified": True,
        "maximum_metric_recomputation_error": max_error, "protected_files_unchanged": True,
        "checks": checks})
    return protocol, scores, predictions


def verify_reloaded(study: Path, protocol: dict, predictions: pd.DataFrame) -> None:
    import joblib
    import tensorflow as tf
    from xgboost import XGBRegressor
    from src.models.experiment import make_sequences
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    datasets, cohort = load_inputs(study)
    config = protocol["config"]
    manifest = json.loads((study / "manifest.json").read_text(encoding="utf-8"))
    artifacts = BASE_DIR / manifest["artifact_dir"]
    for name, sha in manifest["artifact_sha256"].items():
        if file_sha256(artifacts / name) != sha:
            raise ValueError(f"Model/scaler bytes changed: {name}")
    folds = {f["id"]: f for f in config["folds"]}
    exported = {(candidate, phase): part for (candidate, phase), part in predictions.groupby(["candidate", "phase"])}
    checks = []
    for index, candidate in enumerate(manifest["candidates"], 1):
        source, columns = candidate["source"], candidate["feature_columns"]
        features = datasets[source][0]
        masks = fold_masks(cohort, folds[candidate["fold"]])
        train = source_part(cohort.loc[masks["train"]].reset_index(drop=True), source, features)
        if candidate["model"] == "LSTM":
            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(artifacts / candidate["file"], compile=False)
            scalers = joblib.load(artifacts / candidate["scalers"])
            context = context_positions(train.Feature_Position.to_numpy(), config["lookback"])
            x = features.iloc[context][columns].to_numpy()
            y = train[TARGET_COLUMN].to_numpy()
            if not np.array_equal(context, scalers["context_positions"]):
                raise ValueError("Training scaler context changed.")
            for scaler, samples in [(scalers["features"], x), (scalers["target"], y.reshape(-1, 1))]:
                if int(scaler.n_samples_seen_) != len(samples) or not np.allclose(scaler.mean_, samples.mean(axis=0), atol=1e-12):
                    raise ValueError("Scaler was not fitted on registered training-only inputs.")
                if not np.allclose(scaler.var_, samples.var(axis=0), atol=1e-12):
                    raise ValueError("Scaler variance differs from training-only statistics.")
            values = scalers["features"].transform(features[columns])
        elif candidate["model"] == "XGBoost":
            model = XGBRegressor()
            model.load_model(artifacts / candidate["file"])
        else:
            model = joblib.load(artifacts / candidate["file"])
            if not np.allclose(model.steps[0][1].mean_, train[columns].mean().to_numpy(), atol=1e-12):
                raise ValueError("Ridge scaler is not train-only.")
        for phase in ["validation", "evaluation"]:
            part = source_part(cohort.loc[masks[phase]].reset_index(drop=True), source, features)
            reference = exported[(candidate["candidate"], phase)]
            if not np.array_equal(part.Date.to_numpy(), reference.Date.to_numpy()):
                raise ValueError("Reload cohort differs from prediction export.")
            if candidate["model"] == "LSTM":
                tensor = make_sequences(values, part.Feature_Position.to_numpy(), config["lookback"])
                pred = scalers["target"].inverse_transform(model(tensor, training=False).numpy()).ravel()
            else:
                pred = model.predict(part[columns])
            if not np.allclose(pred, reference.predicted_log_return, rtol=1e-5, atol=1e-8):
                raise ValueError(f"Reloaded predictions differ: {candidate['candidate']}")
            checks.append({"candidate": candidate["candidate"], "phase": phase,
                           "max_absolute_difference": float(np.max(np.abs(pred - reference.predicted_log_return))), "verified": True})
        if index % 14 == 0:
            print(f"Reload verified {index}/{len(manifest['candidates'])}", flush=True)
    save_json(study / "reload_verification.json", {"candidates_reloaded": len(manifest["candidates"]),
        "phase_checks": len(checks), "scalers_match_train_only_statistics": True,
        "atol": 1e-8, "rtol": 1e-5, "checks": checks})


def aggregate(study: Path, protocol: dict, scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["fold", "source", "feature_group", "model", "phase"]
    stability = scores.groupby(keys, dropna=False).agg(runs=("rmse", "size"), n=("n", "first"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), mae_mean=("mae", "mean"),
        hit_rate_mean=("hit_rate_pct", "mean"), balanced_accuracy_mean=("balanced_accuracy_pct", "mean"),
        mse_skill_vs_zero_mean=("mse_skill_vs_zero", "mean")).reset_index()
    stability.to_csv(study / "seed_stability.csv", index=False, lineterminator="\n")
    pair_keys = ["fold", "feature_group", "model", "phase"]
    left, right = [stability.loc[stability.source.eq(s)].drop(columns="source") for s in SOURCE_KEYS]
    paired = left.merge(right, on=pair_keys, suffixes=("_yahoo", "_vietcap"), validate="one_to_one")
    paired["rmse_improvement_pct"] = 100 * (1 - paired.rmse_mean_vietcap / paired.rmse_mean_yahoo)
    paired["mae_improvement_pct"] = 100 * (1 - paired.mae_mean_vietcap / paired.mae_mean_yahoo)
    paired.to_csv(study / "paired_fold_comparison.csv", index=False, lineterminator="\n")
    seed_keys = pair_keys + ["seed"]
    left, right = [scores.loc[scores.source.eq(s)].drop(columns="source") for s in SOURCE_KEYS]
    seed_pairs = left.merge(right, on=seed_keys, suffixes=("_yahoo", "_vietcap"), validate="one_to_one")
    seed_pairs["rmse_improvement_pct"] = 100 * (1 - seed_pairs.rmse_vietcap / seed_pairs.rmse_yahoo)
    seed_pairs.to_csv(study / "paired_seed_comparison.csv", index=False, lineterminator="\n")
    primary = {f["id"] for f in protocol["config"]["folds"] if f["role"] == "primary_retrospective"}
    evaluation = stability.loc[stability.phase.eq("evaluation") & stability.fold.isin(primary)]
    overall = evaluation.groupby(["source", "feature_group", "model"]).agg(
        folds=("fold", "size"), mean_fold_rmse=("rmse_mean", "mean"), worst_fold_rmse=("rmse_mean", "max"),
        mean_fold_mae=("mae_mean", "mean"), mean_seed_std=("rmse_std", "mean"),
        positive_skill_folds=("mse_skill_vs_zero_mean", lambda x: int((x > 0).sum()))).reset_index()
    overall.to_csv(study / "primary_summary.csv", index=False, lineterminator="\n")
    return stability, paired


def write_report(study: Path, protocol: dict, stability: pd.DataFrame, paired: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    config = protocol["config"]
    overall = pd.read_csv(study / "primary_summary.csv")
    learned = overall.loc[overall.feature_group.ne("baseline")]
    table = ["| Model / features | Yahoo RMSE | Vietcap RMSE | Thay đổi RMSE | Seed SD Yahoo / Vietcap |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for (group, model), part in learned.groupby(["feature_group", "model"]):
        y, v = [part.loc[part.source.eq(s)].iloc[0] for s in SOURCE_KEYS]
        gain = 100 * (1 - v.mean_fold_rmse / y.mean_fold_rmse)
        sd = "N/A" if pd.isna(y.mean_seed_std) else f"{y.mean_seed_std:.7f} / {v.mean_seed_std:.7f}"
        table.append(f"| {model} / {group} | {y.mean_fold_rmse:.7f} | {v.mean_fold_rmse:.7f} | {gain:+.2f}% | {sd} |")
    zero = overall.loc[overall.model.eq("Zero-Return")].iloc[0]
    baseline = overall.loc[overall.model.eq("Train Mean") & overall.source.eq("vietcap_v3_candidate")].iloc[0]
    stress = paired.loc[paired.fold.eq("eval_2025") & paired.phase.eq("evaluation")]
    stress_table = ["| Model / features | Yahoo RMSE | Vietcap RMSE | Thay đổi RMSE |", "| --- | ---: | ---: | ---: |"]
    for _, row in stress.loc[stress.feature_group.ne("baseline")].iterrows():
        stress_table.append(f"| {row.model} / {row.feature_group} | {row.rmse_mean_yahoo:.7f} | {row.rmse_mean_vietcap:.7f} | {row.rmse_improvement_pct:+.2f}% |")
    splits = pd.read_csv(study / "splits.csv")
    counts = splits.groupby(["fold", "phase"]).size().unstack()
    count_table = ["| Fold | Train mỗi nguồn | Validation | Evaluation |", "| --- | ---: | ---: | ---: |"]
    for fold, row in counts.iterrows():
        count_table.append(f"| {fold} | {row['train']} | {row['validation']} | {row['evaluation']} |")
    cohort = pd.read_csv(study / "cohort.csv")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    folds = [f["id"] for f in config["folds"]]
    for row_index, group in enumerate(config["feature_groups"]):
        for col_index, model in enumerate(["Ridge", "XGBoost", "LSTM"]):
            ax = axes[row_index, col_index]
            for source, label in [("yahoo_v2", "Yahoo v2"), ("vietcap_v3_candidate", "Vietcap candidate")]:
                part = stability.loc[stability.source.eq(source) & stability.feature_group.eq(group) & stability.model.eq(model) & stability.phase.eq("evaluation")].set_index("fold").reindex(folds)
                ax.errorbar(range(len(folds)), part.rmse_mean, yerr=part.rmse_std.fillna(0), marker="o", capsize=3, label=label)
            control = stability.loc[stability.model.eq("Zero-Return") & stability.phase.eq("evaluation")].set_index("fold").reindex(folds)
            ax.plot(range(len(folds)), control.rmse_mean, "k--", linewidth=1, label="Zero return")
            ax.axvspan(2.5, 3.5, color="grey", alpha=.12)
            ax.set_title(f"{model} / {group}")
            ax.set_xticks(range(len(folds)), [f[-4:] for f in folds])
            ax.set_ylabel("RMSE (reference log-return)")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("FPT source comparison: same cohorts and Vietcap reference labels\nBars = seed SD; shaded 2025 = previously inspected diagnostic", fontsize=12)
    fig.tight_layout()
    fig.savefig(study / "source_comparison.png", dpi=170)
    fig.savefig(study / "source_comparison.svg")
    plt.close(fig)
    reload_path = study / "reload_verification.json"
    reload_text = "Chưa kiểm tra nạp lại model." if not reload_path.exists() else "Đã nạp lại toàn bộ 112 model, kiểm tra 224 bộ prediction validation/evaluation và thống kê scaler train-only."
    text = f"""# Đối chứng nguồn dữ liệu FPT — {study.name}

Đây là nghiên cứu hồi cứu để đo tác động đổi nguồn và feature, **không phải chứng minh accuracy chắc chắn tăng**. Snapshot/model serving v2 được giữ nguyên; không tải hoặc đánh giá giá 2026, không tự promote.

## 1. Thiết kế khóa trước training

- Hai nguồn: Yahoo v2 đã khóa và Vietcap ứng viên v3 từ [audit](../../data_audits/{config['audit_id']}/REPORT.md). Raw và toàn bộ features/labels được khóa trong [input_snapshot.zip](input_snapshot.zip), không tải lại mạng trong training.
- Có **{len(cohort)} origin/target pairs chung**, cùng yêu cầu 60 quan sát context. Cả train, validation, evaluation dùng cùng origin **và** target date cho hai nguồn. Không dùng số mẫu có nhãn thêm của Vietcap làm lợi thế về kích thước tập train.
- Nhãn train là nhãn native mỗi nguồn. Validation/early stopping và evaluation dùng **cùng một nhãn Vietcap cố định**. Vì vậy mọi RMSE/MAE trong nghiên cứu so được trên cùng mục tiêu; không so trực tiếp với RMSE của nghiên cứu cũ chấm theo nhãn Yahoo.
- Native context dài 30 quan sát có thể bao phủ các ngày lịch khác nhau giữa hai nguồn; một số dòng chỉ nguồn v3 có vẫn có thể vào rolling features/context. Tác động nguồn bao gồm chất lượng giá/nhãn và độ đầy đủ lịch sử context, không chỉ một hệ số đổi giá. Chưa thử lợi ích tăng thêm labeled training rows của v3.
- Ridge alpha 10, XGBoost depth 2 / 150 cây, LSTM 32 units / Dense 8 / dropout 0,2, lookback 30. Seed 42, 123, 2026 cho XGBoost/LSTM; Ridge xác định chạy một lần. Hai nhóm returns (6) và full (16). Tổng **112 model**; cấu hình đầy đủ trong [protocol.json](protocol.json), source chính xác trong [source_snapshot.zip](source_snapshot.zip).
- LSTM feature scaler chỉ fit hợp các vị trí context thực sự được train dùng, không qua cutoff; target scaler fit native train labels. Ridge scaler chỉ fit vector train. Val labels chung được transform bằng target scaler của train; không fit scaler lên validation.
- Nến Yahoo lỗi và các dòng zero-volume bị loại theo cùng feature builder. Cờ Vietcap/KBS 14/03/2019 được giữ nguyên, không xóa sau khi xem điểm. Corporate actions, lịch HOSE và tính point-in-time chưa được chứng nhận; Vietcap là **nhãn tham chiếu**, không được coi là ground truth của sàn.

{chr(10).join(count_table)}

## 2. So sánh chính: evaluation 2022–2024

Tính mean RMSE qua seed trong từng fold rồi lấy trung bình không trọng số ba fold. Dấu **+** ở cột thay đổi = RMSE Vietcap giảm so với Yahoo, dấu **−** = tăng. Seed SD chỉ phản ánh biến động khi train, không phải confidence interval của lợi thế giao dịch. Ridge không có seed SD vì chỉ chạy một lần.

{chr(10).join(table)}

Baseline trên cùng nhãn/cohort: Zero Return mean-fold RMSE **{zero.mean_fold_rmse:.7f}**; Train Mean từ native Vietcap train **{baseline.mean_fold_rmse:.7f}**. Xem [primary_summary.csv](primary_summary.csv) để biết worst fold và số fold có MSE skill dương; cải thiện so với Yahoo chưa có nghĩa vượt baseline.

## 3. Stress diagnostic 2025 — báo cáo riêng

Năm 2025 đã được nhìn trong phát triển và audit đã phát hiện nhiều bất đồng Yahoo. Đây là kiểm tra chẩn đoán đã biết vấn đề, không phải holdout mới. Không gộp 2025 vào bảng trung bình chính hoặc chọn model phục vụ từ điểm này.

{chr(10).join(stress_table)}

![RMSE theo fold và nguồn](source_comparison.png)

## 4. Kiểm chứng và giới hạn

Mọi candidate được kiểm tra cùng ngày origin/target và cùng reference labels; RMSE/MAE/hit rate được tính lại từ prediction export. Baseline Train Mean chỉ dùng nhãn train native; Return Persistence dùng Log_Return tại origin của nguồn tương ứng. Zero Return không phát hướng nên hit rate N/A. Cần đọc balanced accuracy và tỷ lệ lớp tăng/giảm trước khi diễn giải hit rate.

{reload_text} Xem [verification.json](verification.json), [reload_verification.json](reload_verification.json). Hash 8 file canonical data/model pointers trước và sau trùng nhau.

Các fold dùng kiến trúc/features đã được phát triển trước, có chồng lấn lịch sử train/validation/evaluation giữa các năm. Không dùng seed như các thị trường độc lập, không diễn giải seed SD thành bằng chứng thống kê ngoài mẫu. Chưa thử phí giao dịch, corporate actions point-in-time, drift live hoặc forward holdout mới. Chưa thay model hoặc API/Streamlit.

## 5. Bằng chứng để team đối chiếu

- [metrics.csv](metrics.csv), [predictions.csv.gz](predictions.csv.gz): toàn bộ kết quả và từng forecast, gồm reference label lẫn native label.
- [paired_fold_comparison.csv](paired_fold_comparison.csv), [paired_seed_comparison.csv](paired_seed_comparison.csv): đối chiếu từng fold/seed; không bỏ lượt kém.
- [seed_stability.csv](seed_stability.csv), [splits.csv](splits.csv), [cohort.csv](cohort.csv), [data_quality.json](data_quality.json).
- [lstm_history.json](lstm_history.json), [manifest.json](manifest.json): epochs, artifact hashes, runtime, cấu hình. Weights nghiên cứu lưu local dưới models/research/, không đè serving weights.

```powershell
python -m src.models.source_comparison --config configs/source_comparison_v1.json
python -m src.models.source_comparison_report --study reports/source_comparison/{study.name} --reload-artifacts
```

Lệnh train mới tạo ID riêng. Report/metrics có thể kiểm tra từ file Git; kiểm tra nạp lại cần weights local hoặc tái train. Mỗi lần thử tiếp phải giữ protocol, source snapshot, nhãn đánh giá và cohort để biết thay đổi đến từ đâu.
"""
    write_note(study / 'REPORT.md', text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--reload-artifacts", action="store_true")
    args = parser.parse_args()
    study = args.study.resolve()
    protocol, scores, predictions = verify(study)
    if args.reload_artifacts:
        verify_reloaded(study, protocol, predictions)
    stability, paired = aggregate(study, protocol, scores)
    write_report(study, protocol, stability, paired)
    print(f"Verified and reported: {study}", flush=True)
