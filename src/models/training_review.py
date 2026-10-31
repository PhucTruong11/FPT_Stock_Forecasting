"""Render training-history figures from saved evidence; never train or overwrite studies."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data.contracts import BASE_DIR, file_sha256
from src.models.local_demo import RECENCY, PERIOD, save_json


def build(root=BASE_DIR):
    root = Path(root)
    out = root / "docs/assets/training"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#fafafa", "axes.facecolor": "#fafafa"})
    metrics_path = root / RECENCY / "metrics.csv"
    table = pd.read_csv(metrics_path)
    names = ["frozen_2023", "refit_2025_native", "refit_2025_native_bias60", "monthly_refit_native", "Zero-Return"]
    data = table.loc[table.cohort.eq("vci_full")].set_index("strategy").loc[names]
    labels = ["Ridge train ≤2023", "Ridge train ≤2025", "Ridge ≤2025 + bias", "Ridge monthly refit", "Zero Return"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    axes[0].barh(labels, data.rmse, color=["#94a3b8", "#2563eb", "#0d9488", "#7c3aed", "#64748b"])
    axes[0].invert_yaxis(); axes[0].set_xlabel("RMSE log return · lower is better")
    axes[0].set_xlim(0, .023)
    for i, value in enumerate(data.rmse): axes[0].text(value + .00015, i, f"{value:.6f}", va="center", fontsize=9)
    direction = data.balanced_accuracy_pct
    axes[1].barh(labels[:-1], direction.iloc[:-1], color=["#94a3b8", "#2563eb", "#0d9488", "#7c3aed"])
    axes[1].axvline(50, color="#b45309", ls="--", label="Always down: 50% balanced")
    axes[1].invert_yaxis(); axes[1].set_xlim(0, 70); axes[1].set_xlabel("Balanced accuracy (%) · higher is better")
    for i, value in enumerate(direction.iloc[:-1]): axes[1].text(value + .6, i, f"{value:.2f}%", va="center")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("2026 retrospective · same 202 pairs through 29 Oct (197 nonzero targets)")
    fig.savefig(out / "01_recency_bias.png", dpi=170); plt.close(fig)

    candle_path = root / "reports/candle_ablation/20261030T093521684942Z/metrics.csv"
    candle = pd.read_csv(candle_path)
    ridge = candle.loc[candle.strategy.eq("Ridge_recent")].pivot(index="year", columns="case", values="return_rmse")
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    x = np.arange(len(ridge))
    ax.bar(x - .18, ridge.A_returns6, .36, label="A · 6 return features", color="#2563eb")
    ax.bar(x + .18, ridge.B_gap_body7, .36, label="B · gap/body, 7 features", color="#0d9488", hatch="//")
    ax.set_xticks(x, ridge.index); ax.set_ylabel("RMSE log return · lower is better")
    ax.set_title("Gap/body ablation · same dates within each year; years are different populations")
    ax.legend(); fig.savefig(out / "02_gap_body.png", dpi=170); plt.close(fig)

    predictions_path = root / PERIOD / "predictions.csv.gz"
    p = pd.read_csv(predictions_path)
    p["Target_Date_1d"] = pd.to_datetime(p.Target_Date_1d)
    fig, ax = plt.subplots(figsize=(12, 4.8), layout="constrained")
    actual = p.drop_duplicates("Target_Date_1d").sort_values("Target_Date_1d")
    ax.plot(actual.Target_Date_1d, actual.actual_close_adjusted, color="#111827", lw=2, label="Actual adjusted close")
    for name, label, color in [("LSTM_serving", "LSTM reference", "#7c3aed"),
                                ("refit_2025_native_bias60", "Ridge ≤2025 + causal bias", "#0d9488")]:
        part = p.loc[p.strategy.eq(name)].sort_values("Target_Date_1d")
        ax.plot(part.Target_Date_1d, part.predicted_close_adjusted, label=label, color=color, lw=1.2, ls="--")
    ax.set_ylabel("Provider-adjusted close (VND)")
    ax.set_title("Jun–Oct 2026 · 104 common target dates · one-step historical forecasts")
    ax.legend(); fig.savefig(out / "03_actual_predictions.png", dpi=170); plt.close(fig)

    stages = [
        ("25/10/2026", "Legacy → v2", "Sửa nhãn/split/scaler; 4 run CPU/Colab", "LSTM30 tái lập, không phải accuracy tăng dần", "reports/training_history/runs.json"),
        ("29/10/2026", "Robustness", "63 candidate; feature/seed/fold controls", "Chưa vượt baseline ổn định", "reports/robustness/20261029T131205702813Z/verification.json"),
        ("29/10/2026", "Đối chiếu nguồn", "Audit Yahoo/Vietcap/KBS; 112 candidate", "Vietcap làm ứng viên nguồn nghiên cứu", "reports/source_comparison/20261029T142732805710Z/verification.json"),
        ("29/10/2026", "Training window", "56 fit mới / 112 điều kiện", "Ưu tiên kiểm tra Ridge returns-only", "reports/window_comparison/20261029T150501819490Z/verification.json"),
        ("30/10/2026", "Ridge refinement", "48 fit mới / 80 điều kiện; giữ lượt export lỗi", "Window 3y, alpha100 cho fold mới nhất", "reports/ridge_refinement/20261030T011109863274Z/verification.json"),
        ("30/10/2026", "Frozen 2026", "0 fit; 200 cặp chung / 202 native", "LSTM/Ridge chưa vượt Zero Return", "reports/evaluation_2026/20261030T014012158951Z/verification.json"),
        ("30/10/2026", "Recency + bias", "13 fit mới; cùng 202 cặp", "Refit cải thiện so Ridge cũ; edge vs zero rất nhỏ", RECENCY + "/verification.json"),
        ("30/10/2026", "Phân loại/selective", "20 base + 10 calibrator", "Chưa thay regression; all-down sau calibration", "reports/direction_selective/20261030T090621177786Z/verification.json"),
        ("30/10/2026", "Gap/thân nến", "20 fit B + 20 artifact A tái dùng", "Chưa cải thiện ổn định; giữ A 6 feature", "reports/candle_ablation/20261030T093521684942Z/verification.json"),
        ("30/10/2026", "API/UI local", "0 fit; đóng gói Ridge đến 29/10", "User chọn hướng chính; chưa chứng minh forward edge", "models/local_demo/ridge6_20261029/manifest.json"),
        ("30/10/2026", "Forecast Studio", "0 fit; 4 mục UI, ngày/tuần/tháng recursive", "h=1 đã train; nhiều bước chưa kiểm định riêng", "ARCHITECTURE.md#forecast-studio")]
    save_json(root / "reports/training_history/study_timeline.json", [dict(zip(
        ["Ngày", "Giai đoạn", "Công việc", "Kết luận", "Evidence"], row)) for row in stages])
    save_json(out / "sources.json", {"inputs": {p.relative_to(root).as_posix(): file_sha256(p) for p in [metrics_path, candle_path, predictions_path]},
        "figures": {p.name: file_sha256(p) for p in out.glob("*.png")}, "training_fits": 0,
        "note": "Different research stages are not comparable points on an accuracy-over-time curve."})


if __name__ == "__main__":
    build()
