"""Compare a Colab run with the frozen CPU reference without selecting on test."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.contracts import BASE_DIR
from src.models.experiment import _predict, load_data, metrics, save_json, split_masks


def compare_runs(candidate_artifact_dir: Path, root: Path = BASE_DIR, rmse_rtol: float = 0.05) -> dict:
    root = Path(root)
    pointer = json.loads((root / "models/reference_experiment.json").read_text(encoding="utf-8"))
    reference = json.loads((root / pointer["artifact_dir"] / "manifest.json").read_text(encoding="utf-8"))
    candidate = json.loads((Path(candidate_artifact_dir) / "manifest.json").read_text(encoding="utf-8"))
    checks = {key: candidate[key] == reference[key] for key in (
        "data_sha256", "raw_sha256", "feature_columns", "splits", "seed", "training_config", "source_code_sha256")}
    protocol_match = all(checks.values())
    result = {
        "reference_run": reference["run_id"], "candidate_run": candidate["run_id"],
        "protocol_checks": checks, "protocol_match": protocol_match,
        "reference_runtime": reference["runtime"], "candidate_runtime": candidate["runtime"],
        "package_versions_match": candidate["versions"] == reference["versions"],
        "tensorflow_version_match": candidate["runtime"]["tensorflow"] == reference["runtime"]["tensorflow"],
        "python_version_match": candidate["python"] == reference["python"],
        "same_overall_validation_winner": candidate["selection"]["selected_model"] == reference["selection"]["selected_model"],
        "reference_lstm_lookback": reference["models"].get("LSTM", {}).get("lookback"),
        "candidate_selected_lstm_lookback": candidate["models"].get("LSTM", {}).get("lookback"),
        "rmse_relative_tolerance": rmse_rtol,
        "tolerance_meaning": "Engineering comparison only; not statistical equivalence or evidence of trading profit.",
        "metrics": [],
    }
    if not protocol_match:
        result["status"] = "protocol_mismatch"
        return result
    features, labels, _ = load_data(root)
    reference_rows = {(r["model"], r["split"]): r for r in reference["metrics"]}
    baseline_match = True
    for split, mask in split_masks(labels).items():
        if split == "train":
            continue
        frame = labels.loc[mask]
        for name in reference["models"]:
            if name == "LSTM":
                # Compare the SAME lookback as the CPU reference even if the GPU
                # validation ranking changed. All candidate lookbacks were saved.
                spec = reference["models"][name]
            else:
                spec = candidate["models"][name]
            pred = _predict(spec, features, frame, Path(candidate_artifact_dir))
            actual = metrics(frame["Target_Return_1d"].to_numpy(), pred)
            expected = reference_rows[(name, split)]
            relative = abs(actual["rmse"] - expected["rmse"]) / expected["rmse"]
            exact_enough = bool(np.isclose(actual["rmse"], expected["rmse"], rtol=1e-6, atol=1e-9))
            if name != "LSTM":
                baseline_match = baseline_match and exact_enough
            result["metrics"].append({"model": name, "split": split,
                                      "reference_rmse": expected["rmse"], "candidate_rmse": actual["rmse"],
                                      "rmse_relative_difference": relative,
                                      "within_tolerance": relative <= rmse_rtol,
                                      "candidate_hit_rate_pct": actual["hit_rate_pct"]})
    result["deterministic_baselines_match"] = baseline_match
    result["all_rmse_within_tolerance"] = all(r["within_tolerance"] for r in result["metrics"])
    result["status"] = "protocol_match_numeric_close" if baseline_match and result["all_rmse_within_tolerance"] else "protocol_match_numeric_difference"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "reports/colab_comparison.json")
    args = parser.parse_args()
    candidate = args.candidate
    if candidate is None:
        latest = json.loads((BASE_DIR / "models/latest_experiment.json").read_text())
        candidate = BASE_DIR / latest["artifact_dir"]
    compared = compare_runs(candidate)
    save_json(args.output, compared)
    print(json.dumps(compared, indent=2))
