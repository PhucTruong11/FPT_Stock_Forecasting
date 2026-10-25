"""Rebuild the training audit from saved manifests; never retrain or select on test."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.contracts import BASE_DIR


def collect_runs(root: Path = BASE_DIR) -> list[dict]:
    root = Path(root)
    paths = sorted((root / "reports/training_history/evidence").glob("*/manifest.json"))
    paths += sorted((root / "models/experiments").glob("*/manifest.json"))
    by_id = {}
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        run = manifest["run_id"]
        started = datetime.strptime(run, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
        by_id[run] = {
            "run_id": run,
            "started_utc": started.isoformat(),
            "started_bangkok": started.astimezone(timezone(timedelta(hours=7))).isoformat(),
            "manifest_path": path.relative_to(root).as_posix(),
            "manifest_sha256": hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "manifest_hash_basis": "UTF-8 with LF line endings, matching Git attributes",
            "source_code_sha256": manifest["source_code_sha256"],
            "raw_sha256": manifest["raw_sha256"],
            "data_sha256": manifest["data_sha256"],
            "training_config": manifest["training_config"],
            "splits": manifest["splits"], "seed": manifest["seed"],
            "runtime": manifest["runtime"], "python": manifest["python"],
            "versions": manifest["versions"],
            "selected_model": manifest["selection"]["selected_model"],
            "lstm_lookback": manifest["models"].get("LSTM", {}).get("lookback"),
            "xgboost_format": Path(manifest["models"]["XGBoost"]["file"]).suffix,
            "metrics": manifest["metrics"],
        }
    return [by_id[key] for key in sorted(by_id)]


def build_history(root: Path = BASE_DIR) -> dict:
    root = Path(root)
    out = root / "reports/training_history"
    out.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(root)
    if not runs:
        raise ValueError("No experiment manifests available.")
    pointer = json.loads((root / "models/reference_experiment.json").read_text(encoding="utf-8"))
    reference = next(run for run in runs if run["run_id"] == pointer["run_id"])
    rows, timeline = [], []
    for run in runs:
        for score in run["metrics"]:
            rows.append({"run_id": run["run_id"], "started_bangkok": run["started_bangkok"],
                         "device": run["runtime"]["device"], "xgboost_format": run["xgboost_format"],
                         "selected_model": run["selected_model"], "lstm_lookback": run["lstm_lookback"], **score})
        scores = {(s["model"], s["split"]): s for s in run["metrics"]}
        protocol_keys = ["raw_sha256", "data_sha256", "splits", "seed", "training_config"]
        timeline.append({"run_id": run["run_id"], "started_bangkok": run["started_bangkok"],
                         "device": run["runtime"]["device"], "xgboost_format": run["xgboost_format"],
                         "same_data_and_training_config_as_reference": all(run[k] == reference[k] for k in protocol_keys),
                         "same_source_as_reference": run["source_code_sha256"] == reference["source_code_sha256"],
                         "lstm_lookback": run["lstm_lookback"],
                         "validation_rmse": scores.get(("LSTM", "validation"), {}).get("rmse"),
                         "test_rmse": scores.get(("LSTM", "test"), {}).get("rmse"),
                         "test_mae": scores.get(("LSTM", "test"), {}).get("mae"),
                         "test_hit_rate_pct": scores.get(("LSTM", "test"), {}).get("hit_rate_pct"),
                         "manifest_path": run["manifest_path"]})
    result = {"reference_run": reference["run_id"], "runs": runs, "timeline": timeline,
              "scope": "Development runs on one fixed historical snapshot; not longitudinal market performance.",
              "last_run_started_at": runs[-1]["started_utc"]}
    (out / "runs.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "metrics_over_runs.csv", index=False, float_format="%.12g")
    pd.DataFrame(timeline).to_csv(out / "run_timeline.csv", index=False, float_format="%.12g")

    # Independently recompute the reference metrics from exported predictions.
    preds = pd.read_csv(root / pointer["report_dir"] / "predictions.csv")
    checks = []
    for (model, split), frame in preds.groupby(["model", "split"]):
        actual, predicted = frame["Target_Return_1d"].to_numpy(), frame["predicted_log_return"].to_numpy()
        rmse = float(np.sqrt(np.square(actual - predicted).mean()))
        mae = float(np.abs(actual - predicted).mean())
        nonzero = actual != 0
        issued = (predicted != 0) & nonzero
        hit = float(100 * ((np.sign(actual) == np.sign(predicted)) & nonzero).sum() / nonzero.sum()) if issued.any() else None
        recorded = next(s for s in reference["metrics"] if s["model"] == model and s["split"] == split)
        assert np.isclose(rmse, recorded["rmse"], atol=1e-9, rtol=1e-6)
        assert np.isclose(mae, recorded["mae"], atol=1e-9, rtol=1e-6)
        assert (hit is None and recorded["hit_rate_pct"] is None) or np.isclose(hit, recorded["hit_rate_pct"], atol=1e-7)
        checks.append({"model": model, "split": split, "rmse_recomputed": rmse,
                       "mae_recomputed": mae, "hit_rate_pct_recomputed": hit, "matches_manifest": True})
    (out / "calculation_checks.json").write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(runs)} runs and {len(rows)} model/split measurements to {out}")
    return result


if __name__ == "__main__":
    build_history()
