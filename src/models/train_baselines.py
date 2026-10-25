"""Compatibility entry point for the shared FPT v2 baseline experiment.

Run python -m src.models.experiment for baseline + LSTM experiments.
No second implementation of splitting, metrics or artifact loading is maintained here.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(BASE_DIR))

from src.models.experiment import metrics, run_experiment


def evaluate_predictions(y_true, y_pred):
    score = metrics(y_true, y_pred)
    return score["hit_rate_pct"], score["rmse"], score["mae"]


def calculate_hit_rate(y_true, y_pred):
    value = metrics(y_true, y_pred)["hit_rate_pct"]
    return None if value is None else value / 100.0


def run_benchmark():
    return run_experiment(BASE_DIR, include_lstm=False)


if __name__ == "__main__":
    run_benchmark()
