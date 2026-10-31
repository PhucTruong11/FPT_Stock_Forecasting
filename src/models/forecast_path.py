"""Bounded recursive Ridge scenarios from close history, not synthetic OHLCV bars.

Only h=1 was trained. Week/month output is an unvalidated recursive scenario.
Weekday dates are estimates, not a certified HOSE calendar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.contracts import utc_now
from src.models.local_demo import FEATURES

MAX_STEPS = 23
PATHS = 512
SEED = 20261030


def local_today():
    return pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").tz_localize(None).normalize()


def target_window(origin, horizon, today):
    origin, today = pd.Timestamp(origin).normalize(), pd.Timestamp(today).normalize()
    if origin >= today:
        raise ValueError("Forecast origin must precede the requested local day.")
    if horizon not in {"today", "tomorrow", "week", "month"}:
        raise ValueError("Unknown forecast horizon.")
    start = today + pd.Timedelta(days=1) if horizon == "tomorrow" else today
    end = start
    if horizon == "week":
        end = today + pd.Timedelta(days=6)  # Inclusive seven calendar days.
    elif horizon == "month":
        end = today + pd.DateOffset(months=1) - pd.Timedelta(days=1)
    if horizon in {"today", "tomorrow"}:
        effective_end = pd.offsets.BDay().rollforward(end)
    else:
        effective_end = pd.offsets.BDay().rollback(end)
    dates = pd.bdate_range(origin + pd.Timedelta(days=1), effective_end)
    if not 1 <= len(dates) <= MAX_STEPS:
        raise ValueError(f"Snapshot too old for this horizon ({len(dates)} recursive steps; maximum {MAX_STEPS}). Update the input snapshot before forecasting.")
    return start, end, dates


def return_features(log_closes):
    """Exact Ridge6 semantics; one row per scenario, using the last 21 closes."""
    values = np.asarray(log_closes, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] < 21 or not np.isfinite(values).all():
        raise ValueError("Need at least 21 finite log closes.")
    changes = np.diff(values, axis=1)
    return pd.DataFrame(np.column_stack([changes[:, -1], changes[:, -2], changes[:, -3], changes[:, -6],
        values[:, -1] - values[:, -6], values[:, -1] - values[:, -21]]), columns=FEATURES)


def residual_blocks(residuals, steps):
    """Fixed-seed circular blocks of five; descriptive scenarios, not coverage proof."""
    residuals = np.asarray(residuals, dtype=float)
    if not len(residuals) or not np.isfinite(residuals).all():
        raise ValueError("Invalid residual calibration.")
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, len(residuals), size=(PATHS, (steps + 4) // 5))
    indices = (starts[:, :, None] + np.arange(5)[None, None, :]) % len(residuals)
    return residuals[indices.reshape(PATHS, -1)[:, :steps]]


def forecast(service, horizon="today", model_name=None, today=None):
    name = model_name or service.default_model
    # Apply the existing cutoff, last-bar, source and finite prediction checks first.
    first = service.predict(name)
    today = local_today() if today is None else pd.Timestamp(today).normalize()
    start, end, dates = target_window(first["as_of"], horizon, today)
    close = float(first["latest_close_adjusted"])
    history = np.log(service.features.Close.tail(21).to_numpy(float))
    if len(history) < 21:
        raise ValueError("Insufficient close history for recursive forecasting.")
    point_history = history[None, :].copy()
    sampled_history = np.tile(history, (PATHS, 1))
    bias = first["bias_offset"]
    errors = service.bundle["corrected_errors"] if name == "Ridge+bias60" else service.bundle["raw_errors"]
    noise = residual_blocks(errors[-60:], len(dates))
    rows = []
    for i, day in enumerate(dates):
        if name == "Zero-Return":
            point_return, next_samples = 0., sampled_history[:, -1]
        else:
            point_return = float(service.loaded.predict(return_features(point_history))[0]) + bias
            next_samples = (sampled_history[:, -1] +
                            service.loaded.predict(return_features(sampled_history)) + bias + noise[:, i])
        next_point = float(point_history[0, -1] + point_return)
        with np.errstate(over="ignore", invalid="ignore"):
            price = float(np.exp(next_point))
            samples = np.exp(next_samples)
        if not np.isfinite(price) or not np.isfinite(samples).all() or price <= 0:
            raise ValueError("Recursive scenario exceeds finite price bounds.")
        low, high = np.quantile(samples, [.1, .9]) if name != "Zero-Return" else (None, None)
        rows.append({"step": i + 1, "estimated_date": str(day.date()), "predicted_close_adjusted": price,
                     "step_log_return": point_return, "change_from_origin_pct": float((price / close - 1) * 100),
                     "scenario_lower": float(low) if low is not None else None,
                     "scenario_upper": float(high) if high is not None else None,
                     "in_requested_window": bool(day >= start), "uses_predicted_context": i > 0})
        point_history = np.column_stack([point_history[:, 1:], [next_point]])
        sampled_history = np.column_stack([sampled_history[:, 1:], next_samples])
    return {"ticker": "FPT.VN", "model": name, "model_version": first["model_version"],
        "horizon": horizon, "forecast_method_version": "recursive-ridge6-v1",
        "issued_at_utc": utc_now(), "local_request_date": str(today.date()),
        "origin_date": first["as_of"], "origin_close_adjusted": close,
        "requested_start": str(start.date()), "requested_end": str(end.date()),
        "effective_target_date": rows[-1]["estimated_date"],
        "weekend_shifted": horizon in {"today", "tomorrow"} and str(end.date()) != rows[-1]["estimated_date"],
        "steps": len(rows), "max_steps": MAX_STEPS, "path": rows,
        "trained_horizon_observations": 1, "multi_step_validated": False,
        "calendar_verified": False, "calendar_method": "weekdays_only_no_holiday_calendar",
        "role": "one_step_estimate" if len(rows) == 1 else "unvalidated_recursive_scenario",
        "same_day_is_pre_session_forecast": False, "forward_ledger_recorded": False,
        "price_basis": first["price_basis"], "source": first["source"],
        "training_target_end": first["training_target_end"], "calibration_policy": first["calibration_policy"],
        "uncertainty": {"method": "circular_block_residual_bootstrap", "block_observations": 5,
                        "paths": PATHS if name != "Zero-Return" else 0, "seed": SEED,
                        "displayed_quantiles": [.1, .9], "coverage_guaranteed": False,
                        "meaning": "Descriptive scenario range; ignores parameter/source uncertainty and unobserved future regimes."},
        "limitations": ["The model was trained for one next observation only; recursive horizons have not been separately validated.",
            "Dates exclude weekends only and may include exchange holidays; they are not verified trading sessions.",
            "A same-day estimate is not proof of a forecast issued before the session.",
            "Frozen source and calibration; no live quote, future OHLCV, auto-retrain or ledger write."]}
