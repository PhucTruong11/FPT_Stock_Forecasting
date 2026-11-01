"""Bounded recursive Ridge scenarios from close history, not synthetic OHLCV bars.

Only h=1 was trained. Week/month output is an unvalidated recursive scenario.
Weekday dates are estimates, not a certified HOSE calendar.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from src.data.contracts import BASE_DIR, file_sha256, utc_now
from src.models.local_demo import FEATURES

MAX_STEPS = 23
PATHS = 512
SEED = 20261030
BAND_POLICY = "configs/recursive_band_calibration.json"


def local_today():
    return pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").tz_localize(None).normalize()


def forecast_date_limits(origin, today=None):
    today=local_today() if today is None else pd.Timestamp(today).normalize()
    origin=pd.Timestamp(origin).normalize()
    calendar_end=today+pd.DateOffset(months=1)
    supported_end=origin+pd.offsets.BDay(MAX_STEPS)
    selectable_end=min(calendar_end,supported_end)
    return {'local_today':str(today.date()),'calendar_max_date':str(calendar_end.date()),
        'selectable_min_date':str(today.date()),'selectable_max_date':str(selectable_end.date()),
        'max_effective_date_from_snapshot':str(supported_end.date()),
        'custom_available':bool(origin<today<=selectable_end),'max_steps':MAX_STEPS,
        'calendar_verified':False,'weekend_policy':'roll forward to next weekday; exchange holidays unverified'}


def target_window(origin, horizon, today, target_date=None):
    origin, today = pd.Timestamp(origin).normalize(), pd.Timestamp(today).normalize()
    if origin >= today:
        raise ValueError("Forecast origin must precede the requested local day.")
    if horizon not in {"today", "tomorrow", "week", "month", "custom"}:
        raise ValueError("Unknown forecast horizon.")
    if horizon!='custom' and target_date is not None:
        raise ValueError('target_date is only accepted for a custom forecast.')
    start = today + pd.Timedelta(days=1) if horizon == "tomorrow" else today
    end = start
    if horizon == "week":
        end = today + pd.Timedelta(days=6)  # Inclusive seven calendar days.
    elif horizon == "month":
        end = today + pd.DateOffset(months=1) - pd.Timedelta(days=1)
    elif horizon == 'custom':
        if target_date is None:
            raise ValueError('A custom forecast requires target_date.')
        selected=pd.Timestamp(target_date)
        if selected.tz is not None or pd.isna(selected) or selected!=selected.normalize():
            raise ValueError('target_date must be a calendar date without a time or timezone.')
        if not today<=selected<=today+pd.DateOffset(months=1):
            raise ValueError('Choose a target date from today through one calendar month ahead.')
        end=selected
    if horizon in {"today", "tomorrow", "custom"}:
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


def residual_blocks(residuals, steps, *, center=True):
    """Resample centered innovations; ``center=False`` reproduces historical v2.

    Center the calibration pool, not each sampled path/block: empirical skew,
    variance and within-block dependence are preserved. A finite sample of
    paths need not have exactly zero mean. No future outcomes enter this pool.
    """
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 1 or not len(residuals) or not np.isfinite(residuals).all():
        raise ValueError("Invalid residual calibration.")
    if not isinstance(steps, (int, np.integer)) or not 1 <= steps <= MAX_STEPS:
        raise ValueError("Invalid recursive step count.")
    if center:
        residuals = residuals - residuals.mean()
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, len(residuals), size=(PATHS, (steps + 4) // 5))
    indices = (starts[:, :, None] + np.arange(5)[None, None, :]) % len(residuals)
    return residuals[indices.reshape(PATHS, -1)[:, :steps]]


def recursive_paths(estimator, log_history, noise, *, bias=0., zero_return=False):
    """Shared serving/backtest recursion; return log prices for points and paths.

    Only the initial observed context is accepted; subsequent features use the
    corresponding predicted context. Bias is added at step 1 only, including
    stochastic paths. Noise policy is supplied separately for paired v2/v3 tests.
    """
    history = np.asarray(log_history, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if history.ndim != 1 or len(history) < 21 or not np.isfinite(history).all():
        raise ValueError("Insufficient finite close history for recursive forecasting.")
    if (noise.ndim != 2 or not len(noise) or not 1 <= noise.shape[1] <= MAX_STEPS
            or not np.isfinite(noise).all() or not np.isfinite(bias)):
        raise ValueError("Invalid recursive innovations or bias.")
    point_history = history[-21:][None, :].copy()
    sampled_history = np.tile(point_history, (len(noise), 1))
    points, returns, samples = [], [], []
    for i in range(noise.shape[1]):
        step_bias = bias if i == 0 else 0.
        if zero_return:
            point_return, next_samples = 0., sampled_history[:, -1]
        else:
            point_return = float(estimator.predict(return_features(point_history))[0]) + step_bias
            next_samples = (sampled_history[:, -1] +
                estimator.predict(return_features(sampled_history)) + step_bias + noise[:, i])
        next_point = float(point_history[0, -1] + point_return)
        if not np.isfinite(next_point) or not np.isfinite(next_samples).all():
            raise ValueError("Recursive scenario exceeds finite log-price bounds.")
        points.append(next_point); returns.append(point_return); samples.append(next_samples.copy())
        point_history = np.column_stack([point_history[:, 1:], [next_point]])
        sampled_history = np.column_stack([sampled_history[:, 1:], next_samples])
    return np.asarray(points), np.asarray(returns), np.asarray(samples).T


def load_band_policy(root=BASE_DIR):
    path = root / BAND_POLICY
    if not path.exists():
        return {}
    policy = json.loads(path.read_text(encoding='utf-8'))
    if policy.get('schema') != 'fpt-recursive-band-calibration-v1':
        raise ValueError('Unknown recursive band policy.')
    return {**policy, 'file_sha256': file_sha256(path)}


def band_factor(policy, model, step, origin, model_version):
    """Only named, evaluated horizons may be scaled; no interpolation/extrapolation."""
    if not policy or model == 'Zero-Return':
        return 1.
    if pd.Timestamp(policy['calibration_target_end']) > pd.Timestamp(origin):
        raise ValueError('Band calibration labels mature after forecast origin.')
    if model_version not in policy['allowed_model_versions']:
        raise ValueError('Band policy does not cover this model version.')
    factor = float(policy.get('factors', {}).get(model, {}).get(str(step), 1.))
    if step not in {5, 22} and factor != 1.:
        raise ValueError('Band scaling is limited to explicitly evaluated h=5/h=22.')
    if not np.isfinite(factor) or factor < 1.:
        raise ValueError('Band factor must be finite and at least one.')
    return factor


def widen_band(point, lower, upper, factor):
    """Scale log-price distances about the unchanged point, never price levels."""
    if factor == 1.:
        return lower, upper
    if not (np.isfinite([point, lower, upper, factor]).all() and 0 < lower < point < upper and factor >= 1.):
        raise ValueError('Cannot scale a band that does not surround the point forecast.')
    with np.errstate(over='ignore', invalid='ignore'):
        bounds = np.exp(np.log(point) + factor * (np.log([lower, upper]) - np.log(point)))
    if not np.isfinite(bounds).all() or bounds[0] <= 0:
        raise ValueError('Scaled band exceeds finite price bounds.')
    return float(bounds[0]), float(bounds[1])


def forecast(service, horizon="today", model_name=None, today=None, *, target_date=None, observation_dates=None, band_calibration=None):
    name = model_name or service.default_model
    # Apply the existing cutoff, last-bar, source and finite prediction checks first.
    first = service.predict(name)
    today = local_today() if today is None else pd.Timestamp(today).normalize()
    if observation_dates is None:
        start, end, dates = target_window(first["as_of"], horizon, today, target_date)
    else:
        # Research-only date input. API requests never expose this argument.
        # Dates carry no future prices/features; evaluation uses actual observed sessions.
        if target_date is not None:
            raise ValueError('Historical observation dates cannot be combined with target_date.')
        dates = pd.DatetimeIndex(observation_dates)
        if (not 1 <= len(dates) <= MAX_STEPS or dates.hasnans or not dates.is_unique
                or not dates.is_monotonic_increasing or dates.tz is not None
                or (dates != dates.normalize()).any() or dates.min() <= pd.Timestamp(first['as_of'])
                or pd.Timestamp(first['as_of']) >= today):
            raise ValueError('Invalid historical observation dates.')
        start, end = dates[0], dates[-1]
    policy = load_band_policy(getattr(service, 'root', BASE_DIR)) if band_calibration is None else band_calibration
    close = float(first["latest_close_adjusted"])
    history = np.log(service.features.Close.tail(21).to_numpy(float))
    if len(history) < 21:
        raise ValueError("Insufficient close history for recursive forecasting.")
    bias = first["bias_offset"]
    errors = service.bundle["corrected_errors"] if name == "Ridge+bias60" else service.bundle["raw_errors"]
    pool = np.asarray(errors[-60:], dtype=float)
    noise = residual_blocks(pool, len(dates))
    points, returns, sampled_logs = recursive_paths(service.loaded, history, noise,
        bias=bias, zero_return=name == "Zero-Return")
    rows = []
    for i, day in enumerate(dates):
        with np.errstate(over="ignore", invalid="ignore"):
            price = float(np.exp(points[i]))
            samples = np.exp(sampled_logs[:, i])
        if not np.isfinite(price) or not np.isfinite(samples).all() or price <= 0:
            raise ValueError("Recursive scenario exceeds finite price bounds.")
        low, high = np.quantile(samples, [.1, .9]) if name != "Zero-Return" else (None, None)
        factor = band_factor(policy, name, i + 1, first['as_of'], first['model_version'])
        if low is not None:
            low, high = widen_band(price, low, high, factor)
        rows.append({"step": i + 1, "estimated_date": str(day.date()), "predicted_close_adjusted": price,
                     "step_log_return": float(returns[i]), "change_from_origin_pct": float((price / close - 1) * 100),
                     "scenario_lower": float(low) if low is not None else None,
                     "scenario_upper": float(high) if high is not None else None,
                     "band_scale_factor": factor,
                     "in_requested_window": bool(day >= start), "uses_predicted_context": i > 0})
    return {"ticker": "FPT.VN", "model": name, "model_version": first["model_version"],
        "horizon": horizon, "forecast_method_version": "recursive-ridge6-v4",
        "issued_at_utc": utc_now(), "local_request_date": str(today.date()),
        "origin_date": first["as_of"], "origin_close_adjusted": close,
        "requested_start": str(start.date()), "requested_end": str(end.date()),
        "effective_target_date": rows[-1]["estimated_date"],
        "weekend_shifted": observation_dates is None and horizon in {"today", "tomorrow", "custom"} and str(end.date()) != rows[-1]["estimated_date"],
        "display_start_date": str(start.date()),
        "bridge_steps_before_requested_window": sum(not r['in_requested_window'] for r in rows),
        "steps": len(rows), "max_steps": MAX_STEPS, "path": rows,
        "trained_horizon_observations": 1, "multi_step_validated": False,
        "calendar_verified": False, "calendar_method": "observed_dates_research_only" if observation_dates is not None else "weekdays_only_no_holiday_calendar",
        "role": "one_step_estimate" if len(rows) == 1 else "unvalidated_recursive_scenario",
        "same_day_is_pre_session_forecast": False, "forward_ledger_recorded": False,
        "price_basis": first["price_basis"], "source": first["source"],
        "training_target_end": first["training_target_end"], "calibration_policy": first["calibration_policy"],
        "uncertainty": {"method": "circular_block_residual_bootstrap", "block_observations": 5,
                        "paths": PATHS if name != "Zero-Return" else 0, "seed": SEED,
                        "residual_centering": "subtract calibration-pool mean before block resampling" if name != "Zero-Return" else None,
                        "calibration_residual_mean_removed": float(pool.mean()) if name != "Zero-Return" else None,
                        "residual_pool": ("corrected_errors" if name == "Ridge+bias60" else "raw_errors") if name != "Zero-Return" else None,
                        "base_quantiles": [.1, .9],
                        "displayed_quantiles": [.1, .9] if all(r['band_scale_factor']==1. for r in rows) else None,
                        "coverage_guaranteed": False,
                        "band_calibration": {"study": policy.get('study'), 'file_sha256': policy.get('file_sha256'),
                            'calibration_target_end': policy.get('calibration_target_end'),
                            'evaluated_horizons_observations': policy.get('evaluated_horizons_observations', []),
                            'final_step_factor': rows[-1]['band_scale_factor'],
                            'scope': 'Historical model backtest; transfer to current bundle is unvalidated. Unlisted steps stay unscaled.'},
                        "meaning": "Descriptive scenario range; scaled bounds are no longer literal bootstrap P10/P90. Ignores parameter/source uncertainty and unobserved future regimes."},
        "limitations": ["The model was trained for h=1; historical h=5/h=22 evaluation does not validate the current serving bundle or all calendar windows.",
            "Bias60 is applied to step 1 only; later steps use the raw Ridge response. v3 centers bootstrap innovations without changing point forecasts.",
            "Centering assumes residual mean is not a persistent future drift; pointwise bands are not simultaneous path intervals or full market risk.",
            "Dates exclude weekends only and may include exchange holidays; they are not verified trading sessions.",
            "A same-day estimate is not proof of a forecast issued before the session.",
            "Frozen source and calibration; no live quote, future OHLCV, auto-retrain or ledger write."]}
