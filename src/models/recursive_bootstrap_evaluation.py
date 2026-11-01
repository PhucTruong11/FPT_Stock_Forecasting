"""Paired rolling-origin diagnosis of v2/v3 intervals; never fit or promote a model."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.contracts import BASE_DIR, file_sha256, utc_now
from src.features.build_features import build_dataset
from src.models.forecast_path import PATHS, SEED, recursive_paths, residual_blocks
from src.models.local_demo import FEATURES, LocalRidgeService, save_json

RECENCY = Path('reports/recency_bias/20261030T022440928507Z')
EVALUATION = Path('reports/evaluation_2026/20261030T014012158951Z')
CONFIG = {
    'schema': 'fpt-recursive-bootstrap-evaluation-v1', 'horizons_observations': [5, 21],
    'origin_start': '2026-01-01', 'residual_window': 60, 'minimum_matured': 60,
    'historical_model_id': 'refit_2025_native', 'bias_policy': 'matured raw-error mean at origin; step 1 only',
    'variants': ['v2_uncentered', 'v3_centered'], 'paths': PATHS, 'simulation_seed': SEED,
    'block_observations': 5, 'quantiles': [.1, .9], 'interval_score_alpha': .2,
    'paired_ci': {'resamples': 2000, 'origin_block_length': 21, 'seed': 20261101},
    'common_origins': '60 matured errors and all 21 future observed sessions available; both horizons use identical origins',
    'nonoverlap_sensitivity': 'every h-th origin anchored at first eligible origin; no subset selection',
    'model_refits': 0, 'download': False, 'automatic_promotion': False,
    'role': 'retrospective diagnosis on already inspected adjusted 2026 data; not an untouched holdout or forward test',
    'serving_bundle_evaluated': False,
    'limitations': ['fixed end-2025 model differs from end-October serving bundle and its monthly-model residual pool',
        'daily origins and 5/21-step targets overlap; block CIs are descriptive, not independent-observation tests',
        'actual positive-volume observations define horizons, not 7-day/calendar-month UI windows',
        'adjusted source is not point-in-time certified; centering is a policy assumption, not proof of 80% future coverage'],
    'references': ['https://otexts.com/fpp3/tscv.html',
        'https://www.kevinsheppard.com/files/teaching/mfe/advanced-econometrics/Kreiss_and_lahiri.pdf'],
}


def mature_pool(calibration, origin, window=60):
    """No row whose label matures after origin can affect bias or innovations."""
    prior = calibration.loc[(calibration.Date < origin) &
        (calibration.Target_Date_1d <= origin)].sort_values('Date').tail(window)
    if len(prior) < window:
        raise ValueError('Insufficient matured calibration errors.')
    raw = (prior.Reference_Return - prior.raw_prediction).to_numpy(float)
    corrected = (prior.Reference_Return - prior.prediction).to_numpy(float)
    return raw, corrected, prior.Target_Date_1d.max()


def interval_scores(actual, lower, upper, alpha=.2):
    """Winkler interval score; narrower is rewarded only if misses stay small."""
    actual, lower, upper = map(np.asarray, (actual, lower, upper))
    return upper - lower + (2 / alpha) * (np.maximum(lower - actual, 0) + np.maximum(actual - upper, 0))


def protected_hashes(root, exclude=None):
    return {p.relative_to(root).as_posix(): file_sha256(p)
        for name in ['data', 'models', 'reports'] for p in sorted((root / name).rglob('*'))
        if p.is_file() and (exclude is None or not p.is_relative_to(exclude))}


def load_inputs(root):
    study, source = root / RECENCY, root / EVALUATION
    manifest = json.loads((study / 'manifest.json').read_text())
    assert file_sha256(study / 'protocol.json') == manifest['protocol_sha256']
    spec = next(x for x in manifest['models'] if x['id'] == CONFIG['historical_model_id'])
    model_path = root / spec['path']
    assert file_sha256(model_path) == manifest['artifact_sha256'][spec['path']]
    acquisition = json.loads((source / 'acquisition.json').read_text())
    assert file_sha256(source / 'data_snapshot.zip') == acquisition['zip_sha256']
    with zipfile.ZipFile(source / 'data_snapshot.zip') as archive:
        raw = archive.read('vci.csv')
    assert hashlib.sha256(raw).hexdigest() == acquisition['files']['vci.csv']
    features, _, _, _ = build_dataset(pd.read_csv(io.BytesIO(raw)))
    calibration_path = study / 'refit_2025_native_bias60_calibration.csv'
    calibration = pd.read_csv(calibration_path, parse_dates=['Date', 'Target_Date_1d'])
    assert calibration.Date.is_monotonic_increasing and calibration.Date.is_unique
    assert (calibration.Target_Date_1d > calibration.Date).all()
    assert pd.Timestamp(spec['train_target_end']) < calibration.Date.min()
    estimator = joblib.load(model_path)
    assert list(estimator.feature_names_in_) == FEATURES
    keyed = features.set_index('Date')
    np.testing.assert_allclose(estimator.predict(keyed.loc[calibration.Date, FEATURES]), calibration.raw_prediction, atol=2e-15, rtol=0)
    actual = np.log(keyed.loc[calibration.Target_Date_1d, 'Close'].to_numpy() /
        keyed.loc[calibration.Date, 'Close'].to_numpy())
    np.testing.assert_allclose(actual, calibration.Reference_Return, atol=2e-15, rtol=0)
    # Re-audit the saved causal correction, including its 20-label warm-up.
    for _, row in calibration.iterrows():
        prior = calibration.loc[(calibration.Date < row.Date) & (calibration.Target_Date_1d <= row.Date)].tail(60)
        offset = float((prior.Reference_Return - prior.raw_prediction).mean()) if len(prior) >= 20 else 0.
        np.testing.assert_allclose(row.prediction, row.raw_prediction + offset, atol=2e-15, rtol=0)
    inputs = [study / 'manifest.json', study / 'protocol.json', model_path, calibration_path,
        source / 'data_snapshot.zip', source / 'acquisition.json']
    return estimator, features.reset_index(drop=True), calibration, spec, {p.relative_to(root).as_posix(): file_sha256(p) for p in inputs}


def summarize(frame):
    rows = []
    for (sample, horizon, model, variant), part in frame.groupby(['sample', 'horizon', 'model', 'variant'], sort=False):
        y, pred = part.actual_log_return.to_numpy(), part.predicted_log_return.to_numpy()
        nz = y != 0
        recalls = [(np.sign(pred[y * sign > 0]) == sign).mean() for sign in [1, -1] if (y * sign > 0).any()]
        mse = float(np.mean((y - pred) ** 2)); zero = float(np.mean(y ** 2))
        row = {'sample': sample, 'horizon': int(horizon), 'model': model, 'variant': variant, 'n': len(part),
            'origin_start': part.origin_date.min(), 'origin_end': part.origin_date.max(),
            'target_start': part.target_date.min(), 'target_end': part.target_date.max(),
            'rmse_log_return': np.sqrt(mse), 'mae_log_return': float(np.mean(abs(y - pred))),
            'price_mae_vnd': float(np.mean(abs(part.actual_close - part.predicted_close))),
            'hit_rate_pct': float(100 * (np.sign(pred[nz]) == np.sign(y[nz])).mean()) if model != 'Zero-Return' and nz.any() else None,
            'balanced_accuracy_pct': float(100 * np.mean(recalls)) if model != 'Zero-Return' and len(recalls) == 2 else None,
            'mse_skill_vs_zero': 1 - mse / zero if zero else None,
            'coverage_pct': None, 'lower_miss_pct': None, 'upper_miss_pct': None,
            'mean_width_log_return': None, 'mean_width_pct_origin': None, 'mean_interval_score': None}
        if model != 'Zero-Return':
            lo, hi = part.lower_log_return.to_numpy(), part.upper_log_return.to_numpy()
            row.update(coverage_pct=float(100 * np.mean((y >= lo) & (y <= hi))),
                lower_miss_pct=float(100 * np.mean(y < lo)), upper_miss_pct=float(100 * np.mean(y > hi)),
                mean_width_log_return=float(np.mean(hi - lo)),
                mean_width_pct_origin=float(100 * np.mean(np.exp(hi) - np.exp(lo))),
                mean_interval_score=float(interval_scores(y, lo, hi).mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(predictions):
    rows = []; cfg = CONFIG['paired_ci']
    for (h, model), part in predictions.loc[predictions.model.ne('Zero-Return')].groupby(['horizon', 'model']):
        a = part.loc[part.variant.eq('v2_uncentered')].sort_values('origin_date')
        b = part.loc[part.variant.eq('v3_centered')].sort_values('origin_date')
        assert a.origin_date.to_list() == b.origin_date.to_list()
        np.testing.assert_array_equal(a.predicted_log_return.to_numpy(), b.predicted_log_return.to_numpy())
        y = a.actual_log_return.to_numpy()
        delta = interval_scores(y, b.lower_log_return, b.upper_log_return) - interval_scores(y, a.lower_log_return, a.upper_log_return)
        rng = np.random.default_rng(cfg['seed']); n = len(delta); block = cfg['origin_block_length']
        starts = rng.integers(0, n, size=(cfg['resamples'], (n + block - 1) // block))
        indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(cfg['resamples'], -1)[:, :n]
        lo, hi = np.quantile(delta[indices].mean(axis=1), [.025, .975])
        rows.append({'horizon': int(h), 'model': model, 'n': n,
            'point_forecasts_identical': True, 'interval_score_delta_v3_minus_v2': float(delta.mean()),
            'descriptive_block_ci95_lower': float(lo), 'descriptive_block_ci95_upper': float(hi),
            'origin_block_length': block, 'resamples': cfg['resamples']})
    return pd.DataFrame(rows)


def plot_metrics(metrics, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    colors = ['#7b91a8', '#21a889', '#7b91a8', '#21a889']
    labels = ['Ridge / v2', 'Ridge / v3', 'Ridge+bias / v2', 'Ridge+bias / v3']
    for j, h in enumerate(CONFIG['horizons_observations']):
        data = metrics.loc[metrics['sample'].eq('daily_overlapping') & metrics.horizon.eq(h) & metrics.model.ne('Zero-Return')]
        data = data.set_index(['model', 'variant']).loc[[(m, v) for m in ['Ridge', 'Ridge+bias60'] for v in CONFIG['variants']]]
        for ax, column, title, factor in [(axes[0, j], 'coverage_pct', 'P10-P90 coverage (%)', 1),
                (axes[1, j], 'mean_interval_score', 'Mean interval score (log-return x 100; lower is better)', 100)]:
            values = data[column].to_numpy() * factor
            bars = ax.barh(labels, values, color=colors)
            ax.bar_label(bars, fmt='%.2f', padding=3); ax.margins(x=.2); ax.invert_yaxis()
            ax.set_title(f'{h} observed sessions / n={int(data.n.iloc[0])}\n{title}', fontsize=10)
            ax.spines[['top', 'right']].set_visible(False)
        axes[0, j].axvline(80, color='#c47736', ls='--', label='Nominal 80%')
        axes[0, j].set_xlim(0, 105); axes[0, j].legend(fontsize=8)
    fig.suptitle('FPT: paired rolling-origin residual bootstrap / frozen end-2025 Ridge\nAlready-inspected 2026; same origins; point forecasts unchanged', fontsize=13)
    fig.savefig(destination, dpi=160); plt.close(fig)


def run(root=BASE_DIR, output=None):
    root = Path(root).resolve()
    destination = (root / 'reports/recursive_bootstrap' / pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')) if output is None else Path(output).resolve()
    if not destination.is_relative_to(root / 'reports/recursive_bootstrap'):
        raise ValueError('Evaluation output must stay under reports/recursive_bootstrap.')
    before = protected_hashes(root)
    estimator, features, calibration, spec, inputs = load_inputs(root)
    positions = []
    for i, date in enumerate(features.Date):
        if date < pd.Timestamp(CONFIG['origin_start']) or i < 20 or i + 21 >= len(features):
            continue
        known = calibration.loc[(calibration.Date < date) & (calibration.Target_Date_1d <= date)]
        if len(known) >= CONFIG['minimum_matured']:
            positions.append(i)
    if not positions:
        raise ValueError('No eligible common origins.')
    destination.mkdir(parents=True, exist_ok=False)
    source_files = ['src/models/forecast_path.py', 'src/models/recursive_bootstrap_evaluation.py',
        'src/features/build_features.py', 'src/models/local_demo.py']
    protocol = {'registered_before_evaluation_utc': utc_now(), 'config': CONFIG, 'inputs_sha256': inputs,
        'source_sha256': {p: file_sha256(root / p) for p in source_files},
        'historical_model': {k: v for k, v in spec.items() if k != 'training_dates'},
        'origins': features.Date.iloc[positions].dt.strftime('%Y-%m-%d').to_list()}
    save_json(destination / 'protocol.json', protocol)
    with zipfile.ZipFile(destination / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in source_files:
            archive.write(root / p, p)
        archive.writestr('config.json', json.dumps(CONFIG, indent=2))
    rows = []
    for ordinal, i in enumerate(positions):
        origin = features.iloc[i]; history = np.log(features.Close.iloc[i-20:i+1].to_numpy(float))
        raw, corrected, last_target = mature_pool(calibration, origin.Date)
        assert last_target <= origin.Date and pd.Timestamp(spec['train_target_end']) < origin.Date
        for model in ['Ridge', 'Ridge+bias60', 'Zero-Return']:
            bias = float(raw.mean()) if model == 'Ridge+bias60' else 0.
            pool = corrected if model == 'Ridge+bias60' else raw
            variants = ['baseline'] if model == 'Zero-Return' else CONFIG['variants']
            for variant in variants:
                noise = residual_blocks(pool, 21, center=variant != 'v2_uncentered')
                points, _, paths = recursive_paths(estimator, history, noise, bias=bias, zero_return=model == 'Zero-Return')
                for h in CONFIG['horizons_observations']:
                    actual = features.iloc[i+h]
                    lo, hi = np.quantile(paths[:, h-1] - history[-1], [.1, .9]) if model != 'Zero-Return' else (None, None)
                    rows.append({'origin_ordinal': ordinal, 'origin_date': str(origin.Date.date()),
                        'target_date': str(actual.Date.date()), 'horizon': h, 'model': model, 'variant': variant,
                        'origin_close': float(origin.Close), 'actual_close': float(actual.Close),
                        'predicted_close': float(np.exp(points[h-1])),
                        'actual_log_return': float(np.log(actual.Close / origin.Close)),
                        'predicted_log_return': float(points[h-1] - history[-1]),
                        'lower_log_return': float(lo) if lo is not None else None,
                        'upper_log_return': float(hi) if hi is not None else None,
                        'bias_step1': bias, 'residual_mean_before_centering': float(pool.mean()),
                        'calibration_n': len(pool), 'last_calibration_target': str(last_target.date()),
                        'model_train_target_end': spec['train_target_end']})
        if (ordinal + 1) % 25 == 0:
            print(f'Rolling origins {ordinal + 1}/{len(positions)}', flush=True)
    predictions = pd.DataFrame(rows)
    daily = predictions.assign(sample='daily_overlapping')
    sparse = predictions.loc[predictions.origin_ordinal.mod(predictions.horizon).eq(0)].assign(sample='nonoverlapping_sensitivity')
    metrics = summarize(pd.concat([daily, sparse], ignore_index=True))
    predictions.to_csv(destination / 'predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    metrics.to_csv(destination / 'metrics.csv', index=False)
    comparison = paired_comparisons(predictions)
    comparison.to_csv(destination / 'paired_comparison.csv', index=False)
    plot_metrics(metrics, destination / 'comparison.png')
    # Current serving bundle is shown separately, never used to score historical targets.
    old_path = root / '.cache/bootstrap-v2-before.json'
    live = []
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding='utf-8')); service = LocalRidgeService(root)
        from src.models.forecast_path import forecast
        for date, models in old.items():
            for model, previous in models.items():
                current = forecast(service, 'month', model, today=date)
                np.testing.assert_allclose([p['predicted_close_adjusted'] for p in previous['path']],
                    [p['predicted_close_adjusted'] for p in current['path']], atol=1e-9, rtol=0)
                for result in [previous, current]:
                    live.append({'request_date': date, 'model': model, 'version': result['forecast_method_version'],
                        'origin_date': result['origin_date'], 'steps': result['steps'], 'target_date': result['effective_target_date'],
                        **result['path'][-1], 'role': 'unscored serving scenario; no future actuals available'})
        save_json(destination / 'serving_before_after.json', {'results': live})
    after = protected_hashes(root, exclude=destination)
    assert before == after, 'Protected data/model/historical evidence changed.'
    verification = {'completed_at_utc': utc_now(), 'new_fits': 0, 'downloads': 0, 'automatic_promotion': False,
        'protected_files_unchanged': len(before), 'protected_tree_sha256': hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
        'common_origins': len(positions), 'horizons': CONFIG['horizons_observations'],
        'v2_v3_point_forecasts_identical': True, 'calibration_targets_never_after_origin': True,
        'historical_model_train_target_end': spec['train_target_end'], 'serving_model_used_for_historical_scoring': False,
        'protocol_sha256': file_sha256(destination / 'protocol.json'),
        'outputs_sha256': {p.name: file_sha256(p) for p in destination.iterdir() if p.is_file() and p.name != 'protocol.json'}}
    save_json(destination / 'verification.json', verification)
    print(metrics.to_string(index=False)); print('Evidence:', destination, flush=True)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    run(output=parser.parse_args().output)
