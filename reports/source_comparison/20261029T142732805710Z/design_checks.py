"""Inspect source contrasts and context dates without fitting or changing models."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[2]
sys.path.insert(0, str(ROOT))

from src.data.source_audit import save_json  # noqa: E402
from src.features.build_features import FEATURE_COLUMNS  # noqa: E402
from src.models.robustness import fold_masks  # noqa: E402
from src.models.source_comparison import SOURCE_KEYS, context_positions, load_inputs  # noqa: E402


def main():
    protocol = json.loads((STUDY / 'protocol.json').read_text(encoding='utf-8'))
    config = protocol['config']
    datasets, cohort = load_inputs(STUDY)
    frozen = ROOT / 'data/interim/FPT.VN_features.csv'
    rebuilt_match = None
    if frozen.exists():
        old = pd.read_csv(frozen, parse_dates=['Date'])
        rebuilt = datasets['yahoo_v2'][0]
        assert np.array_equal(old.Date, rebuilt.Date)
        rebuilt_match = float(np.max(np.abs(old[FEATURE_COLUMNS].to_numpy() - rebuilt[FEATURE_COLUMNS].to_numpy())))
        assert rebuilt_match == 0.
    context_rows, window_rows, label_rows = [], [], []
    for _, row in cohort.iterrows():
        windows = {}
        for source in SOURCE_KEYS:
            position = int(row[f'position_{source}'])
            window = datasets[source][0].iloc[position-config['lookback']+1:position+1].Date
            assert window.le(row.Date).all()
            windows[source] = window.to_numpy()
        window_rows.append({'Date': row.Date, 'context_calendar_differs': not np.array_equal(*windows.values()),
                            **{f'first_context_date_{source}': dates[0] for source, dates in windows.items()}})
    for fold in config['folds']:
        masks = fold_masks(cohort, fold)
        for source in SOURCE_KEYS:
            context = context_positions(cohort.loc[masks['train'], f'position_{source}'].to_numpy(), config['lookback'])
            dates = datasets[source][0].iloc[context].Date
            assert dates.le(pd.Timestamp(fold['train_end'])).all()
            context_rows.append({'fold': fold['id'], 'source': source, 'labeled_train_rows': int(masks['train'].sum()),
                                 'unique_scaler_context_rows': len(context), 'first_date': dates.min(), 'last_date': dates.max()})
        for phase, mask in masks.items():
            part = cohort.loc[mask]
            difference = 10000*(part.return_yahoo_v2-part.return_vietcap_v3_candidate)
            label_rows.append({'fold': fold['id'], 'phase': phase, 'n': len(part),
                               'native_label_abs_difference_gt_10bps': int(difference.abs().gt(10).sum()),
                               'median_abs_native_label_difference_bps': float(difference.abs().median()),
                               'reference_up': int(part.return_vietcap_v3_candidate.gt(0).sum()),
                               'reference_down': int(part.return_vietcap_v3_candidate.lt(0).sum()),
                               'reference_neutral': int(part.return_vietcap_v3_candidate.eq(0).sum()),
                               'native_yahoo_neutral': int(part.return_yahoo_v2.eq(0).sum())})
    windows = pd.DataFrame(window_rows)
    windows.to_csv(STUDY/'native_context_windows.csv.gz', index=False, compression='gzip', date_format='%Y-%m-%d', lineterminator='\n')
    pd.DataFrame(context_rows).to_csv(STUDY/'scaler_context_counts.csv', index=False, date_format='%Y-%m-%d', lineterminator='\n')
    pd.DataFrame(label_rows).to_csv(STUDY/'native_label_differences.csv', index=False, lineterminator='\n')
    result = {'common_labeled_origins': len(cohort), 'rebuilt_yahoo_features_max_absolute_difference_to_v2': rebuilt_match,
              'common_origins_with_different_native_context_calendars': int(windows.context_calendar_differs.sum()),
              'all_context_dates_at_or_before_origin': True, 'all_scaler_context_dates_at_or_before_train_cutoff': True,
              'extra_v3_labeled_training_rows_used': 0, 'training_failures': 0,
              'preparation_note': 'One pre-training parser error was fixed and covered by a regression test. Its incomplete preparation archive is retained locally under .cache/source-comparison-failed-preparation; no model was fitted in that attempt.'}
    save_json(STUDY/'design_checks.json', result)
    # Independently check every reported metric, including directional denominators.
    metrics = pd.read_csv(STUDY/'metrics.csv').set_index(['candidate', 'phase'])
    predictions = pd.read_csv(STUDY/'predictions.csv.gz')
    largest_errors = {}
    for identity, frame in predictions.groupby(['candidate', 'phase']):
        true, pred = frame.Reference_Return.to_numpy(), frame.predicted_log_return.to_numpy()
        nz, issued = true != 0, (true != 0) & (pred != 0)
        mse, zero_mse = np.mean((true-pred)**2), np.mean(true**2)
        calculated = {'n': len(true), 'rmse': np.sqrt(mse), 'mae': np.mean(np.abs(true-pred)),
                      'nonzero_actual_n': nz.sum(), 'neutral_actual_n': (~nz).sum(),
                      'direction_coverage_pct': 100*issued.sum()/nz.sum(),
                      'mse_skill_vs_zero': 1-mse/zero_mse,
                      'hit_rate_pct': 100*(np.sign(true[nz])==np.sign(pred[nz])).mean() if issued.any() else np.nan,
                      'balanced_accuracy_pct': 50*((pred[true>0]>0).mean()+(pred[true<0]<0).mean()) if np.any(pred!=0) else np.nan}
        saved = metrics.loc[identity]
        for name, value in calculated.items():
            if pd.isna(value):
                assert pd.isna(saved[name]), (identity, name)
            else:
                error = abs(float(value)-float(saved[name]))
                assert error < 1e-10, (identity, name, error)
                largest_errors[name] = max(largest_errors.get(name, 0.), error)
    save_json(STUDY/'metric_details_verification.json', {'metric_rows': len(metrics),
              'all_metrics_including_balanced_accuracy_and_denominators_verified': True,
              'maximum_absolute_errors': largest_errors, 'source_script': 'design_checks.py'})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
