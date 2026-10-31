"""Local, hash-chained shadow forecasts. No serving promotion or scheduled activity."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.data.contracts import BASE_DIR,file_sha256,utc_now,normalize_ohlcv,invalid_candle_mask
from src.features.build_features import build_dataset


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def read_ledger(path):
    path=Path(path);events=[];previous=''
    if not path.exists():return events
    for line in path.read_text(encoding='utf-8').splitlines():
        event=json.loads(line);checksum=event.pop('event_sha256')
        if event['previous_event_sha256']!=previous or digest(event)!=checksum:raise ValueError('Ledger integrity check failed.')
        event['event_sha256']=checksum;events.append(event);previous=checksum
    return events


def append_event(path,payload):
    if {'sequence','previous_event_sha256','event_sha256'} & payload.keys():raise ValueError('Ledger metadata is generated internally.')
    path=Path(path);events=read_ledger(path);path.parent.mkdir(parents=True,exist_ok=True)
    event={**payload,'sequence':len(events)+1,'previous_event_sha256':events[-1]['event_sha256'] if events else ''}
    event['event_sha256']=digest(event)
    with path.open('a',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    return event


def local_time(now=None):
    now=pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if now.tz is None:raise ValueError('Timezone-aware issuance/scoring time required.')
    return now.tz_convert('Asia/Ho_Chi_Minh')


def validate_issue_dates(origin,training_end,created_at,now):
    local=local_time(now);origin=pd.Timestamp(origin).normalize()
    if origin>local.tz_localize(None).normalize():raise ValueError('Cannot forecast from a future origin.')
    if pd.Timestamp(training_end)>origin:raise ValueError('Model includes labels after the forecast origin.')
    if pd.Timestamp(created_at)>pd.Timestamp(now):raise ValueError('Cannot backdate model issuance.')
    if origin.date()==local.date() and local.hour<16:raise ValueError('Same-day bar is not accepted before conservative 16:00 local EOD cutoff.')


def snapshot_input(path,ledger):
    checksum=file_sha256(Path(path));directory=Path(ledger).parent/'input_snapshots';directory.mkdir(parents=True,exist_ok=True)
    dest=directory/(checksum+'.csv.gz')
    if not dest.exists():dest.write_bytes(gzip.compress(Path(path).read_bytes(),mtime=0))
    return checksum,str(dest.resolve())


def issue(bundle_path,input_path,ledger_path,now=None):
    now=pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    bundle=json.loads(Path(bundle_path).read_text(encoding='utf-8'))
    if bundle['automatic_promotion'] is not False:raise ValueError('Only shadow bundles accepted.')
    model_path=BASE_DIR/bundle['model_path']
    if file_sha256(model_path)!=bundle['model_sha256']:raise ValueError('Frozen model bytes changed.')
    raw=normalize_ohlcv(pd.read_csv(input_path),allow_untraded_missing_prices=True,allow_invalid_candles=True)
    if raw.Date.max().date()>local_time(now).date():raise ValueError('Source contains future dates.')
    features=build_dataset(raw)[0];latest=features.iloc[-1];origin=latest.Date
    if origin!=raw.loc[raw.Volume.gt(0),'Date'].max():raise ValueError('Latest traded bar is invalid; do not silently use an older origin.')
    validate_issue_dates(origin,bundle['training_target_end'],bundle['created_at_utc'],now)
    if bundle['calibration_targets'] and pd.Timestamp(max(bundle['calibration_targets']))>origin:raise ValueError('Calibration includes future labels.')
    model=joblib.load(model_path);point=float(model.predict(features.iloc[[-1]][bundle['feature_columns']])[0])
    errors=np.asarray(bundle['raw_errors'],dtype=float);bias=float(errors.mean()) if len(errors)>=bundle['bias_spec']['minimum_matured'] else 0.
    adjusted=point+bias;interval_errors=np.asarray(bundle['corrected_errors'],dtype=float)
    interval=None
    if len(interval_errors)>=bundle['interval_spec']['minimum_matured']:
        lo,hi=np.quantile(interval_errors,bundle['interval_spec']['quantiles'])
        interval={'nominal_coverage':bundle['interval_spec']['nominal_coverage'],'lower_log_return':float(adjusted+lo),
            'upper_log_return':float(adjusted+hi),'calibration_n':len(interval_errors),'claim':'historical residual quantiles, not guaranteed coverage'}
    bundle_hash=file_sha256(Path(bundle_path));input_hash=file_sha256(Path(input_path))
    duplicate_key=digest({'bundle':bundle_hash,'input':input_hash,'origin':str(origin.date())})
    events=read_ledger(ledger_path)
    if any(e.get('duplicate_key')==duplicate_key for e in events):raise ValueError('This exact forecast input/version was already logged; no overwrite.')
    checksum,snapshot=snapshot_input(input_path,ledger_path)
    payload={'kind':'forecast','issued_at_utc':now.tz_convert('UTC').isoformat(),'issued_local_date':str(local_time(now).date()),
        'origin_date':str(origin.date()),'target_date':None,'target_policy':'next positive-volume source observation; resolved only on maturity',
        'bundle_sha256':bundle_hash,'bundle_path':str(Path(bundle_path).resolve()),'model_sha256':bundle['model_sha256'],
        'model_path':bundle['model_path'],'training_target_end':bundle['training_target_end'],'input_sha256':checksum,'input_snapshot':snapshot,
        'origin_close_adjusted':float(latest.Close),'forecasts':{'Ridge_raw':point,'Ridge_bias_frozen':adjusted,'Zero-Return':0.,'TrainMean_snapshot':bundle['train_mean']},
        'bias_offset':bias,'interval':interval,'role':'pending shadow forecast, not serving recommendation',
        'calibration_policy':'fixed bundle state; no automatic refit/state update','duplicate_key':duplicate_key,
        'forward_eligibility':'unresolved until actual target date; same-target-day issuance excluded from full-session score'}
    event=append_event(ledger_path,payload);print('Logged pending shadow forecast:',event['event_sha256'],flush=True)
    return event


def mature_outcome(forecast,raw,now):
    origin=pd.Timestamp(forecast['origin_date']);traded=raw.loc[raw.Volume.gt(0)].copy()
    later=traded.loc[traded.Date.gt(origin)]
    if later.empty:return None
    target=later.iloc[0]
    if target.Date.date()>=local_time(now).date():return None  # Require a completed prior calendar day.
    earlier=traded.loc[traded.Date.eq(origin)]
    if len(earlier)!=1:raise ValueError('Origin missing from scoring snapshot.')
    if invalid_candle_mask(pd.concat([earlier,later.iloc[[0]]])).any():raise ValueError('Origin/next-observation candle invalid; source review required.')
    origin_close=float(earlier.Close.iloc[0]);target_close=float(target.Close);truth=float(np.log(target_close/origin_close))
    # Current basis on BOTH dates prevents mixing pre/post corporate-action adjustment scales.
    scores={}
    for strategy,pred in forecast['forecasts'].items():
        scores[strategy]={'predicted_log_return':pred,'squared_return_error':float((truth-pred)**2),'absolute_return_error':float(abs(truth-pred)),
            'predicted_close_reanchored':float(origin_close*np.exp(pred)),
            'direction_correct':bool(np.sign(pred)==np.sign(truth)) if truth!=0 and pred!=0 else None}
    interval=forecast.get('interval');covered=None if interval is None else bool(interval['lower_log_return']<=truth<=interval['upper_log_return'])
    return {'kind':'outcome','forecast_event_sha256':forecast['event_sha256'],'scored_at_utc':pd.Timestamp(now).tz_convert('UTC').isoformat(),
        'actual_target_date':str(target.Date.date()),'actual_log_return':truth,'origin_close_current_basis':origin_close,'actual_close_current_basis':target_close,
        'origin_adjustment_ratio':origin_close/forecast['origin_close_adjusted'],
        'full_session_eligible':forecast['issued_local_date']<str(target.Date.date()),'interval_covered':covered,'scores':scores,
        'limits':'same-day target issuance is pre-close shadow only; hash chain is not third-party timestamp certification'}


def score(input_path,ledger_path,now=None):
    now=pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    raw=normalize_ohlcv(pd.read_csv(input_path),allow_untraded_missing_prices=True,allow_invalid_candles=True)
    if raw.Date.max().date()>local_time(now).date():raise ValueError('Scoring input contains future dates.')
    events=read_ledger(ledger_path);resolved={e['forecast_event_sha256'] for e in events if e['kind']=='outcome'};count=0
    for event in events:
        if event['kind']!='forecast' or event['event_sha256'] in resolved:continue
        outcome=mature_outcome(event,raw,now)
        if outcome is not None:
            checksum,snapshot=snapshot_input(input_path,ledger_path);outcome.update(outcome_input_sha256=checksum,outcome_input_snapshot=snapshot)
            append_event(ledger_path,outcome);count+=1
    print(f'New matured outcomes: {count}; no missing future labels invented.',flush=True)
    return count


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    command=sub.add_parser('issue');command.add_argument('--bundle',type=Path,required=True);command.add_argument('--input',type=Path,required=True);command.add_argument('--ledger',type=Path,required=True)
    command=sub.add_parser('score');command.add_argument('--input',type=Path,required=True);command.add_argument('--ledger',type=Path,required=True)
    command=sub.add_parser('verify');command.add_argument('--ledger',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='issue':issue(args.bundle,args.input,args.ledger)
    elif args.action=='score':score(args.input,args.ledger)
    else:print('Verified events:',len(read_ledger(args.ledger)))
