"""Normalize archived Eastmoney NAVs and QLab's exact dividend snapshot.

No remote JavaScript is executed. Missing corporate-action NAVs are reported;
only the two documented 000931 dates are accepted under QLab's existing policy.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

VENDOR_FILES = ["trading/config.py", "trading/engine.py", "trading/ranking.py",
                "data/processor.py", "core/calculator.py", "strategy/mean_reversion.py",
                "config.py", "configs/trading/Fundf_1.json", "configs/trading/Fundg.json",
                "data_files/Fund"]
KNOWN_MISSING = {("000931", "2017-03-22"), ("000931", "2017-11-14")}


def extract(text, name):
    match = re.search(r"var\s+" + re.escape(name) + r"\s*=\s*(.*?);", text)
    if match is None:
        raise ValueError(f"Missing source field {name}")
    return json.loads(match[1])


def import_snapshot(source, raw, output, vendor, commit):
    source, raw, output, vendor = map(Path, (source, raw, output, vendor))
    output.mkdir(parents=True, exist_ok=True)
    assets = json.loads((raw / "catalog.json").read_text())
    common_end = min(datetime.fromtimestamp(json.loads((raw / f"{a['code']}.json").read_text())['last_ts']/1000,
                                           ZoneInfo('Asia/Shanghai')).date().isoformat() for a in assets)
    audit = {r['code']: r for r in json.loads((raw / "dividend_audit.json").read_text())}
    files = list(VENDOR_FILES) + [f"data_files/dividend/fund/{a['code']}.csv" for a in assets]
    hashes = {}
    for relative in files:
        target = vendor / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
        hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    provenance = dict(repository="https://github.com/yqyang733/QLab", commit=commit,
                      files=hashes, scope="Unmodified source subset and dividend/catalog snapshot")
    (vendor / "PROVENANCE.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    coverage = []
    for asset in assets:
        code = asset['code']
        text = (raw / f'{code}.js').read_text(encoding='utf-8-sig')
        if extract(text, 'fS_code') != code:
            raise ValueError(f'Wrong identity {code}')
        values = [v for v in extract(text, 'Data_netWorthTrend')
                  if datetime.fromtimestamp(v['x']/1000, ZoneInfo('Asia/Shanghai')).date().isoformat() <= common_end]
        frame = pd.DataFrame({
            'date': [datetime.fromtimestamp(v['x']/1000, ZoneInfo('Asia/Shanghai')).date().isoformat() for v in values],
            'nav': [v['y'] for v in values], 'dividend': 0.0, 'split': 1.0,
        })
        if (frame.date.duplicated().any() or not frame.date.is_monotonic_increasing
                or not np.isfinite(frame.nav).all() or (frame.nav <= 0).any()):
            raise ValueError(f'Malformed NAV {code}')
        events = pd.read_csv(source / f'data_files/dividend/fund/{code}.csv', dtype=str,
                             encoding='utf-8-sig', keep_default_na=False)
        if events.duplicated(['分红日', '分红类型']).any():
            raise ValueError(f'Duplicate corporate action {code}')
        lookup = {d: i for i, d in enumerate(frame.date)}
        missing = []
        for event in events.to_dict('records'):
            day, kind, value = event['分红日'], event['分红类型'], event['分红']
            if day > frame.date.iloc[-1]:
                continue
            if day not in lookup:
                if (code, day) not in KNOWN_MISSING:
                    raise ValueError(f'Unapproved missing corporate-action NAV {code}: {day}')
                missing.append(event)
                continue
            i = lookup[day]
            if kind == '现金分红':
                frame.at[i, 'dividend'] = float(value)
            elif kind == '份额折算':
                before, after = map(float, value.split(':'))
                if before != 1:
                    raise ValueError('QLab snapshot split must be normalized 1:x')
                frame.at[i, 'split'] = after
            else:
                raise ValueError(f'Unknown action {kind}')
        # All annotated events within NAV history must exist in the snapshot.
        annotated = []
        for value, day in zip(values, frame.date):
            if not value.get('unitMoney'):
                continue
            kind = '现金分红' if '分红' in value['unitMoney'] else '份额折算' if '拆分' in value['unitMoney'] else None
            if kind is None or not ((events['分红日'] == day) & (events['分红类型'] == kind)).any():
                annotated.append({'date': day, 'annotation': value['unitMoney']})
        if annotated:
            raise ValueError(f'New/unmatched source action {code}: {annotated}')
        target = output / f'{code}.csv'
        frame.to_csv(target, index=False, float_format='%.15g')
        origin = json.loads((raw/f'{code}.json').read_text())
        check = audit[code]
        audit_status = ('verified_no_changes' if not check.get('error') and not check.get('added') and not check.get('conflicts')
                        else 'remote_rate_limited' if check.get('error') else 'remote_unreliable_additions_rejected')
        meta = {**asset, 'source_url': origin['source_url'], 'source_sha256': origin['sha256'],
                'retrieved_at': origin['retrieved_at'], 'rows': len(frame),
                'first_date': frame.date.iloc[0], 'last_date': frame.date.iloc[-1],
                'raw_latest_date': datetime.fromtimestamp(origin['last_ts']/1000, ZoneInfo('Asia/Shanghai')).date().isoformat(),
                'data_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                'dividend_source_commit': commit, 'dividend_events': len(events),
                'dividend_audit': audit_status, 'missing_corporate_action_nav': missing,
                'corporate_actions': 'QLab snapshot; cash dividends reinvested at ex-date NAV; split ratios exact',
                'missing_nav_policy': 'Preserve QLab existing omission for the two documented 000931 events; no date shifts or invented NAV',
                'weekend_observations': int((pd.to_datetime(frame.date).dt.dayofweek >= 5).sum())}
        target.with_suffix('.metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        coverage.append(meta)
    manifest = dict(assets=coverage, source=provenance, latest_common_nav_date=min(a['last_date'] for a in coverage),
                    remote_dividend_checks={s: sum(a['dividend_audit']==s for a in coverage) for s in {a['dividend_audit'] for a in coverage}})
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({'funds':len(coverage),'end':manifest['latest_common_nav_date'],
                      'missing_actions':sum(len(a['missing_corporate_action_nav']) for a in coverage),
                      'dividend_checks':manifest['remote_dividend_checks']},ensure_ascii=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('source','raw','output','vendor','commit'): p.add_argument('--'+name,required=True)
    args=p.parse_args();import_snapshot(**vars(args))
