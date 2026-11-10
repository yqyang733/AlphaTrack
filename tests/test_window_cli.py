"""Calendar anchors and data limits must not inherit the old 2010 run dates."""
import json

import pandas as pd
import pytest

from alphatrack import window_cli


def test_windows_anchor_to_requested_date_and_share_actual_data_cutoff(tmp_path, monkeypatch):
    class Rules:
        def __init__(self, source):
            pass
        def catalog(self, types):
            return [{'code':'domestic'}, {'code':'overseas'}]
    monkeypatch.setattr(window_cli, 'QLabRules', Rules)
    (tmp_path/'data').mkdir()
    for code, end in [('domestic','2026-11-09'), ('overseas','2026-11-06')]:
        pd.DataFrame({'date':['2000-01-01',end]}).to_csv(tmp_path/'data'/f'{code}.csv', index=False)
    path = tmp_path/'config.json'
    path.write_text(json.dumps({'source':'source','data_dir':'data','types':['Fundd'],
                                'engine':{'start':'2010-01-01','end':'2025-01-01'}}))
    windows = window_cli.plan_windows(path, as_of='2026-11-10')
    assert [row['start'] for row in windows] == ['2025-11-10','2023-11-10','2021-11-10','2016-11-10','2016-01-01','2010-01-01']
    assert [row['window_id'] for row in windows] == ['1y','3y','5y','10y','since_20160101','since_20100101']
    assert windows[-1]['kind'] == 'fixed' and windows[-1]['years'] is None
    assert {row['end'] for row in windows} == {'2026-11-06'}
    leap = window_cli.plan_windows(path, as_of='2024-02-29', years=[1], fixed_start=None)
    assert len(leap) == 1 and leap[0]['start'] == '2023-02-28' and leap[0]['end'] == '2024-02-29'
    next_anchor = window_cli.plan_windows(path, as_of='2026-11-11')
    assert next_anchor[0]['start'] == '2025-11-11'
    assert [row['start'] for row in next_anchor[-2:]] == ['2016-01-01','2010-01-01']
    with pytest.raises(ValueError, match='Fixed start'):
        window_cli.plan_windows(path, as_of='2015-12-31')


def test_cli_includes_fixed_start_by_default_and_allows_explicit_override(monkeypatch):
    calls = []
    monkeypatch.setattr(window_cli, 'run_windows', lambda *args, **kwargs: calls.append(kwargs))
    window_cli.main(['--as-of','2026-11-10'])
    assert calls[-1]['years'] == [1,3,5,10] and calls[-1]['fixed_start'] == ['2016-01-01','2010-01-01']
    window_cli.main(['--as-of','2026-11-10','--no-fixed-start'])
    assert calls[-1]['fixed_start'] is None
    window_cli.main(['--as-of','2026-11-10','--fixed-start','2018-01-01'])
    assert calls[-1]['fixed_start'] == ['2018-01-01']
    window_cli.main(['--as-of','2026-11-10','--fixed-starts','2016-01-01','2010-01-01'])
    assert calls[-1]['fixed_start'] == ['2016-01-01','2010-01-01']


def test_fixed_start_normalization_preserves_single_date_and_rejects_duplicates():
    assert window_cli.normalize_fixed_starts('2016-01-01') == ['2016-01-01']
    assert window_cli.normalize_fixed_starts(None) == []
    assert window_cli.normalize_fixed_starts(('2016-01-01','2010-01-01')) == ['2016-01-01','2010-01-01']
    with pytest.raises(ValueError, match='unique'):
        window_cli.normalize_fixed_starts(['2016-01-01','20160101'])
