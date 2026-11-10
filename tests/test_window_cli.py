"""Calendar anchors and data limits must not inherit the old 2010 run dates."""
import json

import pandas as pd

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
    assert [row['start'] for row in windows] == ['2025-11-10','2023-11-10','2021-11-10','2016-11-10']
    assert {row['end'] for row in windows} == {'2026-11-06'}
    leap = window_cli.plan_windows(path, as_of='2024-02-29', years=[1])
    assert leap == [{'years':1, 'start':'2023-02-28', 'end':'2024-02-29'}]
