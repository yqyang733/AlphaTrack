"""End-to-end exports retain undefined annualization and reject mixed semantics."""
import json
from pathlib import Path
import pandas as pd
import pytest
from alphatrack.cli import main


def setup_config(tmp_path, navs):
    data=tmp_path/'nav.csv'
    pd.DataFrame({'date':pd.date_range('2021-01-01',periods=len(navs)).strftime('%Y-%m-%d'),'nav':navs}).to_csv(data,index=False)
    cfg={'data':'nav.csv','output':'result.html','report_mode':'lots','strategy':{'name':'fundd','params':{}},'engine':{'withdraw_on_sell':True}}
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    return path


def test_same_day_open_annual_remains_undefined_in_html_csv_and_json(tmp_path):
    path=setup_config(tmp_path,[1.,.9,.9])
    main(['--config',str(path)])
    payload=json.loads((tmp_path/'result.summary.json').read_text())
    assert payload['lot_summary']['open_count']==1
    assert payload['lot_summary']['undefined_annualized_count']==1
    assert payload['lot_summary']['open_annualized_median'] is None
    lots=pd.read_csv(tmp_path/'result.lots.csv')
    assert lots.annualized_return.isna().all()
    assert lots.sell_date.isna().all()
    html=(tmp_path/'result.html').read_text()
    assert 'lot-annual-chart' in html and '同日' in html
    assert '{{TITLE}}' not in html


def test_no_trades_window_produces_valid_report_and_empty_ledger(tmp_path):
    path=setup_config(tmp_path,[1.])
    main(['--config',str(path)])
    payload=json.loads((tmp_path/'result.summary.json').read_text())
    assert payload['lot_summary']['realized_count']==0
    assert payload['lot_summary']['open_count']==0
    assert pd.read_csv(tmp_path/'result.lots.csv').empty
    assert (tmp_path/'result.html').stat().st_size>1000000


def test_legacy_report_rejects_withdrawal_semantics(tmp_path):
    path=setup_config(tmp_path,[1.])
    cfg=json.loads(path.read_text());cfg['report_mode']='account';path.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit):
        main(['--config',str(path)])
