"""Regenerate offline charts from an existing, validated portfolio export."""
import argparse
import json
from pathlib import Path
import pandas as pd
from alphatrack.portfolio_report import render_portfolio_report


def render(report):
    p=Path(report)
    payload=json.loads(p.with_suffix('.summary.json').read_text())
    daily=pd.read_csv(p.with_suffix('.daily.csv'),parse_dates=['date'])
    annual=pd.read_csv(p.with_suffix('.annual.csv'))
    funds=pd.read_csv(p.with_suffix('.funds.csv'),dtype={'code':str})
    groups=pd.read_csv(p.with_suffix('.groups.csv'))
    components=pd.read_csv(p.with_suffix('.components_daily.csv.gz'),usecols=['date','asset_type','daily_pnl'],parse_dates=['date'])
    group_daily={}
    for kind,part in components.groupby('asset_type'):
        table=part.groupby('date',as_index=False).daily_pnl.sum().sort_values('date')
        table['cumulative_pnl']=table.daily_pnl.cumsum()
        group_daily[kind]=table
    render_portfolio_report(daily,annual,funds,{'summary':groups,'daily':group_daily},payload['summary'],payload['metadata'],p)
    print(p.resolve())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',default='reports/qlab_portfolio_2010.html')
    render(parser.parse_args().report)
