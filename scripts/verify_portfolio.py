"""Independent reconciliation of an exported AlphaTrack portfolio report."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def verify(report):
    report=Path(report)
    payload=json.loads(report.with_suffix('.summary.json').read_text())
    summary=payload['summary']
    daily=pd.read_csv(report.with_suffix('.daily.csv'),parse_dates=['date'])
    annual=pd.read_csv(report.with_suffix('.annual.csv'))
    funds=pd.read_csv(report.with_suffix('.funds.csv'),dtype={'code':str})
    groups=pd.read_csv(report.with_suffix('.groups.csv'))
    lots=pd.read_csv(report.with_suffix('.lots.csv'))
    trades=pd.read_csv(report.with_suffix('.trades.csv'),parse_dates=['date','signal_date'])
    components=pd.read_csv(report.with_suffix('.components_daily.csv.gz'),parse_dates=['date'],dtype={'code':str})
    fills=trades[trades.status.isin(['FILLED','PARTIAL'])]
    expected=summary['total_pnl']
    reconciliations={
        'daily_profit':float(daily.daily_pnl.sum()),'annual_profit':float(annual.profit.sum()),
        'fund_profit':float(funds.total_pnl.sum()),'group_profit':float(groups.total_pnl.sum()),
        'fifo_profit':float(lots.pnl.sum()),
        'cashflow_profit':float(summary['final_equity']+summary['total_withdrawals']-summary['total_contributions'])}
    for name,value in reconciliations.items():
        assert np.isclose(value,expected,rtol=1e-10,atol=1e-4),(name,value,expected)
    year_formula=annual.end_assets-annual.beginning_assets-annual.deposits+annual.withdrawals
    assert np.allclose(year_formula,annual.profit,rtol=1e-9,atol=1e-4)
    assert np.allclose(annual.realized_pnl+annual.unrealized_pnl_change,annual.profit,rtol=1e-9,atol=1e-4)
    assert np.allclose(daily.equity+daily.cumulative_withdrawals-daily.total_contributions,daily.cumulative_pnl,rtol=1e-9,atol=1e-4)
    pivot=components.pivot(index='date',columns='code',values='equity').sort_index().ffill().fillna(0).sum(axis=1)
    equity_error=float(np.max(np.abs(pivot.to_numpy()-daily.equity.to_numpy())))
    assert equity_error < 1e-4
    assert (fills.date>fills.signal_date).all()
    assert not ((fills.asset_type!='Fundd')&(fills.action=='SELL')).any()
    assert len(fills[fills.action=='BUY'])==summary['buy_count']
    assert len(fills[fills.action=='SELL'])==summary['sell_count']
    assert annual.year.tolist()==list(range(pd.Timestamp(summary['start_date']).year,pd.Timestamp(summary['end_date']).year+1))
    first=components.groupby('code').first()
    assert np.allclose(first.equity-first.daily_pnl-first.external_flow+first.withdrawal,0,atol=1e-7)
    rate=summary['xirr']
    npv=None
    if rate is not None:
        dates=pd.concat([daily.date,pd.Series([daily.date.iloc[-1]])],ignore_index=True)
        flows=np.r_[daily.withdrawal-daily.external_flow,summary['final_equity']]
        years=(dates-dates.iloc[0]).dt.days.to_numpy()/365
        npv=float(np.sum(flows/(1+rate)**years))
        assert abs(npv)<1e-3
    result={'passed':True,'funds':len(funds),'groups':len(groups),'annual_rows':len(annual),
            'observations':len(daily),'component_rows':len(components),'filled_trades':len(fills),
            'pending_end_signals':int(trades.status.eq('PENDING_END_OF_DATA').sum()),
            'rejected_no_holding':int(trades.status.eq('REJECTED_NO_HOLDING').sum()),
            'weekend_theoretical_fills':int((fills.date.dt.dayofweek>=5).sum()),
            'max_component_equity_error':equity_error,'xirr_npv_residual_yuan':npv,
            'profit_reconciliations':reconciliations}
    report.with_suffix('.validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default='reports/qlab_portfolio_2010.html')
    verify(p.parse_args().report)
