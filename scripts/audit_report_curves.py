"""Independent audit of saved ledger mathematics and actual embedded Plotly arrays.

Does not call production aggregation, performance, XIRR, or rendering functions.
Checks the existing zero-fee, zero-initial-cash, immediate-sale-withdrawal runs.
"""
import base64
import hashlib
import html
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
YEARS = (2010, 2015, 2020, 2025)
PREFIXES = ('qlab_portfolio_windows', 'alphastep_random_windows')
KINDS = ('Fundb', 'Fundc', 'Fundd', 'Fundf_1', 'Fundg')
MONEY_TOL = .0002  # CSV stores only 12 significant figures; much less than one cent.
RATE_TOL = 2e-8
checks = 0
failures = []
maximum_errors = {}


def equal(label, actual, expected, tolerance=MONEY_TOL):
    global checks
    checks += 1
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape:
        failures.append({'check': label, 'shapes': [list(a.shape), list(b.shape)]})
        return
    finite_equal = np.array_equal(np.isnan(a), np.isnan(b))
    error = float(np.nanmax(np.abs(a-b))) if a.size and np.isfinite(a-b).any() else 0.
    category = 'rate' if tolerance == RATE_TOL else 'money_or_quantity'
    maximum_errors[category] = max(maximum_errors.get(category, 0), error)
    if not finite_equal or error > tolerance:
        failures.append({'check': label, 'max_error': error, 'tolerance': tolerance})


def require(label, condition):
    global checks
    checks += 1
    if not condition:
        failures.append({'check': label})


def npv_residual(rate, dates, flows):
    t = (pd.DatetimeIndex(dates)-pd.Timestamp(dates[0])).days.to_numpy()/365.
    values = np.asarray(flows)*np.exp(-math.log1p(rate)*t)
    return float(values.sum()/np.abs(values).sum()) if np.abs(values).sum() else 0.


def independent_irr(dates, flows):
    cash = pd.Series(np.asarray(flows), index=pd.DatetimeIndex(dates)).groupby(level=0).sum()
    cash = cash[cash.abs() > 1e-8]
    if len(cash) < 2 or not (cash.lt(0).any() and cash.gt(0).any()):
        return []
    t = (cash.index-cash.index[0]).days.to_numpy()/365.
    c = cash.to_numpy()/cash.abs().sum()
    def f(log_rate):
        values = c*np.exp(-log_rate*t)
        return float(values.sum()/np.abs(values).sum())
    grid = np.unique(np.r_[np.linspace(-10., 10., 801), np.linspace(-.5, .5, 801)])
    values = np.array([f(x) for x in grid])
    roots = []
    for i in np.flatnonzero(values[:-1]*values[1:] < 0):
        lo, hi = grid[i], grid[i+1]
        for _ in range(65):
            mid = (lo+hi)/2
            if f(lo)*f(mid) <= 0:
                hi = mid
            else:
                lo = mid
        roots.append(math.expm1((lo+hi)/2))
    if abs(f(0)) < 1e-13 and not any(abs(r) < 1e-10 for r in roots):
        roots.append(0.)
    return sorted(roots)


def plots(document):
    decoder = json.JSONDecoder()
    result = {}
    for m in re.finditer(r'Plotly.newPlot\(\s*"((?:portfolio|window)-[^" ]+)"\s*,\s*', document):
        rest = document[m.end():]
        data, end = decoder.raw_decode(rest)
        layout, _ = decoder.raw_decode(rest[end:].lstrip().lstrip(',').lstrip())
        result[m[1]] = (data, layout)
    return result


def values(data):
    if isinstance(data, dict) and 'bdata' in data:
        result = np.frombuffer(base64.b64decode(data['bdata']), dtype=np.dtype(data['dtype']))
        if data.get('shape'):
            result = result.reshape(tuple(map(int, data['shape'].split(','))))
        return result
    return np.asarray(data, dtype=float)


def embedded(document, name):
    m = re.search(r'\b'+name+r'\s*=\s*', document)
    return json.JSONDecoder().raw_decode(document[m.end():])[0]


def component_events(comp, trades, label):
    comp = comp.sort_values(['asset_type', 'code', 'date']).copy()
    keys = ['asset_type', 'code', 'date']
    require(label+' component identity', not comp.duplicated(keys).any())
    require(label+' zero fees', comp.fees.eq(0).all() and trades.fee.eq(0).all())
    require(label+' nonnegative amounts', comp.external_flow.ge(0).all() and comp.withdrawal.ge(0).all())
    filled = trades[trades.status.isin(['FILLED', 'PARTIAL'])].copy()
    require(label+' filled dates', filled.date.notna().all())
    event = filled.set_index(keys)
    out = comp.set_index(keys)
    for action in ['BUY', 'SELL']:
        trade = event[event.action.eq(action)]
        for column in ['amount', 'shares']:
            out[f'{action}_{column}'] = trade.groupby(level=keys)[column].sum().reindex(out.index, fill_value=0.)
    equal(label+' deposits equal actual buys', out.external_flow, out.BUY_amount)
    equal(label+' withdrawals equal actual sales', out.withdrawal, out.SELL_amount)
    prior_shares = out.groupby(level=['asset_type', 'code']).shares.shift(fill_value=0.)
    prior_nav = out.groupby(level=['asset_type', 'code']).nav.shift(fill_value=0.)
    prior_cash = out.groupby(level=['asset_type', 'code']).cash.shift(fill_value=0.)
    previous_assets = prior_cash + prior_shares*prior_nav
    before_trade = prior_cash + prior_shares*out.split*(out.nav+out.dividend)
    expected_shares = prior_shares*out.split*(1+out.dividend/out.nav)+out.BUY_shares-out.SELL_shares
    expected_cash = prior_cash+out.external_flow+out.SELL_amount-out.BUY_amount-out.withdrawal
    equal(label+' corporate actions and trade quantities', out.shares, expected_shares)
    equal(label+' cash from trade flows', out.cash, expected_cash)
    equal(label+' equity from independent holdings', out.equity, expected_cash+expected_shares*out.nav)
    out['market_change'] = before_trade-previous_assets
    equal(label+' component profit from price and corporate actions', out.daily_pnl, out.market_change)
    # Since deposits equal executed buys, reinvested dividends cannot be mistaken for external deposits.
    return out.reset_index(), {
        'component_rows': len(out), 'filled_trade_rows': len(filled),
        'dividend_observations': int(out.dividend.ne(0).sum()),
        'funded_dividend_observations': int((out.dividend.ne(0)&prior_shares.gt(0)).sum()),
        'weekend_observations': int((comp.date.dt.dayofweek >= 5).sum()),
        'weekend_fills': int((filled.date.dt.dayofweek >= 5).sum()),
    }


def independent_aggregate(comp):
    x = comp.copy()
    x['asset'] = x.asset_type+':'+x.code
    x['holdings_value'] = x.cash+x.shares*x.nav
    # Values carry forward across staggered observations; events never carry forward.
    assets = x.pivot(index='date', columns='asset', values='holdings_value').sort_index().ffill().fillna(0.)
    result = x.groupby('date')[['external_flow', 'withdrawal', 'market_change']].sum().sort_index()
    result['equity'] = assets.sum(axis=1)
    result['previous_equity'] = result.equity.shift(fill_value=0.)
    result['daily_return'] = np.divide(result.market_change, result.previous_equity,
                                      out=np.zeros(len(result)), where=result.previous_equity.to_numpy()>1e-9)
    result['unit_nav'] = (1+result.daily_return).cumprod()
    result['drawdown'] = result.unit_nav/result.unit_nav.cummax().clip(lower=1)-1
    result['daily_pnl'] = result.market_change
    result['cumulative_pnl'] = result.market_change.cumsum()
    result['total_contributions'] = result.external_flow.cumsum()
    result['cumulative_withdrawals'] = result.withdrawal.cumsum()
    return result


def scope_metrics(frame, summary, annual, label):
    dates = frame.index
    funded = (frame.equity.gt(1e-9)|frame.external_flow.gt(1e-9)|frame.previous_equity.gt(1e-9))
    first = dates[funded][0]
    rs = frame.loc[first:, 'daily_return']
    compounded = frame.unit_nav.iloc[-1]-1
    years = (dates[-1]-first).days/365.25
    m = dict(final_equity=frame.equity.iloc[-1], total_contributions=frame.external_flow.sum(),
             total_withdrawals=frame.withdrawal.sum(), total_pnl=frame.market_change.sum(),
             total_return=compounded, cagr=(1+compounded)**(1/years)-1,
             max_drawdown=frame.drawdown.min(), annual_volatility=rs.std(ddof=1)*math.sqrt(252),
             sharpe=rs.mean()/rs.std(ddof=1)*math.sqrt(252))
    require(label+' first investment', summary['first_investment_date']==str(first.date()))
    for key, value in m.items():
        equal(label+' '+key, summary[key], value, RATE_TOL if key in ['total_return','cagr','max_drawdown','annual_volatility','sharpe'] else MONEY_TOL)
    flows = (frame.withdrawal-frame.external_flow).to_numpy().copy()
    flows[-1] += frame.equity.iloc[-1]
    roots = independent_irr(dates, flows)
    require(label+' independent XIRR found', len(roots) == 1)
    equal(label+' XIRR', summary['xirr'], roots[0] if len(roots)==1 else np.nan, RATE_TOL)
    equal(label+' XIRR NPV residual', npv_residual(summary['xirr'], dates, flows), 0., RATE_TOL)
    annual_expected = {}
    for year, p in frame.groupby(frame.index.year):
        prev, end = p.previous_equity.iloc[0], p.equity.iloc[-1]
        dep, out = p.external_flow.sum(), p.withdrawal.sum()
        nav = (1+p.daily_return).cumprod()
        expected = dict(beginning_assets=prev, end_assets=end, deposits=dep, withdrawals=out,
                        profit=end-prev-dep+out, twr_return=nav.iloc[-1]-1,
                        max_drawdown=(nav/nav.cummax().clip(lower=1)-1).min())
        a = annual.loc[year]
        for key,value in expected.items():
            equal(label+f' {year} '+key,a[key],value,RATE_TOL if key in ['twr_return','max_drawdown'] else MONEY_TOL)
        if pd.notna(a.xirr):
            ds=list(p.index); cf=list((p.withdrawal-p.external_flow).to_numpy())
            if prev>0:
                ds.insert(0,pd.Timestamp(f'{year}-01-01')-pd.Timedelta(days=1)); cf.insert(0,-prev)
            cf[-1]+=end
            equal(label+f' {year} annual XIRR NPV residual',npv_residual(a.xirr,ds,cf),0.,RATE_TOL)
        annual_expected[int(year)]=expected
    missing_years=annual.index.difference(annual_expected)
    if len(missing_years):
        equal(label+' pre-inception year profits',annual.loc[missing_years,'profit'],np.zeros(len(missing_years)))
    return {'first_investment':str(first.date()),'zero_balance_before_funding_days':int((dates<first).sum()),
            'zero_balance_after_funding_days':int(frame.loc[first:,'equity'].abs().lt(1e-9).sum()),
            'funded_days_without_deposits':int(frame.loc[first:,'external_flow'].eq(0).sum()),
            'observations':len(frame),'independent_xirr':roots,'annual':annual_expected}


def verify_trace(label, trace, frame, column):
    require(label+' dates',trace['x']==frame.index.strftime('%Y-%m-%d').tolist())
    equal(label+' values',values(trace['y']),frame[column],RATE_TOL if column in ['unit_nav','drawdown'] else MONEY_TOL)


def check_window(prefix, year):
    name=f'{prefix}_since_{year}0101'; base=REPORTS/name
    payload=json.loads(base.with_suffix('.summary.json').read_text())
    daily=pd.read_csv(base.with_suffix('.daily.csv'),parse_dates=['date']).set_index('date')
    comp=pd.read_csv(base.with_suffix('.components_daily.csv.gz'),parse_dates=['date'],dtype={'code':str})
    trades=pd.read_csv(base.with_suffix('.trades.csv'),parse_dates=['date'],dtype={'code':str})
    annual=pd.read_csv(base.with_suffix('.annual.csv')).set_index('year')
    groups=pd.read_csv(base.with_suffix('.groups.csv')).set_index('asset_type')
    group_annual=pd.read_csv(base.with_suffix('.group_annual.csv')).set_index(['asset_type','year'])
    comp, facts=component_events(comp,trades,name)
    independent=independent_aggregate(comp)
    require(name+' combined dates',daily.index.equals(independent.index))
    for column in ['equity','previous_equity','external_flow','withdrawal','daily_pnl','cumulative_pnl','total_contributions','cumulative_withdrawals','daily_return','unit_nav','drawdown']:
        equal(name+' combined '+column,daily[column],independent[column],RATE_TOL if column in ['daily_return','unit_nav','drawdown'] else MONEY_TOL)
    scope=scope_metrics(independent,payload['summary'],annual,name)
    group_frames={}; group_scopes={}
    for kind in KINDS:
        g=independent_aggregate(comp[comp.asset_type.eq(kind)]); group_frames[kind]=g
        group_scopes[kind]=scope_metrics(g,groups.loc[kind],group_annual.loc[kind],name+'/'+kind)
    document=base.with_suffix('.html').read_text(); parsed=plots(document)
    require(name+' all four chart IDs',set(parsed)=={'portfolio-annual-chart','portfolio-pnl-chart','portfolio-nav-chart','portfolio-funding-chart'})
    ps,layout=parsed['portfolio-pnl-chart']
    require(name+' complete P&L trace set', [t['name'] for t in ps]==['组合累计盈亏',*KINDS,'组合当日盈亏'])
    require(name+' complete NAV trace set', [t['name'] for t in parsed['portfolio-nav-chart'][0]]==['组合单位净值','回撤'])
    require(name+' complete funding trace set', [t['name'] for t in parsed['portfolio-funding-chart'][0]]==['账户剩余资产','累计投入','累计取出'])
    for tr in ps:
        if tr['name']=='组合累计盈亏': verify_trace(name+'/'+tr['name'],tr,independent,'cumulative_pnl')
        elif tr['name']=='组合当日盈亏': verify_trace(name+'/'+tr['name'],tr,independent,'daily_pnl')
        else: verify_trace(name+'/'+tr['name'],tr,group_frames[tr['name']],'cumulative_pnl')
    require(name+' pnl axis units',layout['yaxis']['title']['text']=='累计盈亏 / 元' and layout['yaxis2']['title']['text']=='当日盈亏 / 元')
    for tr in parsed['portfolio-nav-chart'][0]: verify_trace(name+'/'+tr['name'],tr,independent,'unit_nav' if tr['name']=='组合单位净值' else 'drawdown')
    for tr,column in zip(parsed['portfolio-funding-chart'][0],['equity','total_contributions','cumulative_withdrawals']): verify_trace(name+'/'+tr['name'],tr,independent,column)
    year_trace=parsed['portfolio-annual-chart'][0][0]
    require(name+' annual calendar labels',year_trace['x']==[str(y)+(' YTD' if y==2026 else '') for y in annual.index])
    cards=dict(re.findall(r'<div class="stat-label">([^<]+)</div><div class="stat-value">([^<]+)</div>',document))
    for label,key,style in [('累计盈亏','total_pnl','money'),('资金加权年化 · XIRR','xirr','percent'),('净值年化 · TWR','cagr','percent'),('最大回撤','max_drawdown','percent'),('累计投入','total_contributions','money'),('累计取出','total_withdrawals','money'),('期末剩余资产','final_equity','money'),('夏普比率','sharpe','number')]:
        value=payload['summary'][key]
        expected=f'{value:.2%}' if style=='percent' else ('¥ ' if style=='money' else '')+f'{value:,.2f}'
        require(name+' actual metric card '+label,cards.get(label)==expected)
    equal(name+' annual chart values',values(year_trace['y']),[scope['annual'][y]['profit'] for y in annual.index])
    for i,y in enumerate(annual.index):
        e=scope['annual'][int(y)]
        equal(name+f' annual hover {y}',values(year_trace['customdata'][i]),[e[k] for k in ['deposits','withdrawals','end_assets','twr_return']])
    actual_annual=embedded(document,'annualRows')
    for row in actual_annual:
        expected=scope['annual'][int(row['year'])]
        for shown,source in {'profit':'profit','contributions':'deposits','withdrawals':'withdrawals','ending_assets':'end_assets','twr_return':'twr_return'}.items():
            equal(name+f' displayed annual {row["year"]}/{shown}',row[shown],expected[source])
    actual_groups=embedded(document,'groupRows')
    for row in actual_groups:
        g=group_frames[row['asset_type']]
        for key,value in {'total_pnl':g.cumulative_pnl.iloc[-1],'contributions':g.external_flow.sum(),'withdrawals':g.withdrawal.sum(),'ending_assets':g.equity.iloc[-1]}.items(): equal(name+f' displayed group {row["asset_type"]}/{key}',row[key],value)
    notes={'random_report_has_ranked_default_note':prefix.startswith('alphastep') and '本次五类规则中，Fundb、Fundc 按排名买入' in document,
           'twr_card_says_full_evaluation_period':'剔除外部资金流；按完整评价时间年化' in document}
    scope.pop('annual'); [g.pop('annual') for g in group_scopes.values()]
    return {'window':name,'facts':facts,'portfolio':scope,'groups':group_scopes,'presentation_findings':notes,
            'html_sha256':hashlib.sha256(document.encode()).hexdigest()},independent,annual


def main():
    runs=[]; saved={}
    for prefix in PREFIXES:
        for year in YEARS:
            record,frame,annual=check_window(prefix,year); runs.append(record); saved[(prefix,year)]=(frame,annual)
            print(f'Audited {record["window"]}; cumulative failures={len(failures)}',flush=True)
        document=(REPORTS/f'{prefix}.html').read_text(); parsed=plots(document)
        require(prefix+' overview charts',set(parsed)=={'window-annual-chart','window-pnl-chart'})
        require(prefix+' all four overview P&L traces',len(parsed['window-pnl-chart'][0])==4)
        require(prefix+' all four overview annual traces',len(parsed['window-annual-chart'][0])==4)
        for tr,year in zip(parsed['window-pnl-chart'][0],YEARS): verify_trace(prefix+f'/overview/{year}',tr,saved[(prefix,year)][0],'cumulative_pnl')
        for tr,year in zip(parsed['window-annual-chart'][0],YEARS):
            annual=saved[(prefix,year)][1]
            for x,y in zip(tr['x'],tr['y']):
                if int(x)<year: require(prefix+f'/empty/{year}/{x}',y is None)
                else: equal(prefix+f'/annual/{year}/{x}',y,annual.loc[int(x),'profit'])
        for ident,unit in [('window-pnl-chart','累计盈亏 / 元'),('window-annual-chart','年度盈亏 / 元')]: require(prefix+'/'+ident+' units',parsed[ident][1]['yaxis']['title']['text']==unit)
    output={'passed':not failures,'checks':checks,'failures':failures,'maximum_absolute_errors':maximum_errors,
            'method':'Independent holdings/corporate-action/trade reconstruction, event-only cashflows, end-of-day external-flow TWR, independent log-space XIRR bracketing and annual NPV residuals; actual HTML Plotly arrays decoded including base64 typed arrays.',
            'tolerances':{'money_yuan':MONEY_TOL,'rates':RATE_TOL},'scope':'8 portfolio reports, 40 category accounts, 2 overview HTML files; no production math imported','windows':runs,
            'limitations':['XIRR sign-change root scan is finite and does not prove global uniqueness; reported XIRR NPV residuals are checked separately.','Accounting is verified under existing next-NAV/zero-fee/immediate-withdrawal assumptions; realistic execution and NAV publication timing are outside this audit.','252-observation Sharpe/volatility uses irregular NAV observation dates including weekends; this is the disclosed convention, not an exchange calendar reconstruction.']}
    target=REPORTS/'audit_20261111_curves.json'; target.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':output['passed'],'checks':checks,'failure_count':len(failures),'output':str(target)},ensure_ascii=False),flush=True)
    if failures:
        raise SystemExit(1)

if __name__=='__main__': main()
