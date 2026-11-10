"""Reproducible multi-fund QLab portfolio, with annual cash-flow accounting."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .data import load_data
from .engine import EngineConfig, run_backtest
from .lots import analyze_lots
from .portfolio import aggregate_portfolio, analyze_portfolio
from .portfolio_report import render_portfolio_report
from .strategies.qlab import QLabRules, QLabSignalStrategy, asset_key


def _json_clean(value):
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if value is pd.NaT or value is pd.NA:
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_json_clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def run_config(config_path, *, start=None, end=None, output=None, signal_cache=None, title=None):
    config_path=Path(config_path).resolve()
    conf=json.loads(config_path.read_text())
    unknown=set(conf)-{'title','source','data_dir','types','engine','output'}
    if unknown: raise ValueError(f'Unknown portfolio configuration: {sorted(unknown)}')
    if title is not None: conf['title']=title
    source=(config_path.parent/conf['source']).resolve()
    data_dir=(config_path.parent/conf['data_dir']).resolve()
    output=Path(output).resolve() if output else (config_path.parent/conf['output']).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((source/'PROVENANCE.json').read_text())
    for relative, digest in provenance['files'].items():
        if hashlib.sha256((source/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Reference source changed: {relative}')
    rules=QLabRules(source);assets=rules.catalog(conf['types'])
    frames, coverage, data_hashes={},[],{}
    for asset in assets:
        code=asset['code'];path=data_dir/f'{code}.csv'
        frame,meta=load_data(path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if meta.get('data_sha256')!=digest:raise ValueError(f'Data hash mismatch {code}')
        frames[code]=frame;coverage.append(meta);data_hashes[code]=digest
    cfg_dict=dict(conf['engine'])
    if start:cfg_dict['start']=start
    common_end=min(frame.date.iloc[-1] for frame in frames.values())
    requested_end=pd.Timestamp(end or cfg_dict.get('end') or common_end)
    if requested_end>common_end:raise ValueError(f'Requested end exceeds common NAV coverage {common_end.date()}')
    cfg_dict['end']=str(requested_end.date())
    cfg=EngineConfig(**cfg_dict)
    if cfg.funding_mode!='topup' or cfg.initial_cash!=0 or not cfg.withdraw_on_sell:
        raise ValueError('This portfolio mode requires initial_cash=0, topup and withdraw_on_sell=true')
    conf['engine']=asdict(cfg)
    signals=None
    if signal_cache:
        cache=Path(signal_cache)
        cm=json.loads(cache.with_suffix('.metadata.json').read_text())
        from .strategies import qlab
        expected_adapter=hashlib.sha256(Path(qlab.__file__).read_bytes()).hexdigest()
        if (cm.get('data_hashes')!=data_hashes or cm.get('source_hashes')!=rules.source_hashes
                or cm.get('adapter_sha256')!=expected_adapter or cm.get('start')!=cfg.start
                or cm.get('end')!=cfg.end):
            raise ValueError('Signal cache does not match inputs/source/interval')
        # Explicit local cache only; never deserialize remote downloads.
        with cache.open('rb') as f:signals=pickle.load(f)
        signal_metadata=cm
    if signals is None:
        print(f'Preparing exact QLab signals: {len(assets)} funds',flush=True)
        signals,signal_metadata=rules.build_signals(frames,assets,start=cfg.start,end=cfg.end)
    if set(signals)!={asset_key(a) for a in assets}:raise ValueError('Signal universe mismatch')
    ledgers,trades_by_fund={},{}
    fund_summaries, all_lots, fund_annual=[],[],[]
    for n,asset in enumerate(assets,1):
        key=asset_key(asset);prepared=signals[key]
        if not (prepared.date>=pd.Timestamp(cfg.start)).any():
            raise ValueError(f'No in-range data for {key}')
        daily,trades=run_backtest(prepared,QLabSignalStrategy(prepared),cfg)
        for field in ('code','name','asset_type'):
            daily[field]=asset[field];trades[field]=asset[field]
        ledgers[key]=daily;trades_by_fund[key]=trades
        ad=aggregate_portfolio({key:daily})
        annual,summary=analyze_portfolio(ad,trades,start=cfg.start,end=cfg.end,
                                         periods_per_year=cfg.periods_per_year,risk_free_rate=cfg.risk_free_rate)
        lots,lot_summary=analyze_lots(daily,trades)
        for field in ('code','name','asset_type'):
            lots[field]=asset[field];annual[field]=asset[field]
        all_lots.append(lots);fund_annual.append(annual)
        if not np.isclose(summary['total_pnl'],lot_summary['total_pnl'],rtol=1e-9,atol=1e-6):
            raise ValueError(f'FIFO/account profit mismatch {key}')
        meta=next(m for m in coverage if m['code']==asset['code'])
        fund_summaries.append({**asset,**summary,'first_nav_date':meta['first_date'],
                               'start_date':str(daily.date.iloc[0].date()),'end_date':str(daily.date.iloc[-1].date()),
                               'note':'两条分红日缺NAV，沿用QLab原处理' if meta['missing_corporate_action_nav'] else '',
                               'fifo_realized_pnl':lot_summary['realized_pnl'],
                               'fifo_unrealized_pnl':lot_summary['unrealized_pnl'],
                               'pending_signals':int(trades.status.eq('PENDING_END_OF_DATA').sum())})
        print(f'{n}/{len(assets)} {key}: P&L {summary["total_pnl"]:.2f}, buys {summary["buy_count"]}, sells {summary["sell_count"]}',flush=True)
    combined=aggregate_portfolio(ledgers)
    all_trades=pd.concat(trades_by_fund.values(),ignore_index=True)
    annual,summary=analyze_portfolio(combined,all_trades,start=cfg.start,end=cfg.end,
                                     periods_per_year=cfg.periods_per_year,risk_free_rate=cfg.risk_free_rate)
    group_daily,group_summaries,group_annual={},[],[]
    for kind in conf['types']:
        members={k:v for k,v in ledgers.items() if k.startswith(kind+':')}
        gd=aggregate_portfolio(members)
        gt=all_trades[all_trades.asset_type==kind]
        ga,gs=analyze_portfolio(gd,gt,start=cfg.start,end=cfg.end,
                                periods_per_year=cfg.periods_per_year,risk_free_rate=cfg.risk_free_rate)
        ga['asset_type']=kind;group_annual.append(ga)
        group_daily[kind]=gd;group_summaries.append({'asset_type':kind,'name':kind,'fund_count':len(members),**gs})
    funds=pd.DataFrame(fund_summaries);groups=pd.DataFrame(group_summaries)
    for name,part in [('funds',funds),('groups',groups)]:
        if not np.isclose(part.total_pnl.sum(),summary['total_pnl'],rtol=1e-9,atol=1e-6):
            raise ValueError(f'{name} profit reconciliation failed')
    lots=pd.concat(all_lots,ignore_index=True)
    missing=[dict(code=m['code'],events=m['missing_corporate_action_nav']) for m in coverage if m['missing_corporate_action_nav']]
    metadata={
        'title':conf['title'],'source_url':provenance['repository'],'source_commit':provenance['commit'],
        'qlab_commit':provenance['commit'],'fund_count':len(assets),'types':conf['types'],
        'start_date':cfg.start,'end_date':cfg.end,'coverage':coverage,'data_hashes':data_hashes,
        'signal_metadata':{**signal_metadata,'source_path':conf['source']},'missing_corporate_action_nav':missing,
        'assumptions':[
            '使用当前QLab启用名单回溯，不代表2010年已知的基金池；存在名单选择/幸存者偏差。',
            f'{cfg.start}起空仓；更早历史只预热指标。成立较晚基金不提前交易；债券排名需1/2/3/5年收益均可计算，但源码不验证净值历史连续性。',
            '每个新增净值观察日按QLab生成信号，下一可用净值成交；末日信号不虚构成交。',
            '买入按各类原金额，无资金上限；卖出净回款立即取出。只有Fundd有卖出规则，不添加年末清仓。',
            'QLab分红与拆分记录逐条使用，现金分红按除息日净值再投资；000931两条缺净值事件沿用QLab已有遗漏并单独披露。',
            '000037净值从2013-06-20直接跳至2020-04-22（2498天断档）；沿用QLab原数据和目标日前最近净值算法，未补造历史，可能影响其历史评分与买入。',
            f'本次截止{cfg.end}，不超过快照共同净值截止{common_end.date()}；没有混用期末不同日期的国内和海外净值。',
            '申购赎回费默认0；管理费已反映在净值中。未还原历史限购、暂停申赎、封闭期、确认/到账延迟及QDII实际公布时刻。',
            '净值历史含少量非交易日披露；按净值观察序列建模，不声称所有理论成交在历史上可实际执行。',
            '年度盈亏包含未卖出持仓的市值变化；卖出取回金额含本金，不等同收益。',
            '年度已实现/未实现分解采用平均成本；另附FIFO逐批明细，两种口径总盈亏一致。',
            '账户CAGR按首次投入起ACT/365.25；XIRR按ACT/365；夏普及波动率用252观察期/年、无风险利率0。',
            '当前年份仅统计截至期末的实际金额，不把YTD盈亏外推为全年利润。',
        ],
        'validation':{'annual_pnl_sum':float(annual.profit.sum()),'fund_pnl_sum':float(funds.total_pnl.sum()),
                      'group_pnl_sum':float(groups.total_pnl.sum()),'fifo_pnl_sum':float(lots.pnl.sum()),
                      'cashflow_identity':float(summary['final_equity']+summary['total_withdrawals']-summary['total_contributions'])}}
    for suffix,frame in [('daily.csv',combined),('annual.csv',annual),('funds.csv',funds),('groups.csv',groups),
                         ('group_annual.csv',pd.concat(group_annual,ignore_index=True)),
                         ('fund_annual.csv',pd.concat(fund_annual,ignore_index=True)),('trades.csv',all_trades),('lots.csv',lots)]:
        frame.to_csv(output.with_suffix('.'+suffix),index=False,float_format='%.12g')
    pd.concat(ledgers.values(),ignore_index=True).to_csv(output.with_suffix('.components_daily.csv.gz'),index=False,compression='gzip',float_format='%.12g')
    audit_signals=[]
    for key,frame in signals.items():
        part=frame[frame.date>=pd.Timestamp(cfg.start)][['date','action','amount','fraction','rank','score','reason']].copy()
        part['asset_type'],part['code']=key.split(':');audit_signals.append(part)
    pd.concat(audit_signals,ignore_index=True).to_csv(output.with_suffix('.signals.csv.gz'),index=False,compression='gzip',float_format='%.12g')
    payload={'summary':summary,'config':conf,'metadata':metadata,'annual':annual.to_dict('records'),
             'groups':groups.to_dict('records'),'funds':funds.to_dict('records')}
    write_json(output.with_suffix('.summary.json'),payload)
    render_portfolio_report(combined,annual,funds,{'summary':groups,'daily':group_daily},summary,metadata,output)
    print(json.dumps(_json_clean({'report':str(output),**summary}),ensure_ascii=False,indent=2,allow_nan=False),flush=True)
    return payload


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/qlab_portfolio.json')
    p.add_argument('--start');p.add_argument('--end');p.add_argument('--output')
    p.add_argument('--title',help='Optional display title for this independent run')
    p.add_argument('--signal-cache',help='Explicit trusted local cache; hashes and date range must match')
    args=p.parse_args(argv)
    run_config(args.config,start=args.start,end=args.end,output=args.output,signal_cache=args.signal_cache,title=args.title)


if __name__=='__main__':main()
