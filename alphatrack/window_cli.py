"""Compare independent trailing-year and fixed-start QLab portfolio backtests."""
import argparse
from datetime import date, datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .portfolio_cli import run_config, write_json, strategy_settings
from .strategies.qlab import QLabRules


AUTO_FIXED_STARTS = 'auto'
FIRST_FIXED_START_YEAR = 2010
FIXED_START_STEP_YEARS = 5


def normalize_fixed_starts(fixed_start):
    """Accept the original single-date API as well as multiple fixed starts."""
    values = [] if fixed_start is None else [fixed_start] if isinstance(fixed_start, str) else list(fixed_start)
    starts = [date.fromisoformat(value).isoformat() for value in values]
    if len(starts) != len(set(starts)):
        raise ValueError('Fixed starts must be unique')
    return starts


def plan_windows(config_path, *, as_of, years=(), fixed_start=AUTO_FIXED_STARTS):
    """Anchor starts to the requested date; cap all ends at common NAV coverage."""
    anchor = pd.Timestamp(date.fromisoformat(as_of))
    if any(type(year) is not int or year <= 0 for year in years):
        raise ValueError('Window years must be positive integers')
    if len(set(years)) != len(years):
        raise ValueError('Window years must be unique')
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    source = (config_path.parent / config['source']).resolve()
    data = (config_path.parent / config['data_dir']).resolve()
    assets = QLabRules(source).catalog(config['types'])
    if not assets:
        raise ValueError('No enabled funds in the requested strategy types')
    common_end = min(pd.to_datetime(pd.read_csv(data / f"{asset['code']}.csv", usecols=['date']).date).max()
                     for asset in assets)
    effective_end = min(anchor, common_end)
    fixed_starts = ([f'{year}-01-01' for year in range(
        FIRST_FIXED_START_YEAR, effective_end.year + 1, FIXED_START_STEP_YEARS)]
        if fixed_start == AUTO_FIXED_STARTS else normalize_fixed_starts(fixed_start))
    if not years and not fixed_starts:
        raise ValueError('At least one available fixed start or trailing window is required')
    windows = []
    for year in years:
        start = anchor - pd.DateOffset(years=year)
        if start > effective_end:
            raise ValueError(f'No data in the {year}-year window')
        windows.append({'window_id':f'{year}y', 'kind':'trailing', 'years':year, 'label':f'过去 {year} 年',
                        'start':str(start.date()), 'end':str(effective_end.date())})
    for fixed_date in fixed_starts:
        start = pd.Timestamp(fixed_date)
        if start > effective_end:
            raise ValueError('Fixed start exceeds the available end date')
        windows.append({'window_id':'since_'+fixed_date.replace('-',''), 'kind':'fixed', 'years':None,
                        'label':f'{fixed_date} 至今', 'start':fixed_date, 'end':str(effective_end.date())})
    return windows


def run_windows(config_path, *, as_of, years=(),
                fixed_start=AUTO_FIXED_STARTS, output=None, render_only=False):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    selection = strategy_settings(config)
    if output is None:
        output = ((config_path.parent / config['window_output']) if config.get('window_output')
                  else Path('reports/alphastep_random_windows.html' if selection['selection_mode']=='random'
                            else 'reports/qlab_portfolio_windows.html'))
    output = Path(output).resolve()
    strategy_name = 'AlphaStep · 随机选基版' if selection['selection_mode']=='random' else 'AlphaStep · 原排名版'
    windows = plan_windows(config_path, as_of=as_of, years=years, fixed_start=fixed_start)
    output.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for window in windows:
        report = output.with_name(f'{output.stem}_{window["window_id"]}.html')
        if render_only:
            payload = json.loads(report.with_suffix('.summary.json').read_text())
            summary = payload['summary']
            if summary['start_date'] != window['start'] or summary['end_date'] != window['end']:
                raise ValueError(f'Saved window dates differ from the current plan: {report.name}')
            if summary['beginning_assets'] != 0:
                raise ValueError('Comparison requires independently restarted empty portfolios')
            if strategy_settings(payload['config']) != selection:
                raise ValueError(f'Saved window selection policy differs from requested strategy: {report.name}')
        else:
            print(f'Running {window["label"]}: {window["start"]} to {window["end"]}', flush=True)
            payload = run_config(config_path, start=window['start'], end=window['end'], output=report,
                                 title=f'{strategy_name} · {window["label"]}独立回测')
        runs.append({**window, 'report_path':report, 'payload':payload})

    annual_rows, window_payloads = [], []
    for run in runs:
        payload = run['payload']
        annual_rows.extend({'window_id':run['window_id'], **row} for row in payload['annual'])
        window_payloads.append({key:run[key] for key in ('window_id','kind','years','label','start','end')} | {
                                'report':run['report_path'].name,
                                'summary':payload['summary'], 'annual':payload['annual']})
    annual = pd.DataFrame(annual_rows)
    annual.pivot(index='year', columns='window_id', values='profit').reindex(
        columns=[w['window_id'] for w in windows]).rename(columns={w['window_id']:w['label'] for w in windows}).to_csv(
            output.with_suffix('.annual_matrix.csv'), float_format='%.12g')
    effective_end = windows[0]['end']
    write_json(output.with_suffix('.summary.json'), {
        'as_of':as_of, 'effective_end':effective_end,
        'strategy':selection,
        'fixed_starts':[window['start'] for window in windows if window['kind'] == 'fixed'],
        'fixed_start_policy':({'first_year':FIRST_FIXED_START_YEAR, 'step_years':FIXED_START_STEP_YEARS}
                              if fixed_start == AUTO_FIXED_STARTS else None),
        'method':'Each window starts empty; earlier history warms indicators only. No inherited positions.',
        'windows':window_payloads})
    from .window_report import render_window_report
    render_window_report(runs, output, as_of=as_of, effective_end=effective_end)
    print(f'Window comparison saved: {output}', flush=True)
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/qlab_portfolio.json')
    parser.add_argument('--as-of', default=datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat(),
                        help='YYYY-MM-DD anchor, defaults to today in Asia/Shanghai')
    parser.add_argument('--years', nargs='+', type=int, default=[],
                        help='Optional trailing-year comparisons; disabled by default')
    fixed = parser.add_mutually_exclusive_group()
    fixed.add_argument('--fixed-start', '--fixed-starts', nargs='+',
                       help='Override fixed starts; defaults to Jan 1 every five years from 2010 through available data')
    fixed.add_argument('--no-fixed-start', '--no-fixed-starts', action='store_const', const=None, dest='fixed_start',
                       help='Explicitly omit all fixed-start comparisons')
    parser.set_defaults(fixed_start=AUTO_FIXED_STARTS)
    parser.add_argument('--output', help='Comparison report path; defaults to config window_output or QLab report')
    parser.add_argument('--render-only', action='store_true', help='Rebuild comparison from matching existing window exports')
    args = parser.parse_args(argv)
    run_windows(args.config, as_of=args.as_of, years=args.years, fixed_start=args.fixed_start,
                output=args.output, render_only=args.render_only)


if __name__ == '__main__':
    main()
