"""Compare independently restarted trailing-year QLab portfolio backtests."""
import argparse
from datetime import date, datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .portfolio_cli import run_config, write_json
from .strategies.qlab import QLabRules


def plan_windows(config_path, *, as_of, years=(1, 3, 5, 10)):
    """Anchor starts to the requested date; cap all ends at common NAV coverage."""
    anchor = pd.Timestamp(date.fromisoformat(as_of))
    if not years or any(type(year) is not int or year <= 0 for year in years):
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
    windows = []
    for year in years:
        start = anchor - pd.DateOffset(years=year)
        if start > effective_end:
            raise ValueError(f'No data in the {year}-year window')
        windows.append({'years':year, 'start':str(start.date()), 'end':str(effective_end.date())})
    return windows


def run_windows(config_path, *, as_of, years=(1, 3, 5, 10),
                output='reports/qlab_portfolio_windows.html', render_only=False):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    windows = plan_windows(config_path, as_of=as_of, years=years)
    output.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for window in windows:
        report = output.with_name(f'{output.stem}_{window["years"]}y.html')
        if render_only:
            payload = json.loads(report.with_suffix('.summary.json').read_text())
            summary = payload['summary']
            if summary['start_date'] != window['start'] or summary['end_date'] != window['end']:
                raise ValueError(f'Saved window dates differ from the current plan: {report.name}')
            if summary['beginning_assets'] != 0:
                raise ValueError('Comparison requires independently restarted empty portfolios')
        else:
            print(f'Running {window["years"]} years: {window["start"]} to {window["end"]}', flush=True)
            payload = run_config(config_path, start=window['start'], end=window['end'], output=report,
                                 title=f'QLab 五类基金组合 · 近 {window["years"]} 年独立回测')
        runs.append({'years':window['years'], 'report_path':report, 'payload':payload})

    annual_rows, window_payloads = [], []
    for run in runs:
        payload, year = run['payload'], run['years']
        annual_rows.extend({'window_years':year, **row} for row in payload['annual'])
        window_payloads.append({'years':year, 'report':run['report_path'].name,
                                'summary':payload['summary'], 'annual':payload['annual']})
    annual = pd.DataFrame(annual_rows)
    annual.pivot(index='year', columns='window_years', values='profit').reindex(columns=list(years)).to_csv(
        output.with_suffix('.annual_matrix.csv'), float_format='%.12g')
    effective_end = windows[0]['end']
    write_json(output.with_suffix('.summary.json'), {
        'as_of':as_of, 'effective_end':effective_end,
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
    parser.add_argument('--years', nargs='+', type=int, default=[1, 3, 5, 10])
    parser.add_argument('--output', default='reports/qlab_portfolio_windows.html')
    parser.add_argument('--render-only', action='store_true', help='Rebuild comparison from matching existing window exports')
    args = parser.parse_args(argv)
    run_windows(args.config, as_of=args.as_of, years=args.years, output=args.output, render_only=args.render_only)


if __name__ == '__main__':
    main()
