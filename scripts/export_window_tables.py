"""Export a window report as copyable annual/metric tables in ten thousand yuan."""
import argparse
import csv
import io
import json
from pathlib import Path


METRICS = [
    ('累计盈亏（万元）', 'total_pnl', 'money'),
    ('资金加权年化 · XIRR', 'xirr', 'percent'),
    ('组合净值年化 · TWR', 'cagr', 'percent'),
    ('最大净值回撤', 'max_drawdown', 'percent'),
    ('夏普比率', 'sharpe', 'number'),
    ('年化波动率', 'annual_volatility', 'percent'),
]


def formatted(value, kind='money'):
    if value is None:
        return '—'
    if kind == 'money':
        return f'{value / 10000:.2f}'
    if kind == 'percent':
        return f'{value * 100:.2f}%'
    return f'{value:.2f}'


def tsv(rows):
    stream = io.StringIO()
    csv.writer(stream, delimiter='\t', lineterminator='\n').writerows(rows)
    return stream.getvalue()


def export(report):
    report = Path(report)
    payload = json.loads(report.with_suffix('.summary.json').read_text())
    windows = payload['windows']
    annual = [{int(row['year']): row['profit'] for row in window['annual']} for window in windows]
    years = sorted(set().union(*(set(rows) for rows in annual)))
    header = ['年份／指标', *(window['label'] for window in windows)]
    rows = [[str(year), *(formatted(values.get(year)) for values in annual)] for year in years]
    rows += [[label, *(formatted(window['summary'].get(key), kind) for window in windows)]
             for label, key, kind in METRICS]
    report.with_suffix('.table_wan.tsv').write_text(tsv([header, *rows]), encoding='utf-8')
    markdown = [f'金额单位：万元。数据截止：{payload["effective_end"]}；盈亏含未实现浮盈亏。', '',
                '| ' + ' | '.join(header) + ' |', '|---|' + '---:|' * len(windows)]
    markdown += ['| ' + ' | '.join(row) + ' |' for row in rows]
    report.with_suffix('.table_wan.md').write_text('\n'.join(markdown) + '\n', encoding='utf-8')
    paste_rows = [[*(formatted(window['summary'].get(key), kind) for _, key, kind in METRICS),
                   *(formatted(values.get(year)) for year in years)] for window, values in zip(windows, annual)]
    report.with_suffix('.paste_all_wan.tsv').write_text(tsv(paste_rows), encoding='utf-8')
    paste_header = [*(label for label, _, _ in METRICS), *(str(year) for year in years)]
    report.with_suffix('.paste_header_wan.tsv').write_text(tsv([paste_header]), encoding='utf-8')
    for window, row in zip(windows, paste_rows):
        if window['start'] == '2010-01-01':
            report.with_suffix('.paste_2010_wan.tsv').write_text(tsv([row]), encoding='utf-8')
    print(f'Copyable tables exported: {report.with_suffix(".table_wan.md")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    export(parser.parse_args().report)
