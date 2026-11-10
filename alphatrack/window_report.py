"""Compare independently initialized backtests with calendar-year earnings first."""
from __future__ import annotations

import html
import math
import os
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go

from .portfolio_report import _fmt, _json, _layout


COLORS = ["#42dbc5", "#8da9ff", "#f4c875", "#c2a3f4", "#f7a17d", "#e58cba"]


def _period(start, end, year):
    left = max(pd.Timestamp(start), pd.Timestamp(year, 1, 1))
    right = min(pd.Timestamp(end), pd.Timestamp(year, 12, 31))
    if left > right:
        return None
    partial = left != pd.Timestamp(year, 1, 1) or right != pd.Timestamp(year, 12, 31)
    return dict(start=str(left.date()), end=str(right.date()), partial=partial)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _relative_link(path, directory):
    return html.escape(quote(os.path.relpath(Path(path).resolve(), directory), safe="/"), quote=True)


def render_window_report(runs, output, *, as_of, effective_end):
    """Write an offline dashboard plus long-form annual and summary CSVs.

    Each run is ``{window_id, kind, years, label, start, end, report_path,
    payload}``. ``kind`` is ``trailing`` or ``fixed``; fixed starts need not be
    whole years before ``as_of``. Legacy ``{years, report_path, payload}`` runs
    remain supported. Input order determines display order. ``payload`` is the
    result of ``portfolio_cli.run_config``; the HTML's adjacent ``.daily.csv``
    supplies the curve. All windows must have been initialized independently.
    Cells outside calendar-year coverage are absent, never zero-filled.
    """
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    anchor = pd.Timestamp(as_of).normalize()
    end = pd.Timestamp(effective_end).normalize()
    if end > anchor:
        raise ValueError("Effective data end must not exceed the as-of date")
    ordered = list(runs)
    if not ordered:
        raise ValueError("Windows must be nonempty")

    windows, annual_records, summary_records = [], [], []
    seen_ids = set()
    for i, run in enumerate(ordered):
        kind = run.get("kind", "trailing")
        if kind not in {"trailing", "fixed"}:
            raise ValueError(f"Unknown window kind: {kind}")
        years = int(run["years"]) if kind == "trailing" else None
        if kind == "trailing" and years < 1:
            raise ValueError("Window years must be positive")
        if kind == "fixed" and not run.get("start"):
            raise ValueError("Fixed windows require an explicit start date")
        expected_start = (anchor - pd.DateOffset(years=years) if kind == "trailing"
                          else pd.Timestamp(run["start"]).normalize())
        window_id = str(run.get("window_id") or (f"{years}y" if kind == "trailing"
                                                 else f"since_{expected_start:%Y%m%d}"))
        if window_id in seen_ids:
            raise ValueError("Window IDs must be unique")
        seen_ids.add(window_id)
        label = str(run.get("label") or (f"过去 {years} 年" if kind == "trailing"
                                          else f"{expected_start.date()} 至今"))
        payload = run["payload"]
        summary = dict(payload["summary"])
        metadata = payload.get("metadata", {})
        start = pd.Timestamp(metadata.get("start_date", summary["start_date"]))
        declared_end = pd.Timestamp(metadata.get("end_date", summary["end_date"]))
        if (start != expected_start or declared_end != end
                or pd.Timestamp(summary["start_date"]) != start
                or (run.get("start") and pd.Timestamp(run["start"]) != start)
                or (run.get("end") and pd.Timestamp(run["end"]) != end)):
            raise ValueError(f"Window {window_id} does not match the requested date range")
        if start > end:
            raise ValueError("Window starts after the available data end")
        if abs(float(summary.get("beginning_assets", 0))) > 1e-8:
            raise ValueError("Comparison windows must start with no inherited assets")
        annual = {int(row["year"]): dict(row) for row in payload["annual"]}
        if len(annual) != len(payload["annual"]):
            raise ValueError("Duplicate calendar-year rows")
        expected = set(range(start.year, end.year + 1))
        if set(annual) != expected:
            raise ValueError(f"Missing or out-of-range annual rows in window {window_id}")
        report = Path(run["report_path"])
        daily = pd.read_csv(report.with_suffix(".daily.csv"), parse_dates=["date"])
        if daily.empty or daily.date.min() < start or daily.date.max() > end:
            raise ValueError("Daily ledger does not match the window")
        daily = daily.sort_values("date")
        cumulative = daily["cumulative_pnl"] if "cumulative_pnl" in daily else daily.daily_pnl.cumsum()
        window = dict(window_id=window_id, kind=kind, years=years, label=label,
                      start=str(start.date()), end=str(end.date()),
                      color=COLORS[i % len(COLORS)], summary=summary, annual=annual,
                      report_link=_relative_link(report, output.parent),
                      dates=daily.date.dt.strftime("%Y-%m-%d").tolist(), pnl=cumulative.tolist())
        windows.append(window)
        identity = dict(window_id=window_id, window_label=label, window_kind=kind, window_years=years)
        summary_records.append(dict(**identity, requested_as_of=str(anchor.date()), **summary))
        for year, row in sorted(annual.items()):
            annual_records.append(dict(**identity, **row))

    all_years = list(range(min(pd.Timestamp(w["start"]).year for w in windows), end.year + 1))
    matrix = []
    for year in all_years:
        cells = []
        for window in windows:
            period = _period(window["start"], end, year)
            row = window["annual"].get(year)
            cells.append(None if period is None else dict(
                **period, profit=_number(row["profit"]), deposits=_number(row.get("deposits")),
                withdrawals=_number(row.get("withdrawals")), end_assets=_number(row.get("end_assets")),
                twr_return=_number(row.get("twr_return"))))
        matrix.append(dict(year=year, cells=cells))

    yearly = go.Figure()
    cumulative = go.Figure()
    for i, window in enumerate(windows):
        values = [r["cells"][i]["profit"] if r["cells"][i] is not None else None for r in matrix]
        details = [[c["start"] + " → " + c["end"], c["deposits"], c["withdrawals"]]
                   if (c := r["cells"][i]) is not None else [None, None, None] for r in matrix]
        yearly.add_trace(go.Bar(
            x=[str(y) for y in all_years], y=values, name=window["label"], marker_color=window["color"],
            customdata=details,
            hovertemplate="%{x} · %{customdata[0]}<br><b>盈亏 ¥%{y:,.2f}</b>"
                          "<br>新增投入 ¥%{customdata[1]:,.2f}<br>取回 ¥%{customdata[2]:,.2f}<extra>%{fullData.name}</extra>"))
        cumulative.add_trace(go.Scatter(
            x=window["dates"], y=window["pnl"], name=window["label"], mode="lines",
            line=dict(color=window["color"], width=2), connectgaps=False,
            hovertemplate="%{x}<br>累计盈亏 ¥%{y:,.2f}<extra>%{fullData.name}</extra>"))
    _layout(yearly, 410).update_layout(barmode="group", hovermode="closest")
    yearly.update_xaxes(type="category", title="自然年")
    yearly.update_yaxes(title="年度盈亏 / 元")
    _layout(cumulative, 420)
    cumulative.update_xaxes(title="日期")
    cumulative.update_yaxes(title="累计盈亏 / 元")
    chart_config = {"responsive": True, "displaylogo": False,
                    "modeBarButtonsToRemove": ["sendDataToCloud"],
                    "toImageButtonOptions": {"format": "png", "scale": 2}}
    year_plot = yearly.to_html(full_html=False, include_plotlyjs=True, div_id="window-annual-chart", config=chart_config)
    pnl_plot = cumulative.to_html(full_html=False, include_plotlyjs=False, div_id="window-pnl-chart", config=chart_config)

    overview_rows = []
    for window in windows:
        summary = window["summary"]
        pnl = _number(summary.get("total_pnl"))
        klass = "missing" if pnl is None else "positive" if pnl >= 0 else "negative"
        overview_rows.append(f'''<tr>
          <th scope="row" style="color:{window['color']}">{html.escape(window['label'])}
            <small>{'固定起点' if window['kind']=='fixed' else '滚动区间'}</small></th>
          <td class="period">{window['start']}<small>至 {window['end']}</small></td>
          <td>{_fmt(summary.get('total_contributions'), 'number')}</td>
          <td class="{klass} total">{_fmt(pnl, 'number')}</td>
          <td>{_fmt(summary.get('xirr'), 'percent')}</td>
          <td>{_fmt(summary.get('cagr'), 'percent')}</td>
          <td>{_fmt(summary.get('max_drawdown'), 'percent')}</td>
          <td><a href="{window['report_link']}" aria-label="查看{html.escape(window['label'], quote=True)}的完整报告">查看 ↗</a></td></tr>''')

    table_head = "".join(f'<th style="color:{w["color"]}">{html.escape(w["label"])}</th>' for w in windows)
    body = []
    for row in matrix:
        cells = []
        for cell in row["cells"]:
            if cell is None:
                cells.append('<td class="missing" title="不在这个回测区间内">—</td>')
                continue
            value = cell["profit"]
            klass = "positive" if value is not None and value >= 0 else "negative"
            note = (f'<small>{cell["start"][5:]}—{cell["end"][5:]} · 部分年度</small>' if cell["partial"] else '')
            cells.append(f'<td><span class="{klass}">{_fmt(value,"number")}</span>{note}</td>')
        body.append(f'<tr><th>{row["year"]}</th>{"".join(cells)}</tr>')
    totals = ''.join(f'<td class="total">{_fmt(w["summary"].get("total_pnl"),"number")}</td>' for w in windows)

    metrics = [("累计新增投入", "total_contributions", "number"),
               ("累计卖出取回（含本金）", "total_withdrawals", "number"),
               ("期末持仓市值", "final_equity", "number"),
               ("已实现收益", "realized_pnl", "number"),
               ("未实现浮盈亏", "unrealized_pnl", "number"),
               ("资金加权年化 · XIRR", "xirr", "percent"),
               ("组合净值年化 · TWR", "cagr", "percent"),
               ("最大净值回撤", "max_drawdown", "percent"),
               ("夏普比率", "sharpe", "number"),
               ("年化波动率", "annual_volatility", "percent"),
               ("买入次数", "buy_count", "integer"),
               ("卖出次数", "sell_count", "integer")]
    metric_rows = []
    for label, key, kind in metrics:
        cells = ''.join(f'<td>{_fmt(w["summary"].get(key),kind)}</td>' for w in windows)
        metric_rows.append(f'<tr><th>{label}</th>{cells}</tr>')

    annual_csv = output.with_suffix(".annual.csv")
    summary_csv = output.with_suffix(".summary.csv")
    pd.DataFrame(annual_records).to_csv(annual_csv, index=False, float_format="%.12g")
    pd.DataFrame(summary_records).to_csv(summary_csv, index=False, float_format="%.12g")
    source_commit = windows[0]["summary"].get("source_commit") or ordered[0]["payload"].get("metadata", {}).get("source_commit", "")
    notes = '''每个区间从各自起点空仓开始，已有数据只用于计算历史指标，不继承较早区间的持仓或卖出档位。
    各窗口是一组独立模拟，盈亏不能相加。同一年在不同窗口中，持仓和投入路径可能不同；首尾部分年度也不应与完整年度直接比较。
    年度盈亏包含当年已实现收益及未实现浮盈亏的变化，卖出取回的本金不算利润。没有年末强制清仓，也没有把部分年度金额外推成全年收益。
    无限资金按规则追加、卖出净回款取出；Fundb/Fundc/Fundf_1/Fundg 只有买入规则，Fundd 按原规则买卖。
    零手续费、下一条可用净值成交；未还原历史限购、暂停申赎、确认与到账延迟和 QDII 实际净值公布时刻，属于理论回测。
    使用当前基金名单回放历史，存在名单选择偏差；已知净值断档和分红缺失沿用原报告的处理，并在各区间完整报告披露。'''
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>不同起点的基金组合回测 | AlphaTrack</title>
    <style>
    :root{{color-scheme:dark;--bg:#0b1723;--panel:#132335;--border:#2a3c51;--text:#edf5fa;--muted:#9bb0c7}}
    *{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
    header{{padding:18px 28px;border-bottom:1px solid var(--border);font-size:21px;font-weight:750}}header b{{color:#42dbc5}}
    main{{max-width:1440px;margin:auto;padding:28px}}h1{{font-size:30px;line-height:1.35;margin:8px 0 16px}}h2{{font-size:21px;margin:0 0 8px}}
    .eyebrow{{color:#84d6cb;letter-spacing:3px;font-size:11px}}.muted,p,small{{color:var(--muted)}}p{{margin:8px 0 18px}}a{{color:#8adbd0;text-decoration:none}}a:hover{{text-decoration:underline}}
    .badges{{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0 24px}}.badge{{padding:4px 10px;background:#192b3d;border:1px solid var(--border);border-radius:5px;font-size:12px}}
    section{{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:22px;margin-top:24px;min-width:0}}.table-wrap{{overflow-x:auto;margin:16px 0 8px}}table{{border-collapse:collapse;width:100%;min-width:650px;font-variant-numeric:tabular-nums}}
    th,td{{padding:13px 16px;text-align:right;border-bottom:1px solid var(--border);white-space:nowrap}}th:first-child{{text-align:left}}thead th{{font-size:13px;background:#182a3d}}tbody th{{font-weight:550}}td small{{display:block;font-size:10px}}
    .positive{{color:#79d6c7}}.negative{{color:#ff9b91}}.missing{{color:#667e96}}.total{{font-weight:750}}tfoot th,tfoot td{{background:#1b3044}}
    #window-summary{{min-width:1050px}}#window-summary th:first-child{{position:sticky;left:0;z-index:1;background:var(--panel)}}#window-summary thead th:first-child{{background:#182a3d}}#window-summary th small{{display:block;color:var(--muted);font-size:11px;font-weight:400}}#window-summary .period{{color:var(--muted);font-size:13px}}#window-summary .period small{{font-size:12px}}#window-summary tbody tr:hover{{background:#1a2c40}}.scroll-hint{{display:none}}
    .notice{{border-left:3px solid #60bfb6;background:#102b34;padding:14px 18px;margin:22px 0;color:#c3dcdf;font-size:13px}}
    .actions{{display:flex;gap:18px;flex-wrap:wrap;font-size:13px}}.notes{{white-space:pre-line;font-size:13px;line-height:1.9}}footer{{padding:20px 0;color:var(--muted);font-size:12px}}
    @media(max-width:1200px){{.scroll-hint{{display:block;font-size:12px;margin-bottom:0}}}}@media(max-width:600px){{main{{padding:18px 12px}}h1{{font-size:25px}}section{{padding:16px 12px}}}}
    </style></head><body><header><b>α</b> AlphaTrack</header><main>
    <div class="eyebrow">INDEPENDENT STARTS / CALENDAR YEAR EARNINGS</div>
    <h1>从不同时间开始，每年能赚多少？</h1>
    <p>同一套 QLab 五类基金规则，比较 {html.escape('、'.join(w['label'] for w in windows))}。先看逐年模拟盈亏，再看投入规模和风险。</p>
    <div class="badges"><span class="badge">日期锚点 {anchor.date()}</span><span class="badge">共同净值截止 {end.date()}</span>
    <span class="badge">各区间独立空仓开始</span><span class="badge">无限资金 · 卖出款取出</span><span class="badge">金额单位：人民币元</span></div>
    <section><h2>回测区间总览</h2><p>每行对应一个独立回测区间。金额单位为元，累计盈亏包含未实现浮盈亏；年化收益分别列出 XIRR 与 TWR。</p>
    <div class="table-wrap" tabindex="0" role="region" aria-label="回测区间总览，可横向滚动"><table id="window-summary"><thead><tr>
    <th scope="col">回测区间</th><th scope="col">回测日期</th><th scope="col">累计投入（元）</th><th scope="col">累计盈亏（元）</th>
    <th scope="col">XIRR 年化</th><th scope="col">TWR 年化</th><th scope="col">最大净值回撤</th><th scope="col">完整报告</th>
    </tr></thead><tbody>{''.join(overview_rows)}</tbody></table></div><p class="scroll-hint">可左右滑动查看全部指标，左侧区间名称保持固定。</p></section>
    <div class="notice">“过去 N 年”按日期锚点倒推整年；固定起点区间从标明的日期开始。共同截止日受实际净值覆盖限制。各区间是独立回测，不能把不同区间的利润相加。</div>
    <section><h2>每个自然年的盈亏</h2><p>“—”表示不在该区间内；0.00 才表示该年盈亏为零。首尾不足一整年的单元格注明具体范围，未作全年外推。</p>
    <div class="table-wrap"><table id="annual-matrix"><thead><tr><th>自然年 / 盈亏（元）</th>{table_head}</tr></thead>
    <tbody>{''.join(body)}</tbody><tfoot><tr><th>各区间累计盈亏</th>{totals}</tr></tfoot></table></div>
    <div class="actions"><a href="{_relative_link(annual_csv,output.parent)}" download>下载逐年完整明细 CSV</a><a href="{_relative_link(summary_csv,output.parent)}" download>下载区间指标 CSV</a></div>
    {year_plot}</section>
    <section><h2>投入、回款与收益率一起看</h2><p>相同规则在不同起点会形成不同持仓。XIRR 计入资金投入、取出的金额和时间；TWR 净值年化剔除外部资金流的直接影响。</p>
    <div class="table-wrap"><table><thead><tr><th>指标</th>{table_head}</tr></thead><tbody>{''.join(metric_rows)}</tbody></table></div>
    <p>卖出取回包含本金，不能当成盈利。盈亏恒等式：期末持仓市值 + 累计取回 − 累计投入。XIRR 如无可报告的唯一数值则显示“—”；详细求解状态见各区间报告。</p></section>
    <section><h2>不同起点的累计盈亏路径</h2><p>横轴统一为日历日期。悬停查看日期和金额，单击图例隐藏某条曲线。窗口开始前不补零、不绘制持仓。</p>{pnl_plot}</section>
    <section><h2>计算口径与数据边界</h2><div class="notes">{html.escape(notes)}</div></section>
    <footer>策略来源：QLab · 固定版本 {html.escape(str(source_commit))} · 图表已内置，可离线打开；旁边的完整报告与 CSV 请一并保留。</footer>
    <script type="application/json" id="window-annual-data">{_json(matrix)}</script>
    </main></body></html>'''
    output.write_text(document, encoding="utf-8")
    return output
