"""Offline Plotly report for a portfolio with external contributions/withdrawals.

The annual earnings table is deliberately expressed in currency before returns.
Withdrawn sale proceeds are cash flows, not synonymous with investment profit.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


TEAL, GOLD, BLUE, RED = "#42dbc5", "#f4c875", "#8da9ff", "#ff9b91"
GROUP_COLORS = {"Fundb": TEAL, "Fundc": BLUE, "Fundd": GOLD,
                "Fundf_1": "#c2a3f4", "Fundg": "#f7a17d"}


def _frame(value):
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, dict):
        return pd.DataFrame.from_dict(value, orient="index").reset_index(names="asset_type")
    return pd.DataFrame(value)


def _pick(mapping, *keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _series(frame, *keys, default=0.0):
    for key in keys:
        if key in frame:
            return pd.to_numeric(frame[key], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _fmt(value, kind="money"):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    if kind == "percent":
        return f"{number:.2%}"
    if kind == "integer":
        return f"{number:,.0f}"
    return ("¥ " if kind == "money" else "") + f"{number:,.2f}"


def _json(value):
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple, np.ndarray)):
            return [clean(x) for x in v]
        if isinstance(v, (float, np.floating)):
            return float(v) if math.isfinite(v) else None
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (pd.Timestamp, np.datetime64)):
            return str(pd.Timestamp(v).date())
        if v is pd.NA or v is pd.NaT:
            return None
        return v
    return json.dumps(clean(value), ensure_ascii=False, allow_nan=False, default=str).replace(
        "<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _records(frame):
    data = frame.copy()
    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            data[col] = data[col].dt.strftime("%Y-%m-%d")
    return json.loads(data.to_json(orient="records", force_ascii=False, double_precision=15))


def _layout(fig, height=430):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif", color="#b4c7da", size=12),
        height=height, margin=dict(l=75, r=28, t=45, b=50), hovermode="x unified",
        hoverlabel=dict(bgcolor="#173148", font=dict(color="#f0f7fb")),
        legend=dict(orientation="h", y=1.08, x=0), dragmode="zoom")
    fig.update_xaxes(gridcolor="#1b3044", zerolinecolor="#344d65")
    fig.update_yaxes(gridcolor="#1b3044", zerolinecolor="#54718b")
    return fig


def _normalize_annual(annual):
    result = []
    for row in _records(_frame(annual)):
        year = _pick(row, "year", "calendar_year")
        result.append(dict(
            year=year, period=_pick(row, "period", "year_label", default=str(year)),
            start_date=_pick(row, "start_date"), end_date=_pick(row, "end_date"),
            is_partial=_pick(row, "is_partial", "is_partial_year", "partial_year", default=False),
            beginning_assets=_pick(row, "beginning_assets", "start_equity", "beginning_equity", "opening_equity"),
            contributions=_pick(row, "contributions", "deposits", "total_contributions", "external_flow"),
            withdrawals=_pick(row, "withdrawals", "total_withdrawals", "withdrawal"),
            profit=_pick(row, "profit", "pnl", "total_pnl", "annual_pnl"),
            ending_assets=_pick(row, "ending_assets", "end_assets", "end_equity", "ending_equity", "final_equity"),
            twr_return=_pick(row, "twr_return", "twr", "total_return", "return"),
            xirr=_pick(row, "xirr", "annual_xirr"),
            realized_pnl=_pick(row, "realized_pnl", "realized_change"),
            unrealized_change=_pick(row, "unrealized_change", "unrealized_pnl_change"),
            max_drawdown=_pick(row, "max_drawdown"),
            fees=_pick(row, "fees", "total_fees"),
            xirr_status=_pick(row, "xirr_status", default=""),
        ))
    return result


def _normalize_members(frame):
    result = []
    for row in _records(_frame(frame)):
        result.append(dict(
            code=str(_pick(row, "code", "fund_code", default="")),
            name=_pick(row, "name", "fund_name", default=""),
            asset_type=_pick(row, "asset_type", "group", "type", default=""),
            start_date=_pick(row, "first_nav_date", "first_date", "start_date", default=""),
            end_date=_pick(row, "end_date", "last_date", "last_nav_date", default=""),
            first_buy_date=_pick(row, "first_buy_date", "first_investment_date", default=""),
            contributions=_pick(row, "contributions", "total_contributions", "deposits", "total_buys"),
            withdrawals=_pick(row, "withdrawals", "cumulative_withdrawals", "total_withdrawals"),
            ending_assets=_pick(row, "ending_assets", "end_assets", "final_equity", "equity", "position_value"),
            total_pnl=_pick(row, "total_pnl", "profit", "pnl"),
            realized_pnl=_pick(row, "realized_pnl"),
            unrealized_pnl=_pick(row, "unrealized_pnl"),
            xirr=_pick(row, "xirr"),
            buy_count=_pick(row, "buy_count", "buys"),
            sell_count=_pick(row, "sell_count", "sells"),
            note=_pick(row, "note", "coverage_note", default=""),
        ))
    return result


def render_portfolio_report(daily, annual, funds, groups, summary, metadata, output):
    """Write the report and return its path.

    ``groups`` may be a summary DataFrame/records, or ``{"summary": ..., "daily":
    {"Fundb": daily_frame, ...}}``. Dates and all monetary values use CNY.
    ``metadata.assumptions`` and ``metadata.warnings`` may provide report notes.
    """
    frame = _frame(daily)
    if frame.empty or "date" not in frame:
        raise ValueError("Portfolio report needs a nonempty dated daily ledger")
    frame["date"] = pd.to_datetime(frame.date)
    frame = frame.sort_values("date")
    dates = frame.date.dt.strftime("%Y-%m-%d").tolist()
    metadata = dict(metadata or {})
    summary = dict(summary or {})
    annual_rows = _normalize_annual(annual)
    fund_rows = _normalize_members(funds)
    group_daily = {}
    if isinstance(groups, dict) and "summary" in groups:
        group_daily = groups.get("daily", {})
        group_rows = _normalize_members(groups["summary"])
    else:
        group_rows = _normalize_members(groups)
    if annual_rows and frame.date.iloc[-1].month < 12:
        last_year = int(frame.date.iloc[-1].year)
        for row in annual_rows:
            if str(row["year"]) == str(last_year):
                row["is_partial"] = True
    requested_start = pd.Timestamp(metadata.get("start_date", frame.date.iloc[0]))
    for row in annual_rows:
        year = int(row['year'])
        left = pd.Timestamp(row['start_date']) if row['start_date'] else max(requested_start, pd.Timestamp(year, 1, 1))
        right = pd.Timestamp(row['end_date']) if row['end_date'] else min(frame.date.iloc[-1], pd.Timestamp(year, 12, 31))
        row['period'] = str(year)
        if row['is_partial']:
            row['period'] += (' YTD' if left.month == 1 and left.day == 1 else
                              f'（{left:%m/%d}–{right:%m/%d}）')

    exported = []
    def export(fig, ident):
        fragment = fig.to_html(full_html=False, include_plotlyjs=not exported, div_id=ident,
                               config={"responsive": True, "displaylogo": False,
                                       "toImageButtonOptions": {"format": "png", "scale": 2}})
        exported.append(ident)
        return fragment

    yearly = go.Figure()
    year_labels = [r['period'] for r in annual_rows]
    profits = [r["profit"] for r in annual_rows]
    yearly.add_trace(go.Bar(x=year_labels, y=profits, name="年度盈亏",
                            marker_color=[TEAL if (p or 0) >= 0 else RED for p in profits],
                            customdata=[[r["contributions"], r["withdrawals"], r["ending_assets"], r["twr_return"]] for r in annual_rows],
                            hovertemplate="%{x}<br><b>当年盈亏 ¥%{y:,.2f}</b><br>投入 ¥%{customdata[0]:,.2f}"
                            "<br>取出 ¥%{customdata[1]:,.2f}<br>年末资产 ¥%{customdata[2]:,.2f}"
                            "<br>区间净值收益 %{customdata[3]:.2%}<extra></extra>"))
    _layout(yearly, 370)
    yearly.update_layout(hovermode="closest")
    yearly.update_yaxes(title="当年盈利 / 亏损（元）")
    yearly.update_xaxes(type="category")
    year_plot = export(yearly, "portfolio-annual-chart")

    cumulative = _series(frame, "cumulative_pnl", "total_pnl", default=np.nan)
    if cumulative.isna().all():
        cumulative = _series(frame, "daily_pnl").cumsum()
    pnl = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.14,
                        row_heights=[.65, .35])
    pnl.add_trace(go.Scatter(x=dates, y=cumulative, name="组合累计盈亏", mode="lines",
                            line=dict(color="#ecf4fb", width=2.5),
                            hovertemplate="组合累计盈亏 ¥%{y:,.2f}<extra></extra>"), row=1, col=1)
    for group, color in GROUP_COLORS.items():
        g = _frame(group_daily.get(group)) if isinstance(group_daily, dict) else pd.DataFrame()
        if not g.empty and "date" in g:
            ys = _series(g, "cumulative_pnl", "total_pnl", default=np.nan)
            if ys.isna().all():
                ys = _series(g, "daily_pnl").cumsum()
            xs = pd.to_datetime(g.date).dt.strftime("%Y-%m-%d").tolist()
        elif f"pnl_{group}" in frame:
            ys, xs = frame[f"pnl_{group}"], dates
        else:
            continue
        pnl.add_trace(go.Scatter(x=xs, y=ys, name=group, mode="lines", line=dict(color=color, width=1.5),
                                hovertemplate=group + " 累计盈亏 ¥%{y:,.2f}<extra></extra>"), row=1, col=1)
    pnl.add_trace(go.Scatter(x=dates, y=_series(frame, "daily_pnl"), name="组合当日盈亏", mode="lines",
                            line=dict(color=BLUE, width=1), hovertemplate="当日盈亏 ¥%{y:,.2f}<extra></extra>"), row=2, col=1)
    _layout(pnl, 550)
    pnl.update_yaxes(title="累计盈亏 / 元", row=1, col=1)
    pnl.update_yaxes(title="当日盈亏 / 元", row=2, col=1)
    pnl.update_xaxes(type="date", hoverformat="%Y-%m-%d")
    pnl_plot = export(pnl, "portfolio-pnl-chart")

    nav = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.13, row_heights=[.65, .35])
    nav.add_trace(go.Scatter(x=dates, y=_series(frame, "unit_nav", "unit_net_value", default=np.nan),
                            name="组合单位净值", line=dict(color=TEAL, width=2),
                            hovertemplate="单位净值 %{y:.6f}<extra></extra>"), row=1, col=1)
    drawdown = _series(frame, "drawdown", default=np.nan)
    # A positive-magnitude input still displays downwards from the peak.
    drawdown = -drawdown.abs()
    nav.add_trace(go.Scatter(x=dates, y=drawdown, name="回撤", fill="tozeroy",
                            line=dict(color=RED, width=1), hovertemplate="回撤 %{y:.2%}<extra></extra>"), row=2, col=1)
    _layout(nav, 490)
    nav.update_yaxes(title="净值", row=1, col=1)
    nav.update_yaxes(title="距历史高点", tickformat=".0%", row=2, col=1)
    nav.update_xaxes(type="date", hoverformat="%Y-%m-%d")
    nav_plot = export(nav, "portfolio-nav-chart")

    funding = go.Figure()
    for keys, label, color in [(("equity", "total_value"), "账户剩余资产", GOLD),
                               (("total_contributions", "cumulative_contributions", "net_contributions"), "累计投入", BLUE),
                               (("cumulative_withdrawals", "total_withdrawals"), "累计取出", TEAL)]:
        funding.add_trace(go.Scatter(x=dates, y=_series(frame, *keys), name=label, mode="lines",
                                    line=dict(color=color, width=1.8),
                                    hovertemplate=label + " ¥%{y:,.2f}<extra></extra>"))
    _layout(funding, 370)
    funding.update_yaxes(title="金额 / 元")
    funding.update_xaxes(type="date", hoverformat="%Y-%m-%d")
    funding_plot = export(funding, "portfolio-funding-chart")

    metrics = [
        ("累计盈亏", ("total_pnl", "profit"), "money", "已卖出收益 + 当前持仓浮动收益"),
        ("资金加权年化 · XIRR", ("xirr",), "percent", "按实际投入、取出与期末资产计算；不是逐笔年化均值"),
        ("净值年化 · TWR", ("cagr", "annualized_return", "twr_annualized"), "percent", "剔除外部资金流；从首次投入日期起按实际天数年化"),
        ("最大回撤", ("max_drawdown",), "percent", "组合单位净值相对历史峰值的最大跌幅"),
        ("累计投入", ("total_contributions", "contributions", "total_deposits"), "money", "各次买入所需新增资金的累计金额"),
        ("累计取出", ("cumulative_withdrawals", "withdrawals", "total_withdrawals"), "money", "卖出净回款，包含收回本金，不能视作利润"),
        ("期末剩余资产", ("final_equity", "ending_assets", "equity"), "money", "未假设期末全部卖出；按最近可用净值估值"),
        ("夏普比率", ("sharpe",), "number", "依据组合日净值收益；无风险利率与年化天数见口径"),
    ]
    cards = []
    for title, keys, kind, note in metrics:
        value = _pick(summary, *keys)
        if title == "最大回撤" and value is not None:
            value = -abs(float(value))
        cards.append(f'<div class="stat"><div class="stat-label">{title}</div><div class="stat-value">{_fmt(value, kind)}</div><p>{note}</p></div>')
    audit_metrics = [
        ("最大账户资产", ("max_equity",), "money"),
        ("最大净投入", ("max_net_contributions",), "money"),
        ("年化波动率", ("annual_volatility",), "percent"),
        ("累计交易费用", ("total_fees",), "money"),
        ("买入次数", ("buy_count",), "integer"),
        ("卖出次数", ("sell_count",), "integer"),
    ]
    audit_html = "".join(f'<div>{label}<b>{_fmt(_pick(summary, *keys), kind)}</b></div>'
                         for label, keys, kind in audit_metrics)

    assumptions = metadata.get("assumptions", [])
    if isinstance(assumptions, dict):
        assumptions = [f"{k}：{v}" for k, v in assumptions.items()]
    if isinstance(assumptions, str):
        assumptions = [assumptions]
    warnings = metadata.get("warnings", [])
    if isinstance(warnings, str):
        warnings = [warnings]
    selection_note = (
        "Fundb、Fundc 每日随机选8/5只，沿用原合格候选池与新净值下单限制"
        if metadata.get("strategy", {}).get("selection_mode") == "random"
        else "Fundb、Fundc 按原综合评分排名买入"
    )
    default_notes = [
        f"本次五类规则中，{selection_note}；Fundf_1、Fundg 按历史位置买入；只有 Fundd 配置明确卖出规则。没有擅自给其他类别增加止盈、年度清仓或定期卖出。",
        "基金范围采用本次 QLab 名单并回溯历史，不能视为回测起点当时已知的选基结果；这会带来事后选择和幸存者偏差。",
        "回测起点之前的数据只用于计算历史指标，期初没有回放持仓。成立较晚的基金在有实际数据后参与；不向成立前回填净值。",
        "组合每日估值使用各基金截至当日的最近已知净值。某基金缺少当日净值时只延用估值，不因此重复生成该基金交易；境外基金估值还存在公布延迟。",
        "每年盈亏 = 年末资产 − 年初资产 − 当年投入 + 当年取出。年收益金额包括当年持仓浮盈亏变化，卖出回款不等于已实现利润。",
        "未实现盈亏年度变化可能为负，例如持仓卖出后浮盈转为已实现收益；应与已实现盈亏及总盈亏一起阅读。",
        "本页年度与类别表中的已实现/未实现盈亏按账户加权平均成本分解。另附 lots.csv 使用 FIFO 逐批匹配，两种分解的总盈亏一致，分项不可混用。",
        "年度净值收益是该年区间收益；XIRR 是年化资金收益。未结束年份标记 YTD，不把其盈利金额外推为全年。",
    ]
    notes_html = "".join(f"<li>{html.escape(str(note))}</li>" for note in [*default_notes, *assumptions, *warnings])
    data_notes = [str(note) for note in assumptions if "断档" in str(note)]
    for item in metadata.get("missing_corporate_action_nav", []):
        data_notes.append(f"{item.get('code', '')} 有 {len(item.get('events', []))} 条分红事件缺少当日净值，沿用 QLab 已有处理，未补造净值。")
    data_notes.extend(str(note) for note in warnings)
    data_notice = ('<div class="note data-note"><strong>本次数据边界：</strong>'
                   + '<br>'.join(html.escape(note) for note in dict.fromkeys(data_notes)) + '</div>') if data_notes else ''
    qlab_commit = str(_pick(metadata, "qlab_commit", "source_commit", default="详见数据来源"))
    title = str(metadata.get("title", "QLab 五类基金组合 · 年度收益回测"))
    xirr_note = {
        "unique": "XIRR 已找到唯一解。",
        "unique_detected": "完整现金流在数值搜索范围内找到一个 XIRR 解；含多次投入和取出时，不能据此证明不存在其他解。",
        "multiple_roots": "完整现金流存在多个 XIRR 解，因此不展示单一资金年化值；具体解见指标记录。",
        "undefined": "完整现金流不满足 XIRR 计算条件，因此显示 —。",
        "no_root": "数值搜索范围内未找到 XIRR 解，因此显示 —。",
    }.get(str(summary.get("xirr_status", "")), "XIRR 无法定义或存在多个解时显示 —，详情见指标记录。")
    # Large DataFrames used to draw group curves do not belong in provenance text.
    provenance = {k: v for k, v in metadata.items() if k not in {"group_daily"}}
    start = str(metadata.get("requested_start", metadata.get("start_date", dates[0])))[:10]
    replacements = {
        "TITLE": html.escape(title), "START": html.escape(start), "END": dates[-1],
        "FUND_COUNT": str(len(fund_rows)), "CARDS": "".join(cards), "AUDIT": audit_html,
        "YEAR_PLOT": year_plot, "PNL_PLOT": pnl_plot, "NAV_PLOT": nav_plot,
        "FUNDING_PLOT": funding_plot, "ANNUAL_JSON": _json(annual_rows),
        "FUND_JSON": _json(fund_rows), "GROUP_JSON": _json(group_rows),
        "NOTES": notes_html, "DATA_NOTICE": data_notice, "COMMIT": html.escape(qlab_commit),
        "PROVENANCE": html.escape(json.dumps(provenance, ensure_ascii=False, indent=2, default=str)),
        "SUMMARY": html.escape(json.dumps(summary, ensure_ascii=False, indent=2, default=str)),
        "XIRR_NOTE": html.escape(xirr_note),
    }
    document = _TEMPLATE
    for key, value in replacements.items():
        document = document.replace("{{" + key + "}}", value)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{{TITLE}} | AlphaTrack</title>
<style>
:root{color-scheme:dark;--bg:#091523;--panel:#102035;--border:#263c52;--text:#eef5fd;--muted:#91a7be;--teal:#42dbc5;--gold:#f4c875}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.65 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}header{border-bottom:1px solid var(--border);background:#0d1d30}.header-inner{max-width:1520px;margin:auto;padding:19px 32px;display:flex;justify-content:space-between;gap:20px}.brand{font-size:20px;font-weight:700}.brand b{color:var(--teal)}nav{display:flex;gap:24px}a{color:#bdd1e5;text-decoration:none}a:hover{color:var(--teal)}main{max-width:1520px;margin:auto;padding:34px 32px 45px}.eyebrow{font-size:11px;color:var(--teal);letter-spacing:3px}h1{font-size:31px;line-height:1.35;margin:12px 0}h2{font-size:18px;margin:0}.sub,.muted{color:var(--muted)}.meta{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0 24px}.pill{border:1px solid var(--border);border-radius:5px;padding:4px 9px;font-size:12px;background:#102337;color:#b7cadd}.pill.green{color:var(--teal);border-color:#265e58}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{padding:18px 19px;border:1px solid var(--border);border-radius:10px;background:linear-gradient(135deg,#14293f,#102035);min-height:143px}.stat-label{font-size:12px;color:#afc5dd}.stat-value{font-size:26px;letter-spacing:-.5px;font-variant-numeric:tabular-nums;margin-top:8px}.stat p{font-size:11px;color:var(--muted);margin:8px 0 0}.note{background:#103039;border-left:2px solid #398c82;padding:12px 15px;color:#b1d4d7;font-size:12px;margin:20px 0}.note strong{color:#e7f9f8}.note.data-note{background:#302a20;border-left-color:#d6ac65;color:#dcc9a9}.panel{background:var(--panel);border:1px solid var(--border);padding:23px 22px 13px;border-radius:10px;margin:22px 0;overflow:hidden}.panel-head{display:flex;justify-content:space-between;gap:20px;align-items:start}.panel-head p{font-size:12px;color:var(--muted);margin:7px 0 12px;max-width:1120px}.panel-index{font:12px monospace;color:#5e7893;padding-top:4px;white-space:nowrap}.plotly-graph-div{width:100%}.table-controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0}input,select,button{font:inherit;color:#d4e5f6;background:#173149;border:1px solid #3a5269;border-radius:5px;padding:8px 11px;max-width:100%}input{width:300px}button{cursor:pointer}button:hover:not(:disabled){border-color:var(--teal)}button:disabled{opacity:.4;cursor:default}.table-wrap{overflow:auto;max-height:720px}table{width:100%;border-collapse:collapse;font-size:12px}th{padding:11px;background:#172f47;text-align:right;color:#adc3db;font-weight:500;white-space:nowrap;position:sticky;top:0;z-index:1}td{padding:11px;border-bottom:1px solid #243a50;color:#bfd0e2;white-space:nowrap;font-variant-numeric:tabular-nums;text-align:right}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:#14283d}th:first-child{z-index:2;background:#172f47}td.positive{color:var(--teal)}td.negative{color:#ff9b91}.primary{font-weight:700;font-size:13px}td.text{text-align:left}td.name{max-width:250px;overflow:hidden;text-overflow:ellipsis;text-align:left}td.note-cell{white-space:normal;min-width:200px;max-width:400px;text-align:left;font-size:11px}tr.partial td:first-child{color:var(--gold)}.table-note{font-size:11px;color:var(--muted);margin:12px 0 4px}.pagination{display:flex;justify-content:space-between;gap:12px;align-items:center;margin:15px 0 7px;font-size:12px;color:var(--muted)}.pagination div{display:flex;gap:9px;align-items:center}.formula{font:13px ui-monospace,SFMono-Regular,monospace;background:#0c1a2a;color:#d6e9f7;border:1px solid #243f58;padding:14px;overflow:auto;margin:12px 0}ul.notes{font-size:12px;color:#abc0d5;padding-left:22px}ul.notes li{padding:5px 0}details{border-top:1px solid var(--border);padding:15px 0}summary{cursor:pointer;color:#bbd1e6;font-size:12px}pre{font:11px/1.5 ui-monospace,SFMono-Regular,monospace;overflow:auto;max-height:380px;background:#0c1a2a;padding:15px;color:#b4cbe0}footer{font-size:11px;color:#7891ab;display:flex;justify-content:space-between;gap:20px}.audit{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:12px 0}.audit div{font-size:12px;color:var(--muted);border-bottom:1px solid var(--border);padding:10px 0}.audit b{display:block;color:#d6e6f5;margin-top:4px;font-size:16px;font-weight:500}.split-panels{display:grid;grid-template-columns:1fr 1fr;gap:20px}.split-panels>.panel{min-width:0}@media(max-width:1150px){.stats{grid-template-columns:repeat(2,1fr)}.split-panels{display:block}}@media(max-width:750px){.audit{grid-template-columns:repeat(2,1fr)}nav{display:none}.header-inner{padding:16px 18px}main{padding:25px 14px}.stat{padding:15px}.stat-value{font-size:23px}.panel{padding:18px 10px 10px}h1{font-size:25px}.panel-index{display:none}.pagination{align-items:start;flex-direction:column}footer{flex-direction:column;gap:5px}.table-controls{gap:7px}input{width:100%}.panel-head{padding:0 3px}}@media(max-width:380px){.stat-value{font-size:20px}.stats{gap:8px}.stat{padding:12px}}
</style></head><body>
<header><div class="header-inner"><div class="brand"><b>α</b> AlphaTrack</div><nav><a href="#annual">每年赚多少</a><a href="#pnl">盈亏曲线</a><a href="#members">基金明细</a><a href="#method">回测口径</a></nav></div></header>
<main><div class="eyebrow">PORTFOLIO / CALENDAR YEAR EARNINGS</div><h1>{{TITLE}}</h1><p class="sub">按 QLab 各类别规则逐日判断，汇总 {{FUND_COUNT}} 只基金。先看每年的模拟盈亏金额（含未实现），再看资金使用与风险。</p>
<div class="meta"><span class="pill green">Fundb · Fundc · Fundd · Fundf_1 · Fundg</span><span class="pill">{{START}} → {{END}}</span><span class="pill">无资金上限 · 按需投入</span><span class="pill">卖出款取出</span><span class="pill">人民币 · CNY</span></div>
<section class="stats">{{CARDS}}</section>{{DATA_NOTICE}}
<div class="note"><strong>年度收益怎么看：</strong>年度盈亏包含该年已实现收益与持仓浮盈亏的变化。<strong>投入不算盈利，卖出取回的本金也不算盈利。</strong>期末持仓没有强制卖出。XIRR 与净值年化衡量不同问题，不能与逐批年化均值混用。</div>
<section class="panel" id="annual"><div class="panel-head"><div><h2>每年模拟盈利 / 亏损多少</h2><p>每根柱子对应一个自然年内的盈亏金额。年中开始的首年注明月日范围；YTD 是当年年初至截止日，均未外推为全年。</p></div><span class="panel-index">01 / YEARLY EARNINGS</span></div>{{YEAR_PLOT}}
<div class="table-controls"><button id="export-years">导出年度明细 CSV</button><span class="muted" style="font-size:12px">金额单位：元；表格可横向滚动</span></div><div class="table-wrap"><table><thead id="annual-head"></thead><tbody id="annual-body"></tbody></table></div><p class="table-note">已实现/未实现分解采用加权平均成本；“当年未实现变化”指期末浮盈亏减期初浮盈亏，并非期末浮盈亏余额。年度盈亏与净值收益可能方向不同：金额盈亏受投入规模和时点影响，净值收益按各阶段收益连乘。年度 XIRR 把年初持仓市值视为期初投入，因此和整个组合的 XIRR 不可直接平均。{{XIRR_NOTE}}</p></section>
<section class="panel" id="pnl"><div class="panel-head"><div><h2>组合与各类别 · 累计盈亏和每日盈亏</h2><p>累计盈亏包含已实现与未实现部分。悬停可查看具体日期和金额，单击图例可隐藏某个类别，双击图例可单独查看。</p></div><span class="panel-index">02 / P&amp;L</span></div>{{PNL_PLOT}}</section>
<section class="panel"><div class="panel-head"><div><h2>剔除资金流后的单位净值与回撤</h2><p>追加资金不会凭空增加收益，取出资金不会形成亏损。这里评价整个组合，不是对各基金收益率做简单平均，也不是逐笔交易年化曲线。</p></div><span class="panel-index">03 / RETURN &amp; RISK</span></div>{{NAV_PLOT}}</section>
<section class="panel"><div class="panel-head"><div><h2>投入、取出与剩余资产</h2><p>累计取出包含回收本金。总盈亏可以由「剩余资产 + 累计取出 − 累计投入」复核；无资金上限并不代表没有资金占用。</p></div><span class="panel-index">04 / CAPITAL</span></div><div class="audit">{{AUDIT}}</div>{{FUNDING_PLOT}}</section>
<section class="panel"><div class="panel-head"><div><h2>五类策略分别贡献多少</h2><p>各类别保留各自买卖规则和金额设置，未将 Fundd 的卖出规则套用到其他类别。类别之间的 XIRR 不能按行简单平均。</p></div><span class="panel-index">05 / GROUPS</span></div><div class="table-wrap"><table><thead id="group-head"></thead><tbody id="group-body"></tbody></table></div></section>
<section class="panel" id="members"><div class="panel-head"><div><h2>基金明细与数据覆盖</h2><p>每只基金按实际可用数据参与。净值开始日期不等于第一次买入日期；较晚成立的基金不会出现在早期持仓中。支持按类别筛选、盈亏排序和 CSV 导出。</p></div><span class="panel-index">06 / HOLDINGS</span></div>
<div class="table-controls"><input id="fund-search" aria-label="搜索基金" placeholder="搜索代码 / 基金名称…"><select id="fund-group" aria-label="基金类别"><option value="">全部类别</option><option>Fundb</option><option>Fundc</option><option>Fundd</option><option>Fundf_1</option><option>Fundg</option></select><select id="fund-sort" aria-label="基金排序"><option value="group">类别 / 代码</option><option value="profit_desc">累计盈亏 ↓</option><option value="profit_asc">累计盈亏 ↑</option><option value="contributions">累计投入 ↓</option></select><button id="export-funds">导出筛选 CSV</button></div>
<div class="table-wrap"><table><thead id="fund-head"></thead><tbody id="fund-body"></tbody></table></div><div class="pagination"><span id="fund-count"></span><div><button id="prev-page">上一页</button><span id="page-label"></span><button id="next-page">下一页</button></div></div></section>
<section class="panel" id="method"><div class="panel-head"><div><h2>口径、假设与可复核记录</h2><p>QLab 规则版本：<code>{{COMMIT}}</code>。该报告是历史规则模拟结果，数据覆盖和交易假设都会影响结果。</p></div><span class="panel-index">07 / AUDIT</span></div><div class="formula">当年盈亏 = 年末资产 − 年初资产 − 当年投入 + 当年取出<br>累计盈亏 = 期末资产 + 累计取出 − 累计投入</div><ul class="notes">{{NOTES}}</ul><details><summary>展开完整指标及 XIRR 求解状态</summary><pre>{{SUMMARY}}</pre></details><details><summary>展开规则、数据来源与执行假设</summary><pre>{{PROVENANCE}}</pre></details></section>
<footer><span>AlphaTrack · Calendar-year portfolio analysis</span><span>数据截至 {{END}} · Plotly 单文件离线报告</span></footer></main>
<script>
const annualRows={{ANNUAL_JSON}},allFunds={{FUND_JSON}},groupRows={{GROUP_JSON}};
const el=id=>document.getElementById(id),valid=v=>v!==null&&v!==undefined&&v!==''&&Number.isFinite(Number(v));
const n=(v,d=2)=>valid(v)?Number(v).toLocaleString('zh-CN',{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
const percentKeys=new Set(['twr_return','xirr','max_drawdown']),moneyKeys=new Set(['beginning_assets','contributions','withdrawals','profit','ending_assets','realized_pnl','unrealized_change','fees','total_pnl','unrealized_pnl']);
const annualCols=[['year','年份'],['profit','当年盈亏'],['twr_return','年度净值收益'],['beginning_assets','年初资产'],['contributions','当年投入'],['withdrawals','当年取出'],['ending_assets','年末资产'],['realized_pnl','当年已实现盈亏'],['unrealized_change','当年未实现变化'],['xirr','年度资金 XIRR'],['max_drawdown','年内最大回撤'],['fees','交易费用']];
const groupCols=[['asset_type','类别'],['total_pnl','累计盈亏'],['contributions','累计投入'],['withdrawals','累计取出'],['ending_assets','期末资产'],['realized_pnl','已实现盈亏'],['unrealized_pnl','未实现盈亏'],['xirr','资金 XIRR'],['buy_count','买入次数'],['sell_count','卖出次数']];
const fundCols=[['code','基金代码'],['name','基金名称'],['asset_type','类别'],['total_pnl','累计盈亏'],['contributions','累计投入'],['withdrawals','累计取出'],['ending_assets','期末资产'],['xirr','资金 XIRR'],['buy_count','买入次数'],['sell_count','卖出次数'],['start_date','数据开始'],['first_buy_date','首次买入'],['end_date','最后净值'],['note','覆盖说明']];
function show(v,k,row){if(k==='year')return row.period||String(v);if(percentKeys.has(k))return valid(v)?n(Number(v)*(k==='max_drawdown'?-Math.sign(Number(v)||1):1)*100)+'%':'—';if(moneyKeys.has(k))return n(v);if(k.endsWith('_count'))return n(v,0);return v===null||v===undefined||v===''?'—':String(v);}
function table(name,rows,cols){const trh=document.createElement('tr');for(const [key,label] of cols){const th=document.createElement('th');th.textContent=label;trh.appendChild(th);}el(name+'-head').replaceChildren(trh);const body=el(name+'-body');body.replaceChildren();for(const row of rows){const tr=document.createElement('tr');if(row.is_partial)tr.classList.add('partial');for(const [key] of cols){const td=document.createElement('td');td.textContent=show(row[key],key,row);if(['profit','total_pnl'].includes(key)){td.classList.add('primary');if(valid(row[key]))td.classList.add(Number(row[key])>=0?'positive':'negative');}if(key==='name'){td.classList.add('name');td.title=String(row[key]||'');}if(['asset_type','start_date','end_date','first_buy_date'].includes(key))td.classList.add('text');if(key==='note')td.classList.add('note-cell');tr.appendChild(td);}body.appendChild(tr);}if(!rows.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=cols.length;td.textContent='暂无符合条件的记录';tr.appendChild(td);body.appendChild(tr);}}
function exportCSV(rows,cols,name){const csv=v=>'"'+String(v??'').replaceAll('"','""')+'"';const header=cols.map(([k,label])=>csv(label+(percentKeys.has(k)?'（小数）':''))).join(',');const lines=[header,...rows.map(r=>cols.map(([k])=>csv(k==='year'?show(r[k],k,r):r[k])).join(','))];const url=URL.createObjectURL(new Blob(['\ufeff'+lines.join('\r\n')],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
let currentPage=0,filtered=[];const pageSize=20;
function renderFunds(){const query=el('fund-search').value.trim().toLowerCase(),group=el('fund-group').value,sort=el('fund-sort').value;filtered=allFunds.filter(r=>(!group||r.asset_type===group)&&(!query||(r.code+' '+r.name).toLowerCase().includes(query)));filtered.sort((a,b)=>{if(sort.startsWith('profit'))return ((a.total_pnl||0)-(b.total_pnl||0))*(sort==='profit_asc'?1:-1);if(sort==='contributions')return (b.contributions||0)-(a.contributions||0);return String(a.asset_type).localeCompare(String(b.asset_type))||String(a.code).localeCompare(String(b.code));});const pages=Math.max(1,Math.ceil(filtered.length/pageSize));currentPage=Math.min(currentPage,pages-1);table('fund',filtered.slice(currentPage*pageSize,(currentPage+1)*pageSize),fundCols);el('fund-count').textContent=`共 ${filtered.length} 只基金 · 每页 ${pageSize} 条`;el('page-label').textContent=`${currentPage+1} / ${pages}`;el('prev-page').disabled=currentPage===0;el('next-page').disabled=currentPage===pages-1;}
for(const id of ['fund-search','fund-group','fund-sort'])el(id).addEventListener(id==='fund-search'?'input':'change',()=>{currentPage=0;renderFunds();});el('prev-page').onclick=()=>{currentPage--;renderFunds();};el('next-page').onclick=()=>{currentPage++;renderFunds();};el('export-funds').onclick=()=>exportCSV(filtered,fundCols,'alphatrack-portfolio-funds.csv');el('export-years').onclick=()=>exportCSV(annualRows,annualCols,'alphatrack-portfolio-annual.csv');table('annual',annualRows,annualCols);table('group',groupRows,groupCols);renderFunds();
</script></body></html>'''
