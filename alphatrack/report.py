"""Portable Plotly research report; presentation is independent of the engine."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = {"teal": "#42dbc5", "blue": "#8da9ff", "red": "#ff9b91", "gold": "#f4c875", "muted": "#8194ad"}


def _number(value, kind="number"):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(value):
        return "—"
    if kind == "percent":
        return f"{value:.2%}"
    if kind == "money":
        return f"¥ {value:,.2f}"
    if kind == "integer":
        return f"{value:,.0f}"
    return f"{value:,.3f}"


def _series(frame, column, default=0.0):
    return frame[column] if column in frame else pd.Series(default, index=frame.index)


def _safe_json(value):
    # Script elements must not be terminated by names or rule descriptions.
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=str).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _layout(fig, height=400, *, date_axes=True, margin_bottom=45):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=list(COLORS.values()), height=height,
        margin=dict(l=65, r=28, t=32, b=margin_bottom),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", color="#bccbdc", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.025, x=0, font=dict(size=12)),
        hovermode="x unified", hoverlabel=dict(bgcolor="#17283b", bordercolor="#344a64", font=dict(color="#f4f7fc")),
        dragmode="zoom", uirevision="alphatrack", showlegend=True,
    )
    fig.update_yaxes(gridcolor="#1e3045", zerolinecolor="#41516a", tickfont=dict(size=11))
    fig.update_xaxes(gridcolor="#16283c", tickfont=dict(size=11))
    if date_axes:
        fig.update_xaxes(type="date", hoverformat="%Y-%m-%d", tickformat="%Y-%m")
    return fig


def write_report(daily: pd.DataFrame, trades: pd.DataFrame, metrics: dict,
                 config: dict, metadata: dict, output: Path) -> None:
    """Write one self-contained, offline HTML; no server or CDN is required."""
    if daily.empty:
        raise ValueError("Cannot report an empty backtest")
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    dates = frame.date.dt.strftime("%Y-%m-%d").tolist()
    eng = config.get("engine", {})
    topup = eng.get("funding_mode", metrics.get("funding_mode", "topup")) == "topup"
    title = str(config.get("title", "110020 · Fundd 历史回测"))
    start, end = dates[0], dates[-1]
    first_funded = str(metrics.get("first_investment_date") or "—")[:10]
    plots = []

    def plot(fig, identifier):
        content = fig.to_html(full_html=False, include_plotlyjs=True if not plots else False,
                              div_id=identifier, config={"responsive": True, "displaylogo": False,
                                                        "scrollZoom": False, "toImageButtonOptions": {"format": "png", "scale": 2}})
        plots.append(identifier)
        return content

    common = np.column_stack((_series(frame, "daily_pnl"), _series(frame, "daily_return"),
                              _series(frame, "cumulative_return"), _series(frame, "equity")))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=frame.cumulative_return, name="策略 · 时间加权收益（TWR）", mode="lines",
                             line=dict(color=COLORS["teal"], width=2), customdata=common,
                             hovertemplate="累计 TWR %{y:.2%}<br>当日盈亏 ¥%{customdata[0]:,.2f}<br>当日收益率 %{customdata[1]:.3%}<br>账户资产 ¥%{customdata[3]:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=dates, y=frame.benchmark_return, name="参考 · 回测起点买入持有", mode="lines",
                             line=dict(color=COLORS["blue"], width=1.5), hovertemplate="基金累计收益 %{y:.2%}<extra></extra>"))
    _layout(fig, 440, margin_bottom=65)
    fig.update_yaxes(tickformat=".0%", title="累计收益率")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=.07, bgcolor="#14263a"), rangeselector=dict(
        buttons=[dict(count=1, label="1 年", step="year", stepmode="backward"),
                 dict(count=3, label="3 年", step="year", stepmode="backward"),
                 dict(count=5, label="5 年", step="year", stepmode="backward"), dict(label="全部", step="all")],
        bgcolor="#213750", activecolor="#41736f", font=dict(color="#dce8f5"), x=1, xanchor="right", y=1.05))
    returns_plot = plot(fig, "returns-chart")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12, row_heights=[.62,.38])
    fig.add_trace(go.Scatter(x=dates, y=frame.equity, name="账户总资产", line=dict(color=COLORS["teal"], width=2),
                             hovertemplate="账户资产 ¥%{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=_series(frame, "net_contributions", metrics.get("initial_cash", 0)),
                             name="累计外部本金", line=dict(color=COLORS["gold"], width=1.5, dash="dot"),
                             hovertemplate="累计外部本金 ¥%{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=frame.position_value, name="持仓市值", line=dict(color=COLORS["blue"], width=1),
                             stackgroup="capital", hovertemplate="持仓市值 ¥%{y:,.2f}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=frame.cash, name="账户现金", line=dict(color=COLORS["muted"], width=1),
                             stackgroup="capital", hovertemplate="现金 ¥%{y:,.2f}<extra></extra>"), row=2, col=1)
    _layout(fig, 520)
    fig.update_yaxes(title="人民币 / 元", row=1, col=1)
    fig.update_yaxes(title="资产构成 / 元", row=2, col=1)
    capital_plot = plot(fig, "capital-chart")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12)
    fig.add_trace(go.Scatter(x=dates, y=frame.daily_pnl, name="每日盈亏（剔除外部入金）", mode="lines",
                             line=dict(color=COLORS["teal"], width=1), customdata=common,
                             hovertemplate="当日盈亏 ¥%{y:,.2f}<br>当日收益率 %{customdata[1]:.3%}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=dates, y=frame.daily_return, name="每日资金流调整后收益率",
                         marker_color=np.where(frame.daily_return >= 0, COLORS["teal"], COLORS["red"]).tolist(),
                         hovertemplate="当日收益率 %{y:.3%}<extra></extra>"), row=2, col=1)
    _layout(fig, 500)
    fig.update_yaxes(title="每日盈亏 / 元", row=1, col=1)
    fig.update_yaxes(tickformat=".1%", title="每日收益率", row=2, col=1)
    daily_plot = plot(fig, "daily-chart")

    fig = go.Figure(go.Scatter(x=dates, y=frame.drawdown, name="策略净值回撤", mode="lines", fill="tozeroy",
                               line=dict(color=COLORS["red"], width=1.2), fillcolor="rgba(255,155,145,.14)",
                               hovertemplate="距历史峰值 %{y:.2%}<extra></extra>"))
    _layout(fig, 275)
    fig.update_yaxes(tickformat=".0%", title="回撤")
    drawdown_plot = plot(fig, "drawdown-chart")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.13, row_heights=[.6,.4])
    fig.add_trace(go.Scatter(x=dates, y=frame.nav, name="基金单位净值", line=dict(color=COLORS["blue"], width=1.4),
                             hovertemplate="单位净值 ¥%{y:.4f}<extra></extra>"), row=1, col=1)
    if not trades.empty:
        trade_frame = trades.copy()
        trade_frame["date"] = pd.to_datetime(trade_frame["date"], errors="coerce")
        valid = pd.to_numeric(_series(trade_frame, "shares"), errors="coerce").abs() > 0
        for action, name, color, symbol in [("BUY", "买入成交", COLORS["teal"], "triangle-up"), ("SELL", "卖出成交", COLORS["red"], "triangle-down")]:
            subset = trade_frame[(trade_frame.action == action) & valid]
            if not subset.empty:
                custom = subset.reindex(columns=["amount", "shares", "signal_date", "reason"]).astype(object).where(pd.notna(subset.reindex(columns=["amount", "shares", "signal_date", "reason"])), "").values
                fig.add_trace(go.Scatter(x=subset.date.dt.strftime("%Y-%m-%d"), y=subset.price, name=name, mode="markers",
                                         marker=dict(size=7, color=color, symbol=symbol), customdata=custom,
                                         hovertemplate="成交净值 ¥%{y:.4f}<br>金额 ¥%{customdata[0]:,.2f}<br>份额 %{customdata[1]:,.4f}<br>信号日 %{customdata[2]}<br>%{customdata[3]}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=_series(frame, "signal_metric", np.nan), name="三周期位置最小值",
                             line=dict(color=COLORS["gold"], width=1.5), hovertemplate="位置 %{y:.2%}<extra></extra>"), row=2, col=1)
    params = config.get("strategy", {}).get("params", {})
    buy_limit = params.get("buy_thresholds", [20])[0] / 100
    sell_start = params.get("sell_start", 50) / 100
    fig.add_hline(y=buy_limit, line_dash="dot", line_color=COLORS["teal"], row=2, col=1)
    fig.add_hline(y=sell_start, line_dash="dot", line_color=COLORS["red"], row=2, col=1)
    _layout(fig, 520)
    fig.update_yaxes(title="单位净值 / 元", row=1, col=1)
    fig.update_yaxes(title="历史区间位置", tickformat=".0%", row=2, col=1)
    signal_plot = plot(fig, "signal-chart")

    monthly = pd.DataFrame({"date": frame.date, "return": frame.daily_return}).set_index("date")["return"].resample("MS").apply(lambda s: (1+s).prod()-1 if len(s) else np.nan)
    years = sorted(monthly.index.year.unique())
    z = np.full((len(years), 12), np.nan)
    for day, value in monthly.items():
        z[years.index(day.year), day.month-1] = value
    labels = [["—" if not math.isfinite(v) else f"{v:.1%}" for v in row] for row in z]
    max_abs = max(.01, float(np.nanmax(np.abs(z))))
    fig = go.Figure(go.Heatmap(z=z, x=[f"{i}月" for i in range(1,13)], y=[str(y) for y in years],
                               text=labels, texttemplate="%{text}", textfont=dict(size=11),
                               colorscale=[[0,"#9b514c"],[.5,"#15283e"],[1,"#187f74"]], zmin=-max_abs, zmax=max_abs,
                               xgap=3, ygap=3, showscale=False,
                               hovertemplate="%{y}年 %{x}<br>月收益率 %{z:.2%}<extra></extra>"))
    _layout(fig, max(300, len(years)*30+70), date_axes=False)
    fig.update_layout(showlegend=False, margin=dict(l=65, r=25, t=15, b=40))
    fig.update_yaxes(autorange="reversed", type="category", tickmode="array", tickvals=[str(y) for y in years])
    monthly_plot = plot(fig, "monthly-chart")

    primary_specs = [
        ("累计净投入本金", "net_contributions", "money", "仅外部入金；卖出款循环使用不重复计本金"),
        ("累计总盈亏", "total_pnl", "money", "期末总资产 − 累计净投入本金"),
        ("总投入收益率", "simple_return", "percent", "累计盈亏 ÷ 累计净投入；未考虑入金时点"),
        ("资金加权年化 · XIRR", "xirr", "percent", "按实际外部入金日期与期末资产计算"),
        ("策略年化 · TWR", "cagr", "percent", "资金流调整后净值的复合年化收益"),
        ("最大回撤", "max_drawdown", "percent", "策略单位净值相对历史最高值的最大跌幅"),
    ]
    cards = "".join(f'<article class="stat"><div class="stat-label">{label}</div><div class="stat-value">{_number(metrics.get(key),kind)}</div><p>{note}</p></article>' for label,key,kind,note in primary_specs)
    detail_specs = [
        ("期末账户总资产", "final_equity", "money"), ("累计时间加权收益", "total_return", "percent"),
        ("夏普比率", "sharpe", "number"), ("索提诺比率", "sortino", "number"), ("卡玛比率", "calmar", "number"),
        ("年化波动率", "annual_volatility", "percent"), ("最长回撤持续（自然日）", "max_drawdown_days", "integer"),
        ("最好单日收益率", "best_day", "percent"), ("最差单日收益率", "worst_day", "percent"),
        ("正收益观察日占比", "positive_day_ratio", "percent"), ("平均持仓占比", "exposure_mean", "percent"),
        ("累计买入金额", "total_buys", "money"), ("累计卖出金额", "total_sells", "money"),
        ("买入成交次数", "buy_count", "integer"), ("卖出成交次数", "sell_count", "integer"),
        ("累计交易费用", "total_fees", "money"), ("已实现盈亏", "realized_pnl", "money"),
        ("当前持仓浮动盈亏", "unrealized_pnl", "money"), ("当前持仓收益率", "holding_return", "percent"),
        ("基金参考累计收益", "benchmark_return", "percent"), ("基金参考年化收益", "benchmark_cagr", "percent"),
        ("数据末尾未执行信号", "pending_count", "integer"), ("未成交订单", "rejected_count", "integer"),
    ]
    detail_cards = "".join(f'<div class="metric"><span>{label}</span><b>{_number(metrics.get(key),kind)}</b></div>' for label,key,kind in detail_specs)
    tx = trades.copy()
    for col in tx.columns:
        if pd.api.types.is_datetime64_any_dtype(tx[col]):
            tx[col] = tx[col].dt.strftime("%Y-%m-%d")
    records = json.loads(tx.to_json(orient="records", force_ascii=False, date_format="iso"))
    parameters = html.escape(json.dumps(config, ensure_ascii=False, indent=2, default=str))
    provenance = html.escape(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
    fee_note = ("本次申购、赎回费率均为 0，是未计交易费的理论回测。基金净值已包含其日常管理等费用。"
                if not eng.get("buy_fee",0) and not eng.get("sell_fee",0) else
                f"申购费率 {_number(eng.get('buy_fee',0),'percent')}；赎回费率 {_number(eng.get('sell_fee',0),'percent')}。本 Demo 使用固定费率，不模拟按持有天数阶梯收费。")
    funding_note = ("资金不限额：先用账户现金买入，不足部分当次追加；卖出款留在账户。外部入金从每日盈亏中剔除，累计外部本金不等于累计买入额。"
                    if topup else "固定初始本金：买入使用账户现金，卖出款留在账户；没有自动追加资金，现金不足的订单不能成交。")
    document = _TEMPLATE.replace("{{TITLE}}", html.escape(title)).replace("{{START}}",start).replace("{{END}}",end)
    substitutions = {"ROWS":str(len(frame)), "FIRST_FUNDED":html.escape(first_funded), "CARDS":cards, "DETAILS":detail_cards,
                     "RETURN_PLOT":returns_plot, "CAPITAL_PLOT":capital_plot,"DAILY_PLOT":daily_plot,"DRAWDOWN_PLOT":drawdown_plot,
                     "SIGNAL_PLOT":signal_plot,"MONTHLY_PLOT":monthly_plot,"PARAMETERS":parameters,"PROVENANCE":provenance,
                     "TRADES_JSON":_safe_json(records),"FUNDING_NOTE":funding_note,"FEE_NOTE":fee_note,
                     "RISK_FREE":_number(eng.get("risk_free_rate",0),"percent"),"PERIODS":str(eng.get("periods_per_year",252))}
    for key,value in substitutions.items():
        document=document.replace("{{"+key+"}}",value)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(document,encoding="utf-8")


_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{TITLE}} | AlphaTrack</title>
<style>
:root{color-scheme:dark;--bg:#091523;--panel:#102035;--border:#26384c;--text:#eff5fc;--muted:#91a5bd;--teal:#42dbc5}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif;font-size:14px;line-height:1.65}
header{border-bottom:1px solid var(--border);background:#0c1b2c}header .inner{display:flex;align-items:center;justify-content:space-between;gap:20px;max-width:1440px;padding:19px 32px;margin:auto}.brand{font-size:20px;letter-spacing:-.5px;font-weight:750}.brand i{font-style:normal;color:var(--teal)}.tag{color:var(--muted);font-size:11px;letter-spacing:2px;margin-left:15px}nav{display:flex;gap:23px}a{color:#c8d8eb;text-decoration:none}a:hover{color:var(--teal)}main{max-width:1440px;margin:auto;padding:38px 32px 50px}.eyebrow{font-size:11px;letter-spacing:3px;color:var(--teal);font-weight:650}h1{font-size:32px;letter-spacing:-.8px;margin:9px 0 12px;font-weight:650}h2{font-size:18px;font-weight:600;margin:0}h3{font-size:14px;margin:0 0 6px}.sub{color:var(--muted);margin:0}.meta{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0 24px}.pill{font-size:12px;border:1px solid var(--border);border-radius:5px;padding:5px 10px;color:#afc0d4;background:#0d1e30}.pill.live{border-color:#255950;color:var(--teal)}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:25px}.stat{background:linear-gradient(135deg,#14273d,#102034);border:1px solid var(--border);border-radius:10px;padding:20px 22px;min-height:148px}.stat-label{color:#adc0d6;font-size:12px}.stat-value{font-size:29px;letter-spacing:-.7px;color:#effbfa;margin-top:8px;font-variant-numeric:tabular-nums}.stat p{font-size:11px;color:var(--muted);margin:8px 0 0}.panel{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:23px 24px 12px;margin-bottom:22px;overflow:hidden;scroll-margin-top:20px}.panel-head{display:flex;justify-content:space-between;align-items:start;gap:16px;margin-bottom:12px}.panel-head p{font-size:12px;color:var(--muted);margin:6px 0 0;max-width:960px}.panel-index{font-family:monospace;font-size:12px;color:#5f7690;padding-top:4px}.plotly-graph-div{width:100%}.note{border-left:2px solid #2a756e;padding:8px 13px;background:#0e2a32;color:#a9c9ce;font-size:12px;margin:14px 0 22px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:0 28px;padding-bottom:10px}.metric{display:flex;justify-content:space-between;gap:10px;padding:13px 0;border-bottom:1px solid #20344b;font-size:12px}.metric span{color:var(--muted)}.metric b{font-weight:500;white-space:nowrap;font-variant-numeric:tabular-nums}.table-controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:18px 0 14px}input,select,button{font:inherit;color:#dce8f4;background:#132a40;border:1px solid #344a63;border-radius:5px;padding:8px 12px}input{width:min(360px,100%)}button{cursor:pointer}button:hover:not(:disabled){border-color:var(--teal);color:var(--teal)}button:disabled{opacity:.35;cursor:default}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;min-width:1000px;font-size:12px}th{color:#a8bed5;font-weight:500;text-align:left;border-bottom:1px solid #34475d;background:#13263b;padding:10px 12px;white-space:nowrap}td{padding:11px 12px;border-bottom:1px solid #1e334a;color:#b3c4d9;vertical-align:top;font-variant-numeric:tabular-nums}td.reason{min-width:220px;max-width:420px;font-size:11px}td.buy{color:var(--teal)}td.sell{color:#ff9b91}.pagination{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:16px 0 7px;font-size:12px;color:var(--muted)}.pagination div{display:flex;align-items:center;gap:10px}.method-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px 30px;margin:18px 0}.method-grid p{color:#a5b8cf;font-size:12px;margin:0}details{border-top:1px solid var(--border);padding:15px 0}summary{cursor:pointer;color:#b6cce0;font-size:12px}pre{overflow:auto;max-height:400px;background:#0a192a;padding:16px;font-size:11px;color:#a4c0d4;border-radius:5px}footer{display:flex;justify-content:space-between;color:#69819b;font-size:11px;padding:4px 2px}.footlinks{color:#92abc4}@media(max-width:850px){.tag,nav{display:none}main{padding:25px 14px}.stats{grid-template-columns:repeat(2,1fr)}.metrics{grid-template-columns:repeat(2,1fr)}.method-grid{grid-template-columns:1fr}.panel{padding:18px 10px 10px}h1{font-size:26px}.stat{padding:17px}.stat-value{font-size:24px}header .inner{padding:16px 18px}}@media(max-width:520px){.stats{gap:8px}.stat{padding:13px;min-height:150px}.stat-value{font-size:21px}.metrics{grid-template-columns:1fr}.panel-head{padding:0 4px}.meta{gap:6px}.pagination{align-items:start;flex-direction:column}.panel-index{display:none}}
</style></head><body>
<header><div class="inner"><div class="brand"><i>α</i> AlphaTrack <span class="tag">BACKTEST RESEARCH</span></div><nav><a href="#overview">收益概览</a><a href="#risk">风险指标</a><a href="#transactions">交易明细</a><a href="#method">回测口径</a></nav></div></header>
<main><div class="eyebrow">STRATEGY EXPLORER / DEMO 01</div><h1>{{TITLE}}</h1><p class="sub">从成立开始逐日评估规则，追踪每一笔资金、每一次交易和每一天的收益。</p>
<div class="meta"><span class="pill live">● 可复现历史回测</span><span class="pill">{{START}} → {{END}}</span><span class="pill">{{ROWS}} 个净值观察日</span><span class="pill">首次实际投入 {{FIRST_FUNDED}}</span><span class="pill">Plotly · 离线报告</span></div>
<section class="stats">{{CARDS}}</section>
<div class="note">{{FUNDING_NOTE}} 本报告按下一可用净值理论成交，未还原封闭期、暂停申购赎回、到账延迟及历史阶梯费率，不能视为完全可执行的历史实盘。图表仅使用净值观察日（不是每个自然日），数据截止 {{END}}。悬停查看每日数值，框选缩放，双击恢复全范围。</div>
<section id="overview" class="panel"><div class="panel-head"><div><h2>策略收益与基金参考曲线</h2><p>策略使用时间加权收益，剔除追加本金的机械影响。基金曲线是假定起始日一次性买入并持有的总收益参考；两者现金流和资金利用率不同，不代表同等投入下的超额收益。</p></div><span class="panel-index">01 / PERFORMANCE</span></div>{{RETURN_PLOT}}</section>
<section class="panel"><div class="panel-head"><div><h2>本金、资产与资金利用</h2><p>账户总资产 = 持仓市值 + 现金。总资产与累计外部本金的差额是累计盈亏；追加资金本身不计入收益。</p></div><span class="panel-index">02 / CAPITAL</span></div>{{CAPITAL_PLOT}}</section>
<section class="panel"><div class="panel-head"><div><h2>每日盈亏与每日收益率</h2><p>盈亏以人民币计量；收益率按账户资金流调整。未公布净值的日期不插值、不伪造每日收益，两个观察日之间的变化计入后一个观察日。</p></div><span class="panel-index">03 / DAILY P&amp;L</span></div>{{DAILY_PLOT}}</section>
<section id="risk" class="panel"><div class="panel-head"><div><h2>风险与持有收益指标</h2><p>持仓收益率仅衡量期末剩余份额相对其成本的浮盈比例；与包括现金、已实现收益的全账户收益率不同。比率无法定义时显示「—」。</p></div><span class="panel-index">04 / RISK</span></div><div class="metrics">{{DETAILS}}</div>{{DRAWDOWN_PLOT}}</section>
<section class="panel"><div class="panel-head"><div><h2>净值走势与买卖执行</h2><p>三周期位置指分红再投曲线在近 5 年、近 10 年、成立以来各自高低点之间的位置，再取最小值，不是统计分位排名。标记为实际成交日；信号于前一个净值观察日产生。</p></div><span class="panel-index">05 / EXECUTION</span></div>{{SIGNAL_PLOT}}</section>
<section class="panel"><div class="panel-head"><div><h2>月度收益地图</h2><p>月内每日资金流调整后收益连乘。首尾月份可能是不完整月份；无数据月份为「—」。青色为正收益，暖色为负收益。</p></div><span class="panel-index">06 / MONTHLY</span></div>{{MONTHLY_PLOT}}</section>
<section id="transactions" class="panel"><div class="panel-head"><div><h2>逐笔交易账本</h2><p>同时保留信号日期与执行日期。可搜索规则原因、日期或状态；导出包含当前筛选结果的全部记录。</p></div><span class="panel-index">07 / LEDGER</span></div>
<div class="table-controls"><input id="trade-search" aria-label="搜索交易" placeholder="搜索日期 / 规则 / 状态…"><select id="trade-action" aria-label="交易方向"><option value="">全部方向</option><option value="BUY">买入</option><option value="SELL">卖出</option></select><button id="export-trades">导出筛选 CSV</button></div>
<div class="table-wrap"><table><thead><tr><th>信号日</th><th>执行日</th><th>方向</th><th>金额（元）</th><th>份额</th><th>净值</th><th>费用（元）</th><th>状态</th><th>规则原因</th></tr></thead><tbody id="trade-body"></tbody></table></div>
<div class="pagination"><span id="trade-count"></span><div><button id="prev-page">上一页</button><span id="page-label"></span><button id="next-page">下一页</button></div></div></section>
<section id="method" class="panel"><div class="panel-head"><div><h2>回测口径与可复现参数</h2><p>先明确统计口径，再解释收益。数据、策略和执行假设共同决定结果。</p></div><span class="panel-index">08 / METHODOLOGY</span></div>
<div class="method-grid">
<div><h3>成立日起评估，信号后执行</h3><p>成立第一天缺少历史区间，规则不强制首日买入。每个净值观察日收盘后计算信号，在下一个有净值的观察日按该日净值成交，避免用当日收盘信息反向成交。末日新信号无下一日净值时不成交。</p></div>
<div><h3>资金与费用</h3><p>{{FUNDING_NOTE}} {{FEE_NOTE}} 暂未模拟申赎暂停、确认与到账等待、交易限额和实际份额舍入。</p></div>
<div><h3>收益率有不同含义</h3><p>总投入收益率 = 累计盈亏 / 累计外部本金。TWR 通过剔除外部资金流衡量单位资金的策略收益；XIRR 根据每次外部入金时间及期末总资产求资金加权年化。年化以首次投入后的实际自然日计算，不能把总收益简单除以年数。</p></div>
<div><h3>风险指标与观察频率</h3><p>无风险年利率设为 {{RISK_FREE}}，波动率、夏普和索提诺使用每年 {{PERIODS}} 个观察期的年化因子。成立早期净值存在非每日披露，因此风险年化是近似值。最长回撤持续期使用自然日，未恢复的区间计至回测末日。</p></div>
<div><h3>分红与卖出阶梯</h3><p>策略信号使用分红再投入总收益曲线，避免除息直接扭曲区间位置。买入区间重置卖出档位；本轮每档只触发一次，跨过多档顺序合并执行。参数、分红事件和来源版本保存在下方。</p></div>
<div><h3>使用边界</h3><p>这是历史规则复现和研究 Demo。已实现盈亏采用平均成本口径，未强制期末平仓。历史收益不等于未来结果；更换参数、资金方式或交易费用后，应重新生成报告。</p></div>
</div><details><summary>展开运行配置</summary><pre>{{PARAMETERS}}</pre></details><details><summary>展开数据来源与版本记录</summary><pre>{{PROVENANCE}}</pre></details></section>
<footer><span>AlphaTrack · Reproducible backtesting</span><span>数据截止 {{END}} · 单文件离线交互报告</span></footer></main>
<script>
const allTrades={{TRADES_JSON}};
let tradePage=0;const pageSize=20;let filtered=[];
const columns=['signal_date','date','action','amount','shares','price','fee','status','reason'];
const labels=['信号日','执行日','方向','金额(元)','份额','净值','费用(元)','状态','规则原因'];
const el=id=>document.getElementById(id);
const number=(v,d)=>v===null||v===undefined||v===''?'—':Number(v).toLocaleString('zh-CN',{minimumFractionDigits:d,maximumFractionDigits:d});
function display(v,key){if(key==='action')return v==='BUY'?'买入':v==='SELL'?'卖出':(v||'—');if(key==='status')return ({FILLED:'已成交',PARTIAL:'部分成交',PENDING_END_OF_DATA:'待执行 · 数据已结束',REJECTED_NO_CASH:'未成交 · 现金不足',REJECTED_NO_HOLDING:'未成交 · 无持仓'})[v]||(v??'—');if(['amount','fee'].includes(key))return number(v,2);if(['shares','price'].includes(key))return number(v,4);if(['date','signal_date'].includes(key))return v?String(v).slice(0,10):'—';return v??'—';}
function render(){const text=el('trade-search').value.trim().toLowerCase();const action=el('trade-action').value;filtered=allTrades.filter(t=>(!action||t.action===action)&&(!text||Object.values(t).join(' ').toLowerCase().includes(text)));const pages=Math.max(1,Math.ceil(filtered.length/pageSize));tradePage=Math.min(tradePage,pages-1);el('trade-body').replaceChildren();for(const t of filtered.slice(tradePage*pageSize,(tradePage+1)*pageSize)){const tr=document.createElement('tr');for(const key of columns){const td=document.createElement('td');td.textContent=display(t[key],key);if(key==='reason')td.className='reason';if(key==='action')td.className=t.action==='BUY'?'buy':'sell';tr.appendChild(td);}el('trade-body').appendChild(tr);}if(!filtered.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=columns.length;td.textContent='没有符合条件的交易记录';tr.appendChild(td);el('trade-body').appendChild(tr);}el('trade-count').textContent=`共 ${filtered.length.toLocaleString()} 条记录 · 每页 ${pageSize} 条`;el('page-label').textContent=`${tradePage+1} / ${pages}`;el('prev-page').disabled=tradePage===0;el('next-page').disabled=tradePage>=pages-1;}
el('trade-search').addEventListener('input',()=>{tradePage=0;render();});el('trade-action').addEventListener('change',()=>{tradePage=0;render();});el('prev-page').onclick=()=>{tradePage--;render();};el('next-page').onclick=()=>{tradePage++;render();};
el('export-trades').onclick=()=>{const csv=v=>'"'+String(v??'').replaceAll('"','""')+'"';const rows=[labels.map(csv).join(','),...filtered.map(t=>columns.map(k=>csv(t[k])).join(','))];const url=URL.createObjectURL(new Blob(['\ufeff'+rows.join('\r\n')],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='alphatrack-trades.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};render();
</script></body></html>'''
