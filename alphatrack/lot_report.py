"""Self-contained report of FIFO matched-piece annualized holding returns.

Each point is one sold or remaining piece, never a portfolio cumulative return.
Report statistics are descriptive: medians are not aggregate annualized returns.
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


def _fmt(value, kind="number"):
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
    return f"{value:,.0f}" if kind == "integer" else f"{value:,.2f}"


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=str).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _layout(fig, height=440, date=False):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif",color="#b4c7da",size=12),
                      height=height, margin=dict(l=72,r=28,t=40,b=55), hovermode="closest",
                      hoverlabel=dict(bgcolor="#173148",font=dict(color="#f0f7fb")),
                      legend=dict(orientation="h",y=1.05,x=0), dragmode="zoom")
    fig.update_xaxes(gridcolor="#1b3044",zerolinecolor="#344d65")
    fig.update_yaxes(gridcolor="#1b3044",zerolinecolor="#54718b")
    if date:
        fig.update_xaxes(type="date",tickformat="%Y-%m",hoverformat="%Y-%m-%d")
    return fig


def _records(lots):
    result = lots.copy()
    for col in ("buy_date", "sell_date", "valuation_date"):
        if col in result:
            result[col] = pd.to_datetime(result[col]).dt.strftime("%Y-%m-%d")
    # Pandas serializes all NA/NaN/inf values as null, keeping zero-day rows valid.
    return json.loads(result.to_json(orient="records", force_ascii=False, double_precision=15))


def write_lot_report(daily, trades, lots, lot_summary, config, metadata, output):
    """Export one offline HTML with per-piece annual returns and an audit ledger."""
    if daily.empty:
        raise ValueError("Cannot report an empty observation window")
    frame = daily.copy()
    dates = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d").tolist()
    data = lots.copy()
    if "annualized_return" not in data:
        data["annualized_return"] = pd.Series(dtype=float)
    for col in ("buy_date", "sell_date", "valuation_date"):
        if col not in data:
            data[col] = pd.Series(dtype="datetime64[ns]")
        data[col] = pd.to_datetime(data[col])
    records = _records(data)
    figures = []

    def export(fig, ident):
        fragment = fig.to_html(full_html=False, include_plotlyjs=not figures, div_id=ident,
                               config={"responsive":True,"displaylogo":False,"toImageButtonOptions":{"format":"png","scale":2}})
        figures.append(ident)
        return fragment

    fig = go.Figure()
    duration_fig = go.Figure()
    finite = data[np.isfinite(pd.to_numeric(data.annualized_return,errors="coerce"))]
    max_cost = max(float(finite.cost.max()) if len(finite) else 1, 1e-12)
    hover = ("买入批次 %{customdata[0]}<br>买入 %{customdata[1]}<br>卖出 / 估值 %{customdata[2]}"
             "<br>持有 %{customdata[3]:,.0f} 天 · 份额 %{customdata[4]:,.6f}"
             "<br>分摊成本 ¥%{customdata[5]:,.2f}<br>净回款 / 估值 ¥%{customdata[6]:,.2f}"
             "<br>盈亏 ¥%{customdata[7]:,.2f}<br>持有收益率 %{customdata[8]:.2%}"
             "<br><b>年化收益率 %{y:.2%}</b><extra>%{fullData.name}</extra>")
    for status, label, color, symbol in (("REALIZED","已实现 · 卖出日",TEAL,"circle"),
                                         ("UNREALIZED","未实现 · 期末估值日",GOLD,"diamond-open")):
        sub = finite[finite.status == status] if "status" in finite else finite.iloc[:0]
        end_dates = sub.sell_date.where(sub.status == "REALIZED", sub.valuation_date) if len(sub) else pd.Series(dtype="datetime64[ns]")
        custom = [[r.lot_id, r.buy_date.strftime("%Y-%m-%d"), end_dates.iloc[i].strftime("%Y-%m-%d"),
                   r.holding_days, r.shares, r.cost, r.proceeds, r.pnl, r.holding_return]
                  for i, r in enumerate(sub.itertuples())]
        sizes = (7 + 17*np.sqrt(sub.cost.to_numpy(dtype=float)/max_cost)).tolist() if len(sub) else []
        marker = dict(color=color, symbol=symbol, size=sizes, opacity=.73, line=dict(width=1,color=color))
        fig.add_trace(go.Scatter(x=end_dates.dt.strftime("%Y-%m-%d").tolist(), y=sub.annualized_return.tolist(),
                                name=label, mode="markers", marker=marker, customdata=custom, hovertemplate=hover))
        duration_fig.add_trace(go.Scatter(x=sub.holding_days.tolist(),y=sub.annualized_return.tolist(),
                                         name=label,mode="markers",marker=marker,customdata=custom,hovertemplate=hover))
    _layout(fig,520,date=True)
    fig.update_yaxes(title="每个匹配片段的年化收益率",tickformat=".1%")
    fig.update_xaxes(title="卖出日期 / 期末估值日期",rangeslider=dict(visible=True,thickness=.06,bgcolor="#162d43"))
    if finite.empty:
        fig.add_annotation(text="该区间没有可计算年化的匹配片段",xref="paper",yref="paper",x=.5,y=.5,showarrow=False)
    annual_plot = export(fig,"lot-annual-chart")
    _layout(duration_fig,390)
    duration_fig.update_xaxes(title="持有天数（自然日）")
    duration_fig.update_yaxes(title="年化收益率",tickformat=".1%")
    duration_plot = export(duration_fig,"lot-duration-chart")

    pnl = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.12)
    cumulative = frame.cumulative_pnl if "cumulative_pnl" in frame else frame.daily_pnl.cumsum()
    pnl.add_trace(go.Scatter(x=dates,y=cumulative,name="累计实际盈亏（已实现 + 未实现）",mode="lines",
                             line=dict(color=BLUE,width=2),hovertemplate="累计盈亏 ¥%{y:,.2f}<extra></extra>"),row=1,col=1)
    pnl.add_trace(go.Scatter(x=dates,y=frame.daily_pnl,name="当日盈亏（剔除买入/卖出资金流）",mode="lines",
                             line=dict(color=TEAL,width=1),hovertemplate="当日盈亏 ¥%{y:,.2f}<extra></extra>"),row=2,col=1)
    _layout(pnl,480,date=True)
    pnl.update_layout(hovermode="x unified")
    pnl.update_yaxes(title="累计盈亏 / 元",row=1,col=1)
    pnl.update_yaxes(title="当日盈亏 / 元",row=2,col=1)
    pnl_plot = export(pnl,"lot-pnl-chart")

    cards = [
        ("已实现片段 · 年化中位数","realized_annualized_median","percent","每个已卖出匹配片段等权；不是整体年化"),
        ("未实现片段 · 年化中位数","open_annualized_median","percent","按期末市值估算；每片段等权，尚未卖出"),
        ("已实现 / 未实现片段数",None,None,"一笔买入分次卖出，可以产生多个匹配片段"),
        ("已实现盈亏 · FIFO","realized_pnl","money","只计算已卖出份额；卖出后的现金不继续计时"),
        ("未实现盈亏 · FIFO","unrealized_pnl","money","剩余持仓以期末净值估值，未假设清仓费用"),
        ("全部片段合计盈亏","total_pnl","money","已实现 + 未实现；这是金额，不是年化收益率"),
    ]
    cards_html=[]
    for title,key,kind,note in cards:
        value = (_fmt(lot_summary.get("realized_count"),"integer")+" / "+_fmt(lot_summary.get("open_count"),"integer")) if key is None else _fmt(lot_summary.get(key),kind)
        cards_html.append(f'<div class="stat"><div class="stat-label">{title}</div><div class="stat-value">{value}</div><p>{note}</p></div>')
    title = str(config.get("title", "Fundd 宽基策略"))
    if "逐批年化" not in title:
        title += " · 逐批年化"
    labels = {"TITLE":html.escape(title),"START":dates[0],"END":dates[-1],"CARDS":"".join(cards_html),
              "ANNUAL_PLOT":annual_plot,"DURATION_PLOT":duration_plot,"PNL_PLOT":pnl_plot,
              "LOTS_JSON":_json(records),"PARAMETERS":html.escape(json.dumps(config,ensure_ascii=False,indent=2)),
              "PROVENANCE":html.escape(json.dumps(metadata,ensure_ascii=False,indent=2,default=str)),
              "DAYS":str(lot_summary.get("year_days",365.0)),"SHORT_COUNT":_fmt(lot_summary.get("short_hold_count",0),"integer"),
              "UNDEFINED_COUNT":_fmt(lot_summary.get("undefined_annualized_count",0),"integer"),
              "OBSERVATIONS":str(len(frame)),"REALIZED_COST":_fmt(lot_summary.get("realized_cost"),"money"),
              "OPEN_COST":_fmt(lot_summary.get("open_cost"),"money"),"REALIZED_PROCEEDS":_fmt(lot_summary.get("realized_proceeds"),"money"),
              "OPEN_VALUE":_fmt(lot_summary.get("open_value"),"money")}
    document=_TEMPLATE
    for key,value in labels.items():
        document=document.replace("{{"+key+"}}",value)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(document,encoding="utf-8")


_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{{TITLE}} | AlphaTrack</title>
<style>
:root{color-scheme:dark;--bg:#091523;--panel:#102035;--border:#263c52;--text:#eef5fd;--muted:#91a7be;--teal:#42dbc5;--gold:#f4c875}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.65 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}header{border-bottom:1px solid var(--border);background:#0d1d30}.header-inner{max-width:1440px;margin:auto;padding:19px 32px;display:flex;justify-content:space-between;gap:20px}.brand{font-size:20px;font-weight:700}.brand b{color:var(--teal)}nav{display:flex;gap:25px}a{color:#bdd1e5;text-decoration:none}a:hover{color:var(--teal)}main{max-width:1440px;margin:auto;padding:34px 32px 45px}.eyebrow{font-size:11px;color:var(--teal);letter-spacing:3px}h1{font-size:30px;line-height:1.35;margin:12px 0}h2{font-size:18px;margin:0}h3{font-size:14px;margin:0 0 7px}.sub,.muted{color:var(--muted)}.meta{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0 24px}.pill{border:1px solid var(--border);border-radius:5px;padding:4px 9px;font-size:12px;background:#102337;color:#b7cadd}.pill.green{color:var(--teal);border-color:#265e58}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}.stat{padding:20px 22px;border:1px solid var(--border);border-radius:10px;background:linear-gradient(135deg,#14293f,#102035);min-height:145px}.stat-label{font-size:12px;color:#afc5dd}.stat-value{font-size:29px;letter-spacing:-.5px;font-variant-numeric:tabular-nums;margin-top:8px}.stat p{font-size:11px;color:var(--muted);margin:8px 0 0}.note{background:#103039;border-left:2px solid #398c82;padding:12px 15px;color:#b1d4d7;font-size:12px;margin:20px 0}.note strong{color:#e7f9f8}.panel{background:var(--panel);border:1px solid var(--border);padding:23px 22px 12px;border-radius:10px;margin:22px 0;overflow:hidden}.panel-head{display:flex;justify-content:space-between;gap:20px;align-items:start}.panel-head p{font-size:12px;color:var(--muted);margin:7px 0 12px;max-width:1050px}.panel-index{font:12px monospace;color:#5e7893;padding-top:4px}.plotly-graph-div{width:100%}.audit{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;padding:15px 0 4px}.audit div{border-bottom:1px solid #263d53;padding:10px 0;font-size:12px;color:var(--muted)}.audit b{display:block;color:#d6e6f5;font-weight:500;margin-top:5px}.table-controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0}input,select,button{font:inherit;color:#d4e5f6;background:#173149;border:1px solid #3a5269;border-radius:5px;padding:8px 11px;max-width:100%}input{width:280px}button{cursor:pointer}button:hover:not(:disabled){border-color:var(--teal)}button:disabled{opacity:.4;cursor:default}.table-wrap{overflow:auto}table{width:100%;min-width:1300px;border-collapse:collapse;font-size:12px}th{padding:10px;background:#172f47;text-align:left;color:#adc3db;font-weight:500;white-space:nowrap}td{padding:10px;border-bottom:1px solid #243a50;color:#bfd0e2;white-space:nowrap;font-variant-numeric:tabular-nums}td.annual{font-weight:650;font-size:13px}td.positive{color:var(--teal)}td.negative{color:#ff9b91}td.open{color:var(--gold)}td.note-col{white-space:normal;min-width:140px;font-size:11px}.pagination{display:flex;justify-content:space-between;gap:12px;align-items:center;margin:15px 0 7px;font-size:12px;color:var(--muted)}.pagination div{display:flex;gap:9px;align-items:center}.method-grid{display:grid;grid-template-columns:1fr 1fr;gap:23px 30px;padding:17px 0}.method-grid p{font-size:12px;color:#a6bdd3;margin:0}.formula{font:13px ui-monospace,SFMono-Regular,monospace;background:#0c1a2a;color:#d6e9f7;border:1px solid #243f58;padding:14px;overflow:auto;margin-bottom:8px}details{border-top:1px solid var(--border);padding:15px 0}summary{cursor:pointer;color:#bbd1e6;font-size:12px}pre{font:11px/1.5 ui-monospace,SFMono-Regular,monospace;overflow:auto;max-height:380px;background:#0c1a2a;padding:15px;color:#b4cbe0}footer{font-size:11px;color:#7891ab;display:flex;justify-content:space-between;gap:20px}@media(max-width:850px){nav{display:none}.header-inner{padding:16px 18px}main{padding:25px 14px}.stats{grid-template-columns:repeat(2,1fr)}.stat{padding:16px}.stat-value{font-size:25px}.audit{grid-template-columns:repeat(2,1fr)}.method-grid{grid-template-columns:1fr}.panel{padding:18px 10px 10px}h1{font-size:25px}}@media(max-width:520px){.stats{gap:8px}.stat{padding:13px;min-height:153px}.stat-value{font-size:22px}.panel-index{display:none}.pagination{align-items:start;flex-direction:column}footer{flex-direction:column;gap:5px}.table-controls{gap:7px}input{width:100%}.panel-head{padding:0 3px}}
</style></head><body>
<header><div class="header-inner"><div class="brand"><b>α</b> AlphaTrack</div><nav><a href="#annual">逐批年化</a><a href="#pnl">盈亏金额</a><a href="#lots">批次明细</a><a href="#method">计算口径</a></nav></div></header>
<main><div class="eyebrow">HOLDING PERIOD / ANNUALIZED RETURNS</div><h1>{{TITLE}}</h1><p class="sub">每一批资金，只评价从买入到卖出这一段。未卖出部分单独按期末净值估值。</p>
<div class="meta"><span class="pill green">FIFO · 先进先出</span><span class="pill">{{START}} → {{END}}</span><span class="pill">{{OBSERVATIONS}} 个净值观察日</span><span class="pill">卖出后现金不参与收益评价</span><span class="pill">ACT / {{DAYS}}</span></div>
<section class="stats">{{CARDS}}</section>
<div class="note"><strong>读图方式：</strong>每个点是一段“买入批次 → 卖出匹配”或一段“买入批次 → 期末未卖出持仓”的年化收益，<strong>不是整个账户的累计收益曲线</strong>。同一批买入分次卖出会有多个点。年化中位数只概括片段分布，不能当作整体投资年化。持有不足 30 天的片段有 {{SHORT_COUNT}} 个；年化无法定义的片段有 {{UNDEFINED_COUNT}} 个，保留在明细中并显示「—」。</div>
<section class="panel" id="annual"><div class="panel-head"><div><h2>每笔资金的年化收益 · 按退出日期查看</h2><p>横坐标为卖出日期，未实现片段位于期末估值日期；纵坐标为该片段的年化收益率。青色为已实现，金色为空心未实现；点大小随分摊成本变化。同日多个片段可能重叠，悬停或查阅明细查看；不截断极端年化值。</p></div><span class="panel-index">01 / ANNUALIZED</span></div>{{ANNUAL_PLOT}}</section>
<section class="panel"><div class="panel-head"><div><h2>持有多久，获得怎样的年化</h2><p>每个匹配片段保持独立。短期波动换算成年化可能被明显放大，年化值不表示未来可以按同样速度持续获利。可框选缩放；双击恢复。</p></div><span class="panel-index">02 / DURATION</span></div>{{DURATION_PLOT}}</section>
<section class="panel" id="pnl"><div class="panel-head"><div><h2>收益金额核对</h2><p>卖出后收益保留为已实现盈亏记录，但不再计算现金持有期间的收益。追加资金和卖出转出均不构成盈利；未公布净值的日期不插值。</p></div><span class="panel-index">03 / P&amp;L</span></div><div class="audit"><div>已卖出份额成本<b>{{REALIZED_COST}}</b></div><div>卖出净回款<b>{{REALIZED_PROCEEDS}}</b></div><div>未卖出份额成本<b>{{OPEN_COST}}</b></div><div>期末持仓估值<b>{{OPEN_VALUE}}</b></div></div>{{PNL_PLOT}}</section>
<section class="panel" id="lots"><div class="panel-head"><div><h2>逐批匹配明细 · 以年化为重点</h2><p>部分卖出按 FIFO 分配成本和份额；相同批次编号代表来自同一次买入。未实现行没有卖出日，其期末估值不是实际回款。支持按状态筛选、搜索批次/日期、年化排序及导出完整筛选结果。</p></div><span class="panel-index">04 / LOT LEDGER</span></div>
<div class="table-controls"><input id="lot-search" aria-label="搜索批次" placeholder="搜索批次 / 买入日 / 卖出日…"><select id="lot-status" aria-label="实现状态"><option value="">全部片段</option><option value="REALIZED">已实现</option><option value="UNREALIZED">未实现</option></select><select id="lot-sort" aria-label="明细排序"><option value="date">卖出 / 估值日 ↑</option><option value="annual_desc">年化收益率 ↓</option><option value="annual_asc">年化收益率 ↑</option><option value="days_desc">持有天数 ↓</option></select><button id="export-lots">导出筛选 CSV</button></div>
<div class="table-wrap"><table><thead><tr><th>买入批次</th><th>状态</th><th>买入日期</th><th>卖出日期</th><th>估值日期</th><th>匹配份额</th><th>持有天数</th><th>分摊成本</th><th>净回款 / 估值</th><th>盈亏</th><th>持有收益率</th><th>年化收益率</th><th>说明</th></tr></thead><tbody id="lot-body"></tbody></table></div>
<div class="pagination"><span id="lot-count"></span><div><button id="prev-page">上一页</button><span id="page-label"></span><button id="next-page">下一页</button></div></div></section>
<section class="panel" id="method"><div class="panel-head"><div><h2>计算口径</h2><p>年化使用实际持有自然日，基准为每年 {{DAYS}} 天。出售后不继续累计该片段的持有时间。</p></div><span class="panel-index">05 / METHODOLOGY</span></div>
<div class="formula">年化收益率 = (净回款或期末估值 ÷ 分摊买入成本) ^ ({{DAYS}} ÷ 持有天数) − 1</div>
<div class="method-grid">
<div><h3>逐批配对，而不是全期统一持有</h3><p>每次买入建立一个批次。卖出先匹配最早尚未卖完的批次，跨批次则拆成多条；同一批次分次卖出也拆成多条。各片段独立使用自己的买入日、卖出日和成本，不把所有钱都当成从成立持有到期末。</p></div>
<div><h3>卖出后停止计时</h3><p>该报告只计算资金在持仓中的表现，不将卖出后的闲置或其他用途计入该笔收益。既有盈利仍可汇总作金额核对，但不因此延长已卖出份额的持有天数。</p></div>
<div><h3>未实现年化单独标注</h3><p>未卖出的份额以期末净值估值，并以期末日期计算持有天数。其收益仍可能变化，不等于实际卖出所得；没有虚构期末交易或扣除假设清仓费用。</p></div>
<div><h3>费用、分红和拆分</h3><p>买入成本包含配置的申购费，卖出净回款扣除配置的赎回费。分红按除息日再投资到原批次，拆分调整份额，原批次投入成本及起始日保留。默认 Demo 交易费率为零；参数详见下方。</p></div>
<div><h3>短期年化和无法定义的情形</h3><p>很短的持有期会放大年化值，应与持有天数、实际盈亏一起阅读。持有天数为零等无法定义的情形显示「—」，不填成零收益；不以截断或隐藏异常值美化图表。已实现/未实现中位数仅基于可定义的片段。</p></div>
<div><h3>理论回测边界</h3><p>信号在净值公布后生成，下一可用净值成交。未还原封闭期、暂停申赎、到账延迟、实际份额舍入及历史持有期阶梯费率。FIFO 是归因约定，改变匹配法可能改变各片段结果，但不改变总盈亏。</p></div>
</div><details><summary>展开运行配置</summary><pre>{{PARAMETERS}}</pre></details><details><summary>展开数据来源与策略版本</summary><pre>{{PROVENANCE}}</pre></details></section>
<footer><span>AlphaTrack · FIFO lot-level annualized analysis</span><span>数据截至 {{END}} · 单文件离线交互报告</span></footer></main>
<script>
const allLots={{LOTS_JSON}};
const keys=['lot_id','status','buy_date','sell_date','valuation_date','shares','holding_days','cost','proceeds','pnl','holding_return','annualized_return','annualization_note'];
const headers=['买入批次','状态','买入日期','卖出日期','估值日期','匹配份额','持有天数','分摊成本','净回款或估值','盈亏','持有收益率(小数)','年化收益率(小数)','年化说明'];
const el=id=>document.getElementById(id);let currentPage=0,filtered=[];const pageSize=20;
const valid=v=>v!==null&&v!==undefined&&v!==''&&Number.isFinite(Number(v));
const number=(v,n)=>valid(v)?Number(v).toLocaleString('zh-CN',{minimumFractionDigits:n,maximumFractionDigits:n}):'—';
function show(v,key){if(key==='status')return v==='REALIZED'?'已实现':'未实现';if(['holding_return','annualized_return'].includes(key))return valid(v)?number(v*100,2)+'%':'—';if(['cost','proceeds','pnl'].includes(key))return number(v,2);if(key==='shares')return number(v,6);if(key==='holding_days')return number(v,0);return v===null||v===undefined||v===''?'—':String(v);}
function render(){const query=el('lot-search').value.trim().toLowerCase(),status=el('lot-status').value,sort=el('lot-sort').value;filtered=allLots.filter(r=>(!status||r.status===status)&&(!query||Object.values(r).join(' ').toLowerCase().includes(query)));filtered.sort((a,b)=>{if(sort.startsWith('annual')){if(!valid(a.annualized_return))return valid(b.annualized_return)?1:0;if(!valid(b.annualized_return))return -1;return (a.annualized_return-b.annualized_return)*(sort==='annual_asc'?1:-1);}if(sort==='days_desc')return b.holding_days-a.holding_days;return String(a.sell_date||a.valuation_date).localeCompare(String(b.sell_date||b.valuation_date))||String(a.buy_date).localeCompare(String(b.buy_date));});const pages=Math.max(1,Math.ceil(filtered.length/pageSize));currentPage=Math.min(currentPage,pages-1);el('lot-body').replaceChildren();for(const row of filtered.slice(currentPage*pageSize,(currentPage+1)*pageSize)){const tr=document.createElement('tr');for(const key of keys){const td=document.createElement('td');td.textContent=show(row[key],key);if(key==='annualized_return'){td.classList.add('annual');if(valid(row[key]))td.classList.add(Number(row[key])>=0?'positive':'negative');}if(key==='status'&&row.status==='UNREALIZED')td.classList.add('open');if(key==='annualization_note')td.classList.add('note-col');tr.appendChild(td);}el('lot-body').appendChild(tr);}if(!filtered.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=keys.length;td.textContent='没有符合条件的匹配片段';tr.appendChild(td);el('lot-body').appendChild(tr);}el('lot-count').textContent=`共 ${filtered.length.toLocaleString()} 个匹配片段 · 每页 ${pageSize} 条`;el('page-label').textContent=`${currentPage+1} / ${pages}`;el('prev-page').disabled=currentPage===0;el('next-page').disabled=currentPage===pages-1;}
for(const id of ['lot-search','lot-status','lot-sort'])el(id).addEventListener(id==='lot-search'?'input':'change',()=>{currentPage=0;render();});el('prev-page').onclick=()=>{currentPage--;render();};el('next-page').onclick=()=>{currentPage++;render();};el('export-lots').onclick=()=>{const csv=v=>'"'+String(v??'').replaceAll('"','""')+'"';const rows=[headers.map(csv).join(','),...filtered.map(r=>keys.map(k=>csv(r[k])).join(','))];const url=URL.createObjectURL(new Blob(['\ufeff'+rows.join('\r\n')],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='alphatrack-fifo-annualized-lots.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};render();
</script></body></html>'''
