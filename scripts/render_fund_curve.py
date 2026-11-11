"""Render a fund's dividend-reinvested return, without signals or a backtest.

Reads standard date/nav/dividend/split CSV plus optional .metadata.json through
alphatrack.data.load_data. Produces self-contained Plotly HTML and a PNG using
installed Pillow and a local CJK font; no browser or image-export daemon needed.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import math
from pathlib import Path
from urllib.parse import quote, urlparse

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alphatrack.data import load_data


BG, INK, MUTED = '#f5f8fc', '#152b42', '#63748a'
BLUE, ORANGE, PURPLE, GRID = '#176ec1', '#d48419', '#9364c0', '#dbe4ee'
NOTE = '基金本身收益曲线 · 尚未运行策略回测'
FONT_CANDIDATES = (
    '/System/Library/Fonts/Hiragino Sans GB.ttc',
    '/System/Library/Fonts/STHeiti Medium.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
)


def choose_font(requested=None):
    if requested:
        if not Path(requested).is_file():
            raise ValueError(f'CJK font not found: {requested}')
        return Path(requested)
    for name in FONT_CANDIDATES:
        if Path(name).is_file():
            return Path(name)
    raise ValueError('A CJK font is required for PNG; pass --font /path/to/font')


def limits(values, *, zero=False):
    lo, hi = float(np.min(values)), float(np.max(values))
    if zero:
        lo, hi = min(lo, 0), max(hi, 0)
    span = hi-lo
    pad = (span if span else max(abs(hi), .1))*.07
    lo, hi = lo-pad, hi+pad
    rough = (hi-lo)/5
    base = 10**math.floor(math.log10(rough))
    step = next(v*base for v in [1, 2, 2.5, 5, 10] if v*base >= rough)
    low, high = math.floor(lo/step)*step, math.ceil(hi/step)*step
    return low, high, np.arange(low, high+step*.1, step)


def render_png(frame, title, output, font_path, warning=''):
    width, height = 1800, 1170
    canvas = Image.new('RGB', (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    fonts = {}
    def font(size):
        if size not in fonts:
            fonts[size] = ImageFont.truetype(str(font_path), size)
        return fonts[size]
    def text(x, y, label, size=24, color=INK, anchor='la'):
        draw.text((x,y), str(label), font=font(size), fill=color, anchor=anchor)
    def fit(label, max_width, size):
        while size > 26 and draw.textlength(label,font=font(size)) > max_width:
            size -= 1
        return size
    text(62, 37, title, fit(title, 1676, 40))
    text(64, 103, '诊断预览 · 历史分红缺净值，尚不可用于正式回测' if warning else NOTE,
         26, '#b54a28' if warning else BLUE)
    first, last = frame.date.iloc[0], frame.date.iloc[-1]
    growth = (frame.total_return_index.to_numpy()-1)*100
    days = (frame.date-first).dt.total_seconds().to_numpy()/86400
    duration = max(float(days[-1]), 1.)
    cash = frame.dividend.gt(0).to_numpy()
    split = frame.split.ne(1).to_numpy()
    text(64, 151, f'{first:%Y-%m-%d} → {last:%Y-%m-%d}  |  {len(frame):,} 条真实净值  |  分红 {int(cash.sum())} 次  |  折算 {int(split.sum())} 次', 23, MUTED)
    text(64, 201, '分红再投累计收益率', 26)
    text(1725, 201, f'截至末日  {growth[-1]:+.2f}%', 28, BLUE, 'ra')
    left, right = 148, 1715
    panel_specs = [(250, 665, growth, True), (810, 1015, frame.nav.to_numpy(), False)]
    # At most seven ticks; labels are actual dates on one linear calendar axis.
    tick_days = np.linspace(0, float(days[-1]), min(7, max(2, len(frame))))
    for top, bottom, series, percentage in panel_specs:
        lo, hi, ticks = limits(series, zero=percentage)
        draw.rectangle((left, top, right, bottom), fill='white')
        xy = lambda d,v: (left+(right-left)*float(d)/duration, bottom-(float(v)-lo)/(hi-lo)*(bottom-top))
        for tick in ticks:
            y=xy(0,tick)[1]
            draw.line((left,y,right,y), fill=GRID, width=1)
            label=(f'{tick:g}%' if percentage else f'{tick:g}')
            text(left-18,y,label,21,MUTED,'rm')
        for d in tick_days:
            x=xy(d,lo)[0]
            draw.line((x,top,x,bottom),fill=GRID,width=1)
            text(x,bottom+13,f'{first+pd.Timedelta(days=float(d)):%Y-%m-%d}',18,MUTED,'ma')
        if percentage:
            zero_y=xy(0,0)[1]
            draw.line((left,zero_y,right,zero_y),fill='#aab9c9',width=2)
        points=[xy(d,v) for d,v in zip(days,series)]
        if len(points)>1:
            draw.line(points,fill=BLUE if percentage else '#7692aa',width=3)
        else:
            x,y=points[0];draw.ellipse((x-4,y-4,x+4,y+4),fill=BLUE)
        if percentage:
            for mask,color in [(cash,ORANGE),(split,PURPLE)]:
                for d,v in zip(days[mask],series[mask]):
                    x,y=xy(d,v);draw.ellipse((x-5,y-5,x+5,y+5),fill=color,outline='white',width=1)
            x,y=points[0];draw.ellipse((x-4,y-4,x+4,y+4),fill=BLUE)
    text(64, 735, '原始单位净值（元）', 26)
    draw.ellipse((1330,739,1342,751),fill=ORANGE);text(1355,731,'分红日',21,MUTED)
    draw.ellipse((1500,739,1512,751),fill=PURPLE);text(1525,731,'份额折算日',21,MUTED)
    text(64, 1081, 'QLab口径：首个真实净值日为0%；现金分红按除息日净值再投，份额折算同步调整。', 22, MUTED)
    text(64, 1120, warning or '仅核对基金收益曲线；未生成交易信号、未运行 Fundg 或其他策略。',
         22, '#b54a28' if warning else MUTED)
    canvas.save(output, optimize=True)


def render(data, output, font=None):
    data, output = Path(data).resolve(), Path(output).resolve()
    if output.suffix.lower() != '.html':
        output = output.with_suffix('.html')
    font_path = choose_font(font)
    frame, metadata = load_data(data)
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    if metadata.get('data_sha256') and metadata['data_sha256'] != digest:
        raise ValueError('CSV does not match metadata data_sha256')
    if not np.isclose(frame.total_return_index.iloc[0],1.,rtol=0,atol=1e-12):
        raise ValueError('Return curve must start at zero on the first real NAV date')
    code = str(metadata.get('code') or data.stem)
    name = str(metadata.get('name') or metadata.get('fund_name') or '')
    title = f'{code} {name} · 分红再投收益曲线'.strip()
    warning = str(metadata.get('curve_warning') or '')
    notice = '诊断预览 · 历史分红缺净值，尚不可用于正式回测' if warning else NOTE
    warning_html = (f'<p style="color:#b54a28;font-weight:650">{html.escape(warning)}</p>' if warning else '')
    dates = frame.date.dt.strftime('%Y-%m-%d').tolist()
    returns = ((frame.total_return_index-1)*100).to_numpy()
    events = frame[frame.dividend.ne(0)|frame.split.ne(1)]
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.13,
                           row_heights=[.70,.30], subplot_titles=['分红再投累计收益率（首日为0%）','原始单位净值'])
    figure.add_trace(go.Scatter(x=dates,y=returns,name='分红再投累计收益率',line=dict(color=BLUE,width=2),
                               hovertemplate='%{x}<br>累计收益率 %{y:.2f}%<extra></extra>'),row=1,col=1)
    for column,mask,label,color,symbol in [('dividend',frame.dividend.ne(0),'现金分红日',ORANGE,'circle'),
                                          ('split',frame.split.ne(1),'份额折算日',PURPLE,'diamond')]:
        part=frame[mask]
        if part.empty:
            continue
        details=[[float(row.dividend),float(row.nav),float(row.split)] for row in part.itertuples()]
        figure.add_trace(go.Scatter(x=part.date.dt.strftime('%Y-%m-%d').tolist(),y=(part.total_return_index-1)*100,
                                   mode='markers',name=label,marker=dict(color=color,size=7,symbol=symbol),customdata=details,
                                   hovertemplate='%{x}<br>累计收益率 %{y:.2f}%<br>每份分红 %{customdata[0]:.6f} 元<br>除息日净值 %{customdata[1]:.6f}<br>份额倍数 %{customdata[2]:g}<extra></extra>'),row=1,col=1)
    figure.add_trace(go.Scatter(x=dates,y=frame.nav,name='原始单位净值',line=dict(color='#7692aa',width=1.8),
                               hovertemplate='%{x}<br>单位净值 %{y:.6f} 元<extra></extra>'),row=2,col=1)
    figure.add_hline(y=0,line_width=1,line_color='#aab9c9',row=1,col=1)
    figure.update_layout(height=850,template='plotly_white',paper_bgcolor='white',plot_bgcolor='white',hovermode='x unified',
                         margin=dict(l=85,r=35,t=65,b=60),legend=dict(orientation='h',x=0,y=1.08),
                         font=dict(family='-apple-system,BlinkMacSystemFont,PingFang SC,Microsoft YaHei,sans-serif',size=13,color=INK))
    figure.update_yaxes(title_text='累计收益率（%）',ticksuffix='%',row=1,col=1)
    figure.update_yaxes(title_text='单位净值（元）',row=2,col=1)
    figure.update_xaxes(type='date',tickformat='%Y-%m',hoverformat='%Y-%m-%d',showspikes=True)
    figure.update_xaxes(title_text='真实净值日期',row=2,col=1)
    fragment=figure.to_html(full_html=False,include_plotlyjs=True,div_id='fund-total-return-curve',
                            config={'responsive':True,'displaylogo':False,'toImageButtonOptions':{'format':'png','scale':2}})
    rows=''.join(f'<tr><td>{row.date:%Y-%m-%d}</td><td>{row.nav:.6f}</td><td>{row.dividend:.6f}</td><td>{row.split:g}</td></tr>' for row in events.itertuples())
    source_url=str(metadata.get('source_url',''))
    source_link=(f'<a href="{html.escape(source_url,quote=True)}">净值来源</a>' if urlparse(source_url).scheme in {'http','https'} else '')
    png=output.with_suffix('.png')
    document=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>body{{margin:0;background:{BG};color:{INK};font:16px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}}main{{max-width:1400px;margin:auto;padding:26px 24px}}h1{{font-size:29px;margin:0}}.notice{{color:{BLUE};font-weight:650;margin:10px 0}}.meta,.notes{{color:{MUTED};font-size:14px}}.chart{{background:white;border:1px solid {GRID};border-radius:12px;padding:8px;margin:20px 0}}a{{color:{BLUE}}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{border-bottom:1px solid {GRID};padding:8px 12px;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{overflow-wrap:anywhere}}@media(max-width:600px){{main{{padding:18px 10px}}h1{{font-size:23px}}.chart{{padding:0}}}}</style></head><body><main>
<h1>{html.escape(title)}</h1><p class="notice">{notice}</p>{warning_html}
<p class="meta">{dates[0]} 至 {dates[-1]} · {len(frame):,} 条真实净值 · 累计收益率 {returns[-1]:+.2f}% · 末日单位净值 {frame.nav.iloc[-1]:.6f} 元</p>
<p><a href="{quote(png.name)}" download>下载高清PNG</a>　{source_link}</p><div class="chart">{fragment}</div>
<p class="notes">QLab口径：首个真实净值日累计收益率为0%；现金分红按除息日单位净值转换为新增份额继续持有，份额折算同步调整。主图为基金本身的分红再投累计收益率，副图为未复权的原始单位净值。按真实净值观察日绘图，未补造缺失净值。</p>
<p class="notes">本页只用于核对基金曲线，未生成交易信号，也未运行 Fundg 或其他策略回测。图例可切换曲线显示，拖动图表可缩放。</p>
<details><summary>分红与份额折算记录（{len(events)}条）</summary><table><thead><tr><th>除息/折算日</th><th>单位净值</th><th>每份现金分红（元）</th><th>份额倍数</th></tr></thead><tbody>{rows or '<tr><td colspan="4">当前输入没有分红或折算记录</td></tr>'}</tbody></table></details>
<p class="meta">输入CSV SHA256：<code>{digest}</code></p></main></body></html>'''
    output.parent.mkdir(parents=True,exist_ok=True)
    render_png(frame,title,png,font_path,warning)
    output.write_text(document,encoding='utf-8')
    return {'html':str(output),'png':str(png),'rows':len(frame),'first_date':dates[0],'last_date':dates[-1],
            'cumulative_return':float(returns[-1]/100),'dividend_events':int(frame.dividend.ne(0).sum()),'split_events':int(frame.split.ne(1).sum()),'data_sha256':digest}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path,help='output HTML; PNG uses the same stem')
    parser.add_argument('--font',type=Path,help='optional local CJK font for the PNG')
    args=parser.parse_args()
    import json
    print(json.dumps(render(args.data,args.output,args.font),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
