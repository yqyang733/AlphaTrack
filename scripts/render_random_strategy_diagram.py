"""Render the strategy-two explainer as SVG, high-resolution PNG and offline HTML.

Requires Pillow and a CJK font. On non-macOS systems pass --font /path/to/font.
This is a presentation artifact; the execution rules remain in the QLab adapter.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT, SCALE = 1800, 1420, 2
BG = "#f6f7fb"
INK = "#2b2441"
MUTED = "#6d687e"
PURPLE = "#7546d9"
LAVENDER = "#eee7fc"
LINE = "#a9a3ba"


def render(output: Path, font_path: Path):
    canvas = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), BG)
    draw = ImageDraw.Draw(canvas)
    fonts = {}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
           '<title>AlphaStep · 随机选基版（策略二）</title>',
           '<desc>仅将Fundb前8与Fundc前5的评分排序改为每日随机选择，其他交易规则不变。</desc>',
           f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BG}"/>']

    def font(size):
        if size not in fonts:
            fonts[size] = ImageFont.truetype(str(font_path), size * SCALE)
        return fonts[size]

    def rect(x, y, w, h, *, fill="white", stroke="#dfdce8", radius=18, width=1.5):
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')
        draw.rounded_rectangle((x*SCALE, y*SCALE, (x+w)*SCALE, (y+h)*SCALE),
                               radius=radius*SCALE, fill=fill, outline=stroke, width=round(width*SCALE))

    def text(x, y, value, size=22, *, color=INK, align="left", bold=False):
        anchor = {"left":"start", "center":"middle", "right":"end"}[align]
        panchor = {"left":"lm", "center":"mm", "right":"rm"}[align]
        safe = html.escape(value)
        svg.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="PingFang SC,Hiragino Sans GB,Microsoft YaHei,Noto Sans CJK SC,sans-serif" font-size="{size}" font-weight="{650 if bold else 400}" text-anchor="{anchor}" dominant-baseline="central">{safe}</text>')
        draw.text((x*SCALE, y*SCALE), value, font=font(size), fill=color, anchor=panchor)

    def lines(x, y, values, *, size=22, gap=31, color=INK, align="center"):
        for i, value in enumerate(values):
            text(x, y + i * gap, value, size, color=color, align=align)

    def path(points, *, arrow=True, color=LINE, width=2):
        coords = " ".join(f"{x},{y}" for x,y in points)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>')
        draw.line([(x*SCALE,y*SCALE) for x,y in points], fill=color, width=width*SCALE, joint="curve")
        if arrow:
            x,y = points[-1]
            p = [(x,y), (x-9,y-5), (x-9,y+5)]
            svg.append(f'<polygon points="{" ".join(f"{a},{b}" for a,b in p)}" fill="{color}"/>')
            draw.polygon([(a*SCALE,b*SCALE) for a,b in p], fill=color)

    # Header and legend.
    text(50, 66, "AlphaStep · 随机选基版（策略二）", 40, bold=True)
    text(52, 118, "只替换 Fundb / Fundc 的选基顺序；其余买卖规则、资金口径与成交机制保持一致。", 22, color=MUTED)
    rect(1430, 40, 320, 54, fill="#e9defa", stroke="#d9c6f5", radius=27)
    text(1590, 67, "随机种子 seed = 42", 22, color=PURPLE, align="center", bold=True)

    # Group backgrounds.
    rect(365, 175, 1385, 605, fill="#faf9fd", stroke="#ddd8e8", radius=20)
    text(395, 207, "① 只买不卖：Fundb / Fundc / Fundf_1 / Fundg", 25, bold=True)
    rect(365, 810, 1385, 460, fill="#faf9fd", stroke="#ddd8e8", radius=20)
    text(395, 844, "② Fundd：低位分档买，高位分档卖", 25, bold=True)

    # Upstream observation flow and connectors are drawn behind the nodes.
    path([(300,620),(330,620)], arrow=False)
    path([(330,335),(330,1045)], arrow=False)
    for y in [335,550,700,1045]:
        path([(330,y),(410,y)])
    for y in [335,550,700]:
        path([(770,y),(825,y)])
    path([(1195,335),(1250,335)])
    path([(770,1045),(800,1045),(800,1035),(825,1035)])
    path([(800,1045),(800,1197),(1240,1197),(1240,1035),(1270,1035)])

    rect(50,525,250,190,fill=LAVENDER,stroke="#d4c3f2")
    text(175,559,"每个新净值观察日",24,color=PURPLE,align="center",bold=True)
    lines(175,602,["更新历史指标","逐基金判断","数据不足 → 不交易"],size=22,gap=32,color=PURPLE)

    # The only changed branch.
    rect(390,245,1335,205,fill="#f1eafb",stroke="#bfa2ed",radius=15,width=2)
    text(410,263,"本次唯一修改：评分排名 → 每日随机选择",18,color=PURPLE,bold=True)
    path([(390,335),(410,335)])
    path([(770,335),(825,335)])
    path([(1195,335),(1250,335)])
    rect(410,285,360,105,fill="white",stroke="#d9c8f4",radius=12)
    lines(590,309,["原合格候选池","Fundb / Fundc 各自类别","历史数据及评分须有效"],size=21,gap=28)
    rect(825,275,370,125,fill="#e4d4fa",stroke="#b993e8",radius=12,width=2)
    text(1010,306,"Fundb 随机选 8 只",26,color=PURPLE,align="center",bold=True)
    text(1010,341,"Fundc 随机选 5 只",26,color=PURPLE,align="center",bold=True)
    text(1010,377,"每日重抽 · 不放回 · 可复现",19,color=PURPLE,align="center")
    rect(1250,285,450,105,fill="white",stroke="#d9c8f4",radius=12)
    lines(1475,309,["入选且当日有新净值","每只买入 40 元","当前无自动卖出规则"],size=21,gap=28)
    text(410,424,"候选不足时全选；无新净值不下单、不递补，因此实际发单数可能少于 8 / 5。",19,color=MUTED)

    # Unchanged buy-only branches.
    rect(410,480,360,140,fill=LAVENDER)
    text(590,520,"Fundf_1 · 红利低波",25,color=PURPLE,align="center",bold=True)
    text(590,569,"p = 近 1 年历史位置",22,color=PURPLE,align="center")
    rect(825,480,875,140,fill="white")
    lines(865,503,["50% ≤ p ≤ 60%  →  买 40 元","40% ≤ p < 50%  →  买 80 元","30% ≤ p < 40%  →  买 120 元"],size=21,gap=40,align="left")
    lines(1325,521,["p < 30%  →  买 160 元","p > 60%  →  不买"],size=21,gap=48,align="left")
    rect(410,654,360,95,fill=LAVENDER)
    text(590,680,"Fundg · 美股指数",25,color=PURPLE,align="center",bold=True)
    text(590,719,"p = 近 1 年历史位置",22,color=PURPLE,align="center")
    rect(825,654,875,95,fill="white")
    text(865,683,"p ≤ 60%  →  每只买 10 元",23)
    text(1325,683,"p > 60%  →  不买",23)
    text(865,721,"当前无自动卖出规则",19,color=MUTED)

    # Unchanged Fundd branch.
    rect(410,940,360,220,fill=LAVENDER)
    text(590,976,"Fundd · 三周期最小值",24,color=PURPLE,align="center",bold=True)
    lines(590,1022,["x = min(近 5 年、近 10 年、","成立以来历史位置)"],size=21,gap=31,color=PURPLE)
    text(590,1111,"宽基 B = 50 元；其他 B = 10 元",19,color=PURPLE,align="center")
    rect(825,900,390,280,fill="white")
    text(1020,935,"买入区：x ≤ 20%",25,color=PURPLE,align="center",bold=True)
    lines(853,987,["10% ≤ x ≤ 20%  →  买 1 × B","5% ≤ x < 10%  →  买 2 × B","0 < x < 5%  →  买 3 × B","x ≤ 0  →  买 4 × B"],size=21,gap=37,align="left")
    text(1020,1148,"20% < x < 50%：不买、不卖",19,color=MUTED,align="center")
    rect(1270,900,430,280,fill="white")
    text(1485,935,"卖出区：x ≥ 50%",25,color=PURPLE,align="center",bold=True)
    lines(1294,983,["50%—100% 等距设置 50 档","每档本轮触发一次；跨档合并","档位越高，分配卖出的份额越多","完整比例计划：卖 90%，留 10%"],size=20,gap=32,align="left")
    text(1294,1123,"再进入买入区间 → 重置档位",19,color=PURPLE)
    text(1294,1154,"没有“必须盈利才卖”的硬条件",19,color=MUTED)
    text(410,1234,"指标不足时不交易；卖出比例按执行时持有份额计算，沿用原状态与分档规则。",20,color=MUTED)

    # Shared execution assumptions.
    text(50,1322,"共同执行口径",23,bold=True)
    text(225,1322,"独立空仓开始 · 无限资金追加 · 卖出款取出 · 下一可用净值成交",22,color=MUTED)
    text(50,1365,"沿用原净值观察日历与数据资格条件；随机种子与类别、日期、基金代码共同确定每日抽样。",20,color=MUTED)
    svg.append("</svg>")
    vector = "\n".join(svg)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix(".svg").write_text(vector,encoding="utf-8")
    canvas.save(output.with_suffix(".png"),optimize=True)
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AlphaStep · 随机选基版（策略二）</title><style>
body{{margin:0;background:{BG};color:{INK};font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
nav{{padding:14px 24px;border-bottom:1px solid #ddd8e8;display:flex;gap:24px;flex-wrap:wrap}}a{{color:{PURPLE}}}
main{{max-width:1800px;margin:auto}}svg{{display:block;width:100%;height:auto}}@media print{{nav{{display:none}}body{{background:white}}}}
</style></head><body><nav><b>AlphaTrack · 策略流程图</b><a href="{output.name}.png" download>下载高清 PNG</a><a href="{output.name}.svg" download>下载矢量 SVG</a></nav><main>{vector}</main></body></html>'''
    output.with_suffix(".html").write_text(document,encoding="utf-8")
    print(f"Generated {output}.svg/.png/.html; PNG {WIDTH*SCALE} x {HEIGHT*SCALE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="reports/alphastep_random_strategy")
    parser.add_argument("--font",type=Path,default=Path("/System/Library/Fonts/STHeiti Medium.ttc"))
    args = parser.parse_args()
    if not args.font.is_file():
        parser.error("CJK font missing; pass --font /path/to/a/CJK/font")
    render(Path(args.output),args.font)
