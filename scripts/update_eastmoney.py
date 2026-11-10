"""Refresh a checked public Eastmoney NAV snapshot; never eval remote JavaScript.
Usage: python scripts/update_eastmoney.py --code 110020 --output data/110020.csv
For nonempty dividend/split events, refuses automatic import: provide vetted
normalized dividend/split columns instead of silently ignoring corporate actions.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen


def convert(text, code, output, source_url):
    def extract(name):
        match = re.search(r"var\s+"+name+r"\s*=\s*(.*?);", text)
        if not match:
            raise ValueError(f"Missing {name}")
        return json.loads(match[1])
    if extract("fS_code") != code:
        raise ValueError("Fund identity mismatch")
    values = extract("Data_netWorthTrend")
    accumulated = extract("Data_ACWorthTrend")
    if not values:
        raise ValueError("Empty source")
    # This adapter intentionally limited to verified no-action histories.
    ac_map = {x[0]:x[1] for x in accumulated}
    if any(v.get("unitMoney") or abs(v["y"]-ac_map.get(v["x"],float('inf')))>1e-6 for v in values):
        raise ValueError("Corporate actions found: normalize dividends/splits before backtesting")
    from zoneinfo import ZoneInfo
    rows = [{"date":datetime.fromtimestamp(v["x"]/1000,tz=ZoneInfo("Asia/Shanghai")).date().isoformat(),
             "nav":v["y"],"dividend":0,"split":1} for v in values]
    dates = [r["date"] for r in rows]
    if dates != sorted(set(dates)) or any(not isinstance(r['nav'],(int,float)) or r['nav']<=0 for r in rows):
        raise ValueError("Malformed NAV source")
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["date","nav","dividend","split"])
        writer.writeheader(); writer.writerows(rows)
    meta={"code":code,"name":extract("fS_name"),"source_url":source_url,
          "retrieved_at":datetime.now(timezone.utc).isoformat(),"first_date":dates[0],
          "last_date":dates[-1],"rows":len(rows),
          "source_sha256":hashlib.sha256(text.encode()).hexdigest(),
          "data_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
          "corporate_actions":"No event annotations; cumulative NAV equals unit NAV for all observations",
          "strategy_source":"https://github.com/yqyang733/QLab",
          "strategy_source_commit":"1c1eb37f45a298b0aec3fabb673d269fa95892d5",
          "limitations":["NAV observations, not a complete daily trading calendar",
                          "Early history is weekly; next observation can be more than one trading day away",
                          "No historical subscription/redemption status supplied; theoretical NAV execution"]}
    output.with_suffix('.metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return meta


def main():
    p=argparse.ArgumentParser(); p.add_argument('--code',default='110020'); p.add_argument('--output',default='data/110020.csv'); p.add_argument('--source-file')
    args=p.parse_args()
    if not re.fullmatch(r'\d{6}', args.code):
        p.error('code must have six digits')
    url=f'https://fund.eastmoney.com/pingzhongdata/{args.code}.js'
    if args.source_file:
        text=Path(args.source_file).read_text(encoding='utf-8')
    else:
        req=Request(url,headers={'User-Agent':'AlphaTrack/0.1 public-research'})
        with urlopen(req,timeout=45) as response:
            text=response.read().decode('utf-8-sig')
    print(json.dumps(convert(text,args.code,args.output,url),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
