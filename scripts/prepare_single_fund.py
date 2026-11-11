"""Fetch/normalize one fund for QLab-compatible curve review, never backtest.

Use --from-raw to reproduce a saved nav.js/dividends.html/sources.json snapshot.
The selected trading rule is metadata only until the user approves the curve.
"""
import argparse
from datetime import datetime
from decimal import Decimal
import hashlib
from html import unescape
from io import StringIO
import json
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alphatrack.data import load_data
from alphatrack.strategies.qlab import QLabRules

NUMBER = r"\d+(?:\.\d+)?"
COLUMNS = ["分红日", "分红类型", "分红"]


def extract(text, name):
    match = re.search(r"var\s+" + re.escape(name) + r"\s*=\s*(.*?);", text)
    if match is None:
        raise ValueError(f"Missing source field: {name}")
    return json.loads(match[1])


def parse_events(content, code, as_of):
    """Apply QLab's ex-date, yuan/share and normalized split conventions."""
    title = re.search(r"<title[^>]*>(.*?)</title>", content, re.I | re.S)
    if not title or not re.search(rf"[（(]{re.escape(code)}[)）]", unescape(title[1])):
        raise ValueError("Dividend page identity mismatch")
    tables = pd.read_html(StringIO(content), flavor="lxml")
    cash = [t for t in tables if "除息日" in t.columns]
    splits = [t for t in tables if {"拆分折算日", "拆分折算比例"} <= set(t.columns)]
    if len(cash) != 1 or len(splits) != 1:
        raise ValueError("Cannot uniquely identify both action tables")
    records, aliases = [], []
    for table, kind, date_col, empty in (
        (cash[0], "现金分红", "除息日", "暂无分红信息"),
        (splits[0], "份额折算", "拆分折算日", "暂无拆分信息"),
    ):
        if len(table) == 1 and empty in str(table.iloc[0][date_col]):
            continue
        if table.empty:
            raise ValueError("Empty action table without explicit no-events marker")
        if kind == "现金分红":
            amount_cols = [c for c in table if re.fullmatch(rf"每(?:{NUMBER})?份分红", str(c))]
            if len(amount_cols) != 1:
                raise ValueError("Ambiguous dividend units")
            column = amount_cols[0]
            units = Decimal(re.fullmatch(rf"每({NUMBER})?份分红", str(column))[1] or "1")
        for row in table.to_dict("records"):
            day = str(row[date_col]).strip()
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
                raise ValueError("Invalid action date")
            date = pd.Timestamp(day)
            if kind == "现金分红":
                text = re.sub(r"\s+", "", str(row[column]))
                match = re.fullmatch(rf"每({NUMBER})?份派(?:发)?现金({NUMBER})元", text)
                if match:
                    if Decimal(match[1] or "1") != units:
                        raise ValueError("Dividend header/body units disagree")
                    amount = Decimal(match[2])
                elif re.fullmatch(rf"{NUMBER}(?:元)?", text):
                    amount = Decimal(text.removesuffix("元"))
                else:
                    raise ValueError("Unrecognized dividend amount")
                if units <= 0 or amount <= 0:
                    raise ValueError("Nonpositive dividend")
                value = format(amount / units, "f")
                aliases.append({k: str(row.get(k, "")) for k in ("除息日", "权益登记日", "分红发放日")})
            else:
                text = str(row["拆分折算比例"]).strip().replace("：", ":")
                if not re.fullmatch(rf"{NUMBER}:{NUMBER}", text):
                    raise ValueError("Unrecognized split ratio")
                before, after = map(Decimal, text.split(":"))
                if before <= 0 or after <= 0:
                    raise ValueError("Nonpositive split")
                value = "1:" + format(after / before, "f")
            records.append({"分红日": date, "分红类型": kind, "分红": value})
    result = pd.DataFrame(records, columns=COLUMNS)
    if result.duplicated(["分红日", "分红类型"]).any():
        raise ValueError("Duplicate action date/type")
    if len(result):
        result = result[result["分红日"] <= pd.Timestamp(as_of)].sort_values(["分红日", "分红类型"]).reset_index(drop=True)
    result.attrs["cash_event_dates"] = aliases
    return result


def normalize(nav_text, events, code):
    if events.duplicated(["分红日", "分红类型"]).any():
        raise ValueError("Duplicate action date/type")
    if extract(nav_text, "fS_code") != code:
        raise ValueError("NAV identity mismatch")
    observations = extract(nav_text, "Data_netWorthTrend")
    if not observations:
        raise ValueError("No unit NAV history")
    frame = pd.DataFrame({
        "date": [datetime.fromtimestamp(v["x"] / 1000, ZoneInfo("Asia/Shanghai")).date().isoformat() for v in observations],
        "nav": [v["y"] for v in observations], "dividend": 0., "split": 1.,
    })
    if frame.date.duplicated().any() or not frame.date.is_monotonic_increasing or not np.isfinite(frame.nav).all() or (frame.nav <= 0).any():
        raise ValueError("Malformed NAV history")
    lookup = {day: i for i, day in enumerate(frame.date)}
    event_keys = set()
    for row in events.to_dict("records"):
        day, kind, value = str(row["分红日"].date()), row["分红类型"], row["分红"]
        if day > frame.date.iloc[-1]:
            continue
        if day not in lookup:
            raise ValueError(f"Missing action NAV: {code} {day}; no date shifting or imputation")
        if kind == "现金分红":
            frame.at[lookup[day], "dividend"] = float(value)
        elif kind == "份额折算":
            before, after = map(float, value.split(":"))
            if before != 1:
                raise ValueError("Split must be normalized to 1:x")
            frame.at[lookup[day], "split"] = after
        else:
            raise ValueError("Unknown action type")
        event_keys.add((day, kind))
    for day, observation in zip(frame.date, observations):
        note = observation.get("unitMoney", "")
        if note:
            kind = "现金分红" if "分红" in note else "份额折算" if "拆分" in note else None
            if (day, kind) not in event_keys:
                raise ValueError(f"Unmatched NAV action annotation: {day} {note}")
    return frame


def apply_approved_omissions(nav_text, events, code, approval):
    """Exclude only exact, explicitly accepted events whose ex-date has no NAV."""
    if approval is None:
        return events, []
    if approval.get("code") != code or not str(approval.get("authorization", "")).strip():
        raise ValueError("Omission approval must identify this fund and explicit user authorization")
    requests = approval.get("events")
    if not isinstance(requests, list) or not requests:
        raise ValueError("Omission approval must list exact events")
    if extract(nav_text, "fS_code") != code:
        raise ValueError("NAV identity mismatch")
    nav_dates = {datetime.fromtimestamp(v["x"] / 1000, ZoneInfo("Asia/Shanghai")).date().isoformat()
                 for v in extract(nav_text, "Data_netWorthTrend")}
    removed, seen = [], set()
    remaining = events.copy()
    for request in requests:
        day, kind = request["date"], request["type"]
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) or kind != "现金分红":
            raise ValueError("Only an exact dated cash dividend can be approved for omission")
        key = (day, kind)
        if key in seen:
            raise ValueError("Duplicate approved omission")
        seen.add(key)
        mask = remaining["分红日"].eq(pd.Timestamp(day)) & remaining["分红类型"].eq(kind)
        matches = remaining[mask]
        if len(matches) != 1 or Decimal(str(matches.iloc[0]["分红"])) != Decimal(str(request["value"])):
            raise ValueError("Approved omission does not match exactly one source event")
        if day in nav_dates:
            raise ValueError("Approved omission is no longer applicable: ex-date NAV exists")
        removed.append({"date": day, "type": kind, "value": str(matches.iloc[0]["分红"]),
                        "reason": "User explicitly accepted omitting this missing-NAV cash dividend"})
        remaining = remaining[~mask].copy()
    return remaining.reset_index(drop=True), removed


def prepare(code, rule, directory, *, from_raw=False, approved_omissions=None):
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError("Fund code must contain six digits")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if not from_raw:
        sources = {}
        for filename, url in (("nav.js", f"https://fund.eastmoney.com/pingzhongdata/{code}.js"),
                              ("dividends.html", f"https://fundf10.eastmoney.com/fhsp_{code}.html")):
            if (directory / filename).exists():
                raise ValueError("Snapshot exists; use --from-raw or a new snapshot directory")
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=35) as response:
                body = response.read()
            (directory / filename).write_bytes(body)
            sources[filename] = {"url": url, "sha256": hashlib.sha256(body).hexdigest(),
                                 "retrieved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
        (directory / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2))
    sources = json.loads((directory / "sources.json").read_text())
    for filename in ("nav.js", "dividends.html"):
        if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != sources[filename]["sha256"]:
            raise ValueError("Raw snapshot hash mismatch")
    as_of = pd.Timestamp(sources["dividends.html"]["retrieved_at"]).tz_convert("Asia/Shanghai").tz_localize(None).normalize()
    events = parse_events((directory / "dividends.html").read_text(encoding="utf-8-sig"), code, as_of)
    nav_text = (directory / "nav.js").read_text(encoding="utf-8-sig")
    approval = json.loads(Path(approved_omissions).read_text()) if approved_omissions else None
    included_events, omitted = apply_approved_omissions(nav_text, events, code, approval)
    frame = normalize(nav_text, included_events, code)
    if pd.Timestamp(frame.date.iloc[-1]) > as_of:
        raise ValueError("NAV date is after snapshot date")
    data_path = directory / f"{code}.csv"
    frame.to_csv(data_path, index=False, float_format="%.15g")
    events.to_csv(directory / f"{code}.dividends.csv", index=False, date_format="%Y-%m-%d")
    actual, _ = load_data(data_path)
    rules = QLabRules(ROOT / "vendor/qlab")
    original = rules.calculator.ReturnCalculator.calculate_return_with_dividend_reinvest(
        actual[["date", "nav"]].rename(columns={"date": "净值日期", "nav": "单位净值"}), events)
    delta = float(np.max(np.abs(actual.total_return_index - (1 + original["累计收益率"]))))
    if delta > 1e-12:
        raise ValueError(f"QLab curve parity failed: {delta}")
    meta = {"code": code, "name": extract(nav_text, "fS_name"), "requested_strategy": rule,
            "stage": "curve_review_pending", "backtest_executed": False,
            "rows": len(frame), "first_date": frame.date.iloc[0], "last_date": frame.date.iloc[-1],
            "cash_dividend_count": int((frame.dividend > 0).sum()), "split_count": int((frame.split != 1).sum()),
            "cumulative_return": float(actual.total_return_index.iloc[-1] - 1),
            "final_nav": float(frame.nav.iloc[-1]), "qlab_max_absolute_error": delta,
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "calculator_sha256": rules.source_hashes["core/calculator.py"],
            "sources": sources, "cash_event_dates": events.attrs["cash_event_dates"],
            "curve_basis": "QLab cash dividends reinvested at ex-date unit NAV; normalized split ratios; first observed NAV baseline",
            "planned_start": "2010-01-01", "no_data_before_inception": True}
    if omitted:
        meta.update(approved_omissions=omitted, omission_authorization=approval["authorization"],
                    recorded_cash_dividend_count=int(events["分红类型"].eq("现金分红").sum()),
                    backtest_allowed=True,
                    warnings=[f"经用户确认，本次暂不计入 {code} 在 {item['date']} 的每份 {item['value']} 元分红（除息日缺净值）；其余分红正常计入，未移动日期或补造净值。" for item in omitted])
    data_path.with_suffix(".metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({"data": str(data_path), **{k: v for k, v in meta.items() if k not in {"sources", "cash_event_dates"}}}, ensure_ascii=False, indent=2))
    return data_path, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", required=True)
    parser.add_argument("--rule", required=True, choices=["Fundb", "Fundc", "Fundd", "Fundf_1", "Fundg"])
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--from-raw", action="store_true")
    parser.add_argument("--approved-omissions", type=Path, help="Exact missing-NAV cash events explicitly accepted by the user; no implicit exceptions")
    args = parser.parse_args()
    prepare(args.code, args.rule, args.snapshot_dir, from_raw=args.from_raw,
            approved_omissions=args.approved_omissions)
