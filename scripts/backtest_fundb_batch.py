"""Batch the user's daily-40 single-fund rule using a pinned QLab catalog/actions.

NAV downloads are immutable snapshots. QLab source files are copied byte-for-byte;
only the two previously documented 000931 missing-NAV events follow its existing
lookup policy. All other missing action NAVs or unmatched annotations fail closed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alphatrack.data import load_data
from alphatrack.strategies.qlab import QLabRules
from scripts.backtest_single_fund import run
from scripts.export_window_tables import tsv
from scripts.prepare_single_fund import extract, normalize
from scripts.verify_portfolio import verify

KNOWN_MISSING = {("000931", "2017-03-22"): "0.08", ("000931", "2017-11-14"): "0.04"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(asset, directory):
    folder = directory / asset["code"]
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "nav.js"
    provenance = folder / "nav.source.json"
    if target.exists() and provenance.exists():
        saved = json.loads(provenance.read_text())
        if sha(target) != saved["sha256"]:
            raise ValueError("Saved NAV hash mismatch")
        return
    if target.exists() or provenance.exists():
        raise ValueError("Incomplete existing download; use a new snapshot directory")
    url = f"https://fund.eastmoney.com/pingzhongdata/{asset['code']}.js"
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as response:
                body = response.read()
            if extract(body.decode("utf-8-sig"), "fS_code") != asset["code"]:
                raise ValueError("NAV identity mismatch")
            target.write_bytes(body)
            provenance.write_text(json.dumps(dict(url=url, sha256=sha(target),
                retrieved_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()), indent=2))
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def prepare(asset, source, directory, commit):
    code = asset["code"]
    folder = directory / code
    nav_text = (folder / "nav.js").read_text(encoding="utf-8-sig")
    nav_source = json.loads((folder / "nav.source.json").read_text())
    if sha(folder / "nav.js") != nav_source["sha256"]:
        raise ValueError("NAV source changed")
    original = source / f"data_files/dividend/fund/{code}.csv"
    copied = folder / f"{code}.dividends.csv"
    if copied.exists() and sha(copied) != sha(original):
        raise ValueError("QLab dividend snapshot changed; use a new snapshot directory")
    copied.write_bytes(original.read_bytes())
    events = pd.read_csv(copied, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    events["分红日"] = pd.to_datetime(events["分红日"])
    if events.duplicated(["分红日", "分红类型"]).any():
        raise ValueError("Duplicate action date/type")
    dates = {datetime.fromtimestamp(v["x"] / 1000, ZoneInfo("Asia/Shanghai")).date().isoformat()
             for v in extract(nav_text, "Data_netWorthTrend")}
    omissions = []
    for row in events.to_dict("records"):
        day = row["分红日"].date().isoformat()
        if day <= max(dates) and day not in dates:
            expected = KNOWN_MISSING.get((code, day))
            if expected is None or row["分红类型"] != "现金分红" or float(row["分红"]) != float(expected):
                raise ValueError(f"Unexpected missing action NAV: {code} {day}")
            omissions.append(dict(date=day, type=row["分红类型"], value=row["分红"]))
    included = events[~events["分红日"].dt.strftime("%Y-%m-%d").isin([o["date"] for o in omissions])]
    frame = normalize(nav_text, included, code)
    if frame.date.iloc[-1] > nav_source["retrieved_at"][:10]:
        raise ValueError("Future NAV observation")
    path = folder / f"{code}.csv"
    frame.to_csv(path, index=False, float_format="%.15g")
    normalized, _ = load_data(path)
    rules = QLabRules(ROOT / "vendor/qlab")
    if sha(source / "core/calculator.py") != rules.source_hashes["core/calculator.py"]:
        raise ValueError("Latest QLab calculator differs from pinned calculator")
    reference = rules.calculator.ReturnCalculator.calculate_return_with_dividend_reinvest(
        normalized[["date", "nav"]].rename(columns={"date": "净值日期", "nav": "单位净值"}), events)
    error = float(np.max(np.abs(normalized.total_return_index - (1 + reference["累计收益率"]))))
    if error > 1e-12:
        raise ValueError(f"QLab curve parity failed: {code} {error}")
    gaps = normalized.date.diff().dt.days
    large_gaps = [dict(before=normalized.date.iloc[i-1].date().isoformat(),
                       after=normalized.date.iloc[i].date().isoformat(), days=int(gaps.iloc[i]))
                  for i in range(1, len(normalized)) if gaps.iloc[i] > 60]
    warnings = []
    if omissions:
        warnings.append("沿用QLab现有缺除息日净值处理：000931的2017-03-22每份0.08元、2017-11-14每份0.04元不进入再投计算；原始记录保留，未移动日期或补造净值。")
    for gap in large_gaps:
        warnings.append(f"净值数据存在长缺口：{gap['before']}至{gap['after']}；无净值日不产生买入，未补造数据。")
    meta = dict(code=code, name=asset["name"], source_name=extract(nav_text, "fS_name"),
        requested_strategy="Fundb", stage="batch_authorized", source_kind="qlab_snapshot",
        rows=len(frame), first_date=frame.date.iloc[0], last_date=frame.date.iloc[-1],
        cash_dividend_count=int((frame.dividend > 0).sum()), split_count=int((frame.split != 1).sum()),
        qlab_max_absolute_error=error, data_sha256=sha(path),
        calculator_sha256=rules.source_hashes["core/calculator.py"], qlab_catalog_commit=commit,
        sources={"nav.js": nav_source, "dividends.csv": {"path": str(original), "sha256": sha(copied)},
                 "catalog": {"path": str(source / "data_files/Fund"), "sha256": sha(source / "data_files/Fund")}},
        existing_policy_omissions=omissions, warnings=warnings, large_nav_gaps=large_gaps,
        curve_basis="QLab ex-date yuan/share reinvestment and normalized splits; exact QLab event snapshot",
        authorization="用户要求最新QLab Fundb全部逐只每日40元直接回测，排除003358与161119")
    path.with_suffix(".metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qlab-source", required=True, type=Path)
    parser.add_argument("--qlab-commit", required=True)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--from-snapshot", action="store_true")
    parser.add_argument("--batch-authorized", action="store_true")
    args = parser.parse_args()
    if not args.batch_authorized:
        parser.error("Explicit user batch authorization is required")
    source, directory, output = args.qlab_source.resolve(), args.snapshot_dir.resolve(), args.output_dir.resolve()
    catalog = QLabRules(source).catalog(["Fundb"])
    assets = [a for a in catalog if a["code"] not in {"003358", "161119"}]
    directory.mkdir(parents=True, exist_ok=True)
    if not args.from_snapshot:
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda a: download(a, directory), assets))
    metadata = [prepare(a, source, directory, args.qlab_commit) for a in assets]
    end = min(m["last_date"] for m in metadata)
    manifest = dict(qlab_commit=args.qlab_commit, catalog_sha256=sha(source / "data_files/Fund"),
        excluded=["003358", "161119"], amount=40, start="2010-01-01", common_end=end,
        funds=metadata, count=len(assets))
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(dict(stage="prepared", count=len(assets), common_end=end,
        warnings={m["code"]: m["warnings"] for m in metadata if m["warnings"]}), ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    output.mkdir(parents=True, exist_ok=True)
    rows, headers, summaries = [], None, []
    for asset in assets:
        code = asset["code"]
        prefix = output / f"single_{code}_fundb_2010"
        with redirect_stdout(io.StringIO()):
            payload = run(directory / code / f"{code}.csv", "Fundb", prefix.with_suffix(".html"),
                          end=end, batch_authorized=True, fundb_daily_buy=True)
            validation = verify(prefix.with_suffix(".html"))
        values = prefix.with_suffix(".paste_with_header_wan.tsv").read_text().splitlines()
        if headers is None:
            headers = ["策略"] + values[0].split("\t")
        elif headers[1:] != values[0].split("\t"):
            raise ValueError("Inconsistent output columns")
        rows.append([f"{code}（每交易日40）"] + values[1].split("\t"))
        summaries.append(dict(code=code, name=asset["name"], **payload["summary"]))
        print(json.dumps(dict(code=code, principal=payload["summary"]["total_contributions"],
            profit=payload["summary"]["total_pnl"], xirr=payload["summary"]["xirr"]), ensure_ascii=False), flush=True)
    (output / "fundb_daily40.tsv").write_text(tsv([headers, *rows]))
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    (output / "fundb_daily40.md").write_text("\n".join(lines) + "\n")
    (output / "fundb_daily40.summary.json").write_text(json.dumps(dict(manifest=manifest, results=summaries), ensure_ascii=False, indent=2))
    print(f"Complete: {output / 'fundb_daily40.tsv'}", flush=True)


if __name__ == "__main__":
    main()
