"""Separate single-fund tables: Fundc daily 40; Fundd/Fundf_1 exact QLab rules."""
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import io
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alphatrack.strategies.qlab import QLabRules
from scripts.backtest_fundb_batch import download, prepare, sha
from scripts.backtest_single_fund import run
from scripts.export_window_tables import tsv
from scripts.verify_portfolio import verify
from scripts.prepare_single_fund import extract, parse_events
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.request import Request, urlopen
import pandas as pd

AUTHORIZATION = "用户明确要求最新QLab Fundc每只交易日40元，Fundf_1/Fundd按原交易规则，逐只批量回测并分类型输出；不重复003358/161119"


def refresh_new_events(asset, source, directory, from_snapshot=False):
    """Refresh only unmatched NAV annotations through QLab's own conflict-safe merge."""
    code = asset["code"]
    folder = directory / asset["asset_type"] / code
    event_path = source / f"data_files/dividend/fund/{code}.csv"
    refresh_path = folder / "dividend_refresh.json"
    if refresh_path.exists():
        refresh = json.loads(refresh_path.read_text())
        for filename, digest in refresh["snapshot_hashes"].items():
            if sha(folder / filename) != digest:
                raise ValueError("Dividend refresh snapshot changed")
        if sha(event_path) != refresh["merged_sha256"]:
            raise ValueError("Refreshed QLab dividend source changed")
        return
    events = pd.read_csv(event_path, dtype=str, encoding="utf-8-sig")
    keys = set(zip(events["分红日"], events["分红类型"]))
    unmatched = []
    for row in extract((folder / "nav.js").read_text(encoding="utf-8-sig"), "Data_netWorthTrend"):
        note = row.get("unitMoney", "")
        if not note:
            continue
        day = datetime.fromtimestamp(row["x"] / 1000, ZoneInfo("Asia/Shanghai")).date().isoformat()
        kind = "现金分红" if "分红" in note else "份额折算" if "拆分" in note else None
        if (day, kind) not in keys:
            unmatched.append(dict(date=day, type=kind, annotation=note))
    if not unmatched:
        return
    if from_snapshot:
        raise ValueError(f"Missing saved dividend refresh for {code}")
    if (folder / f"{code}.metadata.json").exists():
        raise ValueError("Cannot refresh an already normalized snapshot; select a new directory")
    url = f"https://fundf10.eastmoney.com/fhsp_{code}.html"
    raw_path = folder / "dividends.refresh.html"
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as response:
        raw_path.write_bytes(response.read())
    retrieved = datetime.now(ZoneInfo("Asia/Shanghai"))
    remote = parse_events(raw_path.read_text(encoding="utf-8-sig"), code, retrieved.date().isoformat())
    spec = importlib.util.spec_from_file_location("batch_qlab_dividends", source / "data/dividends.py")
    module = importlib.util.module_from_spec(spec)
    exec(compile((source / "data/dividends.py").read_bytes(), str(source / "data/dividends.py"), "exec"), module.__dict__)
    original_bytes = event_path.read_bytes()
    local = module.read_local(event_path)
    merged, count, conflicts = module.merge_records(local, remote)
    if conflicts or not count:
        raise ValueError(f"Cannot reconcile new NAV events for {code}: {conflicts}")
    new_keys = set(zip(merged["分红日"].dt.strftime("%Y-%m-%d"), merged["分红类型"]))
    if not all((item["date"], item["type"]) in new_keys for item in unmatched):
        raise ValueError("Unmatched NAV event absent from verified dividend page")
    original_path = folder / "qlab_original.dividends.csv"
    original_path.write_bytes(original_bytes)
    module.write_local(event_path, merged, original_bytes)
    additions = [dict(date=r["分红日"].date().isoformat(), type=r["分红类型"], value=r["分红"])
                 for r in merged.to_dict("records") if (r["分红日"].date().isoformat(), r["分红类型"]) not in keys]
    refresh = dict(code=code, url=url, retrieved_at=retrieved.isoformat(),
                   method="Latest QLab data.dividends.merge_records; preserve all local records; reject conflicts",
                   merger_sha256=sha(source / "data/dividends.py"), additions=additions, conflicts=conflicts,
                   merged_sha256=sha(event_path), snapshot_hashes={p.name: sha(p) for p in (original_path, raw_path)})
    refresh_path.write_text(json.dumps(refresh, ensure_ascii=False, indent=2))
    copied = folder / f"{code}.dividends.csv"
    if copied.exists():
        if copied.read_bytes() != original_bytes:
            raise ValueError("Unexpected pre-normalization dividend copy")
        copied.write_bytes(event_path.read_bytes())
    print(json.dumps(dict(stage="dividend_refresh", code=code, additions=additions), ensure_ascii=False), flush=True)


def execute_one(task):
    asset, directory, output, end, source = task
    code, kind = asset["code"], asset["asset_type"]
    prefix = output / kind / f"single_{code}_{kind.lower()}_2010"
    with redirect_stdout(io.StringIO()):
        result = run(directory / kind / code / f"{code}.csv", kind, prefix.with_suffix(".html"),
                     end=end, batch_authorized=True, category=asset["category"],
                     daily_buy_amount=40 if kind == "Fundc" else None, qlab_source=source)
        validation = verify(prefix.with_suffix(".html"))
    if not validation["passed"]:
        raise ValueError(f"Failed report validation: {kind} {code}")
    return dict(code=code, asset_type=kind, name=asset["name"], **result["summary"])


def export_type(kind, assets, output):
    headers, rows, summaries = None, [], []
    for asset in assets:
        code = asset["code"]
        prefix = output / kind / f"single_{code}_{kind.lower()}_2010"
        values = prefix.with_suffix(".paste_with_header_wan.tsv").read_text().splitlines()
        if headers is None:
            headers = ["策略"] + values[0].split("\t")
        elif headers[1:] != values[0].split("\t"):
            raise ValueError("Inconsistent output columns")
        label = f"{code}（每交易日40）" if kind == "Fundc" else f"{code}（{kind}）"
        rows.append([label] + values[1].split("\t"))
        payload = json.loads(prefix.with_suffix(".summary.json").read_text())
        summaries.append(dict(code=code, name=asset["name"], **payload["summary"]))
    target = output / kind / kind
    target.with_suffix(".tsv").write_text(tsv([headers, *rows]), encoding="utf-8")
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    target.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    target.with_suffix(".summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qlab-source", required=True, type=Path)
    parser.add_argument("--qlab-commit", required=True)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--types", nargs="+", choices=["Fundc", "Fundf_1", "Fundd"], default=["Fundc", "Fundf_1", "Fundd"])
    parser.add_argument("--exclude", nargs="*", default=["003358", "161119"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--from-snapshot", action="store_true")
    parser.add_argument("--refresh-dividends-on-new-event", action="store_true",
                        help="Use QLab merge logic for verified events newly annotated in current NAVs")
    parser.add_argument("--batch-authorized", action="store_true")
    args = parser.parse_args()
    if not args.batch_authorized:
        parser.error("Explicit user batch authorization is required")
    if args.workers < 1:
        parser.error("workers must be positive")
    source, directory, output = args.qlab_source.resolve(), args.snapshot_dir.resolve(), args.output_dir.resolve()
    rules = QLabRules(source)
    assets = [a for a in rules.catalog(args.types) if a["code"] not in args.exclude]
    if not assets:
        raise ValueError("No selected funds")
    directory.mkdir(parents=True, exist_ok=True)
    if not args.from_snapshot:
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda a: download(a, directory / a["asset_type"]), assets))
    if args.refresh_dividends_on_new_event:
        for asset in assets:
            refresh_new_events(asset, source, directory, args.from_snapshot)
    metadata = [prepare(a, source, directory / a["asset_type"], args.qlab_commit,
                        qlab_source=source, authorization=AUTHORIZATION) for a in assets]
    end = min(m["last_date"] for m in metadata)
    manifest = dict(qlab_commit=args.qlab_commit, catalog_sha256=sha(source / "data_files/Fund"),
                    excluded=args.exclude, start="2010-01-01", common_end=end,
                    funds=metadata, assets=assets, count=len(assets),
                    rules={"Fundc": "daily_fixed_40", "Fundf_1": "original_qlab", "Fundd": "original_qlab"})
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(dict(stage="prepared", count=len(assets), common_end=end,
                         warnings={m["code"]: m["warnings"] for m in metadata if m["warnings"]}), ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute_one, (a, directory, output, end, source)) for a in assets]
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps({k: result[k] for k in ("code", "asset_type", "total_contributions", "total_pnl", "xirr")}), flush=True)
    for kind in args.types:
        selected = [a for a in assets if a["asset_type"] == kind]
        if selected:
            export_type(kind, selected, output)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
