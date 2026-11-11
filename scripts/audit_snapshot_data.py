"""Audit local input conversion against archived NAV and verified QLab rules.

This checks faithful transfer and calculation, not external data correctness.
The raw JavaScript is parsed as JSON assignments and is never executed.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from types import ModuleType
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alphatrack.data import load_data
from alphatrack.strategies.qlab import QLabRules


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assignment(text, name):
    match = re.search(r"var\s+" + re.escape(name) + r"\s*=\s*(.*?);", text)
    if match is None:
        raise ValueError(f"Missing archived field {name}")
    return json.loads(match.group(1))


def day(timestamp):
    return datetime.fromtimestamp(timestamp / 1000, ZoneInfo("Asia/Shanghai")).date().isoformat()


def audit(root, raw, qlab_source):
    root, raw, qlab_source = map(Path, (root, raw, qlab_source))
    catalog = json.loads((raw / "catalog.json").read_text())
    reference_catalog = QLabRules(qlab_source).catalog()
    catalog_keys = ("code", "name", "asset_type", "category")
    key = lambda item: tuple(item[name] for name in catalog_keys)
    if sorted(map(key, catalog)) != sorted(map(key, reference_catalog)):
        raise AssertionError("Archived universe differs from enabled QLab catalog")
    if len({item["code"] for item in catalog}) != len(catalog):
        raise AssertionError("Duplicate fund code in archived universe")
    origins = {a["code"]: json.loads((raw / f"{a['code']}.json").read_text()) for a in catalog}
    cutoff = min(day(origin["last_ts"]) for origin in origins.values())
    prior_audit = json.loads((raw / "dividend_audit.json").read_text())
    source_module = ModuleType("independent_qlab_reference_calculator")
    source_path = qlab_source / "core/calculator.py"
    exec(compile(source_path.read_bytes(), str(source_path), "exec"), source_module.__dict__)
    calculator = source_module.ReturnCalculator
    provenance = json.loads((root / "vendor/qlab/PROVENANCE.json").read_text())
    for relative, expected in provenance["files"].items():
        if digest(root / "vendor/qlab" / relative) != expected:
            raise AssertionError(f"Changed vendored reference file: {relative}")
        if digest(qlab_source / relative) != expected:
            raise AssertionError(f"Separate QLab checkout differs: {relative}")
    records, known_missing, event_counts = [], [], Counter()
    warnings = []
    for asset in catalog:
        code = asset["code"]
        original = origins[code]
        js_path = raw / f"{code}.js"
        text = js_path.read_text(encoding="utf-8-sig")
        if assignment(text, "fS_code") != code:
            raise AssertionError(f"Archived fund identity mismatch: {code}")
        values = assignment(text, "Data_netWorthTrend")
        accumulated = dict(assignment(text, "Data_ACWorthTrend"))
        if (len(values), values[0]["x"], values[-1]["x"]) != (
                original["rows"], original["first_ts"], original["last_ts"]):
            raise AssertionError(f"Archived metadata does not describe NAV rows: {code}")
        selected = [value for value in values if day(value["x"]) <= cutoff]
        expected_dates = [day(value["x"]) for value in selected]
        expected_nav = np.array([value["y"] for value in selected], dtype=float)
        csv_path = root / "data/qlab_portfolio" / f"{code}.csv"
        actual, metadata = load_data(csv_path)
        if actual.date.dt.strftime("%Y-%m-%d").tolist() != expected_dates:
            raise AssertionError(f"Normalized dates differ from Shanghai archived dates: {code}")
        if not np.allclose(actual.nav, expected_nav, atol=1e-12, rtol=0):
            raise AssertionError(f"Unit NAV changed during conversion: {code}")
        if not (digest(js_path) == original["sha256"] == metadata["source_sha256"]):
            raise AssertionError(f"Archived source hash mismatch: {code}")
        if digest(csv_path) != metadata["data_sha256"]:
            raise AssertionError(f"Normalized data hash mismatch: {code}")
        if metadata["last_date"] != cutoff or metadata["rows"] != len(actual):
            raise AssertionError(f"Normalized metadata coverage mismatch: {code}")
        dividend_path = qlab_source / f"data_files/dividend/fund/{code}.csv"
        events = pd.read_csv(dividend_path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
        if events.duplicated(["分红日", "分红类型"]).any():
            raise AssertionError(f"Duplicate corporate-action key: {code}")
        expected_dividend = np.zeros(len(actual))
        expected_split = np.ones(len(actual))
        lookup = {date: i for i, date in enumerate(expected_dates)}
        missing, future, applied = [], [], []
        for event in events.to_dict("records"):
            date, kind, value = event["分红日"], event["分红类型"], event["分红"]
            event_counts[kind] += 1
            if date > cutoff:
                future.append(event)
                continue
            if date not in lookup:
                missing.append(event)
                known_missing.append({"code": code, **event})
                continue
            index = lookup[date]
            if kind == "现金分红":
                expected_dividend[index] = float(value)
            elif kind == "份额折算":
                before, after = map(float, value.split(":"))
                if before != 1 or after <= 0:
                    raise AssertionError(f"Unnormalized or invalid split {code}: {event}")
                expected_split[index] = after
            else:
                raise AssertionError(f"Unsupported corporate action {code}: {event}")
            applied.append(event)
        if not np.allclose(actual.dividend, expected_dividend, atol=1e-12, rtol=0):
            raise AssertionError(f"Cash dividend amount or ex-date conversion differs: {code}")
        if not np.allclose(actual.split, expected_split, atol=1e-12, rtol=0):
            raise AssertionError(f"Split ratio or date conversion differs: {code}")
        source_nav = pd.DataFrame({"净值日期": pd.to_datetime(expected_dates), "单位净值": expected_nav})
        reference_events = events.copy()
        reference_events["分红日"] = pd.to_datetime(reference_events["分红日"])
        reference = calculator.calculate_return_with_dividend_reinvest(source_nav, reference_events)
        reference_tri = reference["累计收益率"].to_numpy() + 1
        curve_errors = np.abs(actual.total_return_index.to_numpy() - reference_tri)
        if not np.allclose(actual.total_return_index, reference_tri, atol=1e-12, rtol=0):
            raise AssertionError(f"QLab and AlphaTrack reinvestment curves differ: {code}")
        gaps = actual.date.diff().dt.days
        large_gaps = [{"after": expected_dates[i-1], "before": expected_dates[i], "days": int(gaps.iloc[i])}
                      for i in range(1, len(actual)) if gaps.iloc[i] > 31]
        if large_gaps:
            warnings.append({"code": code, "known_source_history_gaps_over_31_days": large_gaps})
        # This records that unit NAV was used; accumulated NAV is deliberately
        # not used as execution price or compounded a second time.
        ac_differs = sum(abs(value["y"] - accumulated[value["x"]]) > 1e-9
                         for value in selected if accumulated.get(value["x"]) is not None)
        ac_missing = sum(accumulated.get(value["x"]) is None for value in selected)
        records.append({**asset, "rows": len(actual), "first_date": expected_dates[0],
                        "last_date": expected_dates[-1], "source_sha256": digest(js_path),
                        "nav_max_abs_error": float(np.max(np.abs(actual.nav - expected_nav))),
                        "curve_max_abs_error": float(curve_errors.max()),
                        "corporate_actions_total": len(events), "applied_actions": len(applied),
                        "cash_events": int((events["分红类型"] == "现金分红").sum()),
                        "split_events": int((events["分红类型"] == "份额折算").sum()),
                        "missing_nav_events": missing, "after_cutoff_events": future,
                        "rows_where_accumulated_nav_differs": ac_differs,
                        "rows_missing_accumulated_nav": ac_missing,
                        "weekend_observations": int((actual.date.dt.dayofweek >= 5).sum()),
                        "all_conversion_checks_passed": True})
    actual_missing = {(event["code"], event["分红日"]) for event in known_missing}
    expected_missing = {("000931", "2017-03-22"), ("000931", "2017-11-14")}
    if actual_missing != expected_missing:
        raise AssertionError(f"Missing-NAV policy differs from known QLab omissions: {known_missing}")
    return {"passed": True, "scope": "Faithful local conversion and curve parity; upstream financial data accepted as verified by the user",
            "raw_archive": str(raw.resolve()), "qlab_source": str(qlab_source.resolve()),
            "qlab_commit": provenance["commit"], "cutoff": cutoff, "fund_count": len(records),
            "nav_rows_checked": sum(record["rows"] for record in records),
            "cash_events": event_counts["现金分红"], "split_events": event_counts["份额折算"],
            "applied_events": sum(record["applied_actions"] for record in records),
            "after_cutoff_events": sum(len(record["after_cutoff_events"]) for record in records),
            "max_curve_abs_error": max(record["curve_max_abs_error"] for record in records),
            "known_missing_nav_events_preserved": known_missing,
            "known_history_warnings": warnings,
            "prior_remote_audit_unverified_codes": [a["code"] for a in prior_audit if a.get("error")],
            "prior_remote_audit_rejected_changes": [a["code"] for a in prior_audit if a.get("added") or a.get("conflicts")],
            "notes": ["Raw timestamps converted to Asia/Shanghai; normalized NAV uses Data_netWorthTrend.y, not accumulated NAV",
                      "Cash dividends remain QLab yuan per share on the exact ex-date; no extra division by ten",
                      "1:x split ratio multiplies shares by x; no reciprocal conversion",
                      "QLab full event file independently fed to original calculator; all dates and values compared against AlphaTrack load_data",
                      "The two known 000931 missing-NAV cash events are omitted by both calculations; no shift or imputation",
                      "This audit does not treat source equityReturn or empty unitMoney amount annotations as calculation defects"],
            "funds": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT.parent / "tmp/qlab-portfolio-raw")
    parser.add_argument("--qlab-source", type=Path, default=ROOT.parent / "tmp/alphatrack-qlab-source")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/audit_20261111_data.json")
    args = parser.parse_args()
    report = audit(ROOT, args.raw, args.qlab_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"funds", "notes"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
