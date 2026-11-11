"""Check real-data feature prefixes and point-in-time Fundb/Fundc signals.

Uses local normalized observations and existing exports; no backtests, network
requests, business-code changes or historical membership reconstruction.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alphatrack.strategies.qlab import QLabRules, asset_key

DEFAULT_CUTOFFS = ("2017-12-29", "2017-12-30", "2021-01-22", "2025-12-31")
FEATURE_ATOL = 1e-12
SCORE_ATOL = 1e-10  # Existing CSV scores have a finite serialization precision.


def independent_ranking(assets, latest):
    """Recompute QLab's formula without calling either ranking implementation."""
    codes, means, reversions = [], [], []
    for asset in assets:
        row = latest.get(asset_key(asset))
        if row is None:
            continue
        returns = [float(row[f"近{year}年"]) for year in (1, 2, 3, 5)]
        reversion = float(row["mean_reversion_score"])
        if not all(np.isfinite(value) for value in [*returns, reversion]):
            continue
        codes.append(asset["code"])
        means.append(sum(returns) / 4)
        reversions.append(reversion)
    if len(codes) < 2:
        return {}
    means, reversions = np.array(means), np.array(reversions)
    mean_std, reversion_std = means.std(ddof=1), reversions.std(ddof=1)
    if mean_std == 0 or reversion_std == 0:
        return {}
    scores = ((means - means.mean()) / mean_std
              - (reversions - reversions.mean()) / reversion_std)
    ordered = sorted(((code, float(score)) for code, score in zip(codes, scores)
                      if np.isfinite(score)), key=lambda item: (-item[1], item[0]))
    return {code: (rank, score) for rank, (code, score) in enumerate(ordered, 1)}


def independent_random(ranking, kind, date, seed):
    salt = f"alphatrack-daily-random-v1|{seed}|{kind}|{date.date()}|"
    keys = [(hashlib.sha256((salt + code).encode()).hexdigest(), code)
            for code in ranking]
    return {code: (rank, ranking[code][1])
            for rank, (_, code) in enumerate(sorted(keys), 1)}


def compare_frames(actual, expected):
    pd.testing.assert_frame_equal(actual, expected, check_exact=False,
                                  rtol=0, atol=FEATURE_ATOL)
    errors = {}
    for column in actual.select_dtypes(include=np.number):
        if not actual[column].isna().equals(expected[column].isna()):
            raise AssertionError(f"NaN mask changed: {column}")
        finite = np.isfinite(actual[column]) & np.isfinite(expected[column])
        errors[column] = float((actual.loc[finite, column]
                                - expected.loc[finite, column]).abs().max()) if finite.any() else 0.0
    return errors


def compare_signals(export, assets, latest, ranking, kind, date, config):
    actual = export[(export.date == date) & (export.asset_type == kind)]
    fresh = [asset for asset in assets if asset_key(asset) in latest
             and latest[asset_key(asset)].date == date]
    if actual.code.duplicated().any() or set(actual.code) != {a["code"] for a in fresh}:
        raise AssertionError(f"Fresh-NAV signal gate differs at {kind} {date.date()}")
    lookup = actual.set_index("code")
    top_n = getattr(config, f"{kind.lower()}_top_n")
    amount = getattr(config, f"{kind.lower()}_buy_amount")
    max_error, buys = 0.0, []
    for asset in fresh:
        code = asset["code"]
        row = lookup.loc[code]
        expected_rank, expected_score = ranking.get(code, (None, None))
        expected_buy = expected_rank is not None and expected_rank <= top_n
        if expected_rank is None:
            if not pd.isna(row["rank"]) or not pd.isna(row.score):
                raise AssertionError(f"Ineligible fund entered pool: {kind} {code} {date.date()}")
        else:
            error = abs(float(row.score) - expected_score)
            if row["rank"] != expected_rank or not np.isfinite(row.score) or error > SCORE_ATOL:
                raise AssertionError(f"Rank or score differs: {kind} {code} {date.date()}")
            max_error = max(max_error, error)
        if row.action != ("BUY" if expected_buy else "HOLD"):
            raise AssertionError(f"Action differs: {kind} {code} {date.date()}")
        if expected_buy:
            if row.amount != amount:
                raise AssertionError(f"Buy amount differs: {kind} {code} {date.date()}")
            buys.append(code)
        elif not pd.isna(row.amount):
            raise AssertionError(f"Unexpected HOLD amount: {kind} {code} {date.date()}")
        if not pd.isna(row.fraction):
            raise AssertionError(f"Unexpected sell fraction: {kind} {code} {date.date()}")
    stale_eligible = [asset["code"] for asset in assets if asset["code"] in ranking
                      and latest[asset_key(asset)].date < date]
    stale_selected = [code for code in stale_eligible if ranking[code][0] <= top_n]
    return {"fresh_signal_rows": len(fresh), "eligible_pool_size": len(ranking),
            "stale_eligible_codes": stale_eligible, "stale_selected_codes": stale_selected,
            "buy_codes": sorted(buys), "score_max_abs_error": max_error,
            "rank_action_amount_fraction_match": True}


def audit(cutoffs, ranked_path, random_path, random_seed):
    rules = QLabRules(ROOT / "vendor/qlab")
    assets = rules.catalog()
    cutoffs = sorted(set(pd.Timestamp(date).normalize() for date in cutoffs))
    exports = {"ranked": pd.read_csv(ranked_path, parse_dates=["date"], dtype={"code": str}),
               "random": pd.read_csv(random_path, parse_dates=["date"], dtype={"code": str})}
    for mode, frame in exports.items():
        if any(date < frame.date.min() or date > frame.date.max() for date in cutoffs):
            raise ValueError(f"A cutoff is outside the {mode} export range")
    latest_by_date = {date: {} for date in cutoffs}
    cases, total_rows, full_rows = [], 0, 0
    for asset in assets:
        key = asset_key(asset)
        raw = pd.read_csv(ROOT / "data/qlab_portfolio" / f"{asset['code']}.csv",
                          parse_dates=["date"], usecols=["date", "nav", "dividend", "split"])
        full = rules.prepare_features(raw, asset["asset_type"])
        full_rows += len(full)
        for date in cutoffs:
            prefix = raw[raw.date <= date].copy()
            case = {"code": asset["code"], "asset_type": asset["asset_type"],
                    "cutoff": str(date.date()), "raw_prefix_rows": len(prefix)}
            if prefix.empty:
                cases.append({**case, "status": "not_started", "first_data_date": str(raw.date.min().date())})
                continue
            truncated = rules.prepare_features(prefix, asset["asset_type"])
            expected = full[full.date <= date].reset_index(drop=True)
            errors = compare_frames(truncated, expected)
            total_rows += len(prefix)
            latest_by_date[date][key] = truncated.iloc[-1]
            cases.append({**case, "status": "passed", "columns_checked": list(truncated.columns),
                          "last_known_date": str(truncated.date.iloc[-1].date()),
                          "max_abs_error_by_numeric_column": errors,
                          "max_abs_error": max(errors.values(), default=0.0)})
    signal_cases = []
    for date, latest in latest_by_date.items():
        for kind in ("Fundb", "Fundc"):
            group = [asset for asset in assets if asset["asset_type"] == kind]
            expected_ranked = independent_ranking(group, latest)
            expected_random = independent_random(expected_ranked, kind, date, random_seed)
            for mode, expected in (("ranked", expected_ranked), ("random", expected_random)):
                check = compare_signals(exports[mode], group, latest, expected, kind, date, rules.configs[kind])
                signal_cases.append({"date": str(date.date()), "asset_type": kind,
                                     "selection_mode": mode, **check})
    passed = [case for case in cases if case["status"] == "passed"]
    return {"passed": True, "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "fund_count": len(assets), "full_history_rows": full_rows,
            "cutoffs": [str(date.date()) for date in cutoffs], "random_seed": random_seed,
            "feature_prefix_cases_passed": len(passed),
            "not_started_cases": len(cases) - len(passed), "prefix_rows_compared": total_rows,
            "feature_max_abs_error": max(case["max_abs_error"] for case in passed),
            "feature_absolute_tolerance": FEATURE_ATOL, "signal_score_absolute_tolerance": SCORE_ATOL,
            "signal_cases_passed": len(signal_cases),
            "fresh_signal_rows_compared": sum(case["fresh_signal_rows"] for case in signal_cases),
            "signal_score_max_abs_error": max(case["score_max_abs_error"] for case in signal_cases),
            "ranked_export": str(ranked_path.resolve()), "random_export": str(random_path.resolve()),
            "notes": ["Each feature recalculation receives only raw rows through its cutoff; dates, NaN masks and every output column compared with full-history prefix",
                      "Fundb/Fundc scores independently recomputed as z(mean of 1/2/3/5y simple annual returns) minus z(mean reversion), using sample standard deviation and code tie-break",
                      "Independent random ordering uses seed/date/type/code SHA256 over the same eligible pool, including stale candidates",
                      "Signals verified only on fresh NAV rows; stale selected candidates consume slots and emit no duplicate or substitute order",
                      "2017-12-30 deliberately covers an exceptional weekend NAV row with stale peers",
                      "Only sampled historical cutoffs establish empirical causality checks; this does not prove all possible histories or remove current-catalog survivorship bias",
                      "No portfolio backtests rerun; no external-source investigation; existing data and rules unchanged"],
            "feature_cases": cases, "signal_cases": signal_cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoffs", nargs="+", default=DEFAULT_CUTOFFS)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--ranked-signals", type=Path,
                        default=ROOT / "reports/qlab_portfolio_windows_since_20100101.signals.csv.gz")
    parser.add_argument("--random-signals", type=Path,
                        default=ROOT / "reports/alphastep_random_windows_since_20100101.signals.csv.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/audit_20261111_signals.json")
    args = parser.parse_args()
    report = audit(args.cutoffs, args.ranked_signals, args.random_signals, args.random_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"feature_cases", "signal_cases", "notes"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
