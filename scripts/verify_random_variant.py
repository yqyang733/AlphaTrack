"""Audit the random-selection variant against immutable ranked-run exports.

This checks strategy isolation and reproducibility; verify_portfolio.py handles
the separate accounting reconciliations for each exported window.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = {"Fundb": 8, "Fundc": 5}
UNCHANGED = {"Fundd", "Fundf_1", "Fundg"}
IDENTITY = ["asset_type", "code", "date"]
DECISIONS = ["date", "action", "amount", "fraction", "rank", "score", "reason", "asset_type", "code"]


def _read(prefix, suffix):
    return pd.read_csv(str(prefix) + "." + suffix, dtype={"code": str})


def _ordered(frame, keys=IDENTITY):
    return frame.sort_values(keys, kind="stable").reset_index(drop=True)


def _equal(left, right, label, keys=IDENTITY):
    try:
        pd.testing.assert_frame_equal(_ordered(left, keys), _ordered(right, keys),
                                      check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise AssertionError(f"{label}: {exc}") from exc


def _verify_window(baseline, variant, year, seed):
    left = json.loads(Path(str(baseline) + ".summary.json").read_text())
    right = json.loads(Path(str(variant) + ".summary.json").read_text())
    for field in ["source_commit", "data_hashes", "types", "start_date", "end_date"]:
        assert left["metadata"][field] == right["metadata"][field], (year, field)
    for field in ["source_hashes", "configurations", "assets", "universe", "ranking"]:
        assert left["metadata"]["signal_metadata"][field] == right["metadata"]["signal_metadata"][field], (year, field)
    for field in ["source", "data_dir", "types", "engine"]:
        assert left["config"][field] == right["config"][field], (year, field)
    assert left["config"].get("strategy", {}).get("selection_mode", "ranked") == "ranked"
    assert right["config"]["strategy"] == {"selection_mode": "random", "random_seed": seed}
    selection = right["metadata"]["signal_metadata"]
    assert selection["selection_mode"] == "random" and selection["random_seed"] == seed
    assert selection["selection_algorithm"] == "sha256-per-date-type-code-v1"
    assert right["summary"]["beginning_assets"] == left["summary"]["beginning_assets"] == 0

    signals_left = _read(baseline, "signals.csv.gz")
    signals_right = _read(variant, "signals.csv.gz")
    assert not signals_left.duplicated(IDENTITY).any() and not signals_right.duplicated(IDENTITY).any()
    _equal(signals_left[IDENTITY], signals_right[IDENTITY], f"{year} observation membership")
    unaffected_counts = {}
    for suffix in ["signals.csv.gz", "trades.csv", "components_daily.csv.gz"]:
        a = signals_left if suffix == "signals.csv.gz" else _read(baseline, suffix)
        b = signals_right if suffix == "signals.csv.gz" else _read(variant, suffix)
        a = a[a.asset_type.isin(UNCHANGED)].copy()
        b = b[b.asset_type.isin(UNCHANGED)].copy()
        keys = IDENTITY + ["signal_date", "action", "status"] if suffix == "trades.csv" else IDENTITY
        _equal(a, b, f"{year} unchanged groups {suffix}", keys)
        unaffected_counts[suffix] = len(a)

    target_left = _ordered(signals_left[signals_left.asset_type.isin(TARGETS)].copy())
    target_right = _ordered(signals_right[signals_right.asset_type.isin(TARGETS)].copy())
    _equal(target_left[IDENTITY + ["score"]], target_right[IDENTITY + ["score"]], f"{year} original scores")
    assert target_left["rank"].notna().equals(target_right["rank"].notna()), f"{year} rank eligibility changed"
    assert set(target_right.action).issubset({"BUY", "HOLD"})
    assert target_right.loc[target_right.action.eq("BUY"), "amount"].eq(40).all()
    assert target_right.loc[target_right["rank"].isna(), "action"].eq("HOLD").all()
    maxima, buy_counts = {}, {}
    for kind, limit in TARGETS.items():
        part = target_right[target_right.asset_type.eq(kind)]
        expected_buy = part["rank"].le(limit)
        assert part.action.eq("BUY").equals(expected_buy), (year, kind, "selection gate")
        buys = part[part.action.eq("BUY")]
        counts = buys.groupby("date").size()
        assert counts.le(limit).all(), (year, kind, "daily count exceeded")
        maxima[kind] = int(counts.max()) if len(counts) else 0
        buy_counts[kind] = len(buys)
        for date, group in part[part["rank"].notna()].groupby("date"):
            assert group["rank"].ge(1).all() and np.equal(group["rank"] % 1, 0).all()
            assert group["rank"].is_unique
            # The exported table contains only fresh observations. As-of stale
            # peers may occupy omitted ranks, but fresh ranks must retain the
            # same deterministic SHA256 order within the full eligible pool.
            salt = f"alphatrack-daily-random-v1|{seed}|{kind}|{date}|"
            expected_order = sorted(group.code, key=lambda code: (
                hashlib.sha256((salt + code).encode("utf-8")).digest(), code))
            assert group.sort_values("rank").code.tolist() == expected_order, (year, kind, date, "hash order")
    changed_actions = int(target_left.action.ne(target_right.action).sum())
    assert changed_actions > 0, f"{year} random variant made no selection changes"
    daily_counts = pd.concat({
        "ranked": target_left[target_left.action.eq("BUY")].groupby(["date", "asset_type"]).size(),
        "random": target_right[target_right.action.eq("BUY")].groupby(["date", "asset_type"]).size(),
    }, axis=1).fillna(0).astype(int)
    different_counts = daily_counts[daily_counts.ranked.ne(daily_counts.random)].reset_index().to_dict("records")
    trades = _read(variant, "trades.csv")
    target_trades = trades[trades.asset_type.isin(TARGETS)]
    assert set(target_trades.action) == {"BUY"}
    filled = target_trades[target_trades.status.isin(["FILLED", "PARTIAL"])]
    assert filled.amount.eq(40).all()
    assert (pd.to_datetime(filled.date) > pd.to_datetime(filled.signal_date)).all()
    assert np.isclose(left["summary"]["total_withdrawals"], right["summary"]["total_withdrawals"], rtol=0, atol=1e-8)
    a, b = left["summary"], right["summary"]
    result = {
        "start": a["start_date"], "end": a["end_date"],
        "baseline_prefix": str(baseline), "variant_prefix": str(variant),
        "same_source_data_engine": True, "unchanged_group_rows": unaffected_counts,
        "unchanged_groups_exact": sorted(UNCHANGED), "original_scores_and_eligibility_exact": True,
        "target_decision_changes": changed_actions, "random_max_daily_buy_count": maxima,
        "random_buy_signal_count": buy_counts,
        "daily_buy_count_differences": different_counts,
        "baseline_contributions": a["total_contributions"], "variant_contributions": b["total_contributions"],
        "contributions_difference": b["total_contributions"] - a["total_contributions"],
        "baseline_pnl": a["total_pnl"], "variant_pnl": b["total_pnl"],
        "pnl_difference": b["total_pnl"] - a["total_pnl"],
        "baseline_xirr": a["xirr"], "variant_xirr": b["xirr"],
    }
    return result, target_right


def verify(baseline_prefix="reports/qlab_portfolio_windows", variant_prefix="reports/alphastep_random_windows",
           years=(2010, 2015, 2020, 2025), seed=42):
    results, targets = [], {}
    for year in years:
        baseline = Path(f"{baseline_prefix}_since_{year}0101")
        variant = Path(f"{variant_prefix}_since_{year}0101")
        result, target = _verify_window(baseline, variant, year, seed)
        results.append(result)
        targets[year] = target
    overlap = []
    years = sorted(targets)
    for index, earlier in enumerate(years):
        for later in years[index + 1:]:
            a, b = targets[earlier], targets[later]
            start, end = b.date.min(), b.date.max()
            a = a[a.date.between(start, end)]
            _equal(a[DECISIONS], b[DECISIONS], f"{earlier}/{later} overlapping random decisions")
            overlap.append({"earlier_start": earlier, "later_start": later, "identical_decision_rows": len(b)})
    return {"passed": True, "random_seed": seed,
            "scope": "Original as-of eligible candidate pool; fresh NAV emission gate retained; only Fundb/Fundc order randomized",
            "checks": ["source/data/engine equality", "non-target full signal/trade/component equality",
                       "target original scores and eligibility", "40 yuan buys and no sells",
                       "without-replacement daily limits", "deterministic SHA256 ordering of fresh candidates",
                       "same random decisions over overlapping windows"],
            "windows": results, "overlap": overlap}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-prefix", default="reports/qlab_portfolio_windows")
    parser.add_argument("--variant-prefix", default="reports/alphastep_random_windows")
    parser.add_argument("--years", type=int, nargs="+", default=[2010, 2015, 2020, 2025])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/alphastep_random_windows.variant_validation.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = verify(args.baseline_prefix, args.variant_prefix, args.years, args.seed)
    except Exception as exc:
        output.write_text(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        raise
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
