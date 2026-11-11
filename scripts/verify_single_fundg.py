"""Independently audit an approved single-fund Fundg backtest from its fixed CSV.

The independent calculation imports no production backtest implementation.
Decimal arithmetic builds the exact total-return curve and account ledger.
An explicit qlab-binary64 signal mode reproduces QLab's floating-point
operation order while reporting every difference from exact Decimal signals.
That mode also calls the frozen QLab reference functions as separate evidence;
their outputs never drive the independently calculated ledger.
ACT/365 XIRR is solved through a monotone integer-day future-value polynomial.
"""
import argparse
from bisect import bisect_left
import calendar
import csv
from datetime import date
from decimal import Decimal, InvalidOperation, getcontext
import gzip
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
D = Decimal
ZERO, ONE = D(0), D(1)
getcontext().prec = 40


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def previous_year(day):
    return day.replace(year=day.year - 1,
                       day=min(day.day, calendar.monthrange(day.year - 1, day.month)[1]))


def solve_xirr(purchases, terminal_date, terminal_assets):
    """One positive terminal value after only purchases gives a unique root."""
    terms = [((terminal_date - day).days, amount) for day, amount in purchases]
    if not terms or max(days for days, _ in terms) == 0:
        return None, None

    def value(growth):
        return sum((amount * growth ** days for days, amount in terms), ZERO)

    low, high = ZERO, D(2)
    while value(high) < terminal_assets:
        high *= 2
    for _ in range(150):
        middle = (low + high) / 2
        if middle == low or middle == high:
            break
        if value(middle) < terminal_assets:
            low = middle
        else:
            high = middle
    daily_growth = (low + high) / 2
    return daily_growth ** 365 - ONE, value(daily_growth) - terminal_assets


def calculate(path, signal_arithmetic="decimal"):
    if signal_arithmetic not in {"decimal", "qlab-binary64"}:
        raise ValueError("signal_arithmetic must be decimal or qlab-binary64")
    source = rows(path)
    dates = [date.fromisoformat(r["date"]) for r in source]
    assert dates == sorted(set(dates))
    curve, daily, trades, purchases = [], [], [], []
    shares = contribution = economic_pnl = ZERO
    investment_units, investment_unit_price = ZERO, ONE
    curve_multiplier = ONE / D(source[0]["nav"])
    binary_shares = 1.0 / float(source[0]["nav"])
    binary_curve, arithmetic_differences = [], []
    pending = None
    yearly = {year: {"profit": ZERO, "deposits": ZERO} for year in range(2010, dates[-1].year + 1)}
    events = []
    signals = []
    previous_nav = None
    for i, (row, day) in enumerate(zip(source, dates)):
        nav, dividend, split = (D(row[k]) for k in ["nav", "dividend", "split"])
        assert nav > 0 and dividend >= 0 and split > 0
        if i == 0:
            assert dividend == ZERO and split == ONE
        curve_multiplier *= split * (ONE + dividend / nav)
        total_return_value = curve_multiplier * nav
        curve.append(total_return_value)
        # Follow the original binary64 event and subtraction order explicitly.
        # This affects signals only; all holdings and money remain Decimal.
        binary_shares *= float(split)
        if dividend:
            binary_shares += binary_shares * float(dividend) / float(nav)
        binary_curve.append((binary_shares * float(nav) - 1.0) / 1.0)
        left = bisect_left(dates, previous_year(day), 0, i + 1)
        window = curve[left:i + 1]
        decimal_position = binary_position = None
        if len(window) >= 2:
            low, high = min(window), max(window)
            decimal_position = D("0.5") if high == low else (total_return_value - low) / (high - low)
            binary_window = binary_curve[left:i + 1]
            binary_low, binary_high = min(binary_window), max(binary_window)
            binary_position = (0.5 if binary_high == binary_low else
                               (binary_curve[-1] - binary_low) / (binary_high - binary_low))
        decimal_signal = decimal_position is not None and decimal_position <= D("0.6")
        binary_signal = binary_position is not None and binary_position <= 60.0 / 100.0
        if decimal_signal != binary_signal:
            arithmetic_differences.append({
                "date": day.isoformat(), "decimal_position": str(decimal_position),
                "qlab_binary64_position": repr(binary_position),
                "decimal_action": "BUY" if decimal_signal else "HOLD",
                "qlab_binary64_action": "BUY" if binary_signal else "HOLD",
                "next_nav_date": source[i + 1]["date"] if i + 1 < len(source) else None,
                "next_nav": source[i + 1]["nav"] if i + 1 < len(source) else None,
                "window_inputs": [{**source[j], "decimal_total_return_index": str(curve[j]),
                                   "qlab_binary64_cumulative_return": repr(binary_curve[j])}
                                  for j in range(left, i + 1)]})
        position = decimal_position if signal_arithmetic == "decimal" else binary_position
        signal = decimal_signal if signal_arithmetic == "decimal" else binary_signal
        signals.append({"date": day.isoformat(), "buy": signal,
                        "position": str(position) if position is not None else None})

        # Separate economic sources from deposits: all dividends belong only
        # to the position present before a possible close-of-day purchase.
        old_shares = shares
        dividend_cash = old_shares * split * dividend
        pnl = (old_shares * (split * nav - previous_nav) if previous_nav is not None else ZERO) + dividend_cash
        economic_pnl += pnl
        shares = (old_shares * split * nav + dividend_cash) / nav
        if investment_units:
            investment_unit_price = shares * nav / investment_units
        deposit = ZERO
        if pending is not None:
            deposit = D(10)
            bought_shares = deposit / nav
            shares += bought_shares
            contribution += deposit
            investment_units += deposit / investment_unit_price
            purchases.append((day, deposit))
            trades.append({"signal_date": pending.isoformat(), "date": day.isoformat(),
                           "action": "BUY", "status": "FILLED", "amount": deposit,
                           "shares": bought_shares, "price": nav, "fee": ZERO})
        equity = shares * nav
        yearly[day.year]["profit"] += pnl
        yearly[day.year]["deposits"] += deposit
        yearly[day.year]["end_assets"] = equity
        if dividend or split != ONE:
            events.append({"date": day.isoformat(), "dividend_per_share": str(dividend),
                           "split": str(split), "old_shares": str(old_shares),
                           "dividend_cash": str(dividend_cash), "same_day_purchase": bool(deposit)})
        daily.append({"date": day.isoformat(), "nav": nav, "dividend": dividend, "split": split,
                      "shares": shares, "equity": equity,
                      "external_flow": deposit, "withdrawal": ZERO, "cash": ZERO,
                      "daily_pnl": pnl, "cumulative_pnl": economic_pnl,
                      "total_contributions": contribution, "unit_nav": investment_unit_price,
                      "total_return_index": total_return_value})
        pending = day if signal else None
        previous_nav = nav
    if pending is not None:
        trades.append({"signal_date": pending.isoformat(), "date": "", "action": "BUY",
                       "status": "PENDING_END_OF_DATA", "amount": ZERO,
                       "shares": ZERO, "price": None, "fee": ZERO})
    previous_assets = ZERO
    for year, values in yearly.items():
        values["beginning_assets"] = previous_assets
        values.setdefault("end_assets", previous_assets)
        values["withdrawals"] = ZERO
        previous_assets = values["end_assets"]
        assert abs(values["end_assets"] - values["beginning_assets"] - values["deposits"] - values["profit"]) < D("1e-25")
    assert abs(daily[-1]["equity"] - contribution - economic_pnl) < D("1e-25")
    xirr, residual = solve_xirr(purchases, dates[-1], daily[-1]["equity"])
    return {"summary": {"observations": len(daily), "buy_count": len(purchases), "sell_count": 0,
                        "total_contributions": contribution, "total_withdrawals": ZERO,
                        "final_equity": daily[-1]["equity"], "total_pnl": economic_pnl,
                        "xirr": xirr, "first_investment_date": purchases[0][0].isoformat(),
                        "end_date": dates[-1].isoformat()},
            "daily": daily, "trades": trades, "signals": signals, "annual": yearly,
            "corporate_actions": events, "xirr_terminal_value_residual_yuan": residual,
            "signal_arithmetic": signal_arithmetic,
            "signal_arithmetic_differences": arithmetic_differences}


def audit_source_actions(path, metadata):
    """Reconcile every recorded action with the CSV or an exact documented omission."""
    event_path = path.with_suffix(".dividends.csv")
    if not event_path.exists():
        return {"passed": False, "reason": "Missing recorded corporate-action CSV"}
    normalized = rows(path)
    last_date = normalized[-1]["date"]

    def amount(kind, value):
        if kind == "现金分红":
            return D(value)
        if kind == "份额折算":
            before, after = map(D, value.split(":"))
            return after / before
        raise ValueError(f"Unknown corporate-action type: {kind}")

    recorded = {}
    for event in rows(event_path):
        key = (event["分红日"], event["分红类型"])
        if key[0] > last_date:
            continue
        if key in recorded:
            raise ValueError(f"Duplicate recorded corporate action: {key}")
        recorded[key] = amount(key[1], event["分红"])
    included = {}
    for row in normalized:
        if D(row["dividend"]):
            included[(row["date"], "现金分红")] = D(row["dividend"])
        if D(row["split"]) != ONE:
            included[(row["date"], "份额折算")] = D(row["split"])
    omissions = {}
    for event in metadata.get("approved_omissions", []):
        key = (event["date"], event["type"])
        if key in omissions:
            raise ValueError(f"Duplicate documented omission: {key}")
        omissions[key] = amount(key[1], event["value"])
    passed = not (set(included) & set(omissions)) and recorded == {**included, **omissions}
    return {"passed": passed, "recorded_cash_dividends": sum(k[1] == "现金分红" for k in recorded),
            "included_cash_dividends": sum(k[1] == "现金分红" for k in included),
            "included_split_events": sum(k[1] == "份额折算" for k in included),
            "omissions": [{"date": k[0], "type": k[1], "value": str(v)} for k, v in sorted(omissions.items())],
            "exact_dates_types_values_reconciled": passed}


def compare(result, prefix, code="519981"):
    failures, maxima = [], {}
    counts = {"numeric_comparisons": 0, "identity_comparisons": 0}

    def equal(actual, expected, label):
        counts["identity_comparisons"] += 1
        if actual != expected:
            failures.append({"check": label, "actual": str(actual), "expected": str(expected)})

    def number(actual, expected, label):
        counts["numeric_comparisons"] += 1
        if actual is None or expected is None:
            equal(actual, expected, label)
            return
        try:
            actual_number = D(str(actual))
        except (InvalidOperation, ValueError, TypeError):
            failures.append({"check": label, "actual": str(actual), "expected": str(expected),
                             "reason": "Missing or invalid numeric value"})
            return
        if not actual_number.is_finite():
            failures.append({"check": label, "actual": str(actual), "expected": str(expected),
                             "reason": "Nonfinite numeric value"})
            return
        error = abs(actual_number - expected)
        maxima[label.split(":")[0]] = max(maxima.get(label.split(":")[0], ZERO), error)
        tolerance = D("2e-7") + abs(expected) * D("6e-11")
        if error > tolerance:
            failures.append({"check": label, "actual": str(actual), "expected": str(expected), "error": str(error)})

    payload = json.loads(prefix.with_suffix(".summary.json").read_text())
    summary = payload.get("summary", payload)
    for field, expected in result["summary"].items():
        if field in {"first_investment_date", "end_date"}:
            equal(summary[field], expected, "summary." + field)
        else:
            number(summary[field], D(expected) if isinstance(expected, int) else expected, "summary." + field)
    actual_daily = rows(prefix.with_suffix(".daily.csv"))
    equal(len(actual_daily), len(result["daily"]), "daily.row_count")
    for actual, expected in zip(actual_daily, result["daily"]):
        equal(actual["date"], expected["date"], "daily.date")
        for field, value in expected.items():
            if field != "date" and field in actual:
                number(actual[field], value, "daily." + field + ":" + expected["date"])
    actual_components = rows(prefix.with_suffix(".components_daily.csv.gz"))
    equal(len(actual_components), len(result["daily"]), "component.row_count")
    for actual, expected in zip(actual_components, result["daily"]):
        equal(actual["date"], expected["date"], "component.date")
        equal(actual["code"], code, "component.code")
        for field, value in expected.items():
            if field != "date":
                number(actual[field], value, "component." + field + ":" + expected["date"])
    actual_signals = rows(prefix.with_suffix(".signals.csv.gz"))
    equal(len(actual_signals), len(result["signals"]), "signal.row_count")
    for actual, expected in zip(actual_signals, result["signals"]):
        equal(actual["date"], expected["date"], "signal.date")
        equal(actual["code"], code, "signal.code")
        equal(actual["action"], "BUY" if expected["buy"] else "HOLD", "signal.action")
        if expected["position"] is None:
            equal(actual["近1年百分位"], "", "signal.insufficient_history")
        else:
            number(actual["近1年百分位"], D(expected["position"]), "signal.position:" + expected["date"])
        if expected["buy"]:
            number(actual["amount"], D(10), "signal.amount:" + expected["date"])
    actual_trades = rows(prefix.with_suffix(".trades.csv"))
    equal(len(actual_trades), len(result["trades"]), "trade.row_count")
    for actual, expected in zip(actual_trades, result["trades"]):
        for field in ["signal_date", "date", "action", "status"]:
            equal(actual[field], expected[field], "trade." + field)
        for field in ["amount", "shares", "price", "fee"]:
            if expected[field] is not None:
                number(actual[field], expected[field], "trade." + field + ":" + expected["signal_date"])
    actual_annual = rows(prefix.with_suffix(".annual.csv"))
    equal([int(r["year"]) for r in actual_annual], list(result["annual"]), "annual.years")
    for actual in actual_annual:
        expected = result["annual"][int(actual["year"])]
        for field, value in expected.items():
            number(actual[field], value, "annual." + field + ":" + actual["year"])
    # This strategy never sells, so each purchase must remain as one open lot.
    actual_lots = rows(prefix.with_suffix(".lots.csv"))
    filled = [t for t in result["trades"] if t["status"] == "FILLED"]
    equal(len(actual_lots), len(filled), "lot.row_count")
    by_date = {r["date"]: r for r in result["daily"]}
    last = result["daily"][-1]
    for index, (actual, purchase) in enumerate(zip(actual_lots, filled), 1):
        bought = by_date[purchase["date"]]
        value = purchase["amount"] * last["total_return_index"] / bought["total_return_index"]
        equal(actual["lot_id"], f"L{index:05d}", "lot.id")
        equal(actual["buy_date"], purchase["date"], "lot.buy_date")
        equal(actual["status"], "UNREALIZED", "lot.status")
        equal(actual["valuation_date"], last["date"], "lot.valuation_date")
        for field, expected in {"shares": value / last["nav"], "cost": purchase["amount"],
                                "proceeds": value, "pnl": value - purchase["amount"]}.items():
            number(actual[field], expected, "lot." + field + ":" + purchase["date"])
    return {"passed": not failures, **counts, "failure_count": len(failures), "failures": failures[:30],
            "maximum_absolute_errors": {k: str(v) for k, v in maxima.items()}}


def qlab_reference_evidence(data_path, result, code):
    """Check independent binary64 signals against frozen original functions.

    This is a separate reference check. It never supplies trading decisions to
    calculate(), and never calls a production backtest or writes reference files.
    """
    import sys
    from types import ModuleType
    import pandas as pd

    vendor = ROOT / "vendor/qlab"
    namespace = "_single_fundg_audit_original_qlab"
    package = ModuleType(namespace)
    package.__path__ = []
    sys.modules[namespace] = package

    def load(name, relative):
        module = ModuleType(namespace + "." + name)
        module.__file__ = str(vendor / relative)
        sys.modules[module.__name__] = module
        exec(compile((vendor / relative).read_bytes(), module.__file__, "exec"), module.__dict__)
        return module

    config_module = load("config", "trading/config.py")
    engine_module = load("engine", "trading/engine.py")
    calculator = load("calculator", "core/calculator.py")
    nav = pd.read_csv(data_path, parse_dates=["date"])
    events = pd.read_csv(data_path.with_suffix(".dividends.csv"), parse_dates=["分红日"], dtype={"分红": str})
    source_curve = calculator.ReturnCalculator.calculate_return_with_dividend_reinvest(
        nav[["date", "nav"]].rename(columns={"date": "净值日期", "nav": "单位净值"}), events)
    source_positions = calculator.PercentileCalculator.calculate_percentile_position(
        source_curve.drop(columns="单位净值").rename(columns={"累计收益率": "单位净值"}), [1, 2, 3, 5, 10])
    config = config_module.TradingConfig.from_json(vendor / "configs/trading/Fundg.json", asset_type="Fundg")
    engine = engine_module.TradingEngine(config)
    mismatches, boundary_evidence = [], []
    reference_buys = 0
    differing_dates = {item["date"] for item in result["signal_arithmetic_differences"]}
    for index, row in source_positions.iterrows():
        day = str(row["净值日期"].date())
        decision, _ = engine.evaluate("Fundg", code, code, day, row)
        reference_buys += decision.action == "BUY"
        independent = result["signals"][index]
        independent_action = "BUY" if independent["buy"] else "HOLD"
        if day != independent["date"] or decision.action != independent_action:
            mismatches.append({"date": day, "reference_action": decision.action,
                               "independent_action": independent_action})
        if day in differing_dates:
            boundary_evidence.append({"date": day, "original_position_repr": repr(float(row["近1年百分位"])),
                                      "original_action": decision.action, "original_reason": decision.reason})
    files = ["core/calculator.py", "trading/engine.py", "trading/config.py", "configs/trading/Fundg.json"]
    return {"passed": not mismatches, "observations": len(source_positions),
            "reference_buy_signal_count": reference_buys, "action_mismatch_count": len(mismatches),
            "action_mismatches": mismatches, "boundary_evidence": boundary_evidence,
            "source_commit": json.loads((vendor / "PROVENANCE.json").read_text())["commit"],
            "source_hashes": {name: hashlib.sha256((vendor / name).read_bytes()).hexdigest() for name in files},
            "source_locations": {"return_subtraction": "core/calculator.py:64-65",
                                 "position_formula": "core/calculator.py:146-160",
                                 "strict_threshold": "trading/engine.py:67-68"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", help="fund identity; defaults to the input metadata code, otherwise 519981")
    parser.add_argument("--data", type=Path, default=ROOT / "data/single/519981/20261111/519981.csv")
    parser.add_argument("--report-prefix", type=Path, default=ROOT / "reports/single_519981_fundg_2010")
    parser.add_argument("--require-report", action="store_true")
    parser.add_argument("--signal-arithmetic", choices=["decimal", "qlab-binary64"], default="decimal",
                        help="explicit signal arithmetic; money and XIRR always use Decimal")
    args = parser.parse_args()
    metadata_path = args.data.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    code = args.code or metadata.get("code", "519981")
    if not isinstance(code, str) or not re.fullmatch(r"[0-9]{6}", code):
        parser.error("code must contain six digits")
    if metadata.get("code") and metadata["code"] != code:
        parser.error("explicit code disagrees with input metadata identity")
    # Known-answer XIRR: one 100-yuan purchase, 110 yuan after exactly 365 days.
    rate, residual = solve_xirr([(date(2021, 1, 1), D(100))], date(2022, 1, 1), D(110))
    assert abs(rate - D("0.1")) < D("1e-30") and abs(residual) < D("1e-30")
    result = calculate(args.data, args.signal_arithmetic)
    reference = qlab_reference_evidence(args.data, result, code) if args.signal_arithmetic == "qlab-binary64" else None
    counterfactual = None
    if args.signal_arithmetic == "qlab-binary64" and result["signal_arithmetic_differences"]:
        exact = calculate(args.data, "decimal")
        impact_fields = ["buy_count", "total_contributions", "final_equity", "total_pnl", "xirr"]
        counterfactual = {"label": "Exact Decimal signal counterfactual; not the QLab-compatible report",
                          "summary": exact["summary"],
                          "exact_decimal_minus_qlab_compatible": {
                              key: exact["summary"][key] - result["summary"][key] for key in impact_fields}}
    source_actions = audit_source_actions(args.data, metadata)
    report_exists = args.report_prefix.with_suffix(".summary.json").exists()
    comparison = compare(result, args.report_prefix, code) if report_exists else None
    passed = (comparison["passed"] and source_actions["passed"] and (reference is None or reference["passed"])
              if comparison is not None else None)
    stage = ("awaiting_production_report" if comparison is None else
             "verified" if passed else "verification_failed")
    output = {"stage": stage, "passed": passed,
              "passed_scope": "Selected signal arithmetic, independent Decimal ledger and recorded source actions; exact-vs-binary64 differences are separately disclosed",
              "method": __doc__,
              "code": code, "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "decimal_precision": getcontext().prec, "xirr_known_answer_test": "passed",
              "signal_arithmetic": args.signal_arithmetic,
              "exact_decimal_signals_match_qlab": not result["signal_arithmetic_differences"],
              "signal_arithmetic_differences": result["signal_arithmetic_differences"],
              "exact_decimal_counterfactual": counterfactual, "qlab_reference_evidence": reference,
              "summary": result["summary"], "annual": result["annual"],
              "corporate_actions": result["corporate_actions"],
              "source_corporate_action_reconciliation": source_actions,
              "xirr_terminal_value_residual_yuan": result["xirr_terminal_value_residual_yuan"],
              "comparison": comparison}
    path = args.report_prefix.with_suffix(".independent_audit.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"summary": output["summary"], "comparison": comparison,
                      "signal_arithmetic": args.signal_arithmetic,
                      "exact_decimal_signals_match_qlab": output["exact_decimal_signals_match_qlab"],
                      "signal_arithmetic_differences": output["signal_arithmetic_differences"],
                      "exact_decimal_counterfactual": counterfactual, "qlab_reference_evidence": reference,
                      "source_corporate_action_reconciliation": source_actions,
                      "audit": str(path)}, ensure_ascii=False, indent=2, default=str), flush=True)
    raise SystemExit(1 if not source_actions["passed"] or (reference and not reference["passed"]) or (comparison and not comparison["passed"])
                     or (args.require_report and not comparison) else 0)


if __name__ == "__main__":
    main()
