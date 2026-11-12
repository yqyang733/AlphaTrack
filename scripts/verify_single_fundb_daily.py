"""Independent Decimal audit of a user's single-fund fixed daily purchase rule.

All real NAV rows remain valuation observations. Only Monday-Friday NAV rows
can emit fixed-amount BUY signals or execute a pending order; weekends retain the order
for the next eligible NAV. This is the explicitly requested single-fund
override, without peer ranking or a price-position threshold. Defaults preserve
the daily-40-yuan Fundb audit; --amount and --rule select other explicit overrides.
"""
import argparse
from datetime import date
from decimal import Decimal, InvalidOperation, getcontext
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_single_fundg import audit_source_actions as audit_recorded_source_actions, rows, solve_xirr

D = Decimal
ZERO, ONE = D(0), D(1)
getcontext().prec = 40


def positive_amount(value):
    try:
        amount = D(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("Daily amount must be a finite positive number") from exc
    if not amount.is_finite() or amount <= ZERO:
        raise ValueError("Daily amount must be a finite positive number")
    return amount


def calculate(path, amount=D("40"), start="2010-01-01", end=None):
    amount = positive_amount(amount)
    start_date = date.fromisoformat(str(start))
    raw = rows(path)
    dates = [date.fromisoformat(row["date"]) for row in raw]
    assert dates == sorted(set(dates))
    if not dates:
        raise ValueError("No NAV observations on or after the requested start")
    requested_end = date.fromisoformat(str(end)) if end is not None else dates[-1]
    if requested_end < start_date or requested_end > dates[-1]:
        raise ValueError("End must follow start and cannot exceed snapshot coverage")
    raw = [row for row in raw if date.fromisoformat(row["date"]) <= requested_end]
    dates = [date.fromisoformat(row["date"]) for row in raw]
    if not dates or dates[-1] < start_date:
        raise ValueError("No NAV observations on or after the requested start")
    balance = contribution = profit = ZERO
    total_return_shares = ONE / D(raw[0]["nav"])
    account_units, unit_price = ZERO, ONE
    previous_nav, pending = None, None
    daily, signals, trades, purchases, corporate_actions = [], [], [], [], []
    annual = {year: {"profit": ZERO, "deposits": ZERO}
              for year in range(start_date.year, dates[-1].year + 1)}
    weekends, carried_orders = [], []
    for row, day in zip(raw, dates):
        nav, dividend, split = (D(row[name]) for name in ("nav", "dividend", "split"))
        assert nav > 0 and dividend >= 0 and split > 0
        # The source curve keeps its full-history baseline; the investment account
        # starts empty at the cutoff, with no pre-start signal or pending order.
        total_return_shares *= split * (ONE + dividend / nav)
        if day < start_date:
            previous_nav = nav
            continue
        trade_enabled = day.weekday() < 5
        old_shares = balance
        distribution = old_shares * split * dividend
        price_profit = old_shares * (split * nav - previous_nav) if previous_nav is not None else ZERO
        day_profit = price_profit + distribution
        profit += day_profit
        # Corporate events act on prior holdings before any new cash purchase.
        balance = (old_shares * split * nav + distribution) / nav
        if account_units:
            unit_price = balance * nav / account_units
        flow = ZERO
        if trade_enabled and pending is not None:
            flow = amount
            quantity = flow / nav
            balance += quantity
            contribution += flow
            account_units += flow / unit_price
            purchases.append((day, flow))
            trades.append({"signal_date": pending.isoformat(), "date": day.isoformat(),
                           "action": "BUY", "status": "FILLED", "amount": flow,
                           "shares": quantity, "price": nav, "fee": ZERO})
            if any(pending < date.fromisoformat(item["date"]) < day for item in weekends):
                carried_orders.append({"signal_date": pending.isoformat(), "fill_date": day.isoformat(),
                                       "amount": str(flow), "price": str(nav)})
        if dividend or split != ONE:
            corporate_actions.append({"date": day.isoformat(), "dividend_per_share": str(dividend),
                                      "split": str(split), "old_shares": str(old_shares),
                                      "dividend_cash": str(distribution), "same_day_purchase": bool(flow)})
        if not trade_enabled:
            weekends.append({"date": day.isoformat(), "nav": str(nav),
                             "previous_nav": str(previous_nav), "daily_pnl": str(day_profit),
                             "pending_signal_retained": pending.isoformat() if pending else None})
        equity = balance * nav
        annual[day.year]["profit"] += day_profit
        annual[day.year]["deposits"] += flow
        annual[day.year]["end_assets"] = equity
        daily.append({"date": day.isoformat(), "nav": nav, "dividend": dividend, "split": split,
                      "total_return_index": total_return_shares * nav, "trade_enabled": trade_enabled,
                      "shares": balance, "equity": equity, "position_value": equity,
                      "cash": ZERO, "external_flow": flow, "withdrawal": ZERO,
                      "daily_pnl": day_profit, "cumulative_pnl": profit,
                      "total_contributions": contribution, "unit_nav": unit_price})
        signals.append({"date": day.isoformat(), "action": "BUY" if trade_enabled else "HOLD",
                        "amount": amount if trade_enabled else None, "trade_enabled": trade_enabled})
        if trade_enabled:
            pending = day
        previous_nav = nav
    if pending is not None:
        trades.append({"signal_date": pending.isoformat(), "date": "", "action": "BUY",
                       "status": "PENDING_END_OF_DATA", "amount": ZERO,
                       "shares": ZERO, "price": None, "fee": ZERO})
    opening = ZERO
    for year, values in annual.items():
        values["beginning_assets"] = opening
        values.setdefault("end_assets", opening)
        values["withdrawals"] = ZERO
        opening = values["end_assets"]
        assert abs(values["profit"] - (values["end_assets"] - values["beginning_assets"] - values["deposits"])) < D("1e-25")
    assert abs(daily[-1]["equity"] - contribution - profit) < D("1e-25")
    xirr, residual = solve_xirr(purchases, dates[-1], daily[-1]["equity"])
    return {"summary": {"observations": len(daily), "buy_count": len(purchases), "sell_count": 0,
                        "total_contributions": contribution, "total_withdrawals": ZERO,
                        "final_equity": daily[-1]["equity"], "total_pnl": profit,
                        "xirr": xirr, "first_investment_date": purchases[0][0].isoformat() if purchases else None,
                        "end_date": dates[-1].isoformat()},
            "daily_buy_amount": amount, "start": start_date.isoformat(),
            "requested_end": requested_end.isoformat(),
            "warmup_observations": len(raw) - len(daily),
            "daily": daily, "signals": signals, "trades": trades, "annual": annual,
            "corporate_actions": corporate_actions, "weekend_valuation_observations": weekends,
            "pending_orders_carried_over_weekends": carried_orders,
            "trade_enabled_observations": sum(s["trade_enabled"] for s in signals),
            "xirr_terminal_value_residual_yuan": residual}


def audit_source_actions(path, metadata):
    """Audit local QLab snapshots without widening the historical omission policy."""
    policy_omissions = metadata.get("existing_policy_omissions", [])
    if metadata.get("source_kind") != "qlab_snapshot":
        if policy_omissions:
            return {"passed": False, "reason": "Existing QLab omissions require a QLab snapshot source"}
        return audit_recorded_source_actions(path, metadata)

    source = metadata.get("sources", {}).get("dividends.csv", {})
    event_path = path.with_suffix(".dividends.csv")
    if not event_path.exists():
        return {"passed": False, "reason": "Missing copied QLab corporate-action CSV"}
    digest = hashlib.sha256(event_path.read_bytes()).hexdigest()
    if not source.get("path") or source.get("sha256") != digest:
        return {"passed": False, "reason": "Copied QLab corporate-action CSV differs from source fingerprint"}
    if metadata.get("approved_omissions"):
        return {"passed": False, "reason": "Local QLab batch permits only the exact existing missing-NAV policy"}

    known = {("000931", "2017-03-22", "现金分红"): D("0.08"),
             ("000931", "2017-11-14", "现金分红"): D("0.04")}
    nav_dates = {row["date"] for row in rows(path)}
    seen = set()
    for event in policy_omissions:
        key = (metadata.get("code"), event.get("date"), event.get("type"))
        try:
            value = D(str(event.get("value")))
            if not value.is_finite():
                value = None
        except InvalidOperation:
            value = None
        if key in seen or key not in known or value != known.get(key) or event.get("date") in nav_dates:
            return {"passed": False, "reason": "Omission is not an exact existing 000931 event with missing NAV",
                    "rejected_omission": event}
        seen.add(key)
    reconciled = audit_recorded_source_actions(path, {**metadata, "approved_omissions": policy_omissions})
    return {**reconciled, "source_kind": "qlab_snapshot", "source_path": source["path"],
            "source_sha256": digest, "existing_policy_omissions": policy_omissions}


class Checks:
    def __init__(self):
        self.count = 0
        self.failures = []
        self.maxima = {}

    def equal(self, actual, expected, label):
        self.count += 1
        if actual != expected:
            self.failures.append({"check": label, "actual": str(actual), "expected": str(expected)})

    def number(self, actual, expected, label):
        self.count += 1
        try:
            actual, expected = D(str(actual)), D(str(expected))
            assert actual.is_finite() and expected.is_finite()
        except (InvalidOperation, ValueError, TypeError, AssertionError):
            self.failures.append({"check": label, "actual": str(actual), "expected": str(expected),
                                  "reason": "Missing or invalid numeric value"})
            return
        error = abs(actual - expected)
        key = label.split(":")[0]
        self.maxima[key] = max(self.maxima.get(key, ZERO), error)
        if error > D("2e-7") + abs(expected) * D("6e-11"):
            self.failures.append({"check": label, "actual": str(actual), "expected": str(expected), "error": str(error)})

    def boolean(self, actual, expected, label):
        self.equal(str(actual).lower(), str(expected).lower(), label)

    def result(self):
        return {"passed": not self.failures, "comparisons": self.count,
                "failure_count": len(self.failures), "failures": self.failures[:40],
                "maximum_absolute_errors": {key: str(value) for key, value in self.maxima.items()}}


def compare(result, prefix, code, source_hash, rule="Fundb"):
    check = Checks()
    payload = json.loads(prefix.with_suffix(".summary.json").read_text())
    for field, expected in result["summary"].items():
        actual = payload["summary"][field]
        if field in {"first_investment_date", "end_date"} or expected is None:
            check.equal(actual, expected, "summary." + field)
        else:
            check.number(actual, expected, "summary." + field)
    check.equal(payload["metadata"]["approved_curve_data_sha256"], source_hash, "reviewed_snapshot_hash")
    check.equal(payload["metadata"]["data_hashes"][code], source_hash, "report_snapshot_hash")
    check.equal(payload["config"]["engine"]["trade_calendar_column"], "trade_enabled", "explicit_execution_calendar")
    check.equal(payload["config"]["engine"]["start"], result["start"], "requested_start")
    check.equal(payload["config"]["engine"]["end"], result["requested_end"], "requested_end")
    metadata = payload["metadata"]
    override = metadata.get("user_rule_override", {})
    check.equal(override.get("mode"), "daily_fixed", "override.mode")
    check.number(override.get("amount"), result["daily_buy_amount"], "override.amount")
    check.equal(override.get("ranking"), False, "override.no_peer_ranking")
    check.equal(override.get("sell"), False, "override.no_sells")
    check.equal(override.get("calendar"), "weekday_nav", "override.calendar")
    # Legacy daily-40 reports omit this declaration; the full daily BUY comparison
    # below independently verifies that no price-position threshold was applied.
    if "threshold_applied" in override:
        check.equal(override["threshold_applied"], False, "override.no_threshold")
    strategy = metadata.get("strategy", {})
    check.equal(strategy.get("name"), rule, "strategy.rule")
    check.equal(strategy.get("selection_mode"), "daily_fixed", "strategy.selection_mode")
    selection = metadata.get("signal_metadata", {})
    check.equal(selection.get("ranking"), False, "signal_metadata.no_peer_ranking")
    check.number(selection.get("amount"), result["daily_buy_amount"], "signal_metadata.amount")
    for suffix, label in [("daily.csv", "portfolio"), ("components_daily.csv.gz", "component")]:
        actual_rows = rows(prefix.with_suffix("." + suffix))
        check.equal(len(actual_rows), len(result["daily"]), label + ".row_count")
        for actual, expected in zip(actual_rows, result["daily"]):
            check.equal(actual["date"], expected["date"], label + ".date")
            if label == "component":
                check.equal(actual["code"], code, label + ".code")
                check.equal(actual["asset_type"], rule, label + ".rule")
                check.boolean(actual["trade_enabled"], expected["trade_enabled"], label + ".trade_enabled")
            for field, value in expected.items():
                if field in {"date", "trade_enabled"}:
                    continue
                if label == "component" or field in actual:
                    check.number(actual[field], value, label + "." + field + ":" + expected["date"])
    actual_signals = rows(prefix.with_suffix(".signals.csv.gz"))
    check.equal(len(actual_signals), len(result["signals"]), "signal.row_count")
    for actual, expected in zip(actual_signals, result["signals"]):
        for field in ["date", "action"]:
            check.equal(actual[field], expected[field], "signal." + field)
        check.equal(actual["code"], code, "signal.code")
        check.equal(actual["asset_type"], rule, "signal.rule")
        check.boolean(actual["trade_enabled"], expected["trade_enabled"], "signal.trade_enabled")
        check.equal(actual.get("rank", ""), "", "signal.no_peer_ranking")
        if expected["amount"] is not None:
            check.number(actual["amount"], expected["amount"], "signal.amount:" + expected["date"])
    actual_trades = rows(prefix.with_suffix(".trades.csv"))
    check.equal(len(actual_trades), len(result["trades"]), "trade.row_count")
    for actual, expected in zip(actual_trades, result["trades"]):
        check.equal(actual["code"], code, "trade.code")
        check.equal(actual["asset_type"], rule, "trade.rule")
        for field in ["signal_date", "date", "action", "status"]:
            check.equal(actual[field], expected[field], "trade." + field)
        for field in ["amount", "shares", "price", "fee"]:
            if expected[field] is not None:
                check.number(actual[field], expected[field], "trade." + field + ":" + expected["signal_date"])
    actual_annual = rows(prefix.with_suffix(".annual.csv"))
    check.equal([int(row["year"]) for row in actual_annual], list(result["annual"]), "annual.years")
    for actual in actual_annual:
        for field, expected in result["annual"][int(actual["year"])].items():
            check.number(actual[field], expected, "annual." + field + ":" + actual["year"])
    actual_lots = rows(prefix.with_suffix(".lots.csv"))
    filled = [row for row in result["trades"] if row["status"] == "FILLED"]
    check.equal(len(actual_lots), len(filled), "lot.row_count")
    by_date = {row["date"]: row for row in result["daily"]}
    last = result["daily"][-1]
    for index, (actual, purchase) in enumerate(zip(actual_lots, filled), 1):
        bought = by_date[purchase["date"]]
        value = purchase["amount"] * last["total_return_index"] / bought["total_return_index"]
        check.equal(actual["lot_id"], f"L{index:05d}", "lot.id")
        check.equal(actual["buy_date"], purchase["date"], "lot.buy_date")
        check.equal(actual["status"], "UNREALIZED", "lot.status")
        check.equal(actual["valuation_date"], last["date"], "lot.valuation_date")
        for field, expected in {"shares": value / last["nav"], "cost": purchase["amount"],
                                "proceeds": value, "pnl": value - purchase["amount"]}.items():
            check.number(actual[field], expected, "lot." + field + ":" + purchase["date"])
    return check.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/single/003358/20261111/003358.csv")
    parser.add_argument("--report-prefix", type=Path, default=ROOT / "reports/single_003358_fundb_2010")
    parser.add_argument("--amount", type=positive_amount, default=D("40"), help="Fixed daily cash purchase; default 40")
    parser.add_argument("--rule", default="Fundb", help="Expected report asset_type label; default Fundb")
    parser.add_argument("--start", default="2010-01-01", help="Start with empty holdings on or after this date")
    parser.add_argument("--end", help="Last allowed NAV date; defaults to report engine end, else snapshot end")
    parser.add_argument("--require-report", action="store_true")
    args = parser.parse_args()
    metadata = json.loads(args.data.with_suffix(".metadata.json").read_text())
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if metadata["data_sha256"] != digest:
        raise ValueError("Input differs from the reviewed snapshot")
    report_path = args.report_prefix.with_suffix(".summary.json")
    end = args.end
    if end is None and report_path.exists():
        end = json.loads(report_path.read_text())["config"]["engine"]["end"]
    result = calculate(args.data, amount=args.amount, start=args.start, end=end)
    actions = audit_source_actions(args.data, metadata)
    comparison = (compare(result, args.report_prefix, metadata["code"], digest, rule=args.rule)
                  if args.report_prefix.with_suffix(".summary.json").exists() else None)
    passed = comparison["passed"] and actions["passed"] if comparison is not None else None
    output = {"stage": "awaiting_production_report" if comparison is None else "verified" if passed else "verification_failed",
              "passed": passed, "code": metadata["code"], "source_sha256": digest,
              "rule": args.rule, "daily_buy_amount": result["daily_buy_amount"],
              "start": result["start"], "warmup_observations": result["warmup_observations"],
              "requested_end": result["requested_end"],
              "method": __doc__, "decimal_precision": getcontext().prec,
              "summary": result["summary"], "annual": result["annual"],
              "trade_enabled_observations": result["trade_enabled_observations"],
              "weekend_valuation_observations": result["weekend_valuation_observations"],
              "pending_orders_carried_over_weekends": result["pending_orders_carried_over_weekends"],
              "corporate_actions": result["corporate_actions"], "source_corporate_action_reconciliation": actions,
              "xirr_terminal_value_residual_yuan": result["xirr_terminal_value_residual_yuan"],
              "comparison": comparison}
    path = args.report_prefix.with_suffix(".independent_audit.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"summary": output["summary"], "trade_enabled_observations": output["trade_enabled_observations"],
                      "weekend_valuation_count": len(output["weekend_valuation_observations"]),
                      "source_corporate_action_reconciliation": actions, "comparison": comparison,
                      "audit": str(path)}, ensure_ascii=False, indent=2, default=str), flush=True)
    raise SystemExit(1 if not actions["passed"] or (comparison is not None and not passed)
                     or (args.require_report and comparison is None) else 0)


if __name__ == "__main__":
    main()
