"""Independent Decimal reconstruction of saved zero-fee AlphaTrack backtests.

No production engine, data adapter, portfolio, or FIFO code is imported. Input
NAV and corporate actions plus yesterday's exported signals generate positions
and trades from scratch. P&L is accumulated from price movement and dividend
entitlement, independently of the cash-flow identity. Portfolio NAV is obtained
by issuing/redeeming portfolio units at the current marked unit price.
"""
import argparse
from collections import defaultdict, deque
import csv
from decimal import Decimal, getcontext
import gzip
import json
from pathlib import Path


getcontext().prec = 40
D = Decimal
ZERO, ONE = D(0), D(1)
ROOT = Path(__file__).resolve().parents[1]


def read_csv(path, fields=None):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            yield row if fields is None else {key: row[key] for key in fields}


class Checks:
    def __init__(self):
        self.comparisons = 0
        self.failures = []
        self.failure_count = 0
        self.maximum_error = {}
        self.worst_case = {}

    def truth(self, condition, label, context):
        self.comparisons += 1
        if not condition:
            self.failure_count += 1
            if len(self.failures) < 40:
                self.failures.append({"check": label, "context": context})

    def number(self, actual, expected, label, context, absolute="0.00002"):
        self.comparisons += 1
        actual, expected = D(str(actual)), D(str(expected))
        error = abs(actual - expected)
        if error > self.maximum_error.get(label, ZERO):
            self.maximum_error[label] = error
            self.worst_case[label] = context
        # Exported CSVs use 12 significant digits; allow their rounding plus
        # propagation of serialized sell fractions. Keep errors visible below.
        tolerance = D(absolute) + abs(expected) * D("6e-11")
        if error > tolerance:
            self.failure_count += 1
            if len(self.failures) < 40:
                self.failures.append({"check": label, "context": context,
                                      "expected": str(expected), "actual": str(actual),
                                      "error": str(error), "tolerance": str(tolerance)})

    def result(self):
        return {"passed": self.failure_count == 0, "comparisons": self.comparisons,
                "failure_count": self.failure_count, "failures": self.failures,
                "maximum_absolute_errors": {k: float(v) for k, v in self.maximum_error.items()},
                "worst_case": self.worst_case}


def input_history(path):
    """Independent price-relative chain, not the production share accumulator."""
    rows, previous, index = [], None, ONE
    for row in read_csv(path):
        row = {"date": row["date"], **{k: D(row[k]) for k in ("nav", "dividend", "split")}}
        if previous is not None:
            index *= row["split"] * (row["nav"] + row["dividend"]) / previous
        else:
            assert row["dividend"] == ZERO and row["split"] == ONE
        row["total_return_index"] = index
        rows.append(row)
        previous = row["nav"]
    return rows


def replay(raw, signals):
    """Use corporate-action-normalized units to prevent new-buy dividends.

    Existing positions are represented in base units. Corporate events change
    the common units-to-shares multiplier, while new purchases enter at its
    already-adjusted value. Sales remove base units after that day's event.
    """
    base_units = ZERO
    multiplier = ONE
    cost = contributions = withdrawals = realized = economic_profit = ZERO
    previous_nav = None
    pending = None
    for row in raw:
        nav, dividend, split = (row[k] for k in ("nav", "dividend", "split"))
        old_shares = base_units * multiplier
        distribution = old_shares * split * dividend
        price_pnl = (old_shares * (split * nav - previous_nav)
                     if previous_nav is not None else ZERO)
        day_pnl = price_pnl + distribution
        economic_profit += day_pnl
        multiplier *= split * (ONE + dividend / nav)
        deposit = withdrawal = ZERO
        trade = None
        if pending:
            action = pending["action"]
            quantity = amount = ZERO
            status = "FILLED"
            if action == "BUY":
                amount = D(pending["amount"])
                assert amount > 0
                quantity = amount / nav
                base_units += amount / (nav * multiplier)
                deposit = amount
                contributions += amount
                cost += amount
            elif action == "SELL":
                fraction = D(pending["fraction"])
                assert ZERO < fraction <= ONE
                if base_units * multiplier <= D("1e-10"):
                    status = "REJECTED_NO_HOLDING"
                else:
                    sold_units = base_units * fraction
                    quantity = sold_units * multiplier
                    amount = quantity * nav
                    removed_cost = cost * fraction
                    realized += amount - removed_cost
                    cost -= removed_cost
                    base_units -= sold_units
                    withdrawal = amount
                    withdrawals += amount
            else:
                raise AssertionError(action)
            trade = {"signal_date": pending["date"], "date": row["date"],
                     "action": action, "status": status, "amount": amount,
                     "shares": quantity, "price": nav, "fee": ZERO}
        shares = base_units * multiplier
        equity = shares * nav
        expected = {**row, "cash": ZERO, "shares": shares, "position_value": equity,
                    "equity": equity, "holding_cost": cost, "realized_pnl": realized,
                    "unrealized_pnl": equity - cost, "fees": ZERO,
                    "external_flow": deposit, "withdrawal": withdrawal,
                    "total_contributions": contributions, "cumulative_withdrawals": withdrawals,
                    "cumulative_pnl": economic_profit, "daily_pnl": day_pnl,
                    "dividend_cash": distribution, "pre_event_shares": old_shares,
                    "cashflow_profit": equity + withdrawals - contributions,
                    "share_multiplier": multiplier}
        yield expected, trade
        decision = signals[row["date"]]
        pending = decision if decision["action"] != "HOLD" else None
        previous_nav = nav


class IndependentFIFO:
    """Lots retain fixed base units rather than being updated on every event."""
    def __init__(self):
        self.queue = deque()
        self.records = []
        self.next_id = 1
        self.last = None

    def record(self, lot, units, cost, proceeds, event, sold):
        self.records.append({"lot_id": lot["id"], "buy_date": lot["date"],
                             "sell_date": event["date"] if sold else "",
                             "valuation_date": event["date"],
                             "status": "REALIZED" if sold else "UNREALIZED",
                             "shares": units * event["share_multiplier"],
                             "cost": cost, "proceeds": proceeds, "pnl": proceeds - cost})

    def consume(self, event, trade):
        self.last = event
        if trade is None or trade["status"] != "FILLED":
            return
        units = trade["shares"] / event["share_multiplier"]
        if trade["action"] == "BUY":
            self.queue.append({"id": f"L{self.next_id:05d}", "date": event["date"],
                               "units": units, "cost": trade["amount"]})
            self.next_id += 1
        else:
            remaining = units
            while remaining > D("1e-28"):
                lot = self.queue[0]
                matched = min(remaining, lot["units"])
                cost = lot["cost"] * matched / lot["units"]
                proceeds = matched * event["share_multiplier"] * event["nav"]
                self.record(lot, matched, cost, proceeds, event, True)
                remaining -= matched
                lot["units"] -= matched
                lot["cost"] -= cost
                if lot["units"] <= D("1e-28"):
                    self.queue.popleft()

    def finish(self):
        for lot in self.queue:
            value = lot["units"] * self.last["share_multiplier"] * self.last["nav"]
            self.record(lot, lot["units"], lot["cost"], value, self.last, False)
        return self.records


def self_test():
    """Known-answer dividend-buy, split-sell, and first-day-event examples."""
    raw = [dict(date="2020-01-01", nav=D(10), dividend=ZERO, split=ONE),
           dict(date="2020-01-02", nav=D(10), dividend=ZERO, split=ONE),
           dict(date="2020-01-03", nav=D(9), dividend=ONE, split=ONE),
           dict(date="2020-01-04", nav=D("4.5"), dividend=ZERO, split=D(2))]
    signals = {r["date"]: {"date": r["date"], "action": "HOLD"} for r in raw}
    signals[raw[0]["date"]].update(action="BUY", amount="100")
    signals[raw[1]["date"]].update(action="BUY", amount="90")
    signals[raw[2]["date"]].update(action="SELL", fraction="0.5")
    rows = list(replay(raw, signals))
    assert rows[2][0]["dividend_cash"] == D(10)  # only the old 10 shares
    assert abs(rows[2][0]["equity"] - D(190)) < D("1e-30")
    assert abs(rows[3][1]["amount"] - D(95)) < D("1e-30")
    assert abs(rows[3][0]["equity"] - D(95)) < D("1e-30")
    assert abs(rows[3][0]["cumulative_pnl"]) < D("1e-30")
    fifo = IndependentFIFO()
    for event, trade in rows:
        fifo.consume(event, trade)
    lots = fifo.finish()
    assert len(lots) == 3
    assert abs(lots[0]["cost"] - D(95)) < D("1e-30")
    assert abs(sum(lot["pnl"] for lot in lots)) < D("1e-30")
    raw[0]["dividend"] = D(5)
    assert next(replay(raw, signals))[0]["dividend_cash"] == ZERO
    return {"passed": True, "cases": ["dividend entitlement excludes same-day purchase",
            "split and dividend reinvestment preserve zero economic P&L",
            "sale after split values adjusted shares", "empty opening balance has no dividend",
            "normalized-unit FIFO retains dividend ownership and purchase costs"]}


COMPONENT_FIELDS = ["date", "code", "asset_type", "nav", "dividend", "split", "total_return_index",
                    "cash", "shares", "position_value", "equity", "holding_cost", "realized_pnl",
                    "unrealized_pnl", "fees", "external_flow", "withdrawal", "total_contributions",
                    "cumulative_withdrawals", "cumulative_pnl", "daily_pnl"]


def audit_window(root, prefix, histories):
    checks = Checks()
    path = root / "reports" / prefix
    payload = json.loads(path.with_suffix(".summary.json").read_text())
    cfg = payload["config"]["engine"]
    assert cfg["initial_cash"] == cfg["buy_fee"] == cfg["sell_fee"] == 0
    assert cfg["funding_mode"] == "topup" and cfg["withdraw_on_sell"] is True
    start, end = cfg["start"], cfg["end"]
    components, signals, trades, lots = defaultdict(list), defaultdict(dict), defaultdict(list), defaultdict(list)
    for row in read_csv(path.with_suffix(".components_daily.csv.gz"), COMPONENT_FIELDS):
        components[(row["asset_type"], row["code"])].append(row)
    for row in read_csv(path.with_suffix(".signals.csv.gz")):
        key = (row["asset_type"], row["code"])
        checks.truth(row["date"] not in signals[key], "unique_signal_date", str(key))
        signals[key][row["date"]] = row
    for row in read_csv(path.with_suffix(".trades.csv")):
        trades[(row["asset_type"], row["code"])].append(row)
    for row in read_csv(path.with_suffix(".lots.csv")):
        lots[(row["asset_type"], row["code"])].append(row)
    checks.truth(len(components) == 63, "63_funds", prefix)
    checks.truth(set(components) == set(signals), "signal_universe", prefix)
    portfolio_events = defaultdict(list)
    counts = defaultdict(int)
    event_examples = []
    distribution_total = ZERO
    for key, actual_rows in components.items():
        raw = [r for r in histories[key[1]] if start <= r["date"] <= end]
        checks.truth([r["date"] for r in actual_rows] == [r["date"] for r in raw], "observation_dates", str(key))
        checks.truth(set(signals[key]) == {r["date"] for r in raw}, "signal_dates", str(key))
        actual_trades = iter(trades[key])
        fifo = IndependentFIFO()
        for actual, (expected, expected_trade) in zip(actual_rows, replay(raw, signals[key])):
            fifo.consume(expected, expected_trade)
            context = f"{key[0]}:{key[1]}:{actual['date']}"
            counts["component_observations"] += 1
            for field in COMPONENT_FIELDS:
                if field not in {"date", "asset_type", "code"}:
                    checks.number(actual[field], expected[field], f"component.{field}", context,
                                  "0.00000002" if field in {"nav", "dividend", "split", "total_return_index", "shares"} else "0.00002")
            checks.number(expected["cashflow_profit"], expected["cumulative_pnl"], "independent_economic_vs_cashflow_profit", context)
            if expected["dividend"]:
                counts["cash_event_observations"] += 1
                counts["cash_events_with_position"] += expected["pre_event_shares"] > ZERO
                distribution_total += expected["dividend_cash"]
            if expected["split"] != ONE:
                counts["split_event_observations"] += 1
                counts["split_events_with_position"] += expected["pre_event_shares"] > ZERO
            if expected_trade is not None:
                counts["executed_or_rejected_orders"] += 1
                actual_trade = next(actual_trades, None)
                checks.truth(actual_trade is not None, "trade_exists", context)
                if actual_trade is not None:
                    for field in ["signal_date", "date", "action", "status"]:
                        checks.truth(actual_trade[field] == expected_trade[field], f"trade.{field}", context)
                    for field in ["amount", "shares", "price", "fee"]:
                        checks.number(actual_trade[field], expected_trade[field], f"trade.{field}", context)
                if expected["dividend"]:
                    counts["cash_event_with_" + expected_trade["action"].lower()] += 1
                    if len(event_examples) < 12:
                        event_examples.append({"fund": str(key), "date": expected["date"],
                                               "action": expected_trade["action"],
                                               "old_shares": float(expected["pre_event_shares"]),
                                               "dividend_cash": float(expected["dividend_cash"]),
                                               "trade_shares": float(expected_trade["shares"])})
            portfolio_events[expected["date"]].append((key, expected["equity"], expected["daily_pnl"],
                                                       expected["external_flow"], expected["withdrawal"]))
        expected_lots = fifo.finish()
        checks.truth(len(lots[key]) == len(expected_lots), "fifo.row_count", str(key))
        counts["fifo_rows"] += len(expected_lots)
        for actual_lot, expected_lot in zip(lots[key], expected_lots):
            context = f"{key}:{expected_lot['lot_id']}:{expected_lot['valuation_date']}"
            for field in ["lot_id", "buy_date", "sell_date", "valuation_date", "status"]:
                checks.truth(actual_lot[field] == expected_lot[field], "fifo." + field, context)
            for field in ["shares", "cost", "proceeds", "pnl"]:
                checks.number(actual_lot[field], expected_lot[field], "fifo." + field, context)
        remaining = list(actual_trades)
        final_signal = signals[key][raw[-1]["date"]]
        expected_pending = final_signal["action"] != "HOLD"
        checks.truth(len(remaining) == int(expected_pending), "terminal_pending_count", str(key))
        if expected_pending and remaining:
            t = remaining[0]
            checks.truth(t["status"] == "PENDING_END_OF_DATA" and not t["date"]
                         and t["signal_date"] == raw[-1]["date"] and t["action"] == final_signal["action"],
                         "terminal_signal_not_filled", str(key))
            counts["terminal_unfilled_signals"] += 1
    marks, portfolio_units = {}, ZERO
    unit_price = ONE
    contributed = withdrawn = profit = ZERO
    previous_equity, peak = ZERO, ONE
    actual_portfolio = list(read_csv(path.with_suffix(".daily.csv")))
    checks.truth([r["date"] for r in actual_portfolio] == sorted(portfolio_events), "portfolio_calendar", prefix)
    for actual in actual_portfolio:
        date = actual["date"]
        day_pnl = deposit = payout = ZERO
        for key, equity, pnl, inflow, outflow in portfolio_events[date]:
            marks[key] = equity
            day_pnl += pnl
            deposit += inflow
            payout += outflow
        equity = sum(marks.values(), ZERO)
        # Unit issuance/redemption independently reconstructs the TWR curve.
        if portfolio_units:
            unit_price = (previous_equity + day_pnl) / portfolio_units
        portfolio_units += (deposit - payout) / unit_price
        checks.number(equity, portfolio_units * unit_price, "portfolio.unit_ownership_identity", date)
        previous_equity = equity
        peak = max(peak, unit_price)
        contributed += deposit
        withdrawn += payout
        profit += day_pnl
        expected = {"equity": equity, "external_flow": deposit, "withdrawal": payout,
                    "daily_pnl": day_pnl, "cumulative_pnl": profit, "unit_nav": unit_price,
                    "drawdown": unit_price / peak - ONE, "total_contributions": contributed,
                    "cumulative_withdrawals": withdrawn}
        for field, value in expected.items():
            checks.number(actual[field], value, "portfolio." + field, date,
                          "0.00000002" if field in {"unit_nav", "drawdown"} else "0.00002")
    checks.number(payload["summary"]["total_pnl"], profit, "summary.total_pnl", prefix)
    checks.number(payload["summary"]["final_equity"], previous_equity, "summary.final_equity", prefix)
    return {"window": prefix, "start": start, "end": end, "funds": len(components),
            "coverage": dict(counts), "dividend_cash_reinvested": float(distribution_total),
            "independent_total_pnl": float(profit), "independent_final_equity": float(previous_equity),
            "independent_terminal_unit_nav": float(unit_price), "event_examples": event_examples, **checks.result()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/audit_20261111_ledger.json")
    args = parser.parse_args()
    tests = self_test()
    histories = {p.stem: input_history(p) for p in sorted((args.root / "data/qlab_portfolio").glob("*.csv"))}
    results = []
    for strategy in ["qlab_portfolio", "alphastep_random"]:
        for year in [2010, 2015, 2020, 2025]:
            result = audit_window(args.root, f"{strategy}_windows_since_{year}0101", histories)
            results.append(result)
            print(json.dumps({"window": result["window"], "passed": result["passed"],
                              "comparisons": result["comparisons"], "failures": result["failure_count"],
                              "pnl": result["independent_total_pnl"]}), flush=True)
    output = {"passed": all(r["passed"] for r in results), "method": __doc__,
              "self_test": tests, "decimal_precision": getcontext().prec,
              "source_funds": len(histories), "source_observations": sum(map(len, histories.values())),
              "source_cash_dividend_events": sum(bool(r["dividend"]) for h in histories.values() for r in h),
              "source_split_events": sum(r["split"] != ONE for h in histories.values() for r in h),
              "limitations": ["Signals are trusted inputs here; strategy generation is a separate audit.",
                              "Reconstruction retains snapshot omissions; 000931 missing-NAV events are not invented.",
                              "This audit covers the actual zero-fee, cash-topup, sale-withdrawal configuration.",
                              "Dividend reinvestment uses the existing ex-date NAV convention, not historical settlement delays."],
              "windows": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {args.output}; passed={output['passed']}", flush=True)
    raise SystemExit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
