"""Audit single-fund batch reports against QLab and independent accounting.

QLab's calculator and native TradingEngine reproduce original Fundd/Fundf_1
signals without calling the production signal adapter. Decimal normalized-unit
positions and FIFO independently replay those exported signals. A separate
cash-flow/TWR implementation verifies metrics, annual profits and chart arrays.
Fundc fixed purchases reuse the established independent daily-amount audit.
"""
import argparse
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alphatrack.strategies.qlab import QLabRules
from scripts import audit_report_curves as curves
from scripts.audit_ledger_math import Checks, COMPONENT_FIELDS, IndependentFIFO, input_history, read_csv, replay
from scripts.verify_single_fundb_daily import audit_source_actions, calculate as daily_calculate, compare as daily_compare


def native_signals(source, path, payload):
    """Call source calculations directly, with empty sale state at start."""
    rules = QLabRules(source)
    asset = payload["funds"][0]
    kind, code = asset["asset_type"], asset["code"]
    start, end = payload["config"]["engine"]["start"], payload["config"]["engine"]["end"]
    raw = pd.read_csv(path, parse_dates=["date"])
    raw = raw[raw.date <= pd.Timestamp(end)]
    events = pd.read_csv(path.with_suffix(".dividends.csv"), dtype=str, encoding="utf-8-sig")
    events["分红日"] = pd.to_datetime(events["分红日"])
    nav = raw[["date", "nav"]].rename(columns={"date": "净值日期", "nav": "单位净值"})
    reference = rules.calculator.ReturnCalculator.calculate_return_with_dividend_reinvest(nav, events)
    positions = reference.drop(columns="单位净值").rename(columns={"累计收益率": "单位净值"})
    positions = rules.calculator.PercentileCalculator.calculate_percentile_position(positions, [1, 2, 3, 5, 10])
    config = rules.configs[kind]
    if kind == "Fundd" and asset["category"].strip() == "基金>宽基指数":
        config = replace(config, fundd_base_amount=config.fundd_broad_base_amount)
    engine, sold = rules.engine_module.TradingEngine(config), ()
    result = []
    for (_, row), day in zip(positions.iterrows(), raw.date):
        if day < pd.Timestamp(start):
            continue
        decision, sold = engine.evaluate(kind, code, asset["name"], str(day.date()), row, sold_levels=sold)
        result.append(dict(date=str(day.date()), action=decision.action,
                           amount=decision.amount,
                           fraction=float(Fraction(decision.fraction)) if decision.fraction else None))
    return result


def audit_original_rule(source, path, prefix, payload):
    checks = Checks()
    cfg = payload["config"]["engine"]
    assert cfg["initial_cash"] == cfg["buy_fee"] == cfg["sell_fee"] == 0
    assert cfg["funding_mode"] == "topup" and cfg["withdraw_on_sell"] is True
    assert cfg["trade_calendar_column"] is None
    actual_signals = list(read_csv(prefix.with_suffix(".signals.csv.gz")))
    expected_signals = native_signals(source, path, payload)
    asset = payload["funds"][0]
    checks.truth(len(actual_signals) == len(expected_signals), "signal_count", prefix.name)
    for actual, expected in zip(actual_signals, expected_signals):
        checks.truth(actual["code"] == asset["code"] and actual["asset_type"] == asset["asset_type"],
                     "signal_identity", expected["date"])
        for field in ["date", "action"]:
            checks.truth(actual[field] == expected[field], "native_signal." + field, expected["date"])
        for field in ["amount", "fraction"]:
            if expected[field] is None:
                checks.truth(actual[field] == "", "native_signal." + field, expected["date"])
            else:
                checks.number(actual[field], expected[field], "native_signal." + field, expected["date"], "1e-11")
    signals = {s["date"]: s for s in actual_signals}
    raw = [r for r in input_history(path) if cfg["start"] <= r["date"] <= cfg["end"]]
    actual_rows = list(read_csv(prefix.with_suffix(".components_daily.csv.gz")))
    actual_trades = iter(read_csv(prefix.with_suffix(".trades.csv")))
    checks.truth([r["date"] for r in actual_rows] == [r["date"] for r in raw], "ledger_dates", prefix.name)
    fifo, counts = IndependentFIFO(), {"observations": 0, "buys": 0, "sells": 0, "cash_events": 0, "splits": 0}
    for actual, (expected, trade) in zip(actual_rows, replay(raw, signals)):
        day = expected["date"]
        checks.truth(actual["code"] == asset["code"] and actual["asset_type"] == asset["asset_type"],
                     "component_identity", day)
        counts["observations"] += 1
        counts["cash_events"] += bool(expected["dividend"])
        counts["splits"] += expected["split"] != 1
        for field in COMPONENT_FIELDS:
            if field not in {"date", "asset_type", "code"}:
                checks.number(actual[field], expected[field], "component." + field, day,
                              "0.00000002" if field in {"nav", "dividend", "split", "total_return_index", "shares"} else "0.00002")
        fifo.consume(expected, trade)
        if trade is not None:
            if trade["status"] == "FILLED":
                counts["buys" if trade["action"] == "BUY" else "sells"] += 1
            actual_trade = next(actual_trades, None)
            checks.truth(actual_trade is not None, "trade_exists", day)
            if actual_trade is not None:
                for field in ["signal_date", "date", "action", "status"]:
                    checks.truth(actual_trade[field] == trade[field], "trade." + field, day)
                for field in ["amount", "shares", "price", "fee"]:
                    checks.number(actual_trade[field], trade[field], "trade." + field, day)
    pending = list(actual_trades)
    last_signal = signals[raw[-1]["date"]]
    expect_pending = last_signal["action"] != "HOLD"
    checks.truth(len(pending) == int(expect_pending), "pending_count", prefix.name)
    if expect_pending and pending:
        checks.truth(pending[0]["status"] == "PENDING_END_OF_DATA" and not pending[0]["date"]
                     and pending[0]["signal_date"] == raw[-1]["date"]
                     and pending[0]["action"] == last_signal["action"], "unfilled_terminal_signal", prefix.name)
    actual_lots, expected_lots = list(read_csv(prefix.with_suffix(".lots.csv"))), fifo.finish()
    checks.truth(len(actual_lots) == len(expected_lots), "fifo_count", prefix.name)
    for actual, expected in zip(actual_lots, expected_lots):
        for field in ["lot_id", "buy_date", "sell_date", "valuation_date", "status"]:
            checks.truth(actual[field] == expected[field], "fifo." + field, expected["lot_id"])
        for field in ["shares", "cost", "proceeds", "pnl"]:
            checks.number(actual[field], expected[field], "fifo." + field, expected["lot_id"])
    return dict(method="native QLab decisions plus independent Decimal ledger and FIFO", counts=counts, **checks.result())


def audit_metrics_and_charts(prefix, payload):
    curves.checks, curves.failures, curves.maximum_errors = 0, [], {}
    comp = pd.read_csv(prefix.with_suffix(".components_daily.csv.gz"), parse_dates=["date"], dtype={"code": str})
    trades = pd.read_csv(prefix.with_suffix(".trades.csv"), parse_dates=["date"], dtype={"code": str})
    daily = pd.read_csv(prefix.with_suffix(".daily.csv"), parse_dates=["date"]).set_index("date")
    annual = pd.read_csv(prefix.with_suffix(".annual.csv")).set_index("year")
    events, facts = curves.component_events(comp, trades, prefix.name)
    independent = curves.independent_aggregate(events)
    # CSV readers infer integer-valued deposits/zero withdrawals as int64. Keep
    # the independent XIRR's terminal mark from being truncated by in-place +=.
    independent[["external_flow", "withdrawal"]] = independent[["external_flow", "withdrawal"]].astype(float)
    curves.require("daily calendar", daily.index.equals(independent.index))
    for field in ["equity", "previous_equity", "external_flow", "withdrawal", "daily_pnl", "cumulative_pnl", "total_contributions", "cumulative_withdrawals", "daily_return", "unit_nav", "drawdown"]:
        curves.equal(field, daily[field], independent[field], curves.RATE_TOL if field in {"daily_return", "unit_nav", "drawdown"} else curves.MONEY_TOL)
    scope = curves.scope_metrics(independent, payload["summary"], annual, prefix.name)
    document = prefix.with_suffix(".html").read_text()
    charts = curves.plots(document)
    curves.require("four chart IDs", set(charts) == {"portfolio-annual-chart", "portfolio-pnl-chart", "portfolio-nav-chart", "portfolio-funding-chart"})
    for trace in charts["portfolio-pnl-chart"][0]:
        column = "daily_pnl" if "当日" in trace["name"] else "cumulative_pnl"
        curves.verify_trace(trace["name"], trace, independent, column)
    for trace, field in zip(charts["portfolio-nav-chart"][0], ["unit_nav", "drawdown"]):
        curves.verify_trace(trace["name"], trace, independent, field)
    for trace, field in zip(charts["portfolio-funding-chart"][0], ["equity", "total_contributions", "cumulative_withdrawals"]):
        curves.verify_trace(trace["name"], trace, independent, field)
    expected_year_values = annual.profit.where(annual.year_status.ne("not_incepted"), np.nan)
    curves.equal("annual chart values", curves.values(charts["portfolio-annual-chart"][0][0]["y"]), expected_year_values)
    return dict(passed=not curves.failures, checks=curves.checks, failures=curves.failures,
                maximum_errors=curves.maximum_errors, facts=facts,
                independent_xirr=scope["independent_xirr"])


def audit(source, report):
    prefix = report.with_suffix("")
    payload = json.loads(prefix.with_suffix(".summary.json").read_text())
    path = Path(payload["config"]["data"])
    path = path if path.is_absolute() else ROOT / path
    meta = json.loads(path.with_suffix(".metadata.json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == meta["data_sha256"] == payload["metadata"]["data_hashes"][meta["code"]]
    source_actions = audit_source_actions(path, meta)
    kind = payload["funds"][0]["asset_type"]
    if kind == "Fundc":
        ledger = daily_calculate(path, amount=40, start=payload["config"]["engine"]["start"], end=payload["config"]["engine"]["end"])
        accounting = daily_compare(ledger, prefix, meta["code"], digest, rule=kind)
    elif kind in {"Fundd", "Fundf_1"}:
        accounting = audit_original_rule(source, path, prefix, payload)
    else:
        raise ValueError(f"Unsupported batch type {kind}")
    metrics = audit_metrics_and_charts(prefix, payload)
    result = dict(code=meta["code"], asset_type=kind, source_sha256=digest,
                  passed=source_actions["passed"] and accounting["passed"] and metrics["passed"],
                  source_actions=source_actions, accounting=accounting, metrics_and_charts=metrics)
    prefix.with_suffix(".independent_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qlab-source", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if (args.reports_dir is None) == (args.report is None):
        parser.error("Specify exactly one of --reports-dir or --report")
    reports = [args.report] if args.report else sorted(args.reports_dir.glob("*/single_*.html"))
    if not reports:
        parser.error("No individual reports found")
    results = []
    for report in reports:
        result = audit(args.qlab_source, report)
        results.append(dict(code=result["code"], asset_type=result["asset_type"], passed=result["passed"],
                            comparisons=result["accounting"]["comparisons"] + result["metrics_and_charts"]["checks"]))
        print(json.dumps(results[-1]), flush=True)
    output = dict(passed=all(r["passed"] for r in results), funds=len(results),
                  comparisons=sum(r["comparisons"] for r in results), results=results)
    if args.reports_dir:
        (args.reports_dir / "independent_audit_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in output.items() if k != "results"}), flush=True)
    raise SystemExit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
