"""Reproduce Fundd parity against a separately checked-out QLab repository.

Usage: python scripts/verify_qlab.py --qlab /path/to/QLab
No QLab code or state is copied or modified. Only pure calculation/rule modules
are imported; historical live suggestion ledgers are deliberately not used.
"""
import argparse
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qlab", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1]/"data/110020.csv")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from alphatrack.data import load_data
    from alphatrack.strategies.fundd import FunddStrategy
    source = args.qlab.resolve()
    sys.path.insert(0, str(source))
    from trading.config import TradingConfig
    from trading.engine import TradingEngine, FUNDD_COLUMNS
    spec = importlib.util.spec_from_file_location("qlab_reference_calculator", source/"core/calculator.py")
    calculator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calculator)
    frame, _ = load_data(args.data)
    raw = frame[["date", "nav"]].rename(columns={"date": "净值日期", "nav": "单位净值"})
    events = []
    for row in frame.itertuples():
        if row.split != 1:
            events.append({"分红日": row.date, "分红类型": "份额折算", "分红": f"1:{row.split}"})
        if row.dividend != 0:
            events.append({"分红日": row.date, "分红类型": "现金分红", "分红": row.dividend})
    dividends = pd.DataFrame(events) if events else None
    returns = calculator.ReturnCalculator.calculate_return_with_dividend_reinvest(raw, dividends)
    reference = returns.drop(columns="单位净值").rename(columns={"累计收益率": "单位净值"})
    reference = calculator.PercentileCalculator.calculate_percentile_position(reference, [1, 2, 3, 5, 10])
    strategy = FunddStrategy()
    actual = strategy.prepare(frame)
    engine = TradingEngine(TradingConfig(fundd_base_amount=50))
    sold = ()
    counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    errors = []
    for i in range(len(frame)):
        expected, sold = engine.evaluate("Fundd", "110020", "reference", str(frame.date.iloc[i]), reference.iloc[i], sold_levels=sold)
        decision = strategy.decide(actual.iloc[i])
        counts[decision["action"]] += 1
        mismatch = expected.action != decision["action"]
        if expected.action == decision["action"] == "BUY":
            mismatch |= expected.amount != decision["amount"]
        if expected.action == decision["action"] == "SELL":
            mismatch |= abs(float(Fraction(expected.fraction))-decision["fraction"]) > 1e-12
            expected_levels = [step["threshold"] * 100 for step in expected.sell_steps]
            mismatch |= len(expected_levels) != len(decision["levels"])
            if len(expected_levels) == len(decision["levels"]):
                mismatch |= not np.allclose(expected_levels, decision["levels"], rtol=0, atol=1e-12)
        if mismatch:
            errors.append({"date": str(frame.date.iloc[i].date()), "expected": expected.action, "actual": decision})
    maximum_error = 0.
    for current, original in zip(("position_5y", "position_10y", "position_all"), FUNDD_COLUMNS):
        if not np.array_equal(actual[current].isna(), reference[original].isna()):
            raise AssertionError(f"Missing-value mismatch: {current}")
        delta = np.abs(actual[current].to_numpy() - reference[original].to_numpy())
        if np.isfinite(delta).any():
            maximum_error = max(maximum_error, float(np.nanmax(delta)))
    summary = {"observations": len(frame), "start": str(frame.date.iloc[0].date()),
               "end": str(frame.date.iloc[-1].date()), "actions": counts,
               "decision_mismatches": len(errors), "max_position_error": maximum_error,
               "position_tolerance": 1e-12, "first_errors": errors[:5]}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors or maximum_error > 1e-12:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
