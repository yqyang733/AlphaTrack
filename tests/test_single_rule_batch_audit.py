"""Known synthetic rule reports and intentional source-signal corruption."""
import hashlib
import json

import pandas as pd
import pytest

from alphatrack.data import load_data
from alphatrack.strategies.qlab import QLabRules
from scripts.backtest_single_fund import ROOT, run
from scripts.verify_single_rule_batch import audit, audit_original_rule


@pytest.fixture(params=["Fundd", "Fundf_1"])
def rule_report(request, tmp_path):
    kind = request.param
    path = tmp_path / "123456.csv"
    frame = pd.DataFrame({"date": pd.date_range("2010-01-04", periods=30, freq="MS"),
                          "nav": [1., .9, .8, .7, .8, .9, 1.1, 1.2, 1.4, 1.6] * 3,
                          "dividend": [0.] * 30, "split": [1.] * 30})
    frame.loc[4, "dividend"] = .1
    frame.loc[15, "split"] = 1.5
    frame.to_csv(path, index=False)
    event_path = path.with_suffix(".dividends.csv")
    events = pd.DataFrame({"分红日": [frame.date.iloc[4], frame.date.iloc[15]],
                           "分红类型": ["现金分红", "份额折算"], "分红": ["0.1", "1:1.5"]})
    events.to_csv(event_path, index=False)
    rules = QLabRules(ROOT / "vendor/qlab")
    actual, _ = load_data(path)
    reference = rules.calculator.ReturnCalculator.calculate_return_with_dividend_reinvest(
        frame[["date", "nav"]].rename(columns={"date": "净值日期", "nav": "单位净值"}), events)
    meta = dict(code="123456", name="Synthetic", requested_strategy=kind,
                first_date=str(frame.date.iloc[0].date()), last_date=str(frame.date.iloc[-1].date()),
                data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                qlab_max_absolute_error=float((actual.total_return_index - (reference["累计收益率"] + 1)).abs().max()),
                calculator_sha256=rules.source_hashes["core/calculator.py"],
                source_kind="qlab_snapshot", sources={"dividends.csv": {
                    "path": str(event_path), "sha256": hashlib.sha256(event_path.read_bytes()).hexdigest()}})
    path.with_suffix(".metadata.json").write_text(json.dumps(meta))
    prefix = tmp_path / f"single_{kind}"
    payload = run(path, kind, prefix.with_suffix(".html"), start="2009-01-01", curve_confirmed=True,
                  category="基金>宽基指数" if kind == "Fundd" else None)
    return kind, path, prefix, payload


def test_original_rules_ledger_metrics_and_charts_independently_agree(rule_report):
    kind, path, prefix, payload = rule_report
    result = audit(ROOT / "vendor/qlab", prefix.with_suffix(".html"))
    assert result["passed"], result
    assert result["accounting"]["counts"]["buys"] > 0
    if kind == "Fundd":
        assert result["accounting"]["counts"]["sells"] > 0


def test_native_rule_comparison_detects_changed_exported_signal(rule_report):
    kind, path, prefix, payload = rule_report
    signals_path = prefix.with_suffix(".signals.csv.gz")
    signals = pd.read_csv(signals_path)
    index = signals.index[signals.action.eq("BUY")][0]
    signals.loc[index, "amount"] += 1
    signals.to_csv(signals_path, index=False, compression="gzip")
    result = audit_original_rule(ROOT / "vendor/qlab", path, prefix, payload)
    assert not result["passed"]
    assert any(f["check"] == "native_signal.amount" for f in result["failures"])
