"""Known boundary behavior and malformed-report handling in the independent audit."""
from decimal import Decimal

import pytest

from scripts import verify_single_fundg as audit


def test_exact_sixty_percent_and_qlab_float_boundary_are_reported_separately(tmp_path):
    source = tmp_path / "boundary.csv"
    source.write_text(
        "date,nav,dividend,split\n"
        "2012-06-14,1,0,1\n"
        "2012-06-15,1,0,1\n"
        "2012-06-22,.999,0,1\n"
        "2012-06-29,.9995,0,1\n"
        "2012-06-30,.9996,0,1\n"
        "2012-07-06,.9999,0,1\n",
        encoding="utf-8",
    )
    exact = audit.calculate(source, signal_arithmetic="decimal")
    qlab = audit.calculate(source, signal_arithmetic="qlab-binary64")
    exact_signal = next(row for row in exact["signals"] if row["date"] == "2012-06-30")
    qlab_signal = next(row for row in qlab["signals"] if row["date"] == "2012-06-30")
    assert Decimal(exact_signal["position"]) == Decimal("0.6")
    assert exact_signal["buy"] is True
    assert float(qlab_signal["position"]) == pytest.approx(.6000000000000444, rel=0, abs=1e-16)
    assert Decimal(qlab_signal["position"]) > Decimal("0.6")
    assert qlab_signal["buy"] is False

    difference = next(row for row in qlab["signal_arithmetic_differences"]
                      if row["date"] == "2012-06-30")
    assert Decimal(difference["decimal_position"]) == Decimal("0.6")
    assert Decimal(difference["qlab_binary64_position"]) > Decimal("0.6")
    assert difference["decimal_action"] == "BUY"
    assert difference["qlab_binary64_action"] == "HOLD"
    assert [Decimal(row["nav"]) for row in difference["window_inputs"]] == [
        Decimal("1"), Decimal("1"), Decimal(".999"), Decimal(".9995"), Decimal(".9996")]
    # Keeping the original QLab decision changes one next-NAV deposit, not its date.
    assert (exact["summary"]["buy_count"], qlab["summary"]["buy_count"]) == (4, 3)
    assert exact["summary"]["total_contributions"] - qlab["summary"]["total_contributions"] == 10
    assert exact["trades"][-1]["signal_date"] == "2012-06-30"
    assert exact["trades"][-1]["date"] == "2012-07-06"
    assert not any(row["signal_date"] == "2012-06-30" for row in qlab["trades"])


def test_blank_buy_amount_is_a_comparison_failure_not_decimal_exception(tmp_path, monkeypatch):
    prefix = tmp_path / "report"
    prefix.with_suffix(".summary.json").write_text('{"summary": {}}', encoding="utf-8")
    expected = {
        "summary": {},
        "daily": [{"date": "2020-01-02", "nav": Decimal(1), "total_return_index": Decimal(1)}],
        "signals": [{"date": "2020-01-02", "buy": True, "position": "0.5"}],
        "trades": [], "annual": {},
    }
    daily = [{"date": "2020-01-02", "nav": "1", "total_return_index": "1"}]
    exports = {
        "report.daily.csv": daily,
        "report.components_daily.csv.gz": [{**daily[0], "code": "050025"}],
        "report.signals.csv.gz": [{"date": "2020-01-02", "code": "050025", "action": "BUY",
                                    "近1年百分位": "0.5", "amount": ""}],
        "report.trades.csv": [], "report.annual.csv": [], "report.lots.csv": [],
    }
    monkeypatch.setattr(audit, "rows", lambda path: exports[path.name])
    comparison = audit.compare(expected, prefix, code="050025")
    assert comparison["passed"] is False
    assert comparison["failure_count"] == 1
    failure = comparison["failures"][0]
    assert failure["check"] == "signal.amount:2020-01-02"
    assert failure["actual"] == ""
    assert Decimal(failure["expected"]) == 10
    assert failure["reason"]
