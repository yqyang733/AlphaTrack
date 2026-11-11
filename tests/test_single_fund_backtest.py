"""Review gates and inception boundaries for the standalone Fundg workflow."""
import hashlib
import json

import pandas as pd
import pytest

from alphatrack.data import load_data
from alphatrack.strategies.qlab import QLabRules
from scripts import backtest_single_fund


@pytest.fixture
def reviewed_snapshot(tmp_path):
    """A six-observation snapshot with a per-share dividend and real QLab parity."""
    data = tmp_path / "519981.csv"
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2011-03-30", "2011-03-31", "2011-04-01",
                                "2011-04-04", "2011-04-05", "2011-04-06"]),
        "nav": [1.0, 1.0, .9, .9, .8, .85],
        "dividend": [0.0, 0.0, 0.0, 0.0, .1, 0.0],
        "split": 1.0,
    })
    raw.to_csv(data, index=False)
    rules = QLabRules(backtest_single_fund.ROOT / "vendor/qlab")
    reference = rules.calculator.ReturnCalculator.calculate_return_with_dividend_reinvest(
        raw[["date", "nav"]].rename(columns={"date": "净值日期", "nav": "单位净值"}),
        pd.DataFrame({"分红日": [pd.Timestamp("2011-04-05")],
                      "分红类型": ["现金分红"], "分红": [.1]}))
    normalized, _ = load_data(data)
    error = (normalized.total_return_index - (reference["累计收益率"] + 1)).abs().max()
    assert error <= 1e-12
    metadata = {"code": "519981", "name": "Synthetic reviewed fund", "requested_strategy": "Fundg",
                "first_date": "2011-03-30", "last_date": "2011-04-06",
                "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "qlab_max_absolute_error": float(error),
                "calculator_sha256": rules.source_hashes["core/calculator.py"]}
    data.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return data


@pytest.mark.parametrize("failure,confirmed,message", [
    ("unconfirmed", False, "User must confirm"),
    ("changed_csv", True, "previously reviewed QLab-compatible curve"),
    ("nonfinite_parity", True, "previously reviewed QLab-compatible curve"),
    ("negative_parity", True, "previously reviewed QLab-compatible curve"),
    ("diagnostic_only", True, "unresolved data gaps"),
])
def test_unreviewed_or_changed_snapshot_cannot_generate_results(
        reviewed_snapshot, tmp_path, monkeypatch, failure, confirmed, message):
    if failure == "changed_csv":
        changed = pd.read_csv(reviewed_snapshot)
        changed.loc[len(changed) - 1, "nav"] = .86  # Still valid data, but not the reviewed curve.
        changed.to_csv(reviewed_snapshot, index=False)
    elif failure in {"nonfinite_parity", "negative_parity", "diagnostic_only"}:
        path = reviewed_snapshot.with_suffix(".metadata.json")
        metadata = json.loads(path.read_text())
        if failure == "diagnostic_only":
            metadata["backtest_allowed"] = False
        else:
            metadata["qlab_max_absolute_error"] = float("nan") if failure == "nonfinite_parity" else -1.
        path.write_text(json.dumps(metadata))
    output = tmp_path / "should_not_exist" / "report"

    def forbidden_engine(*args, **kwargs):
        pytest.fail("A rejected snapshot must not enter the backtest engine")

    monkeypatch.setattr(backtest_single_fund, "run_backtest", forbidden_engine)
    with pytest.raises(ValueError, match=message):
        backtest_single_fund.run(reviewed_snapshot, "Fundg", output, curve_confirmed=confirmed)
    assert not output.parent.exists()


def test_confirmed_fundg_run_keeps_pre_inception_year_empty(reviewed_snapshot, tmp_path, monkeypatch):
    output = tmp_path / "results" / "report"
    rendered = []
    monkeypatch.setattr(backtest_single_fund, "render_portfolio_report",
                        lambda *args, **kwargs: rendered.append(args))
    payload = backtest_single_fund.run(reviewed_snapshot, "Fundg", output,
                                     start="2010-01-01", curve_confirmed=True)
    assert len(rendered) == 1
    annual = pd.read_csv(output.with_suffix(".annual.csv")).set_index("year")
    empty = annual.loc[2010]
    assert empty.year_status == "not_incepted"
    assert empty[["observations", "beginning_assets", "end_assets", "deposits", "profit"]].eq(0).all()
    assert empty.xirr_status == "no_observations"
    assert pd.isna(empty.xirr)

    component = pd.read_csv(output.with_suffix(".components_daily.csv.gz"), parse_dates=["date"])
    assert len(component) == 6
    assert component.date.min() == pd.Timestamp("2011-03-30")
    assert component.iloc[:2][["shares", "equity", "total_contributions"]].eq(0).all().all()
    assert component.dividend.sum() == pytest.approx(.1)
    trades = pd.read_csv(output.with_suffix(".trades.csv"), parse_dates=["date", "signal_date"])
    first_fill = trades[trades.status == "FILLED"].iloc[0]
    assert first_fill.signal_date == pd.Timestamp("2011-03-31")
    assert first_fill.date == pd.Timestamp("2011-04-01")
    assert payload["summary"]["first_investment_date"] == "2011-04-01"
    assert payload["summary"]["total_contributions"] > 0
    assert payload["metadata"]["approved_curve_data_sha256"] == hashlib.sha256(reviewed_snapshot.read_bytes()).hexdigest()
    assert payload["metadata"]["curve_confirmed_by_user"] is True
    pasted = output.with_suffix(".paste_with_header_wan.tsv").read_text().splitlines()
    headers, values = [line.split("\t") for line in pasted]
    assert values[headers.index("2010")] == "—"
    assert output.with_suffix(".summary.json").exists()
