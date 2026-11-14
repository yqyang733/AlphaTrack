"""Review gates, calendar boundaries and explicit single-fund rule overrides."""
import hashlib
import json
import shutil

import pandas as pd
import pytest

from alphatrack.data import load_data
from alphatrack.strategies.qlab import QLabRules
from scripts import backtest_single_fund
from scripts.export_window_tables import formatted


@pytest.mark.parametrize('yuan,expected', [(5750, '0.58'), (5749.99, '0.57'), (-5750, '-0.58')])
def test_wan_money_rounding_preserves_decimal_halfway_values(yuan, expected):
    assert formatted(yuan) == expected


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


@pytest.fixture
def reviewed_daily_bond_snapshot(tmp_path):
    """Keep a real Saturday year-end valuation/dividend between weekday orders."""
    data = tmp_path / "003358.csv"
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2011-12-29", "2011-12-30", "2011-12-31",
                                "2012-01-02", "2012-01-03", "2012-01-04"]),
        "nav": [1., 1., .8, .8, .8, .8],
        "dividend": [0., 0., .1, 0., 0., 0.], "split": 1.,
    })
    raw.to_csv(data, index=False)
    rules = QLabRules(backtest_single_fund.ROOT / "vendor/qlab")
    reference = rules.calculator.ReturnCalculator.calculate_return_with_dividend_reinvest(
        raw[["date", "nav"]].rename(columns={"date": "净值日期", "nav": "单位净值"}),
        pd.DataFrame({"分红日": [pd.Timestamp("2011-12-31")],
                      "分红类型": ["现金分红"], "分红": [.1]}))
    normalized, _ = load_data(data)
    error = float((normalized.total_return_index - (reference["累计收益率"] + 1)).abs().max())
    assert error <= 1e-12
    metadata = {"code": "003358", "name": "Synthetic reviewed daily bond",
                "requested_strategy": "Fundb", "first_date": "2011-12-29", "last_date": "2012-01-04",
                "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "qlab_max_absolute_error": error,
                "calculator_sha256": rules.source_hashes["core/calculator.py"]}
    data.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return data


@pytest.mark.parametrize("rule,enabled,confirmed", [
    ("Fundb", False, True), ("Fundg", True, True),
    ("Fundc", True, True), ("Fundb", True, False),
])
def test_daily_bond_override_requires_explicit_matching_authorization(
        reviewed_daily_bond_snapshot, tmp_path, monkeypatch, rule, enabled, confirmed):
    output = tmp_path / "rejected_daily_bond" / "report"

    def forbidden_engine(*args, **kwargs):
        pytest.fail("An unauthorized override must not enter the engine")

    monkeypatch.setattr(backtest_single_fund, "run_backtest", forbidden_engine)
    with pytest.raises(ValueError):
        backtest_single_fund.run(reviewed_daily_bond_snapshot, rule, output,
                                curve_confirmed=confirmed, fundb_daily_buy=enabled)
    assert not output.parent.exists()


@pytest.mark.parametrize("start", ["2010-01-01", "2012-01-02"])
def test_daily_bond_override_keeps_valuation_and_starts_without_ranking_or_replay(
        reviewed_daily_bond_snapshot, tmp_path, monkeypatch, start):
    output = tmp_path / "daily_bond" / "report"
    before = reviewed_daily_bond_snapshot.read_bytes()

    def forbidden_ranking(*args, **kwargs):
        pytest.fail("This explicit single-fund override must not rank peers")

    monkeypatch.setattr(QLabRules, "rank_snapshot", forbidden_ranking)
    monkeypatch.setattr(backtest_single_fund, "render_portfolio_report", lambda *args, **kwargs: None)
    payload = backtest_single_fund.run(reviewed_daily_bond_snapshot, "Fundb", output,
                                     start=start, curve_confirmed=True, fundb_daily_buy=True)
    assert reviewed_daily_bond_snapshot.read_bytes() == before
    override = payload["metadata"]["user_rule_override"]
    assert override["amount"] == 40
    assert override["ranking"] is False
    assert override["calendar"] == "weekday_nav"
    assert "2011-12-31" in payload["metadata"]["signal_metadata"]["nontrading_nav_dates"]

    component = pd.read_csv(output.with_suffix(".components_daily.csv.gz"), parse_dates=["date"])
    signals = pd.read_csv(output.with_suffix(".signals.csv.gz"), parse_dates=["date"])
    trades = pd.read_csv(output.with_suffix(".trades.csv"), parse_dates=["date", "signal_date"])
    filled = trades[trades.status == "FILLED"]
    assert filled.action.eq("BUY").all()
    assert filled.amount.eq(40).all()
    assert filled.date.dt.weekday.lt(5).all()
    assert filled.signal_date.dt.weekday.lt(5).all()
    assert signals.loc[signals.date.dt.weekday.lt(5), "action"].eq("BUY").all()
    assert signals.loc[signals.date.dt.weekday.lt(5), "amount"].eq(40).all()
    assert "SELL" not in set(signals.action)
    assert component.iloc[0][["shares", "equity", "total_contributions"]].eq(0).all()

    if start == "2010-01-01":
        assert component.date.tolist() == list(pd.to_datetime([
            "2011-12-29", "2011-12-30", "2011-12-31", "2012-01-02", "2012-01-03", "2012-01-04"]))
        assert signals.iloc[0].action == "BUY"  # No one-year or five-year warm-up requirement.
        assert payload["summary"]["first_investment_date"] == "2011-12-30"
        weekend = component.loc[component.date == pd.Timestamp("2011-12-31")].iloc[0]
        assert weekend.external_flow == 0
        assert weekend.dividend == pytest.approx(.1)
        assert weekend.shares == pytest.approx(45)  # Only Friday's 40 shares receive the dividend.
        assert weekend.equity == pytest.approx(36)
        assert weekend.daily_pnl == pytest.approx(-4)
        assert signals.loc[signals.date == pd.Timestamp("2011-12-31"), "action"].item() == "HOLD"
        monday = filled.loc[filled.date == pd.Timestamp("2012-01-02")].iloc[0]
        assert monday.signal_date == pd.Timestamp("2011-12-30")  # Pending order survives the weekend.
        annual = pd.read_csv(output.with_suffix(".annual.csv")).set_index("year")
        assert annual.loc[2011, "end_assets"] == pytest.approx(36)
        assert annual.loc[2011, "profit"] == pytest.approx(-4)
        assert payload["summary"]["buy_count"] == 4
        assert payload["summary"]["total_contributions"] == 160
    else:
        assert component.date.min() == pd.Timestamp("2012-01-02")
        assert signals.date.min() == pd.Timestamp("2012-01-02")
        assert filled.iloc[0].signal_date == pd.Timestamp("2012-01-02")
        assert filled.iloc[0].date == pd.Timestamp("2012-01-03")
        assert payload["summary"]["first_investment_date"] == "2012-01-03"
        assert payload["summary"]["buy_count"] == 2
        assert payload["summary"]["total_contributions"] == 80


def test_daily_fundg_amount_buys_high_positions_without_changing_default(
        reviewed_snapshot, tmp_path, monkeypatch):
    raw = pd.read_csv(reviewed_snapshot, parse_dates=["date"])
    raw["nav"] = [1., 1., 1.1, 1.2, 1.3, 1.4]
    raw["dividend"] = 0.
    raw.to_csv(reviewed_snapshot, index=False)
    reference = QLabRules(backtest_single_fund.ROOT / "vendor/qlab").prepare_features(raw, "Fundg")
    normalized, _ = load_data(reviewed_snapshot)
    path = reviewed_snapshot.with_suffix(".metadata.json")
    metadata = json.loads(path.read_text())
    metadata["data_sha256"] = hashlib.sha256(reviewed_snapshot.read_bytes()).hexdigest()
    metadata["qlab_max_absolute_error"] = float((normalized.total_return_index - reference.total_return_index).abs().max())
    path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(backtest_single_fund, "render_portfolio_report", lambda *args, **kwargs: None)

    default_prefix = tmp_path / "default" / "report"
    default = backtest_single_fund.run(reviewed_snapshot, "Fundg", default_prefix, curve_confirmed=True)
    default_signals = pd.read_csv(default_prefix.with_suffix(".signals.csv.gz"), parse_dates=["date"])
    assert reference["近1年百分位"].iloc[2:].gt(.6).all()
    assert default_signals.action.tolist() == ["HOLD", "BUY", "HOLD", "HOLD", "HOLD", "HOLD"]
    assert default["summary"]["buy_count"] == 1
    assert "user_rule_override" not in default["metadata"]
    assert "60%" in "\n".join(default["metadata"]["assumptions"])

    def forbidden_original_signals(*args, **kwargs):
        pytest.fail("Explicit daily purchases must not consult original Fundg signals")

    monkeypatch.setattr(QLabRules, "build_signals", forbidden_original_signals)
    daily_prefix = tmp_path / "daily" / "report"
    daily = backtest_single_fund.run(reviewed_snapshot, "Fundg", daily_prefix,
                                   curve_confirmed=True, daily_buy_amount=10)
    daily_signals = pd.read_csv(daily_prefix.with_suffix(".signals.csv.gz"))
    trades = pd.read_csv(daily_prefix.with_suffix(".trades.csv"))
    assert daily_signals.action.eq("BUY").all()
    assert daily_signals.amount.eq(10).all()
    assert trades.loc[trades.status == "FILLED", "amount"].eq(10).all()
    assert daily["summary"]["buy_count"] == 5
    assert daily["summary"]["total_contributions"] == 50
    override = daily["metadata"]["user_rule_override"]
    assert override["mode"] == "daily_fixed" and override["amount"] == 10
    assert override["threshold_applied"] is False
    assert "60%" not in "\n".join(daily["metadata"]["assumptions"])
    assert daily["config"]["rule_configuration"]["mode"] == "daily_fixed"


@pytest.mark.parametrize("rule,amount,legacy_flag", [
    ("Fundg", 0, False), ("Fundg", -1, False),
    ("Fundg", True, False), ("Fundg", False, False),
    ("Fundg", float("nan"), False), ("Fundg", float("inf"), False),
    ("Fundg", float("-inf"), False),
    ("Fundd", 10, False), ("Fundf_1", 10, False), ("Fundc", 0, False),
    ("Fundb", 40, True),
])
def test_daily_amount_rejects_invalid_amount_rules_and_conflicting_flags(
        reviewed_snapshot, tmp_path, monkeypatch, rule, amount, legacy_flag):
    path = reviewed_snapshot.with_suffix(".metadata.json")
    metadata = json.loads(path.read_text())
    metadata["requested_strategy"] = rule
    path.write_text(json.dumps(metadata), encoding="utf-8")
    output = tmp_path / "rejected_daily_amount" / "report"

    def forbidden_engine(*args, **kwargs):
        pytest.fail("Invalid daily purchase settings must be rejected before execution")

    monkeypatch.setattr(backtest_single_fund, "run_backtest", forbidden_engine)
    with pytest.raises(ValueError):
        backtest_single_fund.run(reviewed_snapshot, rule, output, curve_confirmed=True,
                                category="基金>宽基指数", fundb_daily_buy=legacy_flag,
                                daily_buy_amount=amount)
    assert not output.parent.exists()


def test_explicit_batch_authorization_preserves_hash_gate_without_claiming_curve_review(
        reviewed_snapshot, tmp_path, monkeypatch):
    monkeypatch.setattr(backtest_single_fund, "render_portfolio_report", lambda *args, **kwargs: None)
    payload = backtest_single_fund.run(reviewed_snapshot, "Fundg", tmp_path / "batch" / "report",
                                     batch_authorized=True, daily_buy_amount=10)
    assert payload["metadata"]["curve_confirmed_by_user"] is False
    assert payload["metadata"]["batch_authorized_by_user"] is True
    assert "曲线经用户确认" not in "\n".join(payload["metadata"]["assumptions"])
    reviewed_snapshot.write_text(reviewed_snapshot.read_text().replace("0.85", "0.86"))
    with pytest.raises(ValueError, match="previously reviewed"):
        backtest_single_fund.run(reviewed_snapshot, "Fundg", tmp_path / "tampered" / "report",
                                batch_authorized=True, daily_buy_amount=10)


def test_fundc_requires_override_and_daily40_never_calls_ranking(
        reviewed_daily_bond_snapshot, tmp_path, monkeypatch):
    path = reviewed_daily_bond_snapshot.with_suffix(".metadata.json")
    metadata = json.loads(path.read_text())
    metadata["requested_strategy"] = "Fundc"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="no ranking peer universe"):
        backtest_single_fund.run(reviewed_daily_bond_snapshot, "Fundc", tmp_path / "denied",
                                batch_authorized=True)

    def forbidden_signals(*args, **kwargs):
        pytest.fail("Single-fund daily40 must not use Fundc ranking signals")

    monkeypatch.setattr(QLabRules, "build_signals", forbidden_signals)
    monkeypatch.setattr(backtest_single_fund, "render_portfolio_report", lambda *args, **kwargs: None)
    prefix = tmp_path / "fundc_daily40"
    result = backtest_single_fund.run(reviewed_daily_bond_snapshot, "Fundc", prefix,
                                     batch_authorized=True, daily_buy_amount=40)
    trades = pd.read_csv(prefix.with_suffix(".trades.csv"), parse_dates=["date"])
    filled = trades[trades.status == "FILLED"]
    assert len(filled) == 4 and filled.amount.eq(40).all()
    assert filled.date.dt.dayofweek.lt(5).all()
    assert result["summary"]["total_contributions"] == 160
    assert result["summary"]["sell_count"] == 0
    assert result["metadata"]["types"] == ["Fundc"]
    assert result["metadata"]["user_rule_override"]["ranking"] is False


def test_explicit_qlab_source_records_its_commit_and_rejects_changed_rules(
        reviewed_snapshot, tmp_path, monkeypatch):
    source = tmp_path / "latest_qlab"
    shutil.copytree(backtest_single_fund.ROOT / "vendor/qlab", source)
    rules = QLabRules(source)
    path = reviewed_snapshot.with_suffix(".metadata.json")
    metadata = json.loads(path.read_text())
    metadata.update(qlab_rule_hashes=rules.source_hashes, qlab_catalog_commit="f" * 40)
    path.write_text(json.dumps(metadata))
    monkeypatch.setattr(backtest_single_fund, "render_portfolio_report", lambda *args, **kwargs: None)
    result = backtest_single_fund.run(reviewed_snapshot, "Fundg", tmp_path / "latest_result",
                                     batch_authorized=True, daily_buy_amount=10, qlab_source=source)
    assert result["metadata"]["qlab_commit"] == "f" * 40
    assert result["metadata"]["source_commit"] == "f" * 40
    assert result["metadata"]["qlab_source_path"] == str(source)
    changed = source / "configs/trading/Fundf_1.json"
    changed.write_text(changed.read_text().replace('"fundf1_base_amount": 40', '"fundf1_base_amount": 80'))
    with pytest.raises(ValueError, match="Pinned QLab rules changed"):
        backtest_single_fund.run(reviewed_snapshot, "Fundg", tmp_path / "changed_result",
                                batch_authorized=True, daily_buy_amount=10, qlab_source=source)
    assert not (tmp_path / "changed_result.summary.json").exists()


def test_explicit_qlab_source_requires_hash_provenance(reviewed_snapshot, tmp_path, monkeypatch):
    source = tmp_path / "unpinned_qlab"
    shutil.copytree(backtest_single_fund.ROOT / "vendor/qlab", source)
    with pytest.raises(ValueError, match="requires pinned rule hashes"):
        backtest_single_fund.run(reviewed_snapshot, "Fundg", tmp_path / "denied",
                                batch_authorized=True, daily_buy_amount=10, qlab_source=source)
