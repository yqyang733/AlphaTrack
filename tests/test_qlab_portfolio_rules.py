"""Behavioral checks against the actual pinned QLab calculation modules."""
from pathlib import Path
import os

import numpy as np
import pandas as pd
import pytest

from alphatrack.engine import EngineConfig, run_backtest
from alphatrack.strategies.qlab import QLabRules, QLabSignalStrategy, asset_key


@pytest.fixture
def rules():
    root = Path(__file__).resolve().parents[1]
    candidates = [Path(os.environ["QLAB_SOURCE"])] if "QLAB_SOURCE" in os.environ else [
        root / "vendor/qlab", root.parent / "tmp/alphatrack-qlab-source"]
    source = next((path for path in candidates if (path / "trading/engine.py").exists()), None)
    if source is None:
        pytest.skip("Pinned QLab source snapshot is required")
    return QLabRules(source)


def asset(code="000001", kind="Fundb", category="基金>纯债"):
    return dict(code=code, name=code, asset_type=kind, category=category, trade_mode="场外")


def bond_frame(phase=0):
    dates = pd.bdate_range("2004-01-01", "2011-02-01")
    x = np.arange(len(dates))
    values = 1 + 0.00015 * x + .018 * np.sin(x / (19 + phase) + phase)
    return pd.DataFrame(dict(date=dates, nav=values, dividend=0., split=1.))


def test_catalog_preserves_all_enabled_requested_groups(rules):
    assets = rules.catalog()
    counts = pd.Series([row["asset_type"] for row in assets]).value_counts().to_dict()
    assert counts == {"Fundb": 26, "Fundc": 7, "Fundd": 22, "Fundf_1": 6, "Fundg": 2}
    assert len({row["code"] for row in assets}) == 63
    assert "161725" not in {row["code"] for row in assets}  # Commented catalog entry.
    assert "Fundf" not in counts


def test_bond_ranking_returns_match_original_pointwise_calculator(rules):
    frame = bond_frame()
    actual = rules.prepare_features(frame, "Fundb")
    raw = frame[["date", "nav"]].rename(columns={"date": "净值日期", "nav": "单位净值"})
    reference = rules.calculator.ReturnCalculator.calculate_return_with_dividend_reinvest(raw, None)
    for index in [0, 260, 700, 1200, len(frame)-1]:
        for years in (1, 2, 3, 5):
            value = rules.calculator.ReturnCalculator.calculate_annualized_return(reference.iloc[:index+1], years)
            expected = float(f"{value:.4f}") if np.isfinite(value) else np.nan
            assert actual.loc[index, f"近{years}年"] == pytest.approx(expected, nan_ok=True)
    # The upstream scoring definition divides the whole period return by 5;
    # changing it to geometric CAGR would change the source ranking.
    assert np.isnan(actual.loc[1200, "近5年"])


def test_source_lookback_does_not_require_continuous_nav_history(rules):
    # QLab uses the last observation <= target date, even across a large gap.
    # Keep this source behavior explicit; data quality reporting must flag gaps.
    frame = pd.DataFrame({"date": pd.to_datetime(["2013-06-20", "2020-04-22"]),
                          "nav": [1., 1.7]})
    actual = rules.prepare_features(frame, "Fundb")
    assert actual.iloc[-1]["近1年"] == pytest.approx(.7)
    assert actual.iloc[-1]["近5年"] == pytest.approx(.14)


def test_rank_eligibility_and_tie_break_follow_source(rules):
    assets = [asset("000003"), asset("000002"), asset("000001")]
    latest = {}
    for item, ret, mr in zip(assets, [.1, .2, .2], [1., 0., 0.]):
        latest[asset_key(item)] = {**{f"近{years}年": ret for years in (1, 2, 3, 5)},
                                   "mean_reversion_score": mr}
    ranks = rules.rank_snapshot(assets, latest)
    assert ranks["000001"][0] == 1
    assert ranks["000002"][0] == 2
    assert ranks["000003"][0] == 3
    avg = np.array([.1, .2, .2])
    mr = np.array([1., 0., 0.])
    expected = (avg-avg.mean())/avg.std(ddof=1) - (mr-mr.mean())/mr.std(ddof=1)
    assert ranks["000003"][1] == pytest.approx(expected[0])
    latest[asset_key(assets[0])]["近5年"] = np.nan
    assert rules.rank_snapshot([assets[0]], latest) == {}
    # One eligible fund has undefined cross-sectional sample dispersion.
    assert rules.rank_snapshot([assets[1]], latest) == {}


def test_bond_daily_rank_uses_asof_peers_without_stale_duplicate_orders(rules):
    a, b = asset("000001"), asset("000002")
    first, second = bond_frame(0), bond_frame(3)
    missing_day = first.date.iloc[-2]
    second = second[second.date != missing_day].reset_index(drop=True)
    start = first.date.iloc[-3]
    signals, meta = rules.build_signals({a["code"]: first, b["code"]: second}, [a, b], start=start)
    assert missing_day not in set(signals[asset_key(b)].date)
    left = signals[asset_key(a)]
    right = signals[asset_key(b)]
    snapshot = {asset_key(a): left[left.date == missing_day].iloc[0],
                asset_key(b): right[right.date < missing_day].iloc[-1]}
    expected = rules.rank_snapshot([a, b], snapshot)
    assert left[left.date == missing_day].iloc[0]["rank"] == expected[a["code"]][0]
    assert left[left.date < start].action.eq("HOLD").all()
    assert right[right.date < start].action.eq("HOLD").all()
    assert set(left[left.date >= start].action) == {"BUY"}
    assert set(right[right.date >= start].amount) == {40.}
    assert meta["signal_counts"]["SELL"] == 0


def test_fundd_category_base_no_future_signal_and_next_nav_fill(rules):
    dates = pd.bdate_range("2010-01-01", periods=20)
    nav = np.array([1., 1.1, 1.05, .99, .95, 1.01, 1.03, .94, .96, 1.08,
                    1.15, 1.18, 1.14, 1.08, 1.04, .92, .90, .95, 1.1, 1.2])
    frame = pd.DataFrame(dict(date=dates, nav=nav, dividend=0., split=1.))
    broad = asset("110020", "Fundd", "基金>宽基指数")
    sector = asset("000248", "Fundd", "基金>行业指数")
    full, _ = rules.build_signals({"110020": frame, "000248": frame}, [broad, sector], start=dates[0])
    short, _ = rules.build_signals({"110020": frame.iloc[:12]}, [broad], start=dates[0])
    broad_df, sector_df = full[asset_key(broad)], full[asset_key(sector)]
    pd.testing.assert_frame_equal(broad_df.iloc[:12].reset_index(drop=True), short[asset_key(broad)].reset_index(drop=True))
    lows = broad_df["成立以来百分位"] == 0
    assert broad_df.loc[lows, "amount"].eq(200).all()
    assert sector_df.loc[lows, "amount"].eq(40).all()
    assert broad_df.iloc[0].action == "HOLD"
    daily, trades = run_backtest(broad_df, QLabSignalStrategy(broad_df),
                                EngineConfig(initial_cash=0, withdraw_on_sell=True))
    buys = trades[(trades.action == "BUY") & (trades.status == "FILLED")]
    for trade in buys.itertuples():
        index = dates.get_loc(trade.signal_date)
        assert trade.date == dates[index+1]
        assert trade.price == nav[index+1]
    assert daily.iloc[-1].cumulative_pnl == pytest.approx(
        daily.iloc[-1].equity + daily.iloc[-1].cumulative_withdrawals - daily.iloc[-1].total_contributions)


def test_fundf1_and_fundg_configs_and_no_invented_sells(rules):
    dates = pd.bdate_range("2010-01-01", periods=8)
    frame = pd.DataFrame(dict(date=dates, nav=[1., 2., 1., 1.5, 1.6, 1.4, 1.3, 2.], dividend=0., split=1.))
    assets = [asset("005561", "Fundf_1", "基金>红利低波基金"),
              asset("050025", "Fundg", "基金>美股")]
    signals, _ = rules.build_signals({item["code"]: frame for item in assets}, assets, start=dates[0])
    f, g = [signals[asset_key(item)] for item in assets]
    assert f.iloc[2].amount == 160
    assert f.iloc[3].amount == 40  # 50% belongs to the first multiplier tier.
    assert g[g.action == "BUY"].amount.eq(10).all()
    assert "SELL" not in set(f.action) | set(g.action)


def test_missing_execution_observation_fails_instead_of_misaligning(rules):
    frame = pd.DataFrame({"date": pd.date_range("2010-01-01", periods=3), "nav": [1., .9, 1.1]})
    item = asset("110020", "Fundd", "基金>宽基指数")
    prepared, _ = rules.build_signals({"110020": frame}, [item])
    adapter = QLabSignalStrategy(prepared[asset_key(item)])
    with pytest.raises(ValueError, match="observations"):
        adapter.prepare(frame.iloc[1:])
