"""Independent cash-flow, calendar-boundary and XIRR examples."""

import math

import numpy as np
import pandas as pd
import pytest

from alphatrack.portfolio import aggregate_portfolio, analyze_portfolio, cashflow_xirr


def ledger(dates, equity, deposits=None, withdrawals=None, fees=None, realized=None, unrealized=None):
    n = len(dates)
    dep = np.array(deposits if deposits is not None else [0.0]*n, dtype=float)
    out = np.array(withdrawals if withdrawals is not None else [0.0]*n, dtype=float)
    eq = np.array(equity, dtype=float)
    actual_pnl = eq - np.r_[0.0, eq[:-1]] - dep + out
    real = np.array(realized if realized is not None else [0.0]*n, dtype=float)
    unreal = np.array(unrealized if unrealized is not None else np.cumsum(actual_pnl)-real, dtype=float)
    return pd.DataFrame(dict(date=pd.to_datetime(dates), equity=eq, cash=0.0,
                             position_value=eq, holding_cost=eq-unreal, external_flow=dep,
                             withdrawal=out, daily_pnl=actual_pnl, realized_pnl=real,
                             unrealized_pnl=unreal, fees=fees if fees is not None else [0.0]*n))


def test_staggered_assets_forward_fill_values_but_not_flows_or_pnl():
    a = ledger(["2020-01-01", "2020-01-03"], [100, 120], [100, 0])
    b = ledger(["2020-01-02", "2020-01-03"], [300, 270], [300, 0])
    combined = aggregate_portfolio({"a": a, "b": b})
    assert combined.equity.tolist() == [100, 400, 390]
    assert combined.external_flow.tolist() == [100, 300, 0]
    assert combined.daily_pnl.tolist() == [0, 0, -10]
    assert combined.observed_funds.tolist() == [1, 1, 2]
    assert combined.unit_nav.iloc[-1] == pytest.approx(.975)
    # An equal mean of +20% and -10% would incorrectly report +5%.
    assert combined.daily_return.iloc[-1] == pytest.approx(-.025)


def test_fees_on_new_capital_are_not_all_divided_by_old_small_account():
    a = ledger(["2020-01-01", "2020-01-02"], [100, 110], [100, 0])
    b = ledger(["2020-01-02"], [990], [1000], fees=[10])
    combined = aggregate_portfolio({"a": a, "b": b})
    assert combined.daily_pnl.iloc[-1] == pytest.approx(0)
    assert combined.day_fees.iloc[-1] == pytest.approx(10)
    assert combined.unit_nav.iloc[-1] == pytest.approx(1.1*1100/1110)


def test_withdrawal_and_later_new_deposit_do_not_change_nav():
    a = ledger(["2020-01-01", "2020-01-02", "2020-01-03"],
               [100, 50, 100], [100, 0, 50], [0, 60, 0], realized=[0, 10, 10])
    combined = aggregate_portfolio({"a": a})
    assert combined.unit_nav.tolist() == pytest.approx([1, 1.1, 1.1])
    assert combined.cumulative_pnl.iloc[-1] == pytest.approx(10)
    assert combined.equity.iloc[-1]+combined.withdrawal.sum()-combined.external_flow.sum() == pytest.approx(10)


def test_calendar_year_profit_includes_unrealized_change_and_excludes_flows():
    dates = ["2019-12-31", "2020-06-30", "2020-12-31", "2021-01-02"]
    a = ledger(dates, [100, 120, 130, 90], [100, 0, 0, 0], [0, 0, 10, 50],
               realized=[0, 0, 10, 20], unrealized=[0, 20, 30, 30])
    combined = aggregate_portfolio({"a": a})
    annual, summary = analyze_portfolio(combined, start="2019-12-31", end="2021-01-02")
    row = annual.set_index("year").loc[2020]
    assert row.beginning_assets == pytest.approx(100)
    assert row.end_assets == pytest.approx(130)
    assert row.withdrawals == pytest.approx(10)
    assert row.profit == pytest.approx(40)
    assert row.realized_pnl == pytest.approx(10)
    assert row.unrealized_pnl_change == pytest.approx(30)
    assert row.twr_return == pytest.approx(.4)
    assert row.xirr == pytest.approx(1.4**(365/366)-1)
    assert not row.is_partial_year
    assert annual.set_index("year").loc[2021].is_partial_year
    assert summary["total_pnl"] == pytest.approx(50)
    assert annual.profit.sum() == pytest.approx(summary["total_pnl"])


def test_annual_drawdown_includes_drop_on_first_day_of_year():
    a = ledger(["2020-12-31", "2021-01-04", "2021-01-05"], [100, 80, 90], [100, 0, 0])
    annual, summary = analyze_portfolio(aggregate_portfolio({"a": a}), start="2020-12-31", end="2021-01-05")
    assert annual.set_index("year").loc[2021].max_drawdown == pytest.approx(-.2)
    assert summary["max_drawdown"] == pytest.approx(-.2)
    assert summary["max_drawdown_days"] == 5


def test_trade_counts_exclude_rejections_and_pending():
    a = ledger(["2020-01-01", "2020-01-02"], [100, 110], [100, 0])
    trades = pd.DataFrame(dict(date=["2020-01-01"]*4,
                               action=["BUY", "SELL", "BUY", "SELL"],
                               amount=[100, 50, 0, 0], fee=[1, 2, 0, 0],
                               status=["FILLED", "PARTIAL", "REJECTED_NO_CASH", "PENDING_END_OF_DATA"]))
    annual, summary = analyze_portfolio(aggregate_portfolio({"a": a}), trades)
    assert summary["buy_count"] == summary["sell_count"] == 1
    assert summary["buy_amount"] == 100
    assert summary["sell_proceeds"] == 48
    assert annual.buy_count.sum() == 1


def test_xirr_one_root_known_with_intermediate_withdrawal():
    dates = pd.to_datetime(["2021-01-01", "2021-07-01", "2022-01-01"])
    rate = .12
    payout = 30
    end = 100*(1+rate)-payout*(1+rate)**((dates[-1]-dates[1]).days/365)
    result = cashflow_xirr(dates, [-100, payout, end])
    assert result["status"] == "unique"
    assert result["rate"] == pytest.approx(rate, abs=1e-10)


def test_xirr_rejects_choosing_one_of_multiple_roots():
    result = cashflow_xirr(["2021-01-01", "2022-01-01", "2023-01-01"], [-100, 230, -132])
    assert result["status"] == "multiple_roots"
    assert result["rate"] is None
    assert result["roots"] == pytest.approx([.1, .2], abs=1e-9)


def test_xirr_detects_tangent_root_without_sign_crossing():
    result = cashflow_xirr(["2021-01-01", "2022-01-01", "2023-01-01"], [-100, 220, -121])
    assert result["status"] == "unique_detected"
    assert result["rate"] == pytest.approx(.1, abs=1e-7)


def test_xirr_groups_same_day_and_handles_undefined_cashflows():
    result = cashflow_xirr(["2021-01-01", "2021-01-01", "2022-01-01"], [-150, 50, 110])
    assert result["rate"] == pytest.approx(.1, abs=1e-10)
    assert cashflow_xirr(["2021-01-01", "2021-01-01"], [-100, 110])["rate"] is None
    assert cashflow_xirr(["2021-01-01", "2022-01-01"], [-100, -50])["rate"] is None


def test_component_pnl_inconsistency_is_rejected():
    a = ledger(["2020-01-01", "2020-01-02"], [100, 110], [100, 0])
    a.loc[1, "daily_pnl"] = 1000
    with pytest.raises(ValueError, match="component"):
        aggregate_portfolio({"a": a})


def test_empty_investment_has_no_defined_annual_return_or_xirr():
    a = ledger(["2020-01-01", "2021-01-01"], [0, 0])
    annual, summary = analyze_portfolio(aggregate_portfolio({"a": a}))
    assert summary["total_pnl"] == 0
    assert summary["cagr"] is None
    assert summary["xirr"] is None
    assert summary["sharpe"] is None
    assert annual.profit.sum() == 0


def test_requested_years_before_fund_inception_are_reported_without_fake_returns():
    a = ledger(["2022-01-04", "2022-12-30"], [100, 110], [100, 0])
    annual, summary = analyze_portfolio(aggregate_portfolio({"a": a}), start="2020-01-01")
    assert annual.year.tolist() == [2020, 2021, 2022]
    assert annual.profit.tolist() == [0, 0, 10]
    assert annual.observations.tolist() == [0, 0, 2]
    assert pd.isna(annual.iloc[0].twr_return)
    assert summary["first_investment_date"] == "2022-01-04"
