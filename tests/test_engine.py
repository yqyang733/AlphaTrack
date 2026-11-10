"""Economic invariants of the event loop, using deterministic order sequences."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from alphatrack.engine import EngineConfig, run_backtest
from alphatrack.data import load_data
from alphatrack.strategies.fundd import FunddStrategy


class ScriptedStrategy:
    def __init__(self, actions):
        self.actions = actions
        self.index = 0

    def prepare(self, frame):
        return frame.copy()

    def decide(self, row):
        result = self.actions[self.index] if self.index < len(self.actions) else {"action": "HOLD"}
        self.index += 1
        return {"reason": "test fixture", **result}


def market(prices, *, dates=None, dividends=None, splits=None):
    n = len(prices)
    frame = pd.DataFrame({"date": pd.to_datetime(dates) if dates is not None else pd.date_range("2020-01-01", periods=n),
                          "nav": prices, "dividend": dividends or [0.] * n, "split": splits or [1.] * n})
    factor = frame.split * (frame.nav + frame.dividend) / frame.nav.shift(1)
    factor.iloc[0] = 1
    frame["total_return_index"] = factor.cumprod()
    return frame


def buy(amount):
    return {"action": "BUY", "amount": amount}


def sell(fraction):
    return {"action": "SELL", "fraction": fraction}


class EngineTests(unittest.TestCase):
    def test_repeated_price_low_remains_exact_after_data_normalization(self):
        # Actual 2010-06 price sequence: multiplying daily NAV ratios formerly
        # changed a repeated 0.820 low into 0.8200000000000001 and BUY200 to BUY150.
        prices = [1., .82, .821, .845, .835, .838, .833, .82]
        with TemporaryDirectory() as directory:
            path = Path(directory)/"nav.csv"
            pd.DataFrame({"date": pd.date_range("2010-06-01", periods=len(prices)), "nav": prices}).to_csv(path, index=False)
            frame, _ = load_data(path)
        self.assertEqual(frame.total_return_index.iloc[1], frame.total_return_index.iloc[-1])
        strategy = FunddStrategy()
        prepared = strategy.prepare(frame)
        self.assertEqual(prepared.signal_metric.iloc[-1], 0)
        self.assertEqual(strategy.decide(prepared.iloc[-1])["amount"], 200)

    def test_next_nav_execution_and_final_pending_order(self):
        prices = market([1., 2., 3.])
        daily, trades = run_backtest(prices, ScriptedStrategy([buy(100), {"action": "HOLD"}, sell(1)]))
        self.assertEqual(daily.shares.iloc[0], 0)
        self.assertEqual(daily.shares.iloc[1], 50)
        self.assertEqual(trades.iloc[0].signal_date, prices.date.iloc[0])
        self.assertEqual(trades.iloc[0].date, prices.date.iloc[1])
        self.assertEqual(trades.iloc[-1].status, "PENDING_END_OF_DATA")
        self.assertTrue(pd.isna(trades.iloc[-1].date))
        self.assertEqual(daily.equity.iloc[-1], 150)

    def test_topups_recycle_cash_and_reconcile_cost_pnl(self):
        daily, trades = run_backtest(market([1., 1., 1.2, 1.2, 1.]), ScriptedStrategy([buy(100), sell(.5), buy(100)]))
        np.testing.assert_allclose(daily.external_flow, [0, 100, 0, 40, 0], atol=1e-12)
        np.testing.assert_allclose(daily.daily_pnl, [0, 0, 20, 0, -160/6], atol=1e-12)
        np.testing.assert_allclose(daily.daily_return, [0, 0, .2, 0, -1/6], atol=1e-12)
        self.assertAlmostEqual(daily.net_contributions.iloc[-1], 140)
        self.assertAlmostEqual(daily.invested.iloc[-1], 200)
        self.assertAlmostEqual(daily.realized_pnl.iloc[-1], 10)
        np.testing.assert_allclose(daily.equity - daily.net_contributions,
                                   daily.realized_pnl + daily.unrealized_pnl, atol=1e-12)
        self.assertTrue((daily.cash >= 0).all())

    def test_buy_and_sell_fees_reduce_investor_wealth(self):
        config = EngineConfig(buy_fee=.01, sell_fee=.02)
        daily, trades = run_backtest(market([1., 1., 2.]), ScriptedStrategy([buy(100), sell(1)]), config)
        buy_fee = 100 - 100/1.01
        gross_sell = 2*100/1.01
        sell_fee = gross_sell*.02
        self.assertAlmostEqual(daily.daily_return.iloc[1], -buy_fee/100)
        self.assertAlmostEqual(daily.equity.iloc[-1], gross_sell - sell_fee)
        self.assertAlmostEqual(daily.fees.iloc[-1], buy_fee + sell_fee)
        self.assertAlmostEqual(daily.realized_pnl.iloc[-1], gross_sell - sell_fee - 100)
        self.assertAlmostEqual(daily.unit_nav.iloc[-1], (gross_sell-sell_fee)/100)
        self.assertAlmostEqual(daily.holding_cost.iloc[-1], 0)
        self.assertAlmostEqual(daily.unrealized_pnl.iloc[-1], 0)

    def test_large_topup_fee_does_not_charge_old_units_for_new_capital(self):
        cfg = EngineConfig(buy_fee=.1)
        daily, _ = run_backtest(market([1., 1., 1.]), ScriptedStrategy([buy(1), buy(1000)]), cfg)
        first_equity = 1/1.1
        final_equity = first_equity + 1000/1.1
        # New money arrives at close; market return is 0, then all units bear
        # the execution loss proportionally to the now-funded portfolio.
        expected_return = final_equity/(first_equity+1000) - 1
        self.assertAlmostEqual(daily.daily_return.iloc[-1], expected_return)
        self.assertAlmostEqual(daily.unit_nav.iloc[-1], (1/1.1)*(1+expected_return))
        self.assertTrue((daily.unit_nav > 0).all())
        self.assertTrue((daily.daily_return > -1).all())
        self.assertAlmostEqual(daily.daily_pnl.iloc[-1], -1000 + 1000/1.1)

    def test_split_and_reinvested_dividend_conserve_wealth(self):
        prices = market([2., 2., .9], dividends=[0, 0, .1], splits=[1, 1, 2])
        daily, _ = run_backtest(prices, ScriptedStrategy([buy(200)]))
        self.assertAlmostEqual(daily.shares.iloc[-1], 200 + 20/.9)
        self.assertAlmostEqual(daily.equity.iloc[-1], 200)
        self.assertAlmostEqual(daily.daily_pnl.iloc[-1], 0)
        self.assertAlmostEqual(daily.holding_cost.iloc[-1], 200)
        self.assertAlmostEqual(daily.benchmark_return.iloc[-1], 0)

    def test_new_purchase_does_not_receive_prior_dividend(self):
        daily, _ = run_backtest(market([1., .9], dividends=[0, .1]), ScriptedStrategy([buy(90)]))
        self.assertEqual(daily.shares.iloc[-1], 100)
        self.assertEqual(daily.equity.iloc[-1], 90)

    def test_fixed_budget_partial_fill_then_no_cash_and_no_holdings(self):
        cfg = EngineConfig(initial_cash=60, funding_mode="fixed")
        daily, trades = run_backtest(market([1.] * 6), ScriptedStrategy([buy(100), buy(100), sell(1), sell(1)]), cfg)
        self.assertEqual(list(trades.status), ["PARTIAL", "REJECTED_NO_CASH", "FILLED", "REJECTED_NO_HOLDING"])
        self.assertEqual(trades.amount.iloc[0], 60)
        self.assertEqual(daily.external_flow.sum(), 60)
        np.testing.assert_allclose(daily.equity, 60)
        self.assertTrue((daily.cash >= 0).all())

    def test_no_trades(self):
        daily, trades = run_backtest(market([1., 2., .5]), ScriptedStrategy([]))
        self.assertTrue(trades.empty)
        np.testing.assert_allclose(daily.equity, 0)
        np.testing.assert_allclose(daily.daily_pnl, 0)
        np.testing.assert_allclose(daily.unit_nav, 1)

    def test_future_prices_do_not_change_executed_prefix(self):
        actions = [buy(100), sell(.5), buy(200), sell(1)]
        full, full_trades = run_backtest(market([1., 2., 3., 100., .001]), ScriptedStrategy(actions))
        prefix, prefix_trades = run_backtest(market([1., 2., 3.]), ScriptedStrategy(actions))
        pd.testing.assert_frame_equal(full.iloc[:3], prefix)
        pd.testing.assert_frame_equal(full_trades.iloc[:2].reset_index(drop=True),
                                      prefix_trades.iloc[:2].reset_index(drop=True))

    def test_date_range_excludes_orders_before_start(self):
        data = market([1., 1., 2., 3., 4.])
        daily, trades = run_backtest(data, ScriptedStrategy([buy(90)]), EngineConfig(start="2020-01-03", end="2020-01-04"))
        self.assertEqual(len(daily), 2)
        self.assertEqual(daily.shares.iloc[0], 0)
        self.assertEqual(trades.date.iloc[0], pd.Timestamp("2020-01-04"))
        self.assertEqual(daily.equity.iloc[-1], 90)

    def test_invalid_execution_config_and_orders(self):
        for params in ({"initial_cash": -1}, {"funding_mode": "fixed"}, {"funding_mode": "magic"},
                       {"buy_fee": 1}, {"sell_fee": -1}, {"risk_free_rate": -1},
                       {"periods_per_year": True}, {"start": "2020-02-01", "end": "2020-01-01"}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                EngineConfig(**params)
        for action in (buy(0), buy(float("inf")), sell(0), sell(1.1)):
            with self.subTest(action=action), self.assertRaises(ValueError):
                run_backtest(market([1., 1.]), ScriptedStrategy([action]))


if __name__ == "__main__":
    unittest.main()
