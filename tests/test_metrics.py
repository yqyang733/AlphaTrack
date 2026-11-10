"""Known cash-flow examples and independent portfolio-metric identities."""
import math
import unittest

import numpy as np
import pandas as pd

from alphatrack.engine import EngineConfig, run_backtest
from alphatrack.metrics import calculate_metrics, xirr
from test_engine import ScriptedStrategy, market, buy, sell


class MetricsTests(unittest.TestCase):
    def test_xirr_one_contribution(self):
        dates = pd.to_datetime(["2020-01-01", "2021-01-01"])
        expected = 1.1 ** (365.25/366) - 1
        self.assertAlmostEqual(xirr(dates, [-100, 110]), expected, places=12)

    def test_xirr_multiple_contributions_with_known_discount_rate(self):
        dates = pd.to_datetime(["2020-01-01", "2020-06-01", "2022-01-01"])
        rate = .15
        terminal = 100 * (1+rate)**((dates[-1]-dates[0]).days/365.25)
        terminal += 200 * (1+rate)**((dates[-1]-dates[1]).days/365.25)
        self.assertAlmostEqual(xirr(dates, [-100, -200, terminal]), rate, places=12)

    def test_xirr_loss_and_undefined_cases(self):
        dates = pd.to_datetime(["2020-01-01", "2021-01-01"])
        self.assertAlmostEqual(xirr(dates, [-100, 50]), .5**(365.25/366)-1, places=12)
        self.assertIsNone(xirr(dates, [0, 100]))
        self.assertIsNone(xirr(dates, [-100, 0]))
        self.assertIsNone(xirr([dates[0], dates[0]], [-100, 110]))

    def test_cagr_compounds_and_starts_at_first_funding(self):
        dates = ["2018-01-01", "2020-01-01", "2022-01-01"]
        cfg = EngineConfig()
        daily, trades = run_backtest(market([1., 1., 1.44], dates=dates), ScriptedStrategy([buy(100)]), cfg)
        result = calculate_metrics(daily, trades, cfg)
        years = 731/365.25
        expected = 1.44**(1/years)-1
        self.assertAlmostEqual(result["cagr"], expected)
        self.assertNotAlmostEqual(result["cagr"], .44/years)
        self.assertEqual(result["first_investment_date"], "2020-01-01")
        self.assertAlmostEqual(result["xirr"], expected)
        self.assertAlmostEqual(result["total_pnl"], 44)

    def test_sharpe_volatility_and_drawdown_from_known_path(self):
        cfg = EngineConfig(initial_cash=100, funding_mode="fixed", risk_free_rate=.03)
        daily, trades = run_backtest(market([1., 1., 1.2, .9, 1.08]), ScriptedStrategy([buy(100)]), cfg)
        result = calculate_metrics(daily, trades, cfg)
        returns = np.array([0, 0, .2, -.25, .2])
        rf_daily = 1.03**(1/252)-1
        std = returns.std(ddof=1)
        self.assertAlmostEqual(result["annual_volatility"], std*math.sqrt(252))
        self.assertAlmostEqual(result["sharpe"], (returns.mean()-rf_daily)/std*math.sqrt(252))
        self.assertAlmostEqual(result["max_drawdown"], -.25)
        self.assertEqual(result["max_drawdown_days"], 2)
        self.assertEqual(result["buy_count"], 1)
        self.assertEqual(result["sell_count"], 0)

    def test_no_funding_returns_are_undefined_not_infinite(self):
        cfg = EngineConfig()
        daily, trades = run_backtest(market([1., 2., .5]), ScriptedStrategy([]), cfg)
        result = calculate_metrics(daily, trades, cfg)
        for key in ("cagr", "xirr", "sharpe", "sortino", "holding_return", "first_investment_date"):
            self.assertIsNone(result[key], key)
        self.assertEqual(result["total_pnl"], 0)
        self.assertEqual(result["net_contributions"], 0)
        self.assertEqual(result["annual_volatility"], 0)

    def test_recovered_drawdown_duration_includes_recovery_date(self):
        cfg = EngineConfig(initial_cash=100, funding_mode="fixed")
        dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-10"]
        daily, trades = run_backtest(market([1., 1., 1.2, .9, 1.2], dates=dates), ScriptedStrategy([buy(100)]), cfg)
        self.assertEqual(calculate_metrics(daily, trades, cfg)["max_drawdown_days"], 7)

    def test_pending_and_rejected_are_not_fills(self):
        cfg = EngineConfig()
        daily, trades = run_backtest(market([1., 1., 1.]), ScriptedStrategy([sell(1), buy(100), buy(100)]), cfg)
        result = calculate_metrics(daily, trades, cfg)
        self.assertEqual(result["buy_count"], 1)
        self.assertEqual(result["sell_count"], 0)
        self.assertEqual(result["rejected_count"], 1)
        self.assertEqual(result["pending_count"], 1)
        self.assertEqual(result["total_buys"], 100)
        self.assertIsNone(result["cagr"])


if __name__ == "__main__":
    unittest.main()
