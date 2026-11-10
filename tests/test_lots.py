"""Independent known-answer FIFO examples and whole-history reconciliation."""
import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from alphatrack.data import load_data
from alphatrack.engine import run_backtest
from alphatrack.lots import analyze_lots
from alphatrack.strategies.fundd import FunddStrategy


def history(dates, navs, dividends=None, splits=None):
    return pd.DataFrame(dict(date=pd.to_datetime(dates), nav=navs,
                             dividend=dividends or [0.0]*len(dates), split=splits or [1.0]*len(dates)))


def fills(rows):
    return pd.DataFrame([dict(date=pd.Timestamp(date), action=action, shares=shares,
                              amount=amount, fee=fee, status=status)
                         for date, action, shares, amount, fee, status in rows])


class LotTests(unittest.TestCase):
    def test_single_purchase_sale_and_365_day_annualization(self):
        daily = history(["2021-01-01", "2022-01-01", "2024-01-01"], [10, 11, 50])
        trades = fills([("2021-01-01", "BUY", 100, 1000, 0, "FILLED"),
                        ("2022-01-01", "SELL", 100, 1100, 0, "FILLED")])
        lots, summary = analyze_lots(daily, trades)
        self.assertEqual(len(lots), 1)
        self.assertEqual(lots.iloc[0].holding_days, 365)
        self.assertAlmostEqual(lots.iloc[0].annualized_return, .1)
        self.assertAlmostEqual(summary["realized_annualized_median"], .1)
        self.assertEqual(summary["total_pnl"], 100)
        self.assertEqual(summary["open_count"], 0)
        self.assertEqual(lots.iloc[0].valuation_date, pd.Timestamp("2022-01-01"))

    def test_partial_fifo_sale_crosses_lots_and_leaves_open_fragment(self):
        daily = history(["2020-01-01", "2020-07-01", "2021-01-01", "2021-07-01"], [10, 20, 30, 40])
        trades = fills([("2020-01-01", "BUY", 100, 1000, 0, "FILLED"),
                        ("2020-07-01", "BUY", 100, 2000, 0, "FILLED"),
                        ("2021-01-01", "SELL", 150, 4500, 0, "FILLED")])
        lots, summary = analyze_lots(daily, trades)
        self.assertEqual(list(lots.lot_id), ["L00001", "L00002", "L00002"])
        self.assertEqual(list(lots.status), ["REALIZED", "REALIZED", "UNREALIZED"])
        np.testing.assert_allclose(lots.shares, [100, 50, 50])
        np.testing.assert_allclose(lots.cost, [1000, 1000, 1000])
        np.testing.assert_allclose(lots.proceeds, [3000, 1500, 2000])
        self.assertEqual(list(lots.holding_days), [366, 184, 365])
        self.assertAlmostEqual(lots.iloc[1].annualized_return, 1.5**(365/184)-1)
        self.assertTrue(pd.isna(lots.iloc[2].sell_date))
        self.assertEqual(summary["total_pnl"], 3500)
        self.assertEqual(summary["realized_count"], 2)
        self.assertEqual(summary["open_count"], 1)

    def test_multiple_sales_from_same_lot_preserve_cost(self):
        daily = history(["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"], [10, 11, 12, 13])
        trades = fills([("2020-01-01", "BUY", 100, 1000, 0, "FILLED"),
                        ("2021-01-01", "SELL", 20, 220, 0, "FILLED"),
                        ("2022-01-01", "SELL", 30, 360, 0, "FILLED")])
        lots, summary = analyze_lots(daily, trades)
        np.testing.assert_allclose(lots.cost, [200, 300, 500])
        self.assertEqual(list(lots.lot_id), ["L00001"]*3)
        self.assertAlmostEqual(summary["total_pnl"], 230)

    def test_fees_in_cost_and_net_proceeds(self):
        daily = history(["2021-01-01", "2022-01-01"], [10, 12])
        trades = fills([("2021-01-01", "BUY", 100, 1010, 10, "FILLED"),
                        ("2022-01-01", "SELL", 50, 600, 6, "FILLED")])
        lots, summary = analyze_lots(daily, trades)
        self.assertEqual(lots.iloc[0].cost, 505)
        self.assertEqual(lots.iloc[0].proceeds, 594)
        self.assertAlmostEqual(lots.iloc[0].annualized_return, 594/505-1)
        self.assertEqual(lots.iloc[1].proceeds, 600)  # no hypothetical sell fee
        self.assertAlmostEqual(summary["total_pnl"], 184)

    def test_split_and_dividend_reinvestment_before_fill(self):
        daily = history(["2021-01-01", "2021-07-01", "2022-01-01"], [10, 4, 5], [0, 1, 0], [1, 2, 1])
        trades = fills([("2021-01-01", "BUY", 100, 1000, 0, "FILLED"),
                        ("2021-07-01", "BUY", 100, 400, 0, "FILLED"),
                        ("2022-01-01", "SELL", 275, 1375, 0, "FILLED")])
        lots, summary = analyze_lots(daily, trades)
        np.testing.assert_allclose(lots.shares, [250, 25, 75])
        np.testing.assert_allclose(lots.cost, [1000, 100, 300])
        np.testing.assert_allclose(lots.proceeds, [1250, 125, 375])
        self.assertAlmostEqual(lots.iloc[0].annualized_return, .25)
        self.assertEqual(summary["total_pnl"], 350)

    def test_zero_day_is_undefined_not_zero_or_infinity(self):
        daily = history(["2021-01-01"], [10])
        trades = fills([("2021-01-01", "BUY", 100, 1000, 0, "FILLED")])
        lots, summary = analyze_lots(daily, trades)
        self.assertTrue(pd.isna(lots.iloc[0].annualized_return))
        self.assertEqual(summary["undefined_annualized_count"], 1)
        self.assertIsNone(summary["open_annualized_median"])
        self.assertEqual(lots.iloc[0].holding_return, 0)

    def test_overflow_and_short_hold_are_not_hidden_or_capped(self):
        daily = history(["2021-01-01", "2021-01-02"], [1, 100])
        trades = fills([("2021-01-01", "BUY", 1, 1, 0, "FILLED"),
                        ("2021-01-02", "SELL", 1, 100, 0, "FILLED")])
        lots, summary = analyze_lots(daily, trades)
        self.assertTrue(pd.isna(lots.iloc[0].annualized_return))
        self.assertEqual(lots.iloc[0].holding_return, 99)
        self.assertEqual(summary["short_hold_count"], 1)
        self.assertIn("超出数值范围", lots.iloc[0].annualization_note)

    def test_failed_and_pending_not_matched_and_partial_is_matched(self):
        daily = history(["2021-01-01", "2022-01-01"], [1, 2])
        trades = fills([("2021-01-01", "BUY", 0, 0, 0, "REJECTED_NO_CASH"),
                        ("2021-01-01", "BUY", 20, 20, 0, "PARTIAL"),
                        (None, "SELL", 0, 0, 0, "PENDING_END_OF_DATA")])
        lots, summary = analyze_lots(daily, trades)
        self.assertEqual(len(lots), 1)
        self.assertEqual(summary["buy_lot_count"], 1)
        self.assertEqual(lots.iloc[0].status, "UNREALIZED")
        self.assertAlmostEqual(lots.iloc[0].annualized_return, 1)

    def test_empty_history_and_invalid_matches_fail(self):
        with self.assertRaises(ValueError):
            analyze_lots(pd.DataFrame(), pd.DataFrame())
        daily = history(["2021-01-01", "2022-01-01"], [1, 2])
        lots, summary = analyze_lots(daily, pd.DataFrame())
        self.assertTrue(lots.empty)
        self.assertEqual(summary["total_pnl"], 0)
        self.assertIsNone(summary["realized_annualized_median"])
        for trades in [fills([("2021-01-01", "SELL", 1, 1, 0, "FILLED")]),
                       fills([("2023-01-01", "BUY", 1, 1, 0, "FILLED")])]:
            with self.assertRaises(ValueError):
                analyze_lots(daily, trades)
        for year_days in [True, 0, -1, float("inf")]:
            with self.assertRaises(ValueError):
                analyze_lots(daily, pd.DataFrame(), year_days)

    def test_110020_fifo_reconciles_all_fills_and_terminal_holdings(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / "data/110020.csv").is_file():
            self.skipTest("Local data snapshot missing; run scripts/prepare_local_data.py --from /path/to/AlphaTrack")
        frame, _ = load_data(root/"data/110020.csv")
        daily, trades = run_backtest(frame, FunddStrategy())
        lots, summary = analyze_lots(daily, trades)
        valid = trades[trades.status.isin(["FILLED", "PARTIAL"])]
        buys, sells = valid[valid.action == "BUY"], valid[valid.action == "SELL"]
        self.assertEqual(summary["buy_lot_count"], 687)
        self.assertAlmostEqual(lots.cost.sum(), buys.amount.sum(), places=7)
        self.assertAlmostEqual(summary["realized_proceeds"], (sells.amount-sells.fee).sum(), places=7)
        self.assertAlmostEqual(summary["open_value"], daily.position_value.iloc[-1], places=7)
        self.assertAlmostEqual(summary["total_pnl"], daily.realized_pnl.iloc[-1]+daily.unrealized_pnl.iloc[-1], places=7)
        self.assertTrue((lots.holding_days >= 0).all())
        self.assertTrue(np.isfinite(lots.annualized_return.dropna()).all())


if __name__ == "__main__":
    unittest.main()
