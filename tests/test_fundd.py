import unittest

import numpy as np
import pandas as pd

from alphatrack.strategies.fundd import FunddStrategy


class FunddTests(unittest.TestCase):
    def test_exact_buy_boundaries(self):
        cases = [(0.200001, None), (.2, 50), (.1, 50), (.099999, 100),
                 (.05, 100), (.049999, 150), (.000001, 150), (0, 200)]
        for value, amount in cases:
            with self.subTest(value=value):
                result = FunddStrategy().decide({"signal_metric": value})
                self.assertEqual(result.get("amount"), amount)

    def test_jump_composition_and_reset(self):
        strategy = FunddStrategy()
        first = strategy.decide({"signal_metric": .75})
        self.assertEqual(len(first["levels"]), 25)
        self.assertAlmostEqual(first["fraction"], .9 * 325 / 1275)
        for v in (.70, .75):
            self.assertEqual(strategy.decide({"signal_metric": v})["action"], "HOLD")
        last = strategy.decide({"signal_metric": 1})
        self.assertEqual(len(last["levels"]), 25)
        self.assertAlmostEqual((1 - first["fraction"]) * (1 - last["fraction"]), .1)
        strategy.decide({"signal_metric": .2})
        self.assertAlmostEqual(strategy.decide({"signal_metric": 1})["fraction"], .9)

    def test_missing_does_not_reset(self):
        strategy = FunddStrategy()
        strategy.decide({"signal_metric": .5})
        strategy.decide({"signal_metric": np.nan})
        self.assertEqual(strategy.decide({"signal_metric": .5})["action"], "HOLD")

    def test_no_lookahead_and_short_history(self):
        frame = pd.DataFrame({"date": pd.date_range("2009-08-26", periods=5),
                              "total_return_index": [1., .998, 1.005, .99, 100.]})
        all_rows = FunddStrategy().prepare(frame)
        prefix = FunddStrategy().prepare(frame.iloc[:4])
        pd.testing.assert_frame_equal(all_rows.iloc[:4], prefix)
        self.assertTrue(np.isnan(all_rows.signal_metric.iloc[0]))
        self.assertEqual(all_rows.signal_metric.iloc[1], 0)
        self.assertEqual(FunddStrategy().decide(all_rows.iloc[0])["action"], "HOLD")

    def test_calendar_window_and_minimum(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2009-01-01", "2019-01-01", "2024-01-01", "2024-01-02"]),
                              "total_return_index": [1., 3., 2., 2.5]})
        rows = FunddStrategy().prepare(frame)
        self.assertEqual(rows.position_5y.iloc[-2], 0)
        self.assertEqual(rows.position_5y.iloc[-1], 1)
        self.assertEqual(rows.position_10y.iloc[-1], .5)
        self.assertEqual(rows.position_all.iloc[-1], .75)
        self.assertEqual(rows.signal_metric.iloc[-1], .5)

    def test_flat_window(self):
        frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=2), "total_return_index": [1., 1.]})
        self.assertEqual(FunddStrategy().prepare(frame).signal_metric.iloc[1], .5)

    def test_custom_parameters_and_buy_priority(self):
        strategy = FunddStrategy(base_amount=2, buy_thresholds=[70, 30, 0],
                                 buy_multipliers=[2, 3, 4], sell_start=50, sell_count=2, sell_total="1/2")
        self.assertEqual(strategy.decide({"signal_metric": .6})["amount"], 4)
        self.assertEqual(strategy.decide({"signal_metric": 1})["fraction"], .5)
        strategy = FunddStrategy(sell_total=1)
        self.assertEqual(strategy.decide({"signal_metric": 1})["fraction"], 1)

    def test_invalid_parameters(self):
        invalid = [{"base_amount": 0}, {"base_amount": True}, {"base_amount": float("nan")},
                   {"buy_thresholds": [20, 20]}, {"buy_thresholds": [20, 10]},
                   {"buy_multipliers": [1, 2, 3, 0]}, {"sell_count": 1}, {"sell_count": True},
                   {"sell_start": 100}, {"sell_end": 101}, {"sell_total": 0}, {"sell_total": "1/0"}]
        for params in invalid:
            with self.subTest(params=params), self.assertRaises(ValueError):
                FunddStrategy(**params)


if __name__ == "__main__":
    unittest.main()
