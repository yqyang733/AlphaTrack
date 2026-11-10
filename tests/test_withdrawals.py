"""Sell proceeds leave immediately and never affect future held-capital returns."""
import unittest
import numpy as np
from alphatrack.engine import EngineConfig, run_backtest
from alphatrack.metrics import calculate_metrics
from test_engine import ScriptedStrategy, market, buy, sell


class WithdrawalTests(unittest.TestCase):
    def test_full_sale_cash_is_external_and_future_buy_is_fresh(self):
        cfg=EngineConfig(withdraw_on_sell=True)
        d,t=run_backtest(market([1.,1.,1.2,1.2,1.2]),ScriptedStrategy([buy(100),sell(1),buy(80)]),cfg)
        np.testing.assert_allclose(d.external_flow,[0,100,0,80,0])
        np.testing.assert_allclose(d.withdrawal,[0,0,120,0,0])
        np.testing.assert_allclose(d.cash,0,atol=1e-10)
        np.testing.assert_allclose(d.daily_return,[0,0,.2,0,0],atol=1e-10)
        np.testing.assert_allclose(d.daily_pnl.cumsum(),d.cumulative_pnl,atol=1e-10)
        self.assertAlmostEqual(d.cumulative_pnl.iloc[-1],20)
        m=calculate_metrics(d,t,cfg)
        self.assertEqual(m['total_pnl'],20)
        self.assertIsNone(m['xirr'])

    def test_withdrawal_uses_net_proceeds_and_partial_sale_fee(self):
        cfg=EngineConfig(withdraw_on_sell=True,sell_fee=.01)
        d,t=run_backtest(market([1.,1.,2.,2.]),ScriptedStrategy([buy(100),sell(.5)]),cfg)
        self.assertAlmostEqual(d.withdrawal.iloc[2],99)
        self.assertAlmostEqual(d.equity.iloc[2],100)
        self.assertAlmostEqual(d.daily_pnl.iloc[2],99)
        self.assertAlmostEqual(d.daily_return.iloc[2],.99)
        np.testing.assert_allclose(d.cumulative_pnl,d.realized_pnl+d.unrealized_pnl,atol=1e-10)

    def test_fixed_budget_does_not_reuse_withdrawn_cash(self):
        cfg=EngineConfig(initial_cash=100,funding_mode='fixed',withdraw_on_sell=True)
        d,t=run_backtest(market([1.]*4),ScriptedStrategy([buy(100),sell(1),buy(100)]),cfg)
        self.assertEqual(t.status.iloc[-1],'REJECTED_NO_CASH')
        self.assertEqual(d.equity.iloc[-1],0)

    def test_sell_cash_policy_does_not_change_unlimited_fundd_fills(self):
        from alphatrack.data import load_data
        from alphatrack.strategies.fundd import FunddStrategy
        from pathlib import Path
        frame,_=load_data(Path(__file__).resolve().parents[1]/'data/110020.csv')
        keep,a=run_backtest(frame,FunddStrategy(),EngineConfig())
        withdraw,b=run_backtest(frame,FunddStrategy(),EngineConfig(withdraw_on_sell=True))
        np.testing.assert_allclose(a.amount,b.amount,atol=1e-8)
        np.testing.assert_allclose(a.shares,b.shares,atol=1e-8)
        self.assertAlmostEqual(keep.cumulative_pnl.iloc[-1],withdraw.cumulative_pnl.iloc[-1],places=7)
        self.assertAlmostEqual(withdraw.total_contributions.iloc[-1],61400)
        self.assertTrue((withdraw.cash < 1e-7).all())
