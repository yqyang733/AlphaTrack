"""Daily event loop with next-observation fills and cash-flow-neutral NAV.

Strategies see a causal prepared history, and must not inspect future rows.
No external cash flow is counted as P&L. Sale proceeds may be withdrawn.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EngineConfig:
    initial_cash: float = 0.0
    funding_mode: str = "topup"
    withdraw_on_sell: bool = False
    start: str | None = None
    end: str | None = None
    buy_fee: float = 0.0
    sell_fee: float = 0.0
    risk_free_rate: float = 0.0
    periods_per_year: int = 252

    def __post_init__(self):
        if self.funding_mode not in {"fixed", "topup"}:
            raise ValueError("funding_mode must be fixed or topup")
        if type(self.withdraw_on_sell) is not bool:
            raise ValueError("withdraw_on_sell must be boolean")
        for key in ("initial_cash", "buy_fee", "sell_fee", "risk_free_rate"):
            v = getattr(self, key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"Invalid {key}")
        if self.initial_cash < 0 or (self.funding_mode == "fixed" and self.initial_cash <= 0):
            raise ValueError("Fixed funding requires positive initial_cash")
        if not 0 <= self.buy_fee < 1 or not 0 <= self.sell_fee < 1 or self.risk_free_rate <= -1:
            raise ValueError("Invalid fee or risk-free rate")
        if type(self.periods_per_year) is not int or self.periods_per_year < 1:
            raise ValueError("periods_per_year must be positive integer")
        if self.start and self.end and pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError("start must precede end")


TRADE_COLUMNS = ["signal_date", "date", "action", "amount", "shares", "price", "fee", "reason", "status"]


def run_backtest(frame, strategy, config=None):
    cfg = config or EngineConfig()
    prepared = strategy.prepare(frame.copy())
    active = prepared.copy()
    if cfg.start:
        active = active[active.date >= pd.Timestamp(cfg.start)]
    if cfg.end:
        active = active[active.date <= pd.Timestamp(cfg.end)]
    if active.empty:
        raise ValueError("No data in requested date range")
    active = active.reset_index(drop=True)
    cash, shares, cost = cfg.initial_cash, 0.0, 0.0
    contributed, realized, fees, invested = cfg.initial_cash, 0.0, 0.0, 0.0
    withdrawn = 0.0
    prev_equity, unit, peak = cfg.initial_cash, 1.0, 1.0
    pending = None
    daily, trades = [], []
    reference = float(active.iloc[0].total_return_index)
    for index, row in active.iterrows():
        nav = float(row.nav)
        flow, day_fee, withdrawal = 0.0, 0.0, 0.0
        # Apply corporate actions to holdings brought into this observation.
        shares *= float(row.split)
        if row.dividend and shares:
            distribution = shares * float(row.dividend)
            shares += distribution / nav
            # Cost unchanged: holding P&L includes reinvested distributions.
        pre_flow_equity = cash + shares * nav
        if pending:
            decision, signal_date = pending
            action = decision["action"]
            amount, quantity, fee, status = 0.0, 0.0, 0.0, "FILLED"
            if action == "BUY":
                requested = float(decision["amount"])
                if not math.isfinite(requested) or requested <= 0:
                    raise ValueError("BUY amount must be positive")
                if cfg.funding_mode == "topup":
                    flow = max(0.0, requested - cash)
                    cash += flow
                    contributed += flow
                amount = min(requested, cash)
                if amount <= 1e-10:
                    amount, status = 0.0, "REJECTED_NO_CASH"
                else:
                    status = "PARTIAL" if amount < requested - 1e-8 else "FILLED"
                    # Subscription rate charged on net subscription (gross/(1+r)).
                    net = amount / (1 + cfg.buy_fee)
                    fee = amount - net
                    quantity = net / nav
                    shares += quantity
                    cash -= amount
                    cost += amount
                    invested += amount
            elif action == "SELL":
                fraction = float(decision["fraction"])
                if not math.isfinite(fraction) or not 0 < fraction <= 1:
                    raise ValueError("SELL fraction must be in (0,1]")
                if shares <= 1e-10:
                    status = "REJECTED_NO_HOLDING"
                else:
                    quantity = shares * fraction
                    amount = quantity * nav
                    fee = amount * cfg.sell_fee
                    removed_cost = cost * fraction
                    cost -= removed_cost
                    realized += amount - fee - removed_cost
                    shares -= quantity
                    cash += amount - fee
                    if cfg.withdraw_on_sell:
                        withdrawal = amount - fee
                        cash -= withdrawal
                        withdrawn += withdrawal
            else:
                raise ValueError(f"Unsupported action {action}")
            day_fee += fee
            fees += fee
            trades.append(dict(signal_date=signal_date, date=row.date, action=action,
                               amount=amount, shares=quantity, price=nav, fee=fee,
                               reason=decision["reason"], status=status))
        equity = cash + shares * nav
        pnl = equity - prev_equity - flow + withdrawal
        # Link returns immediately before and after close-of-day contributions.
        # New-capital fees must not be divided only by the small old portfolio.
        market_factor = pre_flow_equity / prev_equity if prev_equity > 1e-10 else 1.0
        post_flow_base = pre_flow_equity + flow
        trade_factor = (equity + withdrawal) / post_flow_base if post_flow_base > 1e-10 else 1.0
        ret = market_factor * trade_factor - 1
        unit *= 1 + ret
        peak = max(peak, unit)
        unrealized = shares * nav - cost
        record = row.to_dict()
        record.update(cash=max(0.0, cash), shares=shares, position_value=shares*nav,
                      equity=equity, daily_pnl=pnl, daily_return=ret,
                      cumulative_return=unit-1, unit_nav=unit, drawdown=unit/peak-1,
                      benchmark_return=row.total_return_index/reference-1,
                      holding_cost=cost, unrealized_pnl=unrealized,
                      holding_return=unrealized/cost if cost > 1e-10 else np.nan,
                      realized_pnl=realized, fees=fees, invested=invested,
                      external_flow=flow + (cfg.initial_cash if index == 0 else 0),
                      net_contributions=contributed,
                      total_contributions=contributed, withdrawal=withdrawal,
                      cumulative_withdrawals=withdrawn,
                      net_external_contributions=contributed-withdrawn,
                      cumulative_pnl=equity+withdrawn-contributed,
                      simple_return=(equity+withdrawn-contributed)/contributed if contributed else 0.0)
        daily.append(record)
        prev_equity = equity
        decision = strategy.decide(row)
        pending = (decision, row.date) if decision["action"] != "HOLD" else None
    # The final signal has no next NAV; preserve it in audit output without filling.
    if pending:
        decision, signal_date = pending
        trades.append(dict(signal_date=signal_date, date=pd.NaT, action=decision["action"],
                           amount=0.0, shares=0.0, price=np.nan, fee=0.0,
                           reason=decision["reason"], status="PENDING_END_OF_DATA"))
    return pd.DataFrame(daily), pd.DataFrame(trades, columns=TRADE_COLUMNS)
