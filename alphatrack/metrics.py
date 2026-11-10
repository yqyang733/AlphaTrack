"""Portfolio metrics use unitized returns; investor return uses dated cash flows."""
import math
import numpy as np
import pandas as pd


def xirr(dates, flows):
    """Unique root for contributions followed by one terminal liquidation value."""
    times = np.array([(pd.Timestamp(d)-pd.Timestamp(dates[0])).days/365.25 for d in dates])
    flows = np.asarray(flows, dtype=float)
    if times[-1] <= 0 or not (flows < 0).any() or not (flows > 0).any():
        return None
    # Discount all deposits to terminal time, monotonic for this cash-flow pattern.
    def residual(log_rate):
        with np.errstate(over="ignore", invalid="ignore"):
            return float(np.sum(flows * np.exp(log_rate*(times[-1]-times))))
    low, high = -30.0, 30.0
    if residual(low)*residual(high) > 0:
        return None
    for _ in range(160):
        mid = (low+high)/2
        if residual(mid) > 0:
            low = mid
        else:
            high = mid
    return math.expm1((low+high)/2)


def calculate_metrics(daily, trades, cfg):
    end = daily.iloc[-1]
    funded = daily[daily.net_contributions > 0]
    start = funded.iloc[0] if len(funded) else daily.iloc[0]
    years = (end.date-start.date).days/365.25
    r = funded.daily_return if len(funded) else pd.Series(dtype=float)
    scale = cfg.periods_per_year
    sd = float(r.std(ddof=1)) if len(r)>1 else 0.0
    rf = (1+cfg.risk_free_rate)**(1/scale)-1
    excess = r-rf
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0)**2))) if len(r) else 0.0
    cagr = float(end.unit_nav**(1/years)-1) if years>0 and len(funded) else None
    dd = float(daily.drawdown.min())
    duration, peak_date, underwater = 0, daily.iloc[0].date, False
    for row in daily.itertuples():
        if row.drawdown >= -1e-12:
            if underwater:
                duration = max(duration, (row.date-peak_date).days)
            peak_date = row.date
            underwater = False
        else:
            underwater = True
            duration = max(duration, (row.date-peak_date).days)
    fills = trades[trades.status.isin(["FILLED", "PARTIAL"])]
    buys, sells = fills[fills.action=="BUY"], fills[fills.action=="SELL"]
    deposits = daily[daily.external_flow > 0]
    irr = xirr(list(deposits.date)+[end.date], list(-deposits.external_flow)+[end.equity]) if len(deposits) else None
    benchmark_years = (end.date-daily.iloc[0].date).days/365.25
    result = dict(initial_cash=cfg.initial_cash, funding_mode=cfg.funding_mode,
                  first_investment_date=str(start.date.date()) if len(funded) else None,
                  final_equity=float(end.equity), total_pnl=float(end.equity-end.net_contributions),
                  total_return=float(end.unit_nav-1), simple_return=float(end.simple_return),
                  net_contributions=float(end.net_contributions), cagr=cagr, xirr=irr,
                  annual_volatility=sd*math.sqrt(scale),
                  sharpe=float(excess.mean()/sd*math.sqrt(scale)) if sd>1e-15 else None,
                  sortino=float(excess.mean()/downside*math.sqrt(scale)) if downside>1e-15 else None,
                  max_drawdown=dd, calmar=cagr/abs(dd) if cagr is not None and dd<0 else None,
                  max_drawdown_days=duration,
                  best_day=float(r.max()) if len(r) else None,
                  worst_day=float(r.min()) if len(r) else None,
                  positive_day_ratio=float((r>0).mean()) if len(r) else None,
                  buy_count=len(buys), sell_count=len(sells), total_fees=float(end.fees),
                  total_buys=float(buys.amount.sum()), total_sells=float(sells.amount.sum()),
                  realized_pnl=float(end.realized_pnl), unrealized_pnl=float(end.unrealized_pnl),
                  holding_return=float(end.holding_return),
                  exposure_mean=float((funded.position_value/funded.equity).mean()) if len(funded) else 0.0,
                  benchmark_return=float(end.benchmark_return),
                  benchmark_cagr=float((1+end.benchmark_return)**(1/benchmark_years)-1) if benchmark_years>0 else None,
                  pending_count=int((trades.status=="PENDING_END_OF_DATA").sum()),
                  rejected_count=int(trades.status.str.startswith("REJECTED").sum()),
                  start_date=str(daily.iloc[0].date.date()), end_date=str(end.date.date()), observations=len(daily))
    return {k:(None if isinstance(v,float) and not math.isfinite(v) else v) for k,v in result.items()}
