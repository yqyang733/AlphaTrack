"""Cash-flow-aware aggregation and calendar-year analysis of fund ledgers.

Stocks are carried forward only after a fund's first observation. Cash flows,
fees and P&L are events and are never carried forward. Portfolio returns are
computed from the combined assets, not an average of component returns.
"""

from collections.abc import Mapping
import math

import numpy as np
import pandas as pd


STOCK_FIELDS = (
    "equity", "cash", "position_value", "holding_cost", "realized_pnl",
    "unrealized_pnl", "fees", "invested",
)
EPS = 1e-10


def _numeric(frame, column, default=0.0):
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    result = pd.to_numeric(frame[column], errors="raise").astype(float)
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite ledger column: {column}")
    return result


def aggregate_portfolio(ledgers: Mapping[str, pd.DataFrame], *, start=None, end=None):
    """Combine independent zero-opening-balance backtests on observed dates.

    ``external_flow`` contains positive deposits (including initial cash),
    ``withdrawal`` contains positive payouts and ``fees`` is cumulative.
    All trades are valued at the day's NAV. External contributions occur
    after market movement; transaction fees and sale withdrawals follow.
    ``start``/``end`` filter output after the complete ledger is aggregated,
    retaining previous_equity for a correct opening balance.
    """
    if not ledgers:
        raise ValueError("At least one fund ledger is required")
    frames = {}
    for key, value in ledgers.items():
        if value.empty:
            continue
        if not {"date", "equity", "external_flow", "withdrawal"}.issubset(value):
            raise ValueError(f"Missing required ledger fields for {key}")
        frame = value.copy()
        frame["date"] = pd.to_datetime(frame.date, errors="raise")
        if frame.date.isna().any() or frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
            raise ValueError(f"Ledger dates must be unique and increasing: {key}")
        frames[key] = frame.set_index("date")
    if not frames:
        raise ValueError("All fund ledgers are empty")
    dates = pd.DatetimeIndex(sorted(set().union(*(set(f.index) for f in frames.values()))), name="date")
    combined = pd.DataFrame(0.0, index=dates, columns=[*STOCK_FIELDS, "external_flow", "withdrawal", "source_daily_pnl"])
    have_source_pnl = all("daily_pnl" in frame for frame in frames.values())
    observations = pd.Series(0, index=dates, dtype=int)
    for frame in frames.values():
        for field in STOCK_FIELDS:
            combined[field] += _numeric(frame, field).reindex(dates).ffill().fillna(0.0)
        for field in ("external_flow", "withdrawal"):
            values = _numeric(frame, field)
            if (values < -EPS).any():
                raise ValueError(f"{field} must use positive amounts")
            combined[field] += values.reindex(dates, fill_value=0.0)
        if "daily_pnl" in frame:
            combined["source_daily_pnl"] += _numeric(frame, "daily_pnl").reindex(dates, fill_value=0.0)
        observations.loc[frame.index] += 1

    combined["observed_funds"] = observations
    combined["previous_equity"] = combined.equity.shift(1, fill_value=0.0)
    combined["day_fees"] = combined.fees.diff().fillna(combined.fees.iloc[0])
    if (combined.day_fees < -1e-7).any():
        raise ValueError("Cumulative fees decreased")
    combined["daily_pnl"] = combined.equity - combined.previous_equity - combined.external_flow + combined.withdrawal
    if have_source_pnl and not np.allclose(combined.daily_pnl, combined.source_daily_pnl, rtol=1e-9, atol=1e-6):
        raise ValueError("Portfolio daily P&L disagrees with component ledgers")
    combined["pre_flow_equity"] = combined.equity - combined.external_flow + combined.withdrawal + combined.day_fees
    if (combined.pre_flow_equity < -1e-6).any():
        raise ValueError("Negative pre-flow assets are not supported")
    combined["pre_flow_equity"] = combined.pre_flow_equity.clip(lower=0.0)
    market = np.divide(combined.pre_flow_equity, combined.previous_equity,
                       out=np.ones(len(combined)), where=combined.previous_equity.to_numpy() > EPS)
    post_flow_base = combined.pre_flow_equity + combined.external_flow
    trading = np.divide(combined.equity + combined.withdrawal, post_flow_base,
                        out=np.ones(len(combined)), where=post_flow_base.to_numpy() > EPS)
    combined["daily_return"] = market * trading - 1.0
    combined["unit_nav"] = (1.0 + combined.daily_return).cumprod()
    combined["cumulative_return"] = combined.unit_nav - 1.0
    combined["drawdown"] = combined.unit_nav / combined.unit_nav.cummax().clip(lower=1.0) - 1.0
    combined["total_contributions"] = combined.external_flow.cumsum()
    combined["cumulative_withdrawals"] = combined.withdrawal.cumsum()
    combined["net_external_contributions"] = combined.total_contributions - combined.cumulative_withdrawals
    combined["cumulative_pnl"] = combined.equity - combined.net_external_contributions
    combined["realized_daily_pnl"] = combined.realized_pnl.diff().fillna(combined.realized_pnl.iloc[0])
    combined["unrealized_daily_change"] = combined.unrealized_pnl.diff().fillna(combined.unrealized_pnl.iloc[0])
    if not np.allclose(combined.daily_pnl.cumsum(), combined.cumulative_pnl, rtol=1e-9, atol=1e-6):
        raise ValueError("Portfolio cumulative P&L failed cash-flow reconciliation")
    combined = combined.drop(columns="source_daily_pnl").reset_index()
    if start is not None:
        combined = combined[combined.date >= pd.Timestamp(start)]
    if end is not None:
        combined = combined[combined.date <= pd.Timestamp(end)]
    if combined.empty:
        raise ValueError("No portfolio dates in requested interval")
    return combined.reset_index(drop=True)


def cashflow_xirr(dates, amounts, year_days=365.0):
    """Search dated-cash-flow IRR roots, explicitly reporting ambiguity.

    The scan covers log(1+r) in [-27.63, 27.63], i.e. approximately
    -100% < r < 10^12. Both sign-crossing roots and stationary zeroes are
    tested. One sign change gives a provably unique root; with more sign
    changes a single detected root is labelled unique_detected, not proven
    globally unique. Multiple detected roots return rate=None.
    """
    result = {"rate": None, "roots": [], "status": "undefined", "sign_changes": 0,
              "search_log_rate_bounds": [-27.6310211159, 27.6310211159]}
    if not math.isfinite(year_days) or year_days <= 0:
        raise ValueError("year_days must be positive")
    if len(dates) != len(amounts):
        raise ValueError("Cash-flow dates and amounts must have equal lengths")
    if not len(dates):
        result["status"] = "no_cashflows"
        return result
    frame = pd.DataFrame({"date": pd.to_datetime(dates, errors="raise"), "amount": np.asarray(amounts, dtype=float)})
    if frame.date.isna().any() or not np.isfinite(frame.amount).all():
        raise ValueError("Invalid cash-flow data")
    flows = frame.groupby("date", sort=True).amount.sum()
    flows = flows[flows.abs() > EPS]
    if len(flows) < 2 or flows.index[-1] == flows.index[0]:
        result["status"] = "insufficient_dates"
        return result
    cf = flows.to_numpy()
    if not (cf < 0).any() or not (cf > 0).any():
        result["status"] = "no_sign_change"
        return result
    result["sign_changes"] = int(np.count_nonzero(np.sign(cf[1:]) != np.sign(cf[:-1])))
    times = (flows.index - flows.index[0]).days.to_numpy(dtype=float) / year_days
    cf = cf / np.max(np.abs(cf))

    def evaluate(y, derivative=False):
        exponents = -y * times
        weights = np.exp(exponents - exponents.max())
        numerator = np.sum(cf * weights * (-times if derivative else 1.0))
        return float(numerator / np.sum(np.abs(cf) * weights))

    def bisect(a, b, derivative=False):
        fa = evaluate(a, derivative)
        for _ in range(100):
            mid = (a + b) / 2
            fm = evaluate(mid, derivative)
            if abs(b - a) < 1e-13:
                return mid
            if fa * fm <= 0:
                b = mid
            else:
                a, fa = mid, fm
        return (a + b) / 2

    lower, upper = result["search_log_rate_bounds"]
    # Dense around practical rates, with broad tails for extreme cash flows.
    grid = np.unique(np.concatenate((np.linspace(lower, upper, 2049), np.linspace(-2, 2, 2049), [0.0])))
    values = np.array([evaluate(x) for x in grid])
    candidates = [float(grid[i]) for i in np.flatnonzero(np.abs(values) < 1e-13)]
    for i in np.flatnonzero(values[:-1] * values[1:] < 0):
        candidates.append(bisect(float(grid[i]), float(grid[i+1])))
    if result["sign_changes"] > 1:
        derivatives = np.array([evaluate(x, True) for x in grid])
        for i in np.flatnonzero(derivatives[:-1] * derivatives[1:] < 0):
            stationary = bisect(float(grid[i]), float(grid[i+1]), True)
            if abs(evaluate(stationary)) < 1e-11:
                candidates.append(stationary)
            # An extremum can hide two close roots within one grid cell.
            for a, b in ((float(grid[i]), stationary), (stationary, float(grid[i+1]))):
                if evaluate(a) * evaluate(b) < 0:
                    candidates.append(bisect(a, b))
    roots = []
    for candidate in sorted(candidates):
        if abs(evaluate(candidate)) <= 1e-9 and not any(abs(candidate - old) < 1e-7 for old in roots):
            roots.append(candidate)
    result["roots"] = [math.expm1(x) for x in roots]
    if len(roots) == 1:
        result["rate"] = result["roots"][0]
        result["status"] = "unique" if result["sign_changes"] == 1 else "unique_detected"
    elif len(roots) > 1:
        result["status"] = "multiple_roots"
    else:
        result["status"] = "no_root_in_search_range"
    return result


def _period_xirr(part, opening_date=None):
    dates = list(part.date)
    amounts = list((part.withdrawal - part.external_flow).astype(float))
    beginning = float(part.previous_equity.iloc[0])
    if beginning > EPS:
        dates.insert(0, pd.Timestamp(opening_date) if opening_date is not None else part.date.iloc[0] - pd.Timedelta(days=1))
        amounts.insert(0, -beginning)
    dates.append(part.date.iloc[-1])
    amounts.append(float(part.equity.iloc[-1]))
    return cashflow_xirr(dates, amounts)


def _period_drawdown(returns):
    path = np.cumprod(1 + np.asarray(returns, dtype=float))
    peaks = np.maximum.accumulate(np.concatenate(([1.0], path)))[1:]
    return float(np.min(path / peaks - 1.0)) if len(path) else 0.0


def _filled_trades(trades):
    if trades is None:
        return pd.DataFrame(columns=["date", "action", "amount", "fee"])
    if isinstance(trades, Mapping):
        trades = pd.concat(trades.values(), ignore_index=True) if trades else pd.DataFrame()
    if trades.empty:
        return pd.DataFrame(columns=["date", "action", "amount", "fee"])
    result = trades.copy()
    if "status" in result:
        result = result[result.status.isin(["FILLED", "PARTIAL"])]
    result["date"] = pd.to_datetime(result.date, errors="raise")
    return result.dropna(subset=["date"])


def _trade_totals(trades):
    buys, sells = trades[trades.action == "BUY"], trades[trades.action == "SELL"]
    return {"buy_count": len(buys), "sell_count": len(sells),
            "buy_amount": float(_numeric(buys, "amount").sum()),
            "sell_proceeds": float((_numeric(sells, "amount") - _numeric(sells, "fee")).sum())}


def analyze_portfolio(daily, trades=None, *, start=None, end=None, periods_per_year=252, risk_free_rate=0.0):
    """Return (calendar-year dataframe, summary) from an aggregated ledger."""
    if daily.empty:
        raise ValueError("Nonempty portfolio ledger is required")
    if type(periods_per_year) is not int or periods_per_year < 1 or not math.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("Invalid annualization settings")
    df = daily.copy()
    df["date"] = pd.to_datetime(df.date)
    if df.date.duplicated().any() or not df.date.is_monotonic_increasing:
        raise ValueError("Portfolio dates must be unique and increasing")
    scope_start = pd.Timestamp(start) if start is not None else df.date.iloc[0]
    scope_end = min(pd.Timestamp(end), df.date.iloc[-1]) if end is not None else df.date.iloc[-1]
    if scope_start > scope_end:
        raise ValueError("start must precede end")
    df = df[(df.date >= scope_start) & (df.date <= scope_end)].copy()
    if df.empty:
        raise ValueError("No portfolio dates in requested interval")
    fills = _filled_trades(trades)
    if len(fills):
        fills = fills[(fills.date >= scope_start) & (fills.date <= scope_end)]
    rows = []
    grouped = {int(year): part for year, part in df.groupby(df.date.dt.year, sort=True)}
    for year in range(scope_start.year, scope_end.year+1):
        ystart, yend = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
        part = grouped.get(year)
        if part is None:
            prior = df[df.date < ystart]
            carried_assets = float(prior.equity.iloc[-1]) if len(prior) else float(df.previous_equity.iloc[0])
            rows.append(dict(year=year, start_date=str(max(scope_start, ystart).date()),
                             end_date=str(min(scope_end, yend).date()),
                             is_partial_year=bool(scope_start > ystart or scope_end < yend),
                             beginning_assets=carried_assets, end_assets=carried_assets,
                             deposits=0.0, withdrawals=0.0, net_flow=0.0, profit=0.0,
                             realized_pnl=0.0, unrealized_pnl_change=0.0,
                             twr_return=None, xirr=None, xirr_status="no_observations", max_drawdown=None,
                             fees=0.0, observations=0, **_trade_totals(fills.iloc[0:0])))
            continue
        opening_date = max(ystart - pd.Timedelta(days=1), scope_start - pd.Timedelta(days=1))
        irr = _period_xirr(part, opening_date)
        beginning, terminal = float(part.previous_equity.iloc[0]), float(part.equity.iloc[-1])
        deposits, withdrawals = float(part.external_flow.sum()), float(part.withdrawal.sum())
        profit = terminal - beginning - deposits + withdrawals
        if not math.isclose(profit, float(part.daily_pnl.sum()), abs_tol=1e-6, rel_tol=1e-9):
            raise ValueError(f"Annual P&L reconciliation failed for {year}")
        row = dict(year=int(year), start_date=str(max(scope_start, ystart).date()),
                   end_date=str(part.date.iloc[-1].date()),
                   is_partial_year=bool(scope_start > ystart or scope_end < yend),
                   beginning_assets=beginning, end_assets=terminal,
                   deposits=deposits, withdrawals=withdrawals, net_flow=deposits-withdrawals,
                   profit=profit, realized_pnl=float(part.realized_daily_pnl.sum()),
                   unrealized_pnl_change=float(part.unrealized_daily_change.sum()),
                   twr_return=float((1+part.daily_return).prod()-1), xirr=irr["rate"],
                   xirr_status=irr["status"], max_drawdown=_period_drawdown(part.daily_return),
                   fees=float(part.day_fees.sum()), observations=len(part))
        year_trades = fills[fills.date.dt.year == year] if len(fills) else fills
        row.update(_trade_totals(year_trades))
        rows.append(row)
    annual = pd.DataFrame(rows)
    if not math.isclose(float(annual.profit.sum()), float(df.daily_pnl.sum()), rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("Annual profits do not sum to portfolio P&L")
    funded_mask = (df.external_flow > EPS) | (df.equity > EPS) | (df.previous_equity > EPS)
    first_investment = df.loc[funded_mask, "date"].iloc[0] if funded_mask.any() else None
    returns = df[df.date >= first_investment].daily_return if first_investment is not None else pd.Series(dtype=float)
    sd = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    rf_daily = (1+risk_free_rate)**(1/periods_per_year)-1
    years = (df.date.iloc[-1] - first_investment).days/365.25 if first_investment is not None else 0.0
    total_return = float((1+df.daily_return).prod()-1)
    cagr = (1+total_return)**(1/years)-1 if years > 0 else None
    irr = _period_xirr(df, scope_start-pd.Timedelta(days=1))
    end_row = df.iloc[-1]
    peak, peak_date, max_duration, underwater = 1.0, first_investment or df.date.iloc[0], 0, False
    for row in df.itertuples():
        if first_investment is not None and row.date < first_investment:
            continue
        if row.unit_nav >= peak*(1-1e-12):
            if underwater:
                max_duration = max(max_duration, (row.date-peak_date).days)
            peak, peak_date, underwater = max(peak, row.unit_nav), row.date, False
        else:
            underwater = True
            max_duration = max(max_duration, (row.date-peak_date).days)
    summary = dict(start_date=str(scope_start.date()), end_date=str(df.date.iloc[-1].date()),
                   first_investment_date=str(first_investment.date()) if first_investment is not None else None,
                   observations=len(df), beginning_assets=float(df.previous_equity.iloc[0]),
                   final_equity=float(end_row.equity),
                   total_contributions=float(df.external_flow.sum()), total_withdrawals=float(df.withdrawal.sum()),
                   net_contributions=float(df.external_flow.sum()-df.withdrawal.sum()),
                   total_pnl=float(df.daily_pnl.sum()), realized_pnl=float(df.realized_daily_pnl.sum()),
                   unrealized_pnl=float(end_row.unrealized_pnl),
                   unrealized_pnl_change=float(df.unrealized_daily_change.sum()),
                   total_return=total_return, cagr=cagr,
                   xirr=irr["rate"], xirr_status=irr["status"], xirr_roots=irr["roots"],
                   xirr_sign_changes=irr["sign_changes"],
                   sharpe=float((returns.mean()-rf_daily)/sd*math.sqrt(periods_per_year)) if sd > EPS else None,
                   annual_volatility=sd*math.sqrt(periods_per_year), max_drawdown=_period_drawdown(df.daily_return),
                   max_drawdown_days=int(max_duration), max_position_value=float(df.position_value.max()),
                   max_equity=float(df.equity.max()),
                   max_net_contributions=float(max(0.0, df.net_external_contributions.max())),
                   total_fees=float(df.day_fees.sum()), risk_free_rate=float(risk_free_rate),
                   periods_per_year=periods_per_year, account_cost_basis="average_cost",
                   cagr_basis="ACT/365.25 from first investment; linked account returns",
                   xirr_basis="ACT/365; contributions negative, withdrawals and terminal marked value positive",
                   risk_metrics_note="Observation-date returns annualized with configured periods_per_year; stale prices carried forward, no invented weekend returns",
                   xirr_note="unique_detected means one root found in a broad numerical scan, not proof of global uniqueness; multiple roots are not reduced to one number")
    summary.update(_trade_totals(fills))
    return annual, {key: None if isinstance(value, float) and not math.isfinite(value) else value for key, value in summary.items()}
