"""FIFO purchase-lot attribution, with annualization only during ownership.

A realized row is one purchase lot matched to one sale, so a partially sold lot
can have several realized rows and one terminal unrealized row. Reinvested
cash dividends and splits stay with their original lot and preserve its cost.
Annualized medians describe rows; they are not portfolio returns or XIRRs.
"""
from collections import deque
import math

import numpy as np
import pandas as pd


LOT_COLUMNS = [
    "lot_id", "buy_date", "sell_date", "valuation_date", "status", "shares",
    "holding_days", "cost", "proceeds", "pnl", "holding_return",
    "annualized_return", "annualization_note",
]


def _annualize(cost, proceeds, days, year_days):
    if days <= 0:
        return np.nan, "同日买入估值或卖出，持有不足一天，不计算年化"
    if proceeds == 0:
        return -1.0, "持有不足30天，年化对短期波动敏感" if days < 30 else ""
    # Subtract logs so extreme finite amounts do not overflow their ratio.
    exponent = (math.log(proceeds) - math.log(cost)) * year_days / days
    try:
        result = math.expm1(exponent)
    except OverflowError:
        return np.nan, "年化结果超出数值范围；请查看持有收益率"
    if not math.isfinite(result):
        return np.nan, "年化结果超出数值范围；请查看持有收益率"
    return result, "持有不足30天，年化对短期波动敏感" if days < 30 else ""


def analyze_lots(daily, trades, year_days=365.0):
    """Return matched chunks and summary, excluding cash after every sale.

    ``proceeds`` is net sale cash for REALIZED rows and terminal marked value
    for UNREALIZED rows. Open shares do not assume a hypothetical redemption
    fee. Purchase costs include subscription fees. Dates use actual days and
    a configurable year basis (default ACT/365, matching Excel's convention).
    """
    if isinstance(year_days, bool) or not isinstance(year_days, (int, float)) or not math.isfinite(year_days) or year_days <= 0:
        raise ValueError("year_days must be positive and finite")
    if daily.empty or not {"date", "nav"}.issubset(daily):
        raise ValueError("Nonempty daily date/nav history is required")
    history = daily.copy()
    history["date"] = pd.to_datetime(history.date, errors="raise")
    if history.date.isna().any() or history.date.duplicated().any() or not history.date.is_monotonic_increasing:
        raise ValueError("Daily dates must be unique, nonmissing and increasing")
    if not np.isfinite(history.nav).all() or (history.nav <= 0).any():
        raise ValueError("NAV must be positive and finite")
    if trades.empty:
        fills = trades.copy()
        grouped = {}
    else:
        required = {"status", "action", "date", "shares", "amount", "fee"}
        if not required.issubset(trades):
            raise ValueError(f"Trade columns missing: {sorted(required-set(trades))}")
        fills = trades[trades.status.isin(["FILLED", "PARTIAL"])].copy()
        fills["date"] = pd.to_datetime(fills.date, errors="raise")
        if fills.date.isna().any() or not fills.date.isin(history.date).all():
            raise ValueError("Filled trade date must occur in NAV history")
        if not fills.date.is_monotonic_increasing:
            raise ValueError("Filled trades must be in execution order")
        grouped = {date: rows for date, rows in fills.groupby("date", sort=False)}
    queue, records = deque(), []
    bought_cost, sold_net, next_id = 0.0, 0.0, 1

    def record(lot, shares, cost, proceeds, date, status):
        days = (date-lot["buy_date"]).days
        if cost <= 0:
            raise ValueError("Matched purchase cost must be positive")
        annual, note = _annualize(cost, proceeds, days, year_days)
        records.append(dict(lot_id=lot["lot_id"], buy_date=lot["buy_date"],
                            sell_date=date if status == "REALIZED" else pd.NaT,
                            valuation_date=date, status=status, shares=shares,
                            holding_days=days, cost=cost, proceeds=proceeds,
                            pnl=proceeds-cost, holding_return=proceeds/cost-1,
                            annualized_return=annual, annualization_note=note))

    for row in history.itertuples(index=False):
        split, dividend = float(getattr(row, "split", 1.0)), float(getattr(row, "dividend", 0.0))
        if not math.isfinite(split) or split <= 0 or not math.isfinite(dividend) or dividend < 0:
            raise ValueError("Invalid corporate action")
        for lot in queue:
            lot["shares"] *= split
            lot["shares"] += lot["shares"] * dividend / row.nav
        day_trades = grouped.get(row.date)
        if day_trades is None:
            continue
        for trade in day_trades.itertuples(index=False):
            qty, amount, fee = float(trade.shares), float(trade.amount), float(trade.fee)
            if not all(math.isfinite(v) for v in (qty, amount, fee)) or qty <= 0 or amount <= 0 or not 0 <= fee <= amount:
                raise ValueError("Filled trade has invalid quantity, amount or fee")
            if trade.action == "BUY":
                queue.append(dict(lot_id=f"L{next_id:05d}", buy_date=row.date, shares=qty, cost=amount))
                next_id += 1
                bought_cost += amount
            elif trade.action == "SELL":
                available = math.fsum(lot["shares"] for lot in queue)
                tolerance = max(1e-10, available*1e-12)
                if qty > available+tolerance:
                    raise ValueError("Sale exceeds available FIFO holdings")
                if math.isclose(qty, available, rel_tol=1e-12, abs_tol=1e-10):
                    qty = available
                if qty <= 0:
                    raise ValueError("Cannot sell without a purchase lot")
                net, remaining = amount-fee, qty
                while remaining > 0 and queue:
                    lot = queue[0]
                    matched = min(lot["shares"], remaining)
                    cost = lot["cost"] * (matched/lot["shares"])
                    proceeds = net * (matched/qty)
                    record(lot, matched, cost, proceeds, row.date, "REALIZED")
                    remaining -= matched
                    if matched == lot["shares"]:
                        queue.popleft()
                    else:
                        lot["shares"] -= matched
                        lot["cost"] -= cost
                if remaining > tolerance:
                    raise ValueError("FIFO quantity matching failed")
                sold_net += net
            else:
                raise ValueError(f"Unsupported filled action: {trade.action}")

    final_date, final_nav = history.iloc[-1].date, float(history.iloc[-1].nav)
    for lot in queue:
        record(lot, lot["shares"], lot["cost"], lot["shares"]*final_nav, final_date, "UNREALIZED")
    result = pd.DataFrame(records, columns=LOT_COLUMNS)
    for column in ("buy_date", "sell_date", "valuation_date"):
        result[column] = pd.to_datetime(result[column])
    closed, opened = result[result.status == "REALIZED"], result[result.status == "UNREALIZED"]
    def total(part, column):
        return float(part[column].sum()) if len(part) else 0.0
    def median(part):
        valid = pd.to_numeric(part.annualized_return, errors="coerce").dropna()
        return float(valid.median()) if len(valid) else None
    realized_cost, open_cost = total(closed, "cost"), total(opened, "cost")
    open_value = total(opened, "proceeds")
    if not math.isclose(realized_cost+open_cost, bought_cost, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError("FIFO cost conservation failed")
    if not math.isclose(total(closed, "proceeds"), sold_net, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError("FIFO sale proceeds conservation failed")
    total_pnl = total(result, "pnl")
    if not math.isclose(total_pnl, sold_net+open_value-bought_cost, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError("FIFO P&L conservation failed")
    # If a ledger is present, also verify that all terminal shares were matched.
    if "shares" in history and not math.isclose(total(opened, "shares"), float(history.iloc[-1].shares), rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError("FIFO terminal holdings differ from execution ledger")
    summary = dict(matching_method="FIFO", year_days=float(year_days),
                   realized_count=len(closed), open_count=len(opened), buy_lot_count=next_id-1,
                   realized_pnl=total(closed,"pnl"), unrealized_pnl=total(opened,"pnl"), total_pnl=total_pnl,
                   realized_cost=realized_cost, open_cost=open_cost,
                   realized_proceeds=total(closed,"proceeds"), open_value=open_value,
                   positive_realized_cost_share=total(closed[closed.pnl>0],"cost")/realized_cost if realized_cost else None,
                   realized_annualized_median=median(closed), open_annualized_median=median(opened),
                   min_hold_days=int(result.holding_days.min()) if len(result) else None,
                   short_hold_count=int((result.holding_days<30).sum()),
                   undefined_annualized_count=int(result.annualized_return.isna().sum()))
    return result, summary
