"""QLab Fundd v2, adapted as a stateful signal plugin, without execution logic.

Source: yqyang733/QLab, commit 5455e50154d994eee157780cd300c03214d1fce7,
trading/{config,engine,service}.py and core/calculator.py. The Fundd rules also
match locally cached origin/main 841293e3b74af3a3ab04c1bad9c00868243aa4f8.
The so-called percentiles are min/max price positions, not quantile ranks.
"""

from fractions import Fraction
import math

import numpy as np
import pandas as pd


def _number(value, name, *, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} outside permitted range")
    return float(value)


class FunddStrategy:
    """Make one decision per observed NAV date; instantiate anew for each run.

    Buy resets the sell ladder when the *signal* occurs (as in QLab), even if
    an execution layer later rejects the order. An order is not a fill.
    Default base 50 CNY matches 110020's broad-index classification.
    """

    def __init__(self, *, base_amount=50, buy_thresholds=(20, 10, 5, 0),
                 buy_multipliers=(1, 2, 3, 4), sell_start=50, sell_end=100,
                 sell_count=50, sell_total=0.9):
        self.base_amount = _number(base_amount, "base_amount")
        if self.base_amount == 0:
            raise ValueError("base_amount must be positive")
        if not isinstance(buy_thresholds, (list, tuple)) or not isinstance(buy_multipliers, (list, tuple)):
            raise ValueError("buy thresholds and multipliers must be arrays")
        if len(buy_thresholds) < 2 or len(buy_thresholds) != len(buy_multipliers):
            raise ValueError("buy thresholds and multipliers need equal lengths >= 2")
        self.buy_thresholds = tuple(_number(v, "buy_threshold", maximum=100) for v in buy_thresholds)
        self.buy_multipliers = tuple(_number(v, "buy_multiplier") for v in buy_multipliers)
        if any(a <= b for a, b in zip(self.buy_thresholds, self.buy_thresholds[1:])):
            raise ValueError("buy thresholds must be strictly decreasing")
        if any(v == 0 for v in self.buy_multipliers):
            raise ValueError("buy multipliers must be positive")
        start = _number(sell_start, "sell_start", maximum=100)
        end = _number(sell_end, "sell_end", maximum=100)
        if start >= end:
            raise ValueError("sell_start must be less than sell_end")
        if type(sell_count) is not int or not 2 <= sell_count <= 1000:
            raise ValueError("sell_count must be an integer from 2 to 1000")
        if isinstance(sell_total, bool):
            raise ValueError("sell_total must be a number or fraction in (0, 1]")
        try:
            total = Fraction(str(sell_total))
        except (ValueError, ZeroDivisionError):
            raise ValueError("sell_total must be a number or fraction in (0, 1]") from None
        if not 0 < total <= 1:
            raise ValueError("sell_total must be in (0, 1]")
        unit = total / (sell_count * (sell_count + 1) // 2)
        remaining = Fraction(1)
        self.sell_levels = []
        for i in range(1, sell_count + 1):
            threshold = float(Fraction(str(start)) + (Fraction(str(end)) - Fraction(str(start))) * Fraction(i - 1, sell_count - 1))
            allocation = unit * i
            self.sell_levels.append((threshold, allocation / remaining))
            remaining -= allocation
        self.sold_levels = set()

    def prepare(self, frame):
        """Compute causal positions using all available history, including today.

        Windows use calendar years with their left boundary included. Before a
        full 5/10 years exists, use the available prefix. At least two points
        are needed; an entirely flat window gets 0.5, matching QLab.
        """
        required = {"date", "total_return_index"}
        if not required.issubset(frame.columns):
            raise ValueError(f"prepare requires {sorted(required)}")
        result = frame.copy()
        dates = pd.to_datetime(result["date"], errors="raise")
        values = np.asarray(result["total_return_index"], dtype=float)
        if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
            raise ValueError("dates must be valid, unique and increasing")
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("total_return_index must be finite and positive")
        date_array = dates.to_numpy()
        for years in (5, 10):
            positions = np.full(len(result), np.nan)
            for i, date in enumerate(dates):
                start = np.datetime64(date - pd.DateOffset(years=years))
                j = np.searchsorted(date_array, start, side="left")
                window = values[j:i + 1]
                if len(window) >= 2:
                    low, high = window.min(), window.max()
                    positions[i] = .5 if high == low else (values[i] - low) / (high - low)
            result[f"position_{years}y"] = positions
        prices = pd.Series(values, index=frame.index)
        low = prices.expanding(min_periods=2).min()
        high = prices.expanding(min_periods=2).max()
        result["position_all"] = ((prices - low) / (high - low)).mask(high == low, .5)
        result["signal_metric"] = result[["position_5y", "position_10y", "position_all"]].min(axis=1, skipna=False)
        return result

    def decide(self, row):
        """Return a serializable signal. Fractions refer to remaining holdings."""
        value = row.get("signal_metric")
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value):
            return {"action": "HOLD", "reason": "三周期位置数据不足", "levels": []}
        multiple = None
        if value <= self.buy_thresholds[0] / 100:
            multiple = self.buy_multipliers[-1]
            for i, lower in enumerate(self.buy_thresholds[1:]):
                exclusive = i == len(self.buy_thresholds) - 2
                if value > lower / 100 or (not exclusive and value == lower / 100):
                    multiple = self.buy_multipliers[i]
                    break
        if multiple is not None:
            self.sold_levels.clear()
            return {"action": "BUY", "amount": self.base_amount * multiple,
                    "reason": f"三周期最小值 {value:.2%}；基数 {self.base_amount:g} × {multiple:g}；重置卖出档位",
                    "levels": []}
        pending = [(threshold, ratio) for threshold, ratio in self.sell_levels
                   if value >= threshold / 100 and threshold not in self.sold_levels]
        if not pending:
            return {"action": "HOLD", "reason": "不在买入区间；没有新的卖出档位", "levels": []}
        remaining = Fraction(1)
        for threshold, ratio in pending:
            remaining *= 1 - ratio
            self.sold_levels.add(threshold)
        return {"action": "SELL", "fraction": float(1 - remaining),
                "reason": f"三周期最小值 {value:.2%}；新触发 {len(pending)} 个卖出档位",
                "levels": [threshold for threshold, _ in pending]}
