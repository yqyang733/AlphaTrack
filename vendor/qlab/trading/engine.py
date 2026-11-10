"""Pure trading rules. Amounts are CNY; sale fractions refer to current holdings."""

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from typing import Optional, Tuple

from .config import TradingConfig


FUNDD_COLUMNS = ("近5年百分位", "近10年百分位", "成立以来百分位")
SUPPORTED_TYPES = {"Fundb", "Fundc", "Fundf_1", "Fundd", "Funde", "Fundg"}
SELL_NOTE = "卖出需结合持有收益判断；比例以执行时持有份额为基数。"


@dataclass(frozen=True)
class TradeDecision:
    asset_type: str
    code: str
    name: str
    data_date: str
    action: str = "HOLD"
    amount: Optional[float] = None
    fraction: str = ""
    rank: Optional[int] = None
    score: Optional[float] = None
    buy_metric: Optional[float] = None
    sell_metric: Optional[float] = None
    threshold: Optional[float] = None
    sell_steps: tuple = ()
    used_sell_levels: tuple = ()
    reason: str = ""
    note: str = "规则建议，非实际成交记录。"

    @property
    def display(self):
        if self.action == "BUY":
            return f"+{self.amount:g} 元"
        if self.action == "SELL":
            return " → ".join(f"-{step['fraction']}" for step in self.sell_steps) + f"（合计 -{float(Fraction(self.fraction)):.2%}）"
        return "—"

    def to_dict(self):
        return asdict(self)


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class TradingEngine:
    def __init__(self, config: TradingConfig, *, fundd_rule_version=2):
        # Version 1 is used only to replay observations saved before the change
        # from the three-period maximum to the minimum. New signals use v2.
        if type(fundd_rule_version) is not int or fundd_rule_version not in (1, 2):
            raise ValueError("不支持的 Fundd 规则版本")
        self.config = config
        self.fundd_rule_version = fundd_rule_version

    def buy_multiple(self, prefix, value):
        thresholds = getattr(self.config, prefix + "_buy_thresholds")
        multiples = getattr(self.config, prefix + "_buy_multipliers")
        if value > thresholds[0] / 100:
            return None
        for index, lower in enumerate(thresholds[1:]):
            # Fundd's final boundary belongs to the deepest tier (default x<=0).
            exclusive = prefix == "fundd" and index == len(thresholds) - 2
            if value > lower / 100 or (not exclusive and value == lower / 100):
                return multiples[index]
        return multiples[-1]

    def evaluate(self, asset_type, code, name, data_date, values, *, rank=None,
                 score=None, sold_levels=()) -> Tuple[TradeDecision, tuple]:
        fields = dict(asset_type=asset_type, code=code, name=name,
                      data_date=data_date, rank=rank, score=finite(score))
        sold = tuple(sold_levels)
        if asset_type in ("Fundb", "Fundc"):
            top_n = getattr(self.config, f"{asset_type.lower()}_top_n")
            amount = getattr(self.config, f"{asset_type.lower()}_buy_amount")
            if rank is None or fields["score"] is None:
                return TradeDecision(**fields, reason="收益或评分数据不足，未参与总分排名"), sold
            reason = f"综合评分排名 {rank} / 买入前 {top_n} 名"
            if rank <= top_n:
                return TradeDecision(**fields, action="BUY", amount=amount, reason=reason), sold
            return TradeDecision(**fields, reason=reason + "，本次不买入"), sold
        if asset_type in ("Fundf_1", "Fundg"):
            prefix = "fundg" if asset_type == "Fundg" else "fundf1"
            base = getattr(self.config, prefix + "_base_amount")
            thresholds = getattr(self.config, prefix + "_buy_thresholds")
            value = finite(values.get("近1年百分位"))
            if value is None:
                return TradeDecision(**fields, reason="近1年百分位缺失，暂不生成买入建议"), sold
            fields["buy_metric"] = value
            multiple = self.buy_multiple(prefix, value)
            if multiple is None:
                return TradeDecision(**fields, reason=f"近1年 {value:.2%} > {thresholds[0]:g}%，不买入"), sold
            return TradeDecision(**fields, action="BUY", amount=base * multiple,
                                 reason=f"近1年 {value:.2%}；基数 {base:g} × {multiple}"), sold
        if asset_type in ("Fundd", "Funde"):
            metrics = [finite(values.get(column)) for column in FUNDD_COLUMNS]
            if any(value is None for value in metrics):
                return TradeDecision(**fields, reason="近5年/近10年/成立以来数据不全，保留已有卖出状态"), sold
            minimum = min(metrics)
            sell_metric = minimum if self.fundd_rule_version == 2 else max(metrics)
            sell_label = "最小值" if self.fundd_rule_version == 2 else "最大值"
            fields.update(buy_metric=minimum, sell_metric=sell_metric)
            # Preserve buy priority if user-configured buy and sell tiers overlap.
            multiple = self.buy_multiple("fundd", minimum)
            sell_levels = [(level.threshold / 100, level.fraction) for level in self.config.fundd_sell_levels]
            if multiple is not None:
                reason = f"三周期最小值 {minimum:.2%}；基数 {self.config.fundd_base_amount:g} × {multiple}；重置卖出档位"
                if sell_metric >= sell_levels[0][0]:
                    reason += "（同时满足卖出阈值，本次买入优先）"
                return TradeDecision(**fields, action="BUY", amount=self.config.fundd_base_amount * multiple,
                                     reason=reason), ()
            reached = [(level, fraction) for level, fraction in sell_levels if sell_metric >= level]
            if not reached:
                return TradeDecision(**fields, reason=f"最小值 {minimum:.2%} > {self.config.fundd_buy_thresholds[0]:g}%，{sell_label} {sell_metric:.2%} < {sell_levels[0][0]:.2%}，观望"), sold
            pending = [(level, fraction) for level, fraction in reached if level not in sold]
            if not pending:
                return TradeDecision(**fields, threshold=reached[-1][0],
                                     reason=f"{sell_label} {sell_metric:.2%}；已达到的档位本轮均已使用，尚未重新进入买入区间，不重复卖出",
                                     note=SELL_NOTE + " 规则建议，非成交记录。"), sold
            remaining = Fraction(1)
            steps = []
            for level, fraction in pending:
                steps.append({"threshold": level, "fraction": fraction})
                remaining *= 1 - Fraction(fraction)
            updated = tuple(sorted(set(sold) | {item[0] for item in reached}))
            return TradeDecision(**fields, action="SELL", fraction=str(1 - remaining), threshold=pending[-1][0],
                                 sell_steps=tuple(steps),
                                 reason=f"三周期{sell_label} {sell_metric:.2%}，依次触发 " + "、".join(f"{level * 100:g}%" for level, _ in pending),
                                 note=SELL_NOTE + " 规则建议，非成交记录。"), updated
        return TradeDecision(**fields, reason="此资产类型未配置买卖规则"), sold
