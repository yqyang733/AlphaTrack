"""Validated, user-editable trading parameters; percentage thresholds use 0–100."""

from dataclasses import asdict, dataclass, fields
from fractions import Fraction
import json
import math
from pathlib import Path


def resolve_config_path(asset_type, override=None):
    """Explicit overrides win; per-strategy defaults are relative to the repo."""
    if override is not None:
        return Path(override).resolve()
    if asset_type in ("Fundf_1", "Fundg"):
        return Path(__file__).resolve().parents[1] / "configs" / "trading" / f"{asset_type}.json"
    return None


def numeric(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数值")
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} 必须是{'非负' if allow_zero else '大于0的'}有限数值")
    return float(value)


@dataclass(frozen=True)
class SellLevel:
    threshold: float
    fraction: str

    def __post_init__(self):
        threshold = numeric(self.threshold, "卖出阈值", allow_zero=True)
        if threshold > 100:
            raise ValueError("卖出阈值不能超过100%，100%以上由用户自行判断")
        if isinstance(self.fraction, bool):
            raise ValueError("卖出比例必须是0到1之间的分数，例如 1/10")
        try:
            ratio = Fraction(str(self.fraction))
        except (ValueError, ZeroDivisionError):
            raise ValueError("卖出比例必须是有效分数或小数，例如 1/10 或 0.1") from None
        if not 0 < ratio <= 1:
            raise ValueError("卖出比例必须大于0且不超过1")
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "fraction", str(ratio))


def increasing_sell_levels(start=50, end=100, count=50, total="9/10"):
    """Equal threshold spacing; original-share weights 1..N, not equal ratios.

    Fractions stay exact: multiplying all remaining-share ratios leaves 1-total.
    Only thresholds are converted to float, matching the existing rule engine.
    """
    start = numeric(start, "卖出起始阈值", allow_zero=True)
    end = numeric(end, "卖出结束阈值")
    if not 0 <= start < end <= 100:
        raise ValueError("卖出区间必须满足 0 ≤ 起点 < 终点 ≤ 100")
    if type(count) is not int or not 2 <= count <= 1000:
        raise ValueError("卖出档数必须是2到1000的整数")
    target = Fraction(SellLevel(0, total).fraction)
    unit = target / (count * (count + 1) // 2)
    remaining = Fraction(1)
    levels = []
    first, last = Fraction(str(start)), Fraction(str(end))
    for index in range(1, count + 1):
        allocation = unit * index
        threshold = float(first + (last - first) * Fraction(index - 1, count - 1))
        levels.append(SellLevel(threshold, str(allocation / remaining)))
        remaining -= allocation
    return tuple(levels)


DEFAULT_SELL_LEVELS = increasing_sell_levels()


@dataclass(frozen=True)
class TradingConfig:
    fundb_top_n: int = 8
    fundb_buy_amount: float = 40.0
    fundc_top_n: int = 5
    fundc_buy_amount: float = 40.0
    fundf1_base_amount: float = 40.0
    fundg_base_amount: float = 10.0
    fundd_base_amount: float = 10.0
    fundd_broad_base_amount: float = 50.0
    fundf1_buy_thresholds: tuple = (60, 50, 40, 30)
    fundf1_buy_multipliers: tuple = (1, 2, 3, 4)
    fundg_buy_thresholds: tuple = (60, 50, 40, 30)
    fundg_buy_multipliers: tuple = (1, 1, 1, 1)
    fundd_buy_thresholds: tuple = (20, 10, 5, 0)
    fundd_buy_multipliers: tuple = (1, 2, 3, 4)
    fundd_sell_start: float = 50
    fundd_sell_end: float = 100
    fundd_sell_count: int = 50
    fundd_sell_total: str = "9/10"
    # Explicit levels override the generator, including frozen legacy records.
    fundd_sell_levels: tuple | None = None

    def __post_init__(self):
        for name in ("fundb_top_n", "fundb_buy_amount", "fundc_top_n", "fundc_buy_amount",
                     "fundf1_base_amount", "fundg_base_amount", "fundd_base_amount", "fundd_broad_base_amount"):
            value = getattr(self, name)
            numeric(value, name)
            if name.endswith("top_n") and not isinstance(value, int):
                raise ValueError(f"{name} 必须是正整数")
        for prefix in ("fundf1", "fundg", "fundd"):
            thresholds = getattr(self, prefix + "_buy_thresholds")
            multiples = getattr(self, prefix + "_buy_multipliers")
            if not isinstance(thresholds, (list, tuple)) or not isinstance(multiples, (list, tuple)):
                raise ValueError(f"{prefix} 买入阈值和倍数必须是数组")
            if len(thresholds) < 2 or len(thresholds) != len(multiples):
                raise ValueError(f"{prefix} 买入阈值和倍数数量必须相同且至少有2档")
            thresholds = tuple(numeric(v, prefix + "买入阈值", allow_zero=True) for v in thresholds)
            multiples = tuple(numeric(v, prefix + "买入倍数") for v in multiples)
            if thresholds[0] > 100 or any(a <= b for a, b in zip(thresholds, thresholds[1:])):
                raise ValueError(f"{prefix} 买入阈值必须在0到100之间且严格递减")
            object.__setattr__(self, prefix + "_buy_thresholds", thresholds)
            object.__setattr__(self, prefix + "_buy_multipliers", multiples)
        generated = increasing_sell_levels(self.fundd_sell_start, self.fundd_sell_end,
                                           self.fundd_sell_count, self.fundd_sell_total)
        object.__setattr__(self, "fundd_sell_total", SellLevel(0, self.fundd_sell_total).fraction)
        levels = generated if self.fundd_sell_levels is None else self.fundd_sell_levels
        if not isinstance(levels, (list, tuple)) or not levels:
            raise ValueError("fundd_sell_levels 必须是非空数组")
        normalized = []
        for level in levels:
            if isinstance(level, SellLevel):
                normalized.append(level)
            elif isinstance(level, dict) and set(level) == {"threshold", "fraction"}:
                normalized.append(SellLevel(**level))
            else:
                raise ValueError("每个卖出档位必须包含 threshold 和 fraction")
        if any(a.threshold >= b.threshold for a, b in zip(normalized, normalized[1:])):
            raise ValueError("卖出阈值必须严格递增，不能重复")
        object.__setattr__(self, "fundd_sell_levels", tuple(normalized))

    @classmethod
    def from_json(cls, path: Path, *, asset_type=None):
        with Path(path).open(encoding="utf-8-sig") as stream:
            values = json.load(stream)
        if not isinstance(values, dict):
            raise ValueError("交易配置必须是 JSON 对象")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"未知交易配置项: {', '.join(sorted(unknown))}")
        if asset_type == "Fundg" and any(not key.startswith("fundg_") for key in values):
            raise ValueError("Fundg 必须使用独立配置文件，只允许 fundg_* 参数，不再读取 fundf1_* 参数")
        if asset_type == "Fundf_1" and any(key.startswith("fundg_") for key in values):
            raise ValueError("Fundf_1 不能加载 Fundg 参数，请使用 Fundf_1.json 或原有 fundf1_* 配置")
        return cls(**values)

    def to_dict(self):
        return asdict(self)
