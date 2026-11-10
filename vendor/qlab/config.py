# ============================================================================
# 文件: config.py
# ============================================================================
"""配置管理模块"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
from trading.config import TradingConfig


@dataclass
class SystemConfig:
    """系统配置类"""
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)
    data_dir: Path = field(init=False)
    dividend_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    nav_cache_dir: Path = field(init=False)
    trading_dir: Path = field(init=False)
    trading: TradingConfig = field(default_factory=TradingConfig)
    offline: bool = False
    
    input_csv: str = "Fund"
    output_html: Dict[str, str] = field(default_factory=lambda: {
        "Fundb": "Fundb.html",
        "Fundc": "Fundc.html", 
        "Fundd": "Fundd.html"
    })
    
    # Fundb 策略参数（纯债）
    fundb_ma_windows: List[int] = field(default_factory=lambda: [5, 10, 30, 365])
    fundb_ma_weights: Dict[int, float] = field(default_factory=lambda: {
        5: 0.1, 10: 0.2, 30: 0.3, 365: 0.4,
    })
    
    # Fundc 策略参数（混债）
    fundc_ma_windows: List[int] = field(default_factory=lambda: [10, 30, 90, 365])
    fundc_ma_weights: Dict[int, float] = field(default_factory=lambda: {
        10: 0.1, 30: 0.2, 90: 0.3, 365: 0.4,
    })

    # Fundf 策略参数（场外红利低波）
    fundf_ma_windows: List[int] = field(default_factory=lambda: [10, 30, 90, 365])
    fundf_ma_weights: Dict[int, float] = field(default_factory=lambda: {
        10: 0.1, 30: 0.2, 90: 0.3, 365: 0.4,
    })
    
    # 收益率计算周期
    return_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    
    # Fundd 百分位计算周期
    percentile_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    
    def __post_init__(self):
        self.data_dir = self.project_root / "data_files"
        self.dividend_dir = self.data_dir / "dividend"
        self.nav_cache_dir = self.data_dir / "nav_cache"
        self.results_dir = self.project_root / "out"
        self.trading_dir = self.data_dir / "trading"
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要目录存在"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dividend_dir.mkdir(parents=True, exist_ok=True)
        (self.dividend_dir / "fund").mkdir(parents=True, exist_ok=True)
        self.nav_cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def get_input_path(self) -> Path:
        return self.data_dir / self.input_csv
    
    def get_dividend_path(self, symbol: str) -> Path:
        from data.cache import fund_identity
        return self.dividend_dir / f"{fund_identity(symbol)}.csv"
    
    def get_output_path(self, asset_type: str) -> Path:
        filename = self.output_html.get(asset_type, f"{asset_type}.html")
        return self.results_dir / filename
    
    def get_ma_params(self, asset_type: str):
        """根据资产类型获取均值回归参数"""
        if asset_type == "Fundb":
            return self.fundb_ma_windows, self.fundb_ma_weights
        elif asset_type == "Fundc":
            return self.fundc_ma_windows, self.fundc_ma_weights
        elif asset_type == "Fundf":
            return self.fundf_ma_windows, self.fundf_ma_weights
