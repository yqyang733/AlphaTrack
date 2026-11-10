# ============================================================================
# 文件: strategy/mean_reversion.py
# ============================================================================
"""均值回归策略"""

import pandas as pd
import numpy as np
from typing import List, Dict


class MeanReversionStrategy:
    """均值回归策略"""
    
    def __init__(self, windows: List[int], weights: Dict[int, float]):
        self.windows = windows
        self.weights = weights
    
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算均值回归信号"""
        result_df = df.copy()
        
        for w in self.windows:
            result_df[f"ma_{w}"] = result_df["累计收益率"].rolling(w).mean()
            result_df[f"std_{w}"] = result_df["累计收益率"].rolling(w).std()
            
            with np.errstate(divide='ignore', invalid='ignore'):
                result_df[f"z_{w}"] = (
                    (result_df["累计收益率"] - result_df[f"ma_{w}"]) / 
                    result_df[f"std_{w}"]
                )
        
        result_df["mean_reversion_score"] = 0.0
        for w in self.windows:
            weight = self.weights.get(w, 0)
            result_df["mean_reversion_score"] += weight * result_df[f"z_{w}"].fillna(0)
        
        return result_df