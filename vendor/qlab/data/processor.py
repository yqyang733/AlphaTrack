# ============================================================================
# 文件: data/processor.py
# ============================================================================
"""数据处理模块"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class DataProcessor:
    """数据处理器"""
    
    @staticmethod
    def calculate_period_average(metrics_dict: Dict[str, List], 
                                 year_columns: List[str] = None) -> List[str]:
        """计算指定周期的平均收益率"""
        if year_columns is None:
            year_columns = ["近1年", "近2年", "近3年", "近5年"]
        
        avg_values = []
        n_rows = len(next(iter(metrics_dict.values())))

        for i in range(n_rows):
            vals = []
            for col in year_columns:
                if col in metrics_dict:
                    try:
                        v = float(metrics_dict[col][i])
                        vals.append(v)
                    except:
                        pass
            
            if len(vals) == len(year_columns):
                avg_values.append(f"{sum(vals)/len(vals):.4f}")
            else:
                avg_values.append("N/A")
        
        return avg_values
    
    @staticmethod
    def prepare_score_analysis_data(
        metrics_dict: Dict[str, List],
        results_dict: Dict[Tuple[str, str], Dict],
        year_columns: List[str] = None
    ) -> pd.DataFrame:
        """准备综合评分分析数据"""
        if year_columns is None:
            year_columns = ["近1年", "近2年", "近3年", "近5年"]
        
        records = []

        for (code, name, categray), result in results_dict.items():
            try:
                idx = metrics_dict["基金代码"].index(code)
            except ValueError:
                continue

            vals = []
            for col in year_columns:
                if col in metrics_dict:
                    try:
                        v = float(metrics_dict[col][idx])
                        vals.append(v)
                    except:
                        pass

            if len(vals) != len(year_columns):
                continue

            avg_ret = sum(vals) / len(vals)
            mr_score = result.get("signals", {}).get("mean_reversion_score", np.nan)
            
            if np.isnan(mr_score):
                continue

            records.append({
                "code": code,
                "name": name,
                "avg_ret": avg_ret,
                "mr": mr_score
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["avg_ret_norm"] = (df["avg_ret"] - df["avg_ret"].mean()) / df["avg_ret"].std()
        df["mr_norm"] = (df["mr"] - df["mr"].mean()) / df["mr"].std()
        df["score"] = df["avg_ret_norm"] - df["mr_norm"]
        df = df.sort_values("score", ascending=False)

        return df