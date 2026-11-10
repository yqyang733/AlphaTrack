# ============================================================================
# 文件: core/calculator.py
# ============================================================================
"""收益率计算模块"""

import pandas as pd
import numpy as np
from typing import Optional, List


class ReturnCalculator:
    """收益率计算器"""
    
    @staticmethod
    def calculate_return_with_dividend_reinvest(
        nav_df: pd.DataFrame,
        dividend_df: pd.DataFrame,
        initial_investment: float = 1.0
    ) -> pd.DataFrame:
        """
        计算考虑分红再投资的累计收益率（支持现金分红和份额折算）

        使用 numpy 数组 + dict 查找替代 iterrows + DataFrame 过滤，
        避免每行都做一次 O(m) 的 DataFrame 布尔索引。
        """
        df = nav_df.copy()
        dates = df["净值日期"].values
        navs = df["单位净值"].values
        n = len(df)

        initial_nav = navs[0]
        shares = initial_investment / initial_nav

        # 按日期(天精度整数)建分红查找表，O(1) 查找
        div_lookup = {}
        if dividend_df is not None and len(dividend_df) > 0:
            d_dates = dividend_df["分红日"].values.astype("datetime64[D]").astype("int64")
            d_types = dividend_df["分红类型"].values
            d_vals = dividend_df["分红"].values
            for k in range(len(dividend_df)):
                key = d_dates[k]
                if key not in div_lookup:
                    div_lookup[key] = []
                div_lookup[key].append((str(d_types[k]), str(d_vals[k])))

        dates_int = dates.astype("datetime64[D]").astype("int64")

        cumulative_returns = np.empty(n)
        shares_history = np.empty(n)

        for i in range(n):
            current_nav = navs[i]

            divs = div_lookup.get(dates_int[i])
            if divs is not None:
                for div_type, div_value in divs:
                    if div_type == "现金分红":
                        cash_per_share = float(div_value)
                        shares += shares * cash_per_share / current_nav
                    elif div_type == "份额折算":
                        if ":" in div_value:
                            shares *= float(div_value.split(":")[1])

            current_value = shares * current_nav
            cumulative_returns[i] = (current_value - initial_investment) / initial_investment
            shares_history[i] = shares

        df["累计收益率"] = cumulative_returns
        df["持有份额"] = shares_history

        return df
    
    @staticmethod
    def calculate_annualized_return(
        return_df: pd.DataFrame,
        years: Optional[int] = None
    ) -> float:
        """计算年化收益率"""
        if return_df is None or len(return_df) == 0:
            return np.nan
        
        df = return_df.sort_values("净值日期").copy()
        
        end_date = df["净值日期"].iloc[-1]
        end_return = df["累计收益率"].iloc[-1]
        
        if years is not None:
            target_date = end_date - pd.DateOffset(years=years)
            mask = df["净值日期"] <= target_date
            df_past = df[mask]
            
            if len(df_past) == 0:
                return np.nan
            
            start_return = df_past["累计收益率"].iloc[-1]
            actual_years = float(years)
        else:
            start_date = df["净值日期"].iloc[0]
            start_return = df["累计收益率"].iloc[0]
            
            days_diff = (end_date - start_date).days
            if days_diff == 0:
                return np.nan
            actual_years = days_diff / 365.25
        
        if actual_years == 0:
            return np.nan
        
        annualized_return = (end_return - start_return) / (1 + start_return) / actual_years
        
        return annualized_return


class PercentileCalculator:
    """百分位计算器（用于 Fundd）"""

    @staticmethod
    def calculate_percentile_position(
        price_df: pd.DataFrame,
        periods: List[int]
    ) -> pd.DataFrame:
        """
        计算当前价格在历史区间的百分位位置

        使用 numpy + searchsorted 替代逐行 pandas 过滤，
        对 3000+ 行数据可提速 10-50 倍。

        Args:
            price_df: 价格数据，需包含 '净值日期' 和 '单位净值'
            periods: 回溯周期列表（年）

        Returns:
            添加百分位列的DataFrame
        """
        df = price_df.copy().sort_values("净值日期").reset_index(drop=True)

        dates = df["净值日期"].values
        prices = df["单位净值"].values
        n = len(df)

        for period in periods:
            col_name = f"近{period}年百分位"
            percentiles = np.full(n, np.nan)
            offset = pd.DateOffset(years=period)

            for i in range(n):
                start_np = np.datetime64(pd.Timestamp(dates[i]) - offset)
                j = np.searchsorted(dates, start_np, side="left")

                window = prices[j : i + 1]
                if len(window) < 2:
                    continue

                min_p = window.min()
                max_p = window.max()

                if max_p == min_p:
                    percentiles[i] = 0.5
                else:
                    percentiles[i] = (prices[i] - min_p) / (max_p - min_p)

            df[col_name] = percentiles

        # 成立以来百分位（expanding 向量化，O(n)）
        price_series = df["单位净值"]
        exp_min = price_series.expanding(min_periods=2).min()
        exp_max = price_series.expanding(min_periods=2).max()
        pct = (price_series - exp_min) / (exp_max - exp_min)
        df["成立以来百分位"] = pct.mask(exp_max == exp_min, 0.5)

        # 综合分数
        percentile_cols = [f"近{p}年百分位" for p in [1, 2, 3, 5]]
        df["综合分数"] = df[percentile_cols].mean(axis=1, skipna=True)

        return df