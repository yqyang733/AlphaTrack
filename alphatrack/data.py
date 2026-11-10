"""Validated NAV adapter: geometric total return, explicit corporate actions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def load_data(path):
    path = Path(path)
    df = pd.read_csv(path, dtype={"code": str})
    if not {"date", "nav"}.issubset(df):
        raise ValueError("CSV requires date,nav; optional dividend,split columns")
    df["date"] = pd.to_datetime(df.date, errors="raise")
    if df.date.duplicated().any() or not df.date.is_monotonic_increasing:
        raise ValueError("NAV dates must be unique and increasing")
    if df.empty or df.date.isna().any():
        raise ValueError("Empty data or missing dates")
    for col, default in (("nav", None), ("dividend", 0.0), ("split", 1.0)):
        if col not in df:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all() or (df[col] < 0).any():
            raise ValueError(f"Invalid {col}")
    if (df.nav <= 0).any() or (df.split <= 0).any():
        raise ValueError("NAV and split must be positive")
    if df.iloc[0].dividend != 0 or df.iloc[0].split != 1:
        raise ValueError("First row must precede corporate actions")
    # Split occurs first; dividend per post-split share, reinvest at ex-date NAV.
    # Accumulate share-changing events only. Daily chained NAV ratios introduce
    # rounding drift that can turn an exact repeated low (position=0) positive.
    shares = 1.0 / float(df.nav.iloc[0])
    values = []
    for row in df.itertuples():
        shares *= row.split
        shares += shares * row.dividend / row.nav
        values.append(shares * row.nav)
    df["total_return_index"] = values
    meta_path = path.with_suffix(".metadata.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return df, meta
