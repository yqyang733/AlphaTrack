"""Consume the existing score calculation without changing its formula."""

import pandas as pd

from data.processor import DataProcessor
from .engine import finite


def prepare_scores(metrics, results):
    if not results or not metrics.get("基金代码"):
        return pd.DataFrame()
    # Keep invalid observations out of the new trading layer, rather than
    # changing the existing analyzer/calculator/processor implementations.
    valid = {}
    for key, result in results.items():
        try:
            index = metrics["基金代码"].index(key[0])
            returns = [finite(metrics[f"近{years}年"][index]) for years in (1, 2, 3, 5)]
        except (KeyError, IndexError, ValueError):
            continue
        mr = finite(result.get("signals", {}).get("mean_reversion_score"))
        if all(value is not None for value in returns) and mr is not None:
            valid[key] = {**result, "signals": {**result.get("signals", {}), "mean_reversion_score": mr}}
    if not valid:
        return pd.DataFrame()
    scores = DataProcessor.prepare_score_analysis_data(metrics, valid)
    if scores.empty:
        return scores
    scores = scores[scores["score"].map(finite).notna()]
    return scores.sort_values(["score", "code"], ascending=[False, True], kind="stable")
