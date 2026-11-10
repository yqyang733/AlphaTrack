"""Window comparisons preserve missing years, partial periods, and independent starts."""
import json
import re

import pandas as pd
import pytest

from alphatrack.window_report import render_window_report


def _run(tmp_path, years, profits):
    start = f"{2026-years}-11-10"
    path = tmp_path / f"window_{years}y.html"
    pd.DataFrame({"date": [start, "2026-11-06"], "cumulative_pnl": [0., sum(profits)]}).to_csv(
        path.with_suffix(".daily.csv"), index=False)
    return dict(years=years, report_path=path, payload=dict(
        summary=dict(start_date=start, end_date="2026-11-06", beginning_assets=0.,
                     total_pnl=sum(profits), total_contributions=100., total_withdrawals=20.,
                     final_equity=80.+sum(profits), xirr=None, cagr=.03, max_drawdown=-.05),
        metadata=dict(start_date=start, end_date="2026-11-06", source_commit="test<commit>"),
        annual=[dict(year=2026-years+i, profit=p, deposits=100. if i == 0 else 0.,
                     withdrawals=20. if i == len(profits)-1 else 0.) for i,p in enumerate(profits)]))


def test_window_report_preserves_absent_zero_and_partial_calendar_years(tmp_path):
    runs = [_run(tmp_path, 1, [0., 12.]), _run(tmp_path, 3, [1., -2., 3., 14.])]
    output = render_window_report(runs, tmp_path / "comparison.html", as_of="2026-11-10", effective_end="2026-11-06")
    document = output.read_text()
    matrix = json.loads(re.search(r'id="window-annual-data">(.*?)</script>', document, re.S).group(1))
    assert matrix[0]["year"] == 2023
    assert matrix[0]["cells"][0] is None
    assert matrix[0]["cells"][1]["partial"]
    assert matrix[0]["cells"][1]["start"] == "2023-11-10"
    assert matrix[2]["cells"][0]["profit"] == 0.
    assert matrix[2]["cells"][0]["partial"]
    assert not matrix[2]["cells"][1]["partial"]
    assert matrix[-1]["cells"][0]["end"] == "2026-11-06"
    assert all(cell["partial"] for cell in matrix[-1]["cells"])
    assert "test&lt;commit&gt;" in document
    assert 'href="window_1y.html"' in document
    assert '<script src=' not in document
    assert all(f'id="window-{name}-chart"' in document for name in ["annual", "pnl"])
    annual = pd.read_csv(output.with_suffix(".annual.csv"))
    assert annual.groupby("window_years").profit.sum().to_dict() == {1: 12., 3: 16.}


def test_window_report_rejects_inherited_assets_or_wrong_anchor(tmp_path):
    run = _run(tmp_path, 1, [0., 12.])
    run["payload"]["summary"]["beginning_assets"] = 100.
    with pytest.raises(ValueError, match="inherited"):
        render_window_report([run], tmp_path / "bad.html", as_of="2026-11-10", effective_end="2026-11-06")
    run["payload"]["summary"]["beginning_assets"] = 0.
    with pytest.raises(ValueError, match="date range"):
        render_window_report([run], tmp_path / "bad.html", as_of="2026-11-09", effective_end="2026-11-06")


def _fixed_run(tmp_path):
    run = _run(tmp_path, 10, [1.] * 11)
    path = tmp_path / "since_20160101.html"
    pd.DataFrame({"date": ["2016-01-01", "2026-11-06"], "cumulative_pnl": [0., 11.]}).to_csv(
        path.with_suffix(".daily.csv"), index=False)
    run.update(window_id="since_20160101", kind="fixed", years=None, label="2016-01-01 至今",
               start="2016-01-01", end="2026-11-06", report_path=path)
    run["payload"]["summary"]["start_date"] = "2016-01-01"
    run["payload"]["metadata"]["start_date"] = "2016-01-01"
    return run


def test_fixed_and_trailing_windows_keep_input_order_and_distinct_coverage(tmp_path):
    fixed = _fixed_run(tmp_path)
    trailing = _run(tmp_path, 10, [2.] * 11)
    trailing.update(window_id="10y", kind="trailing", label="过去 10 年",
                    start="2016-11-10", end="2026-11-06")
    output = render_window_report([fixed, trailing], tmp_path / "mixed.html",
                                  as_of="2026-11-10", effective_end="2026-11-06")
    document = output.read_text()
    matrix = json.loads(re.search(r'id="window-annual-data">(.*?)</script>', document, re.S).group(1))
    assert matrix[0]["cells"][0]["start"] == "2016-01-01"
    assert not matrix[0]["cells"][0]["partial"]
    assert matrix[0]["cells"][1]["start"] == "2016-11-10"
    assert matrix[0]["cells"][1]["partial"]
    annual = pd.read_csv(output.with_suffix(".annual.csv"))
    summary = pd.read_csv(output.with_suffix(".summary.csv"))
    assert summary.window_id.tolist() == ["since_20160101", "10y"]
    assert summary.window_kind.tolist() == ["fixed", "trailing"]
    assert pd.isna(summary.window_years.iloc[0])
    assert summary.window_years.iloc[1] == 10
    assert annual.groupby("window_id").profit.sum().to_dict() == {"10y": 22., "since_20160101": 11.}
    assert "2016-01-01 至今" in document and "固定起点" in document
    assert "None 年" not in document


def test_fixed_window_rejects_mismatched_start_and_duplicate_window_id(tmp_path):
    fixed = _fixed_run(tmp_path)
    fixed["start"] = "2016-02-01"
    with pytest.raises(ValueError, match="date range"):
        render_window_report([fixed], tmp_path / "bad.html", as_of="2026-11-10", effective_end="2026-11-06")
    fixed["start"] = "2016-01-01"
    trailing = _run(tmp_path, 1, [0., 12.])
    trailing["window_id"] = "since_20160101"
    with pytest.raises(ValueError, match="IDs must be unique"):
        render_window_report([fixed, trailing], tmp_path / "bad.html", as_of="2026-11-10", effective_end="2026-11-06")
