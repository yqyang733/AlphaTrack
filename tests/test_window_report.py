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
