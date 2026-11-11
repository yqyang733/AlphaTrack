"""Report integration contracts: annual earnings must not become withdrawals."""
import json
import re

import pandas as pd
import pytest

from alphatrack.portfolio_report import render_portfolio_report


@pytest.mark.parametrize("selection_mode", ["ranked", "random"])
def test_offline_report_preserves_calendar_earnings_and_escapes_metadata(tmp_path, selection_mode):
    daily = pd.DataFrame({
        "date": pd.to_datetime(["2020-12-31", "2021-01-04", "2021-11-09"]),
        "equity": [100., 110., 55.], "daily_pnl": [0., 10., 5.],
        "cumulative_pnl": [0., 10., 15.], "unit_nav": [1., 1.1, 1.15],
        "drawdown": [0., 0., -.01], "total_contributions": [100., 100., 100.],
        "cumulative_withdrawals": [0., 0., 60.],
    })
    annual = pd.DataFrame([dict(year=2021, start_date="2021-06-01", end_date="2021-11-09", is_partial_year=True, beginning_assets=100.,
                               end_assets=55., deposits=0., withdrawals=60., profit=15.,
                               realized_pnl=10., unrealized_pnl_change=5., twr_return=.15,
                               xirr=None, xirr_status="multiple_roots", fees=0., max_drawdown=-.01)])
    funds = pd.DataFrame([dict(code="000001", name="测试 <基金>", asset_type="Fundb",
                              total_pnl=15., total_contributions=100., total_withdrawals=60.,
                              final_equity=55., first_investment_date="2020-12-31",
                              start_date="2010-01-01", first_nav_date="2009-01-05", end_date="2021-11-09")])
    summary = dict(total_pnl=15., xirr=None, cagr=.1, max_drawdown=-.01,
                   total_contributions=100., total_withdrawals=60., final_equity=55., sharpe=.5)
    path = render_portfolio_report(daily, annual, funds, funds, summary,
                                   {"title": "测试 </script>", "qlab_commit": "abc123",
                                    "strategy": {"selection_mode": selection_mode}}, tmp_path / "report.html")
    document = path.read_text()
    rows = json.loads(re.search(r"const annualRows=(.*?),allFunds=", document).group(1))
    assert rows[0]["profit"] == 15. and rows[0]["withdrawals"] == 60.
    assert rows[0]["ending_assets"] == 55. and rows[0]["is_partial"]
    assert rows[0]['period'] == '2021（06/01–11/09）'
    assert rows[0]["xirr"] is None
    assert rows[0]["unrealized_change"] == 5.
    members = json.loads(re.search(r",allFunds=(.*?),groupRows=", document).group(1))
    assert members[0]["start_date"] == "2009-01-05"
    assert members[0]["first_buy_date"] == "2020-12-31"
    assert "测试 &lt;/script&gt;" in document
    assert "测试 \\u003c基金\\u003e" in document
    assert "<script src=" not in document
    assert "cdn.plot.ly" not in re.sub(r"<script.*?</script>", "", document, flags=re.S)
    assert "年度清仓" in document and "包含收回本金" in document
    assert "每年模拟盈利" in document and "加权平均成本" in document
    assert all(f'id="portfolio-{name}-chart"' in document for name in ["annual", "pnl", "nav", "funding"])

    assert "从首次投入日期起按实际天数年化" in document
    assert "按完整评价时间年化" not in document
    if selection_mode == "random":
        assert "Fundb、Fundc 每日随机选8/5只" in document
        assert "Fundb、Fundc 按原综合评分排名买入" not in document
    else:
        assert "Fundb、Fundc 按原综合评分排名买入" in document
        assert "Fundb、Fundc 每日随机选8/5只" not in document


def test_single_fund_scope_and_pre_inception_year(tmp_path):
    daily = pd.DataFrame({"date": pd.to_datetime(["2011-04-08", "2011-12-30"]),
                          "equity": [100., 105.], "daily_pnl": [0., 5.],
                          "unit_nav": [1., 1.05]})
    annual = [dict(year=2010, year_status="not_incepted", profit=0.,
                   contributions=0., twr_return=0., max_drawdown=0.),
              dict(year=2011, profit=5., contributions=100., twr_return=.05)]
    funds = [dict(code="519981", name="测试 <基金>", asset_type="Fundg")]
    path = render_portfolio_report(
        daily, annual, funds, funds, {},
        {"scope": "single_fund", "types": ["Fundg"], "code": "519981",
         "name": "测试 <基金>", "start_date": "2010-01-01",
         "assumptions": ["本次假设 <严格沿用>"]}, tmp_path / "single.html")
    document = path.read_text()
    rows = json.loads(re.search(r"const annualRows=(.*?),allFunds=", document).group(1))
    assert rows[0]["period"] == "2010（未成立）"
    assert rows[0]["year_status"] == "not_incepted"
    assert all(rows[0][key] is None for key in ["profit", "contributions", "twr_return", "max_drawdown"])
    assert rows[1]["profit"] == 5. and rows[1]["twr_return"] == .05
    assert annual[0]["profit"] == 0.  # Display treatment does not alter the source ledger.
    assert '<span class="pill green">Fundg</span>' in document
    assert '<option>Fundg</option>' in document and '<option>Fundb</option>' not in document
    assert "519981 测试 &lt;基金&gt;" in document
    assert "本次假设 &lt;严格沿用&gt;" in document
    assert "Fundg 策略账户汇总" in document and "策略账户单位净值" in document
    assert "五类" not in document and "本次 QLab 名单" not in document
    assert "Fundb、Fundc" not in document and "{{" not in re.sub(r"<script.*?</script>", "", document, flags=re.S)
