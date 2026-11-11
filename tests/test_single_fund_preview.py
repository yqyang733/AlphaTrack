"""Critical unit/date/identity checks for curve-only single-fund inputs."""
import json

import pandas as pd
import pytest

from scripts.prepare_single_fund import normalize, parse_events


CODE = "519981"
AS_OF = pd.Timestamp("2026-11-11")


def page(cash_rows=None, split_rows=None, *, code=CODE, cash_header="每10份分红"):
    cash_rows = cash_rows if cash_rows is not None else [
        ["暂无分红信息", "暂无分红信息", "暂无分红信息", "暂无分红信息"]]
    split_rows = split_rows if split_rows is not None else [["暂无拆分信息", "暂无拆分信息"]]

    def table(columns, rows):
        header = "".join(f"<th>{name}</th>" for name in columns)
        body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
        return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"

    return (f"<html><head><title>基金分红送配({code})</title></head><body>"
            + table(["权益登记日", "除息日", cash_header, "分红发放日"], cash_rows)
            + table(["拆分折算日", "拆分折算比例"], split_rows) + "</body></html>")


def nav_text(dates, *, code=CODE, notes=None):
    records = [{"x": int(pd.Timestamp(day, tz="Asia/Shanghai").timestamp() * 1000),
                "y": 1.0, "unitMoney": (notes or {}).get(day, "")} for day in dates]
    return (f"var fS_code = {json.dumps(code)};\n"
            f"var Data_netWorthTrend = {json.dumps(records)};")


def test_cash_dividend_is_per_share_on_ex_date_only():
    events = parse_events(page(cash_rows=[
        ["2025-01-02", "2025-01-03", "每10份派现金0.50元", "2025-01-07"],
        ["2026-12-01", "2026-12-02", "每10份派现金0.20元", "2026-12-08"],
    ]), CODE, AS_OF)
    assert len(events) == 1
    assert events.iloc[0]["分红日"] == pd.Timestamp("2025-01-03")
    assert float(events.iloc[0]["分红"]) == 0.05
    actual = normalize(nav_text(["2025-01-02", "2025-01-03", "2025-01-07"]), events, CODE)
    assert actual.dividend.tolist() == [0.0, 0.05, 0.0]
    per_share = parse_events(page(cash_rows=[
        ["2025-01-02", "2025-01-03", "每份派现金0.05元", "2025-01-07"]],
        cash_header="每份分红"), CODE, AS_OF)
    assert float(per_share.iloc[0]["分红"]) == float(events.iloc[0]["分红"])


def test_split_ratio_is_normalized_to_one_original_share():
    events = parse_events(page(split_rows=[["2025-01-03", "2：3"]]), CODE, AS_OF)
    assert events.iloc[0]["分红类型"] == "份额折算"
    assert events.iloc[0]["分红"] == "1:1.5"
    actual = normalize(nav_text(["2025-01-02", "2025-01-03"]), events, CODE)
    assert actual.split.tolist() == [1.0, 1.5]
    assert actual.dividend.eq(0).all()


def test_no_events_requires_both_explicit_markers_not_empty_or_error_page():
    assert parse_events(page(), CODE, AS_OF).empty
    for content in [page(cash_rows=[]), page(split_rows=[]),
                    f"<title>基金({CODE})</title><p>请求过于频繁，请稍后再试</p>",
                    page().replace("暂无拆分信息", "服务繁忙")]:
        with pytest.raises(ValueError):
            parse_events(content, CODE, AS_OF)


def test_dividend_page_and_nav_must_belong_to_requested_fund():
    with pytest.raises(ValueError, match="identity mismatch"):
        parse_events(page(code="000001"), CODE, AS_OF)
    events = parse_events(page(), CODE, AS_OF)
    with pytest.raises(ValueError, match="identity mismatch"):
        normalize(nav_text(["2025-01-02"], code="000001"), events, CODE)


@pytest.mark.parametrize("second_amount", ["0.50元", "0.60元"])
def test_duplicate_events_are_rejected_even_when_values_match(second_amount):
    rows = [["2025-01-02", "2025-01-03", "0.50元", "2025-01-07"],
            ["2025-01-02", "2025-01-03", second_amount, "2025-01-07"]]
    with pytest.raises(ValueError, match="Duplicate action"):
        parse_events(page(cash_rows=rows), CODE, AS_OF)


def test_event_without_exact_nav_and_unmatched_annotation_are_rejected():
    events = parse_events(page(cash_rows=[
        ["2025-01-02", "2025-01-03", "0.50元", "2025-01-07"]]), CODE, AS_OF)
    # A later NAV must never stand in for the absent ex-date NAV.
    with pytest.raises(ValueError, match="Missing action NAV"):
        normalize(nav_text(["2025-01-02", "2025-01-07"]), events, CODE)
    with pytest.raises(ValueError, match="Unmatched NAV action annotation"):
        normalize(nav_text(["2025-01-02", "2025-01-03"],
                           notes={"2025-01-03": "分红：每份派现金0.05元"}),
                  parse_events(page(), CODE, AS_OF), CODE)
