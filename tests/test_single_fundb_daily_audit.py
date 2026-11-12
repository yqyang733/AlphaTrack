"""Independent audit cutoffs and exact local QLab omission policy."""
import copy
import hashlib
from decimal import Decimal

import pytest

from scripts import verify_single_fundb_daily as audit


@pytest.fixture
def warmed_nav(tmp_path):
    path = tmp_path / "fund.csv"
    path.write_text(
        "date,nav,dividend,split\n"
        "2009-12-29,2,0,1\n"
        "2009-12-30,1,0,2\n"
        "2009-12-31,1,1,1\n"
        "2010-01-02,1,0,1\n"
        "2010-01-04,1,0,1\n"
        "2010-01-05,1,1,1\n"
        "2010-01-06,2,0,1\n", encoding="utf-8")
    return path


def test_start_warms_curve_but_does_not_carry_prestart_holdings_or_orders(warmed_nav):
    result = audit.calculate(warmed_nav, end="2010-01-05")
    assert result["warmup_observations"] == 3
    assert result["summary"]["observations"] == 3
    assert list(result["annual"]) == [2010]
    first, monday, tuesday = result["daily"]
    assert first["total_return_index"] == monday["total_return_index"] == 2
    assert tuesday["total_return_index"] == 4
    assert first["shares"] == monday["shares"] == 0
    assert [row["action"] for row in result["signals"]] == ["HOLD", "BUY", "BUY"]
    assert result["trades"][0]["signal_date"] == "2010-01-04"
    assert result["trades"][0]["date"] == "2010-01-05"
    assert result["trades"][-1]["status"] == "PENDING_END_OF_DATA"
    assert result["summary"]["total_contributions"] == result["summary"]["final_equity"] == 40
    assert result["summary"]["total_pnl"] == 0
    assert result["corporate_actions"][0]["dividend_cash"] == "0"


def test_end_excludes_future_fills_and_amount_scales_only_cash(warmed_nav):
    pending = audit.calculate(warmed_nav, end="2010-01-04")
    assert pending["summary"]["buy_count"] == 0
    assert len(pending["trades"]) == 1
    assert pending["trades"][0]["status"] == "PENDING_END_OF_DATA"
    full = audit.calculate(warmed_nav)
    small = audit.calculate(warmed_nav, amount="10")
    for field in ("total_contributions", "final_equity", "total_pnl"):
        assert full["summary"][field] == small["summary"][field] * 4
    assert abs(full["summary"]["xirr"] / small["summary"]["xirr"] - 1) < Decimal("1e-30")
    assert full["summary"]["buy_count"] == small["summary"]["buy_count"] == 2
    assert small["summary"]["final_equity"] == 30
    with pytest.raises(ValueError, match="No NAV"):
        audit.calculate(warmed_nav, start="2010-01-03", end="2010-01-03")


@pytest.fixture
def local_snapshot(tmp_path):
    path = tmp_path / "000931.csv"
    path.write_text(
        "date,nav,dividend,split\n"
        "2017-03-21,1,0,1\n"
        "2017-11-13,1,0,1\n"
        "2019-09-26,1,.03,1\n"
        "2019-11-01,1,.03,1\n"
        "2019-11-29,1,.03,1\n", encoding="utf-8")
    event_path = path.with_suffix(".dividends.csv")
    event_path.write_text(
        "分红日,分红类型,分红\n"
        "2017-03-22,现金分红,0.08\n"
        "2017-11-14,现金分红,0.04\n"
        "2019-09-26,现金分红,0.03\n"
        "2019-11-01,现金分红,0.03\n"
        "2019-11-29,现金分红,0.03\n", encoding="utf-8-sig")
    metadata = {"code": "000931", "source_kind": "qlab_snapshot",
                "sources": {"dividends.csv": {"path": "/qlab/data_files/dividend/fund/000931.csv",
                    "sha256": hashlib.sha256(event_path.read_bytes()).hexdigest()}},
                "existing_policy_omissions": [
                    {"date": "2017-03-22", "type": "现金分红", "value": "0.08"},
                    {"date": "2017-11-14", "type": "现金分红", "value": "0.04"}]}
    return path, metadata


def test_local_csv_reconciles_only_the_two_existing_exact_omissions(local_snapshot):
    path, metadata = local_snapshot
    result = audit.audit_source_actions(path, metadata)
    assert result["passed"] is True
    assert result["recorded_cash_dividends"] == 5
    assert result["included_cash_dividends"] == 3
    assert result["included_split_events"] == 0
    assert len(result["omissions"]) == 2
    assert result["source_sha256"] == metadata["sources"]["dividends.csv"]["sha256"]


@pytest.mark.parametrize("change", ["code", "value", "date", "duplicate", "restored_nav"])
def test_existing_policy_cannot_expand_or_ignore_a_restored_nav(local_snapshot, change):
    path, original = local_snapshot
    metadata = copy.deepcopy(original)
    if change == "code":
        metadata["code"] = "999999"
    elif change == "value":
        metadata["existing_policy_omissions"][0]["value"] = "0.09"
    elif change == "date":
        metadata["existing_policy_omissions"][0]["date"] = "2017-03-23"
    elif change == "duplicate":
        metadata["existing_policy_omissions"].append(metadata["existing_policy_omissions"][0])
    else:
        path.write_text(path.read_text().replace("2017-11-13", "2017-03-22"))
    result = audit.audit_source_actions(path, metadata)
    assert result["passed"] is False
    assert "exact existing 000931" in result["reason"]


def test_local_source_hash_and_unrecognized_missing_event_fail(local_snapshot):
    path, metadata = local_snapshot
    event_path = path.with_suffix(".dividends.csv")
    with event_path.open("a", encoding="utf-8") as stream:
        stream.write("2018-01-03,现金分红,0.01\n")
    assert "fingerprint" in audit.audit_source_actions(path, metadata)["reason"]
    metadata["sources"]["dividends.csv"]["sha256"] = hashlib.sha256(event_path.read_bytes()).hexdigest()
    assert audit.audit_source_actions(path, metadata)["passed"] is False
    metadata["approved_omissions"] = [{"date": "2018-01-03", "type": "现金分红", "value": "0.01"}]
    assert "only the exact existing" in audit.audit_source_actions(path, metadata)["reason"]


def test_duplicate_source_actions_are_rejected(local_snapshot):
    path, metadata = local_snapshot
    event_path = path.with_suffix(".dividends.csv")
    with event_path.open("a", encoding="utf-8") as stream:
        stream.write("2019-11-29,现金分红,0.03\n")
    metadata["sources"]["dividends.csv"]["sha256"] = hashlib.sha256(event_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Duplicate recorded corporate action"):
        audit.audit_source_actions(path, metadata)
