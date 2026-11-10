"""Selection identity must survive caches and report rendering boundaries."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from alphatrack import portfolio_cli, window_cli, window_report
from alphatrack.strategies import qlab


@pytest.mark.parametrize("requested,cached", [
    ({"selection_mode": "random", "random_seed": 42}, {"selection_mode": "ranked", "random_seed": None}),
    ({"selection_mode": "random", "random_seed": 42}, {"selection_mode": "random", "random_seed": 43}),
    ({"selection_mode": "ranked", "random_seed": None}, {"selection_mode": "random", "random_seed": 42}),
    ({"selection_mode": "random", "random_seed": 42}, {}),
])
def test_wrong_selection_cache_is_rejected_before_deserialization(tmp_path, monkeypatch, requested, cached):
    source, data = tmp_path / "source", tmp_path / "data"
    source.mkdir()
    data.mkdir()
    (source / "PROVENANCE.json").write_text(json.dumps({"files": {}}))
    nav_path = data / "000001.csv"
    nav_path.write_text("date,nav\n2025-01-01,1.0\n2025-01-02,1.1\n")
    digest = hashlib.sha256(nav_path.read_bytes()).hexdigest()
    frame = pd.read_csv(nav_path, parse_dates=["date"])
    monkeypatch.setattr(portfolio_cli, "load_data", lambda path: (frame, {"data_sha256": digest}))

    class Rules:
        source_hashes = {"reference": "unchanged"}

        def __init__(self, source):
            pass

        def catalog(self, types):
            return [{"code": "000001", "asset_type": "Fundb", "name": "Example"}]

    monkeypatch.setattr(portfolio_cli, "QLabRules", Rules)
    config = dict(title="Selection isolation", source="source", data_dir="data", types=["Fundb"],
                  strategy=requested, output="out.html",
                  engine={"initial_cash": 0, "funding_mode": "topup", "withdraw_on_sell": True,
                          "start": "2025-01-01", "end": "2025-01-02"})
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    cache = tmp_path / "signals.pickle"
    cache.write_bytes(b"do not deserialize this cache")
    metadata = dict(data_hashes={"000001": digest}, source_hashes=Rules.source_hashes,
                    adapter_sha256=hashlib.sha256(Path(qlab.__file__).read_bytes()).hexdigest(),
                    start="2025-01-01", end="2025-01-02", **cached)
    cache.with_suffix(".metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(portfolio_cli.pickle, "load", lambda stream: pytest.fail("Unsafe cache was deserialized"))
    with pytest.raises(ValueError, match="selection policy"):
        portfolio_cli.run_config(config_path, signal_cache=cache)
    assert not (tmp_path / "out.html").exists()


def _window():
    return {"window_id": "since_20250101", "kind": "fixed", "years": None,
            "label": "2025-01-01 至今", "start": "2025-01-01", "end": "2026-11-06"}


def _random_config(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/qlab_portfolio_random.json").read_text())
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "random.json"
    path.write_text(json.dumps(config))
    return path, config


@pytest.mark.parametrize("configured_output", [True, False])
def test_random_default_outputs_leave_baseline_report_intact(tmp_path, monkeypatch, configured_output):
    config_path, config = _random_config(tmp_path)
    if not configured_output:
        config.pop("window_output")
        config_path.write_text(json.dumps(config))
    monkeypatch.chdir(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    baseline = reports / "qlab_portfolio_windows.html"
    baseline.write_text("original ranked results")
    assert (config_path.parent / config["output"]).resolve() != reports / "qlab_portfolio_2010.html"
    monkeypatch.setattr(window_cli, "plan_windows", lambda *args, **kwargs: [_window()])
    calls = []

    def run(config_path, **kwargs):
        calls.append(kwargs)
        return {"config": config, "summary": {"start_date": "2025-01-01", "end_date": "2026-11-06",
                                               "beginning_assets": 0},
                "annual": [{"year": 2025, "profit": 1.}, {"year": 2026, "profit": 2.}]}

    monkeypatch.setattr(window_cli, "run_config", run)
    rendered = []
    monkeypatch.setattr(window_report, "render_window_report",
                        lambda runs, output, **kwargs: rendered.append(output))
    window_cli.run_windows(config_path, as_of="2026-11-10")
    output = reports / "alphastep_random_windows.html"
    assert rendered == [output]
    assert calls[0]["output"] == reports / "alphastep_random_windows_since_20250101.html"
    assert baseline.read_text() == "original ranked results"
    assert json.loads(output.with_suffix(".summary.json").read_text())["strategy"] == {
        "selection_mode": "random", "random_seed": 42}


@pytest.mark.parametrize("saved", [
    {},  # Legacy ranked run without a strategy field.
    {"strategy": {"selection_mode": "ranked", "random_seed": None}},
    {"strategy": {"selection_mode": "random", "random_seed": 43}},
])
def test_render_only_rejects_other_mode_or_seed_without_overwriting(tmp_path, monkeypatch, saved):
    config_path, _ = _random_config(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    output = reports / "alphastep_random_windows.html"
    output.write_text("keep existing comparison")
    window_report_path = reports / "alphastep_random_windows_since_20250101.html"
    payload = {"config": saved,
               "summary": {"start_date": "2025-01-01", "end_date": "2026-11-06", "beginning_assets": 0}}
    window_report_path.with_suffix(".summary.json").write_text(json.dumps(payload))
    monkeypatch.setattr(window_cli, "plan_windows", lambda *args, **kwargs: [_window()])
    monkeypatch.setattr(window_report, "render_window_report", lambda *args, **kwargs: pytest.fail("Wrong report rendered"))
    with pytest.raises(ValueError, match="selection policy"):
        window_cli.run_windows(config_path, as_of="2026-11-10", render_only=True)
    assert output.read_text() == "keep existing comparison"


@pytest.mark.parametrize("seed", [True, 42., "42", None])
def test_random_seed_rejects_ambiguous_types(seed):
    with pytest.raises(ValueError, match="integer"):
        portfolio_cli.strategy_settings({"strategy": {"selection_mode": "random", "random_seed": seed}})
