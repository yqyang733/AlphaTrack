"""Replay QLab's five requested strategy families without rewriting its rules.

Only pure modules from an explicit, separately checked-out QLab source tree are
loaded. No live recommendation ledger, network fetcher or QLab output is opened.
Ranking uses each fund's latest known observation as of the evaluation date;
orders are emitted only for funds with a fresh NAV observation on that date.
"""
from contextlib import contextmanager
from dataclasses import fields, replace
from fractions import Fraction
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd


DEFAULT_TYPES = ("Fundb", "Fundc", "Fundd", "Fundf_1", "Fundg")
SIGNAL_COLUMNS = ("action", "amount", "fraction", "rank", "score", "reason")


@contextmanager
def _temporary_aliases(aliases):
    """Allow QLab's absolute internal imports without polluting caller imports."""
    missing = object()
    previous = {key: sys.modules.get(key, missing) for key in aliases}
    sys.modules.update(aliases)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


class QLabRules:
    """Load frozen reference rules and create causal historical suggestions."""

    def __init__(self, source):
        self.source = Path(source).resolve()
        self.namespace = "_alphatrack_qlab_" + hashlib.sha256(str(self.source).encode()).hexdigest()[:12]
        self.source_hashes = {}
        self._package(self.namespace)
        for part in ("trading", "data", "core", "strategy"):
            self._package(f"{self.namespace}.{part}")
        self.config_module = self._load("trading.config", "trading/config.py")
        self.engine_module = self._load("trading.engine", "trading/engine.py")
        self.processor = self._load("data.processor", "data/processor.py")
        self.calculator = self._load("core.calculator", "core/calculator.py")
        self.mean_reversion = self._load("strategy.mean_reversion", "strategy/mean_reversion.py")
        with _temporary_aliases({"data": sys.modules[f"{self.namespace}.data"],
                                 "data.processor": self.processor,
                                 "trading": sys.modules[f"{self.namespace}.trading"],
                                 "trading.config": self.config_module}):
            self.ranking = self._load("trading.ranking", "trading/ranking.py")
            system = self._load("system_config", "config.py")
        # Read dataclass default factories, without constructing SystemConfig:
        # its __post_init__ creates directories in the source checkout.
        self.ma_parameters = {}
        defaults = {field.name: field for field in fields(system.SystemConfig)}
        for kind in ("Fundb", "Fundc"):
            prefix = kind.lower()
            self.ma_parameters[kind] = (
                defaults[f"{prefix}_ma_windows"].default_factory(),
                defaults[f"{prefix}_ma_weights"].default_factory(),
            )
        self.configs = {}
        for kind in DEFAULT_TYPES:
            path = self.config_module.resolve_config_path(kind)
            if path is None:
                config = self.config_module.TradingConfig()
            else:
                config = self.config_module.TradingConfig.from_json(path, asset_type=kind)
                self._hash(Path(path).relative_to(self.source))
            self.configs[kind] = config

    def _package(self, name):
        module = ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
        return module

    def _hash(self, relative):
        path = self.source / relative
        self.source_hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()

    def _load(self, name, relative):
        path = self.source / relative
        self._hash(relative)
        full_name = f"{self.namespace}.{name}"
        spec = importlib.util.spec_from_file_location(full_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        # Compile directly rather than exec_module: SourceFileLoader may write
        # __pycache__ into the reference checkout, which must stay read-only.
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
        return module

    def catalog(self, types=DEFAULT_TYPES):
        """Read only enabled catalog lines; keep category labels exactly."""
        unknown = set(types) - set(DEFAULT_TYPES)
        if unknown:
            raise ValueError(f"Unsupported QLab strategy types: {sorted(unknown)}")
        self._hash("data_files/Fund")
        records = []
        with (self.source / "data_files/Fund").open(encoding="utf-8-sig") as stream:
            next(stream, None)
            for line in stream:
                if line.startswith(";") or not line.strip():
                    continue
                parts = [part.strip() for part in line.strip().split(",")]
                if len(parts) < 5 or not parts[0] or parts[2] not in types:
                    continue
                code, name, kind, category, mode = parts[:5]
                records.append(dict(code=code, name=name, asset_type=kind,
                                    category=category, trade_mode=mode))
        if len({asset_key(item) for item in records}) != len(records):
            raise ValueError("Duplicate QLab type/code catalog entry")
        return records

    def prepare_features(self, frame, asset_type):
        """Use exact source total-return, MA and price-position calculations."""
        if asset_type not in DEFAULT_TYPES:
            raise ValueError(f"Unsupported strategy: {asset_type}")
        required = {"date", "nav"}
        if not required.issubset(frame):
            raise ValueError("QLab features require date and nav")
        result = frame.copy().reset_index(drop=True)
        result["date"] = pd.to_datetime(result.date, errors="raise")
        if result.empty or result.date.isna().any() or result.date.duplicated().any() or not result.date.is_monotonic_increasing:
            raise ValueError("QLab NAV dates must be nonempty, unique and increasing")
        if not np.isfinite(result.nav).all() or (result.nav <= 0).any():
            raise ValueError("QLab NAV values must be finite and positive")
        raw = result[["date", "nav"]].rename(columns={"date": "净值日期", "nav": "单位净值"})
        events = []
        for row in result.itertuples(index=False):
            split = float(getattr(row, "split", 1.0))
            dividend = float(getattr(row, "dividend", 0.0))
            if split != 1:
                events.append({"分红日": row.date, "分红类型": "份额折算", "分红": f"1:{split}"})
            if dividend:
                events.append({"分红日": row.date, "分红类型": "现金分红", "分红": dividend})
        reference = self.calculator.ReturnCalculator.calculate_return_with_dividend_reinvest(
            raw, pd.DataFrame(events) if events else None)
        result["total_return_index"] = reference["累计收益率"] + 1
        if asset_type in ("Fundb", "Fundc"):
            windows, weights = self.ma_parameters[asset_type]
            reference = self.mean_reversion.MeanReversionStrategy(windows, weights).calculate_signals(reference)
            result["mean_reversion_score"] = reference.mean_reversion_score
            dates = reference["净值日期"].to_numpy()
            values = reference["累计收益率"].to_numpy()
            # QLab annualized ranking input is SIMPLE period return / years,
            # not CAGR. Its analyzer rounds each return to 4 decimal places
            # before cross-sectional scoring; preserve both choices exactly.
            for years in (1, 2, 3, 5):
                targets = (reference["净值日期"] - pd.DateOffset(years=years)).to_numpy()
                indexes = np.searchsorted(dates, targets, side="right") - 1
                valid = indexes >= 0
                annual = np.full(len(result), np.nan)
                initial = values[indexes[valid]]
                annual[valid] = (values[valid] - initial) / (1 + initial) / float(years)
                result[f"近{years}年"] = [float(f"{value:.4f}") if np.isfinite(value) else np.nan for value in annual]
        else:
            positions = reference.drop(columns="单位净值").rename(columns={"累计收益率": "单位净值"})
            positions = self.calculator.PercentileCalculator.calculate_percentile_position(positions, [1, 2, 3, 5, 10])
            for column in [f"近{year}年百分位" for year in (1, 2, 3, 5, 10)] + ["成立以来百分位"]:
                result[column] = positions[column]
        return result

    def rank_snapshot(self, assets, latest):
        """Return original QLab scores/ranks for one category's as-of snapshot."""
        metrics = {key: [] for key in ("基金代码", "近1年", "近2年", "近3年", "近5年")}
        results = {}
        for asset in assets:
            row = latest.get(asset_key(asset))
            if row is None:
                continue
            metrics["基金代码"].append(asset["code"])
            for years in (1, 2, 3, 5):
                metrics[f"近{years}年"].append(row.get(f"近{years}年", np.nan))
            results[(asset["code"], asset["name"], asset["category"])] = {
                "signals": {"mean_reversion_score": row.get("mean_reversion_score", np.nan)}}
        scores = self.ranking.prepare_scores(metrics, results)
        return {row["code"]: (i, float(row["score"]))
                for i, row in enumerate(scores.to_dict("records"), 1)}

    def build_signals(self, frames, assets=None, *, start="2010-01-01", end=None):
        """Return per-type/fund frames and reproducibility metadata.

        History before ``start`` warms indicators and peer rankings only. The
        Fundd sale ladder starts empty with the initially empty portfolio.
        Latest peer data are carried forward for ranking, never backwards.
        A stale fund can influence the peer ranking but never gets a duplicate
        order on a day without a new NAV row.
        """
        assets = self.catalog() if assets is None else list(assets)
        if len({asset_key(asset) for asset in assets}) != len(assets):
            raise ValueError("Duplicate strategy/fund entry")
        start = pd.Timestamp(start)
        end = pd.Timestamp(end) if end is not None else None
        if end is not None and end < start:
            raise ValueError("end precedes start")
        prepared, engines, sold, events = {}, {}, {}, {}
        for asset in assets:
            key = asset_key(asset)
            frame = self.prepare_features(frames[asset["code"]], asset["asset_type"])
            if end is not None:
                frame = frame[frame.date <= end].copy()
            prepared[key] = frame
            config = self.configs[asset["asset_type"]]
            if asset["asset_type"] == "Fundd" and asset["category"].strip() == "基金>宽基指数":
                config = replace(config, fundd_base_amount=config.fundd_broad_base_amount)
            engines[key] = self.engine_module.TradingEngine(config)
            sold[key] = ()
            for index, row in frame.iterrows():
                events.setdefault(row.date, []).append((asset, index, row))
            for column, initial in (("action", "HOLD"), ("amount", np.nan), ("fraction", np.nan),
                                    ("rank", np.nan), ("score", np.nan), ("reason", "回测起点之前，仅用于指标预热")):
                frame[column] = initial
        latest, counts = {}, {"BUY": 0, "SELL": 0, "HOLD": 0}
        by_type = {kind: [asset for asset in assets if asset["asset_type"] == kind] for kind in DEFAULT_TYPES}
        for date in sorted(events):
            fresh = events[date]
            for asset, index, row in fresh:
                latest[asset_key(asset)] = row
            if date < start:
                continue
            active_rank_types = {asset["asset_type"] for asset, _, _ in fresh} & {"Fundb", "Fundc"}
            rankings = {kind: self.rank_snapshot(by_type[kind], latest) for kind in active_rank_types}
            for asset, index, row in fresh:
                key, kind = asset_key(asset), asset["asset_type"]
                rank, score = rankings.get(kind, {}).get(asset["code"], (None, None))
                decision, sold[key] = engines[key].evaluate(
                    kind, asset["code"], asset["name"], str(date.date()), row,
                    rank=rank, score=score, sold_levels=sold[key])
                values = dict(action=decision.action, amount=decision.amount,
                              fraction=float(Fraction(decision.fraction)) if decision.fraction else np.nan,
                              rank=decision.rank, score=decision.score, reason=decision.reason)
                for column, value in values.items():
                    prepared[key].at[index, column] = np.nan if value is None else value
                counts[decision.action] += 1
        metadata = {
            "source_path": str(self.source), "source_hashes": dict(self.source_hashes),
            "start": str(start.date()), "end": str(end.date()) if end is not None else None,
            "types": list(dict.fromkeys(asset["asset_type"] for asset in assets)),
            "assets": assets, "signal_counts": counts,
            "configurations": {kind: self.configs[kind].to_dict() for kind in by_type if by_type[kind]},
            "ranking": "latest known NAV per peer; emit only at each fund's fresh NAV; all 1/2/3/5y return inputs finite using target-date-or-earlier NAV; continuity not required by source; returns rounded to four decimals",
            "universe": "current enabled QLab catalog replayed historically; not historical point-in-time membership",
            "signal_start": "empty Fundd sell state at requested start; earlier history warms indicators only",
        }
        return prepared, metadata


def asset_key(asset):
    return f"{asset['asset_type']}:{asset['code']}"


class QLabSignalStrategy:
    """Adapter for AlphaTrack's existing next-NAV single-fund execution loop."""

    def __init__(self, signals):
        self.signals = signals.copy()
        if self.signals.date.duplicated().any():
            raise ValueError("Duplicate signal date")

    def prepare(self, frame):
        # Keep source-derived features, and verify the execution data align.
        if not pd.DatetimeIndex(frame.date).equals(pd.DatetimeIndex(self.signals.date)):
            raise ValueError("Execution and signal observations must match")
        if not np.array_equal(frame.nav.to_numpy(), self.signals.nav.to_numpy()):
            raise ValueError("Execution NAV differs from signal NAV")
        return self.signals.copy()

    def decide(self, row):
        action = row["action"]
        decision = {"action": action, "reason": row["reason"]}
        if action == "BUY":
            decision["amount"] = float(row["amount"])
        elif action == "SELL":
            decision["fraction"] = float(row["fraction"])
        return decision
