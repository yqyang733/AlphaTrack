"""CLI: config paths resolve relative to the JSON, overrides relative to cwd."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from .data import load_data
from .engine import EngineConfig, run_backtest
from .metrics import calculate_metrics
from .strategies import create_strategy
from .report import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="AlphaTrack reproducible backtesting")
    parser.add_argument("--config", default="configs/110020_fundd.json")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--data")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    conf = json.loads(config_path.read_text(encoding="utf-8"))
    unknown = set(conf)-{"title","data","output","strategy","engine"}
    if unknown:
        parser.error(f"Unknown configuration keys: {sorted(unknown)}")
    conf.setdefault("engine", {})
    for key in ("start", "end"):
        if getattr(args,key):
            conf["engine"][key] = getattr(args,key)
    cfg = EngineConfig(**conf["engine"])
    conf["engine"] = asdict(cfg)
    data_path = Path(args.data).resolve() if args.data else (config_path.parent/conf["data"]).resolve()
    output = Path(args.output).resolve() if args.output else (config_path.parent/conf["output"]).resolve()
    frame, meta = load_data(data_path)
    meta.update(data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                execution="Signal on NAV date; fill at next available NAV; no terminal liquidation",
                settlement="Immediate fractional shares; no fund subscription suspensions or settlement lag modelled",
                fees="Configurable constant subscription/redemption rates; zero in demo, NOT historical fee schedule",
                funding="Recycle sale proceeds; inject only cash shortage at purchase close",
                risk="Daily means per NAV observation, 252 observations/year; early history is sparse")
    strategy = create_strategy(conf["strategy"]["name"], conf["strategy"].get("params"))
    daily, trades = run_backtest(frame,strategy,cfg)
    metrics = calculate_metrics(daily,trades,cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output.with_suffix(".daily.csv"), index=False, float_format="%.12g")
    trades.to_csv(output.with_suffix(".trades.csv"), index=False, float_format="%.12g")
    payload = dict(metrics=metrics, config=conf, metadata=meta)
    output.with_suffix(".summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    write_report(daily,trades,metrics,conf,meta,output)
    print(json.dumps(dict(report=str(output), **metrics),ensure_ascii=False,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
