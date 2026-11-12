"""Run one fund only after its exact dividend-reinvested curve is confirmed.

Uses the existing QLab rules, next-NAV engine and cash-flow portfolio metrics.
Ranking strategies require a peer universe. Explicit fixed-daily overrides
replace the original entry conditions with a user-specified purchase amount.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alphatrack.data import load_data
from alphatrack.engine import EngineConfig, run_backtest
from alphatrack.lots import analyze_lots
from alphatrack.portfolio import aggregate_portfolio, analyze_portfolio
from alphatrack.portfolio_cli import write_json
from alphatrack.portfolio_report import render_portfolio_report
from alphatrack.strategies.qlab import QLabRules, QLabSignalStrategy, asset_key
from scripts.export_window_tables import formatted, tsv


def export_tables(output, annual, summary, warnings=()):
    metrics = [
        ("累计投入本金（万元）", "total_contributions", "money"),
        ("累计盈亏（万元）", "total_pnl", "money"),
        ("资金加权年化·XIRR", "xirr", "percent"),
        ("净值年化·TWR", "cagr", "percent"),
        ("最大净值回撤", "max_drawdown", "percent"),
        ("夏普比率", "sharpe", "number"),
        ("年化波动率", "annual_volatility", "percent"),
    ]
    headers = [label for label, _, _ in metrics] + [str(int(y)) for y in annual.year]
    row = [formatted(summary[key], kind) for _, key, kind in metrics]
    row += ["—" if r.year_status == "not_incepted" else formatted(r.profit) for r in annual.itertuples()]
    # The original six-metric paste layout is retained separately; the main
    # requested layout adds total principal before the old columns.
    for suffix, values in (("paste_header_wan.tsv", [headers]), ("paste_2010_wan.tsv", [row]),
                           ("paste_with_header_wan.tsv", [headers, row]),
                           ("paste_legacy_wan.tsv", [row[1:]])):
        output.with_suffix("." + suffix).write_text(tsv(values), encoding="utf-8")
    annual_rows = []
    for r in annual.itertuples():
        label = str(r.year) + ("（未成立）" if r.year_status == "not_incepted" else " YTD" if r.is_partial_year else "")
        annual_rows.append([label, *(["—"] * 3 if r.year_status == "not_incepted" else
                                     [f"{r.deposits:.2f}", f"{r.profit:.2f}", f"{r.end_assets:.2f}"])])
    annual_headers = ["年份", "当年投入（元）", "当年盈亏（元）", "年末资产（元）"]
    lines = [f"截止 {summary['end_date']}；盈亏包含持仓浮盈亏。", *warnings, "",
             "| " + " | ".join(annual_headers) + " |", "|---|---:|---:|---:|"]
    lines += ["| " + " | ".join(values) + " |" for values in annual_rows]
    lines += ["", "指标汇总（累计金额为万元）：", "", "| 指标 | 数值 |", "|---|---:|"]
    lines += [f"| {label} | {formatted(summary[key], kind)} |" for label, key, kind in metrics]
    output.with_suffix(".table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    output.with_suffix(".annual_yuan.tsv").write_text(tsv([annual_headers, *annual_rows]), encoding="utf-8")


def run(data, rule, output, *, start="2010-01-01", end=None, curve_confirmed=False, category=None,
        fundb_daily_buy=False, daily_buy_amount=None, batch_authorized=False):
    if not curve_confirmed and not batch_authorized:
        raise ValueError("User must confirm this snapshot's curve before backtesting")
    if fundb_daily_buy and rule != "Fundb":
        raise ValueError("fundb_daily_buy applies only to single-fund Fundb")
    if fundb_daily_buy and daily_buy_amount is not None:
        raise ValueError("Use either fundb_daily_buy or daily_buy_amount, not both")
    if daily_buy_amount is not None:
        if isinstance(daily_buy_amount, bool) or not isinstance(daily_buy_amount, (int, float)) or not np.isfinite(daily_buy_amount) or daily_buy_amount <= 0:
            raise ValueError("daily_buy_amount must be finite and positive")
        if rule not in {"Fundb", "Fundg"}:
            raise ValueError("daily_buy_amount supports only explicitly selected Fundb or Fundg snapshots")
    fixed_amount = 40 if fundb_daily_buy else daily_buy_amount
    daily_fixed = fixed_amount is not None
    if rule == "Fundb" and not daily_fixed:
        raise ValueError("Single-fund Fundb requires explicit fundb_daily_buy; no ranking peer universe was supplied")
    if rule not in {"Fundb", "Fundd", "Fundf_1", "Fundg"}:
        raise ValueError("Fundb/Fundc require an explicitly defined ranking peer universe")
    data, output = Path(data).resolve(), Path(output).resolve()
    frame, source_meta = load_data(data)
    if source_meta.get("backtest_allowed") is False:
        raise ValueError("Diagnostic curve has unresolved data gaps and cannot be backtested")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    parity_error = source_meta.get("qlab_max_absolute_error")
    valid_parity = (type(parity_error) in (int, float) and np.isfinite(parity_error)
                    and 0 <= parity_error <= 1e-12)
    if source_meta.get("data_sha256") != digest or not valid_parity:
        raise ValueError("Data must match the previously reviewed QLab-compatible curve")
    if source_meta.get("requested_strategy") != rule:
        raise ValueError("Selected rule differs from the reviewed request")
    rules = QLabRules(ROOT / "vendor/qlab")
    provenance = json.loads((ROOT / "vendor/qlab/PROVENANCE.json").read_text())
    for relative, value in rules.source_hashes.items():
        if provenance["files"].get(relative) != value:
            raise ValueError(f"Pinned QLab rules changed: {relative}")
    if source_meta.get("calculator_sha256") != rules.source_hashes["core/calculator.py"]:
        raise ValueError("Curve calculator differs from the reviewed snapshot")
    if rule == "Fundd" and category is None:
        raise ValueError("Fundd requires --category to select its original broad-index/other amount")
    terminal = pd.Timestamp(end) if end else frame.date.iloc[-1]
    if terminal > frame.date.iloc[-1]:
        raise ValueError("End exceeds the reviewed snapshot coverage")
    asset = dict(code=source_meta["code"], name=source_meta["name"], asset_type=rule,
                 category=category or "用户指定单基金", trade_mode="场外")
    key = asset_key(asset)
    user_override = None
    if daily_fixed:
        prepared = frame[frame.date <= terminal].copy()
        prepared["trade_enabled"] = prepared.date.dt.dayofweek < 5
        active_buy = prepared.trade_enabled & (prepared.date >= pd.Timestamp(start))
        prepared["action"] = np.where(active_buy, "BUY", "HOLD")
        prepared["amount"] = np.where(active_buy, fixed_amount, np.nan)
        prepared["fraction"] = np.nan
        prepared["rank"] = np.nan
        prepared["score"] = np.nan
        prepared["reason"] = np.where(active_buy, f"用户指定单基金每日定投：每个交易日买入{fixed_amount:g}元，无买入门槛",
                                      "不生成买入：起点之前或非交易日，仅估值")
        user_override = dict(mode="daily_fixed", amount=fixed_amount, ranking=False, threshold_applied=False,
                             calendar="weekday_nav", sell=False,
                             execution="next enabled NAV; valuation-only dates retain pending orders")
        signal_metadata = dict(selection_algorithm="user-defined-single-fund-daily-fixed",
                               ranking=False, amount=fixed_amount, threshold_applied=False,
                               nontrading_nav_dates=prepared.loc[~prepared.trade_enabled, "date"].dt.strftime("%Y-%m-%d").tolist())
    else:
        signals, signal_metadata = rules.build_signals({asset["code"]: frame}, [asset], start=start, end=terminal)
        prepared = signals[key]
    cfg = EngineConfig(initial_cash=0, funding_mode="topup", withdraw_on_sell=True,
                       start=start, end=str(terminal.date()), buy_fee=0, sell_fee=0,
                       trade_calendar_column="trade_enabled" if daily_fixed else None)
    component, trades = run_backtest(prepared, QLabSignalStrategy(prepared), cfg)
    for field in ("code", "name", "asset_type"):
        component[field] = asset[field]
        trades[field] = asset[field]
    daily = aggregate_portfolio({key: component})
    annual, summary = analyze_portfolio(daily, trades, start=start, end=cfg.end)
    annual["year_status"] = ["not_incepted" if year < frame.date.iloc[0].year else "observed" for year in annual.year]
    lots, lot_summary = analyze_lots(component, trades)
    for field in ("code", "name", "asset_type"):
        lots[field] = asset[field]
    if not np.isclose(summary["total_pnl"], lot_summary["total_pnl"], rtol=1e-10, atol=1e-6):
        raise ValueError("FIFO and account profit disagree")
    funds = pd.DataFrame([{**asset, **summary, "first_nav_date": source_meta["first_date"]}])
    groups = pd.DataFrame([{**summary, "asset_type": rule, "name": rule, "fund_count": 1}])
    signal_metadata.update(universe="one user-selected fund; no peer ranking used")
    if not daily_fixed:
        signal_metadata["selection_algorithm"] = "original-qlab-independent-fund-rule"
    assumptions = [
        f"用户指定 {asset['code']} {asset['name']}，采用 QLab 原 {rule} 买卖规则。",
        f"曲线经用户确认后使用同一份数据，截止 {cfg.end}；无二次更新或混用新数据。",
        f"从{start}空仓开始；首个真实净值日为{source_meta['first_date']}，成立前不生成持仓或收益。",
        "新净值观察日生成信号，下一条可用净值成交；末日信号没有下一净值时保留待成交。",
        "资金按需追加；不设资金上限；申购赎回费为0，不虚构年末清仓。",
        "现金分红按除息日净值再投资；新买入份额不重复领取当天旧持仓的分红。",
        "年度盈亏包含持仓浮盈亏变化；当前年份仅统计至数据截止日，不外推全年。",
        "XIRR按实际日期ACT/365计算；TWR年化从首次投入日起ACT/365.25计算。",
        "风险指标按252净值观察期/年，无风险利率0；未建模QDII实际公布时刻、确认到账延迟和限购。",
    ]
    if rule == "Fundg" and not daily_fixed:
        assumptions.insert(1, "Fundg：分红再投曲线的近一年高低位置≤60%时，每次买10元；原规则没有卖出。高低位置采用已有历史，不要求先等待完整一年。")
    if daily_fixed:
        assumptions[0] = f"用户指定 {asset['code']} {asset['name']}，每个交易日固定买入{fixed_amount:g}元，无买入门槛，不参与排名，不卖出。"
        assumptions[3] = f"有真实净值的周一至周五作为可执行交易日；当日产生{fixed_amount:g}元买入信号，下一可执行净值成交。周末保留待成交订单，只计算估值与分红；末日新信号不虚构成交。"
        assumptions.insert(1, "这是用户明确设定的单基金固定买入规则；基金净值与分红再投资计算沿用QLab口径。")
    if batch_authorized and not curve_confirmed:
        assumptions = [f"用户直接授权按QLab口径批量回测；数据已通过QLab收益曲线逐点一致性校验，截止 {cfg.end}。"
                       if note.startswith("曲线经用户确认") else note for note in assumptions]
    display_rule = f"每日{fixed_amount:g}元定投" if daily_fixed else rule
    metadata = dict(scope="single_fund", title=f"{asset['code']} · {asset['name']} · {display_rule} 单基金回测",
                    code=asset["code"], name=asset["name"], types=[rule], fund_count=1,
                    start_date=start, end_date=cfg.end, curve_confirmed_by_user=bool(curve_confirmed),
                    batch_authorized_by_user=bool(batch_authorized),
                    approved_curve_data_sha256=digest, data_hashes={asset["code"]: digest},
                    qlab_commit=provenance["commit"], source_commit=provenance["commit"],
                    qlab_catalog_commit=source_meta.get("qlab_catalog_commit"),
                    source_url=provenance["repository"], coverage=[source_meta],
                    signal_metadata=signal_metadata, assumptions=assumptions,
                    warnings=source_meta.get("warnings", []),
                    approved_omissions=source_meta.get("approved_omissions", []),
                    existing_policy_omissions=source_meta.get("existing_policy_omissions", []),
                    strategy={"name": rule, "scope": "single_fund", **({"selection_mode": "daily_fixed"} if daily_fixed else {})},
                    validation={"annual_pnl_sum": float(annual.profit.sum()),
                                "fund_pnl_sum": summary["total_pnl"], "group_pnl_sum": summary["total_pnl"],
                                "fifo_pnl_sum": lot_summary["total_pnl"],
                                "cashflow_identity": summary["final_equity"] + summary["total_withdrawals"] - summary["total_contributions"]})
    if user_override is not None:
        metadata["user_rule_override"] = user_override
    config = dict(data=str(data.relative_to(ROOT)) if data.is_relative_to(ROOT) else str(data),
                  engine=asdict(cfg), strategy={"name": rule, "category": asset["category"]},
                  rule_configuration=user_override if user_override is not None else rules.configs[rule].to_dict())
    payload = dict(summary=summary, config=config, metadata=metadata, annual=annual.to_dict("records"),
                   funds=funds.to_dict("records"), groups=groups.to_dict("records"))
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, table in (("daily.csv", daily), ("annual.csv", annual), ("funds.csv", funds),
                          ("groups.csv", groups), ("trades.csv", trades), ("lots.csv", lots),
                          ("fund_annual.csv", annual.assign(**asset)), ("group_annual.csv", annual.assign(asset_type=rule))):
        table.to_csv(output.with_suffix("." + suffix), index=False, float_format="%.12g")
    component.to_csv(output.with_suffix(".components_daily.csv.gz"), index=False, compression="gzip", float_format="%.12g")
    prepared[prepared.date >= pd.Timestamp(start)].assign(**asset).to_csv(
        output.with_suffix(".signals.csv.gz"), index=False, compression="gzip", float_format="%.12g")
    write_json(output.with_suffix(".summary.json"), payload)
    render_portfolio_report(daily, annual, funds, {"summary": groups, "daily": {rule: daily}}, summary, metadata, output)
    export_tables(output, annual, summary, metadata["warnings"])
    print(json.dumps({"report": str(output), **summary}, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--rule", required=True, choices=["Fundb", "Fundd", "Fundf_1", "Fundg"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end")
    parser.add_argument("--category", help="Fundd: 基金>宽基指数 selects the original broad-index amount")
    parser.add_argument("--curve-confirmed", action="store_true", help="Use only after the user confirms this exact snapshot's curve")
    parser.add_argument("--batch-authorized", action="store_true", help="User explicitly requested direct batch execution with QLab-compatible data")
    parser.add_argument("--fundb-daily-buy", action="store_true", help="User-specified single-fund Fundb: 40 yuan each weekday NAV, no ranking or sales")
    parser.add_argument("--daily-buy-amount", type=float, help="Explicit fixed daily purchase amount, replacing the selected Fundb/Fundg entry conditions")
    args = parser.parse_args()
    run(args.data, args.rule, args.output, start=args.start, end=args.end,
        curve_confirmed=args.curve_confirmed, category=args.category, fundb_daily_buy=args.fundb_daily_buy,
        daily_buy_amount=args.daily_buy_amount, batch_authorized=args.batch_authorized)
