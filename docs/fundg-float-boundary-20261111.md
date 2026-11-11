# Fundg 60% 边界：050025 的 QLab 兼容性说明

本次050025继续使用已核对的快照与固定版本QLab原规则，保留原始浮点比较，没有修改买入条件、加入容差或补买交易。

## 原始行为

2012-06-30之前的可用单位净值依次为：2012-06-14的1、06-15的1、06-21的0.999、06-29的0.9995、06-30的0.9996。此时尚无分红。

按十进制精确数计算，近一年高低位置为 `(0.9996 - 0.999) / (1 - 0.999) = 0.6`。原QLab先以二进制浮点生成累计收益，再求高低位置，结果为 `0.6000000000000444`。原交易函数直接判断是否大于 `60 / 100`，因此当天为HOLD。

该结果可仅用Python标准浮点运算复现：

```python
navs = [1.0, 1.0, 0.999, 0.9995, 0.9996]
returns = [value - 1.0 for value in navs]
position = (returns[-1] - min(returns)) / (max(returns) - min(returns))
assert position == 0.6000000000000444
assert position > 60 / 100
```

对应原源码是 `vendor/qlab/core/calculator.py` 的累计收益与高低位置计算，以及 `vendor/qlab/trading/engine.py` 的阈值比较。直接调用这些原函数逐日检查3,417条记录，生产动作与原QLab没有差异。

## 对本次结果的影响

十进制精确信号与原QLab信号仅此一日不同。若改用十进制精确阈值，会在下一净值观察日2012-07-06（净值0.9999）额外买入10元。持有至2026-11-10，该笔额外买入将增加资产约57.15元，其中利润约47.15元。

当前正式结果保留原QLab的503次买入、累计本金5,030元；精确阈值的504次买入属于另一种数值判定口径，并非本次正式结果。独立审计需明确记录这项差异，并独立复核原QLab兼容信号下的分红、账本、年度盈亏与XIRR，不能把精确信号差异静默忽略。

本次输入CSV的SHA256为 `76c42e31b3666fae97027f349aac6d4e041b4ff701fe6148fa40ab4b068d2bf0`。原始数据和详细审计结果仅保存在本地，不随代码上传。

## 复现与审计

使用同一份已确认快照运行：

```bash
python scripts/backtest_single_fund.py --data data/single/050025/20261111/050025.csv --rule Fundg --curve-confirmed --output reports/single_050025_fundg_2010.html
python scripts/verify_portfolio.py --report reports/single_050025_fundg_2010.html
python scripts/verify_single_fundg.py --code 050025 --data data/single/050025/20261111/050025.csv --report-prefix reports/single_050025_fundg_2010 --signal-arithmetic qlab-binary64 --require-report
```

审计脚本默认仍使用十进制精确信号。显式的 `--signal-arithmetic qlab-binary64` 独立复算QLab运算次序下的浮点信号，交易账本和XIRR仍用Decimal计算。此模式的通过只代表与原QLab兼容的信号和账本核对通过；`exact_decimal_signals_match_qlab`、差异明细和精确十进制反事实结果会单独保留，不将两种判定说成完全相同。
