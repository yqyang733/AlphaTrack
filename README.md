# AlphaTrack

**默认组合回测区间：2010、2015、2020、2025 年的1月1日至今。**运行 `python -m alphatrack.window_cli` 即计算这四组独立空仓回测；打开 `reports/qlab_portfolio_windows.html` 查看每个自然年的盈亏对照。起点从2010年起每隔5年生成，共同截止日受已保存净值覆盖限制；日期和数据覆盖到2030年后自动加入2030-01-01，之后以此类推。

**仓库存储：**GitHub 只保存源码、测试、配置和使用说明。`data/` 的净值及元数据、参考分红数据、`reports/` 的全部图表/汇总/账本均只保存在本地，由 `.gitignore` 排除。基金名单、交易规则、固定源码版本及数据哈希配置保留，用于复核复现条件。新克隆请先按下方说明准备数据，再运行回测；本机已有完整结果可继续使用。报告生成说明见 [reports/README.md](reports/README.md)。

## 数据准备（新克隆必读）

实测数据不随代码下载，也不把最新网络数据当成本次历史快照。要精确复现本次结果，使用单独保管的 AlphaTrack 数据快照目录；其中需要保留 `data/` 和 `vendor/qlab/data_files/dividend/` 的相对路径。当前本机工作目录中的数据已保留。

```bash
# 从已有完整 AlphaTrack 目录或相同目录结构的数据备份恢复
python scripts/prepare_local_data.py --from /path/to/saved/AlphaTrack
# 核验当前本地数据与固定快照哈希一致
python scripts/prepare_local_data.py --check
```

`configs/data_snapshot.json` 只记录所需文件路径和 SHA256，不包含净值。恢复程序先校验源文件，再补齐缺失文件；现有内容不匹配时拒绝覆盖。没有匹配的独立数据快照，只有代码不能精确重现本文数值。更新行情可用已有 `update_eastmoney.py`（只接受无分红/拆分的单基金）或 `import_qlab_snapshot.py`（已审核的原始归档）；更新后须另存快照及哈希，不能把新结果声称为旧快照复现。

纯代码副本可直接运行 `pytest -q`：两项依赖真实快照的集成测试在未准备数据时会明确跳过；恢复本地数据后可运行全部测试。

## 单基金：先核对曲线，再回测

先提供基金代码和买卖规则。预览阶段只展示基金自身的分红再投资累计收益，不执行策略；用户确认曲线后再运行指定规则，汇总2010年至今年度盈亏、累计投入和XIRR等指标。成立前无数据年份标为未成立，当前年份只统计最新实际净值日期。

```bash
pip install -e '.[preview]'
# 保存最新可获得的原始净值及分红页面；使用新的快照目录
python scripts/prepare_single_fund.py --code 519981 --rule Fundg --snapshot-dir data/single/519981/20261111
# 已保存原始响应时，用 --from-raw 离线重建，不再次下载
python scripts/prepare_single_fund.py --code 519981 --rule Fundg --snapshot-dir data/single/519981/20261111 --from-raw
python scripts/render_fund_curve.py --data data/single/519981/20261111/519981.csv --output reports/single_519981_fundg_curve_20261111.html
```

分红只使用除息日，现金换算为元/份，份额折算归一为1:x；新遇到除息日缺净值、重复事件、基金身份不符或净值注释未匹配时停止处理。标准化曲线与原QLab计算器逐点核验，通过后输出离线HTML和PNG。原始响应、净值和报告只保存在本地，不上传GitHub。`--rule`在预览阶段仅记录下一步所需规则，不会运行交易。

用户明确确认这份曲线后，使用**同一数据快照**回测：

```bash
python scripts/backtest_single_fund.py --data data/single/519981/20261111/519981.csv --rule Fundg --curve-confirmed --output reports/single_519981_fundg_2010.html
python scripts/verify_portfolio.py --report reports/single_519981_fundg_2010.html
# 对本次519981的Fundg结果做独立Decimal信号、分红、账本及XIRR核算
python scripts/verify_single_fundg.py
```

`--curve-confirmed`仅在用户已核对该快照时使用；数据或原QLab计算器哈希不一致会拒绝运行。单基金入口复用原买卖引擎与包含提款的XIRR计算，支持 Fundg、Fundf_1，以及明确给定 `--category` 的 Fundd；Fundb/c 需要同类排名池，不能把一只基金伪装成有效排名池。起点默认2010-01-01，截止默认已核对快照的最后净值日。

Fundg 保持近一年高低位置≤60%时买10元、不卖出的原规则（不足一年时使用已有历史）。结果输出年度盈亏、投入、期末资产、交易、FIFO、信号和账户曲线；`.paste_with_header_wan.tsv` 为“累计本金、累计盈亏、XIRR、TWR、回撤、夏普、波动率、2010年至今各年盈亏”的两行表头/数据，`.paste_2010_wan.tsv` 为仅数据的一行，金额单位万元。`.annual_yuan.tsv` 和 `.table.md` 保留元口径，避免单基金小额年度盈亏被万元舍入掩盖。2010等成立前年份显示“未成立/—”，原始账本不补造净值。

原QLab直接比较二进制浮点高低位置，不增加阈值容差；恰好60%的十进制数学边界可能得到略大于60%的浮点值。本次050025出现一处这种情况，正式结果保留QLab原行为，独立审计另列精确数学信号的差异与影响，见 [050025浮点边界说明](docs/fundg-float-boundary-20261111.md)。

096001 的用户确认例外仅为 **2015-01-19 每份0.0187元现金分红**：该除息日缺净值，本次暂不计入，其余12笔正常处理。`configs/096001_fundg_omissions.json` 保存这项明确授权；处理器逐项匹配基金、日期、类型、金额，只允许省略缺少除息日净值的指定事件。如果日后补齐该日真实净值，旧例外会拒绝执行，须重新核对曲线。原始13笔记录继续保留，报告展示该项说明。

```bash
# 使用已核对的原始快照离线复现，不刷新数据
python scripts/prepare_single_fund.py --code 096001 --rule Fundg --snapshot-dir data/single/096001/20261111 --from-raw --approved-omissions configs/096001_fundg_omissions.json
python scripts/backtest_single_fund.py --data data/single/096001/20261111/096001.csv --rule Fundg --curve-confirmed --output reports/single_096001_fundg_2010.html
python scripts/verify_portfolio.py --report reports/single_096001_fundg_2010.html
python scripts/verify_single_fundg.py --code 096001 --data data/single/096001/20261111/096001.csv --report-prefix reports/single_096001_fundg_2010 --require-report
```

本次已核对输入CSV的SHA256为 `f05d89e0ec551269c99e9bebd167b2dd4db8ac9350ba7587d1de7e6af332c80b`，覆盖2011-03-23至2026-11-09共3735条净值。需要相同原始快照才能精确复现；净值、原始响应与结果文件仍不上传GitHub。

## QLab 五类基金组合回测

新增 `Fundb + Fundc + Fundd + Fundf_1 + Fundg` 共63只基金组合，从 **2010-01-01** 空仓开始，到共同净值截止 **2026-11-06**。直接加载固定版本 QLab 的原始评分、排名和买卖函数，不重新发明策略。资金无限追加，卖出净回款立即取出，金额采用各类别原配置。

```bash
python -m alphatrack.portfolio_cli --config configs/qlab_portfolio.json
# 时间区间可以自定义，但不能超过已保存数据的共同截止日
python -m alphatrack.portfolio_cli --config configs/qlab_portfolio.json --start 2015-01-01 --end 2025-12-31 --output reports/portfolio_custom.html
```

打开 **`reports/qlab_portfolio_2010.html`**：年度盈亏表、年度投入/取出、年末资产、每日盈亏、组合净值与回撤、五类贡献及63只基金明细，均可离线查看。

同名输出包括 `.annual.csv`、`.daily.csv`、`.groups.csv`、`.funds.csv`、`.group_annual.csv`、`.fund_annual.csv`、`.trades.csv`、`.lots.csv`、`.summary.json`；全部信号和组件账本另存 `.signals.csv.gz`、`.components_daily.csv.gz`。逐批明细仍使用FIFO；组合已实现/未实现分解为平均成本，两者总盈亏核对一致。

| 类别 | 数量 | 买入规则摘要 | 自动卖出 |
|---|---:|---|---|
| Fundb | 26 | 原始综合评分前8名，每只每次40元 | 无 |
| Fundc | 7 | 原始综合评分前5名，每只每次40元 | 无 |
| Fundd | 22 | 三周期位置最小值；宽基基数50元，其余10元；1/2/3/4倍 | 原始50档累计卖出本轮90% |
| Fundf_1 | 6 | 近一年位置，基数40元、1/2/3/4倍 | 无 |
| Fundg | 2 | 近一年位置≤60%时每次10元 | 无 |

Fundb/c 的排名要求1/2/3/5年收益输入均有效、横截面评分可计算；仍沿用 QLab 的简单年化、四位小数舍入和标准化。源码使用目标日期或更早的最近净值，不保证中间历史连续。较晚成立基金只在有历史后参与；起点前历史只预热指标，不回放持仓。当前基金名单回溯存在名单选择/幸存者偏差，并非2010年事先可知的基金池。

**年度盈亏 = 年末资产 − 年初资产 − 当年投入 + 当年取出**，包含持仓浮盈亏变化，不能当成全部已取走现金。只有 Fundd 有自动卖出，不给其他类别添加清仓。报告将账户TWR、含提款的完整现金流XIRR、逐批年化分开；2026金额仅为截至11月6日的YTD。

本次快照结果：累计投入1,969,000元，取出572,360.13元，期末资产2,161,028.63元，总盈亏764,388.75元；账户净值年化5.26%，资金XIRR7.71%，净值最大回撤37.04%。结果是**净值观察序列的理论回测**：费率为0，未建模历史限购/暂停、确认与到账时差、QDII实际净值公布时刻；每个信号在下一净值观察日成交，存在157笔周末观察日理论成交。风险指标按252观察期/年，无风险利率0。

数据局限：000037从2013-06-20跳至2020-04-22，存在2498天断档；000931在2017-03-22、2017-11-14两次分红日无NAV，沿用QLab已有遗漏（主要影响历史评分）。不移动事件日期、不补造净值、不扩展到其他缺失情况。分红采用QLab快照；远程复核37只一致、25只限流、1只疑似错误新增未采纳，所有截止日前NAV事件注释均已与本地记录匹配（上述两条已知缺失除外）。

`vendor/qlab/PROVENANCE.json` 固定源提交 `1c1eb37f45a298b0aec3fabb673d269fa95892d5` 和73个文件哈希。`data/qlab_portfolio/manifest.json` 固定63份净值、来源与质量说明。这些数据文件不在 GitHub 当前版本内；按数据准备说明恢复后，上面的回测命令无需联网或外部QLab目录。

新增模块：`strategies/qlab.py` 精确规则适配；`portfolio.py` 组合聚合、年度指标与多根XIRR检测；`portfolio_cli.py` 运行与导出；`portfolio_report.py` 离线报告。

本次两个策略、四个窗口的独立计算复核见 [计算审计说明](docs/calculation-audit-20261111.md)。可复跑脚本从原始净值、分红与交易记录重建账本、现金流和图表，审计结果保存在本地 `reports/audit_20261111_*.json`。

## 默认固定区间：从2010年起，每5年一个起点

打开 `reports/qlab_portfolio_windows.html`，顶部用表格展示四个区间，每个区间一行，列出起止日期、累计投入、累计盈亏、XIRR、净值年化、最大回撤及完整报告入口；窄屏支持横向滚动并固定区间名称。下方保留**每个自然年的盈亏金额**矩阵、详细指标和收益曲线。

```bash
python -m alphatrack.window_cli --config configs/qlab_portfolio.json --as-of 2026-11-10 --output reports/qlab_portfolio_windows.html
# 已完成计算后，只重新生成汇总页面
python -m alphatrack.window_cli --as-of 2026-11-10 --render-only
# 自定义固定起点，覆盖默认五年梯度
python -m alphatrack.window_cli --fixed-starts 2015-01-01 2020-01-01
# 如需恢复滚动窗口，可显式指定；此例只运行滚动窗口
python -m alphatrack.window_cli --years 1 3 5 10 --no-fixed-start
```

省略 `--as-of` 时使用上海当天日期。默认不添加滚动窗口，只生成从2010年开始、间隔5年的固定起点，按时间升序排列；起点不得晚于实际共同截止日。因此即使运行日期到了2030年，数据仍停在2029年时也不会提前生成2030区间。`--fixed-starts`（原 `--fixed-start` 也可用）覆盖默认起点；`--years` 可额外添加滚动年数，配合 `--no-fixed-start` 可只运行滚动窗口。所有区间共同截止于“日期锚点和全部基金最新净值的较早者”。本次日期锚点为2026-11-10，实际共同净值截止2026-11-06。

**四组分别从空仓、零资金和清空的卖出档位开始，各区间结果不能通过截取更长窗口账本得到，也不能把四组利润相加。**起点前数据只用于指标预热；各组继续按原金额无限追加资金，卖出净回款取出。四组首年分别为完整的2010、2015、2020、2025年，2026年仅截至11月6日，不将部分年度金额外推为全年。

| 窗口 | 固定起点 | 本次实际截止日 |
|---|---|---|
| 2010年至今 | 2010-01-01 | 2026-11-06 |
| 2015年至今 | 2015-01-01 | 2026-11-06 |
| 2020年至今 | 2020-01-01 | 2026-11-06 |
| 2025年至今 | 2025-01-01 | 2026-11-06 |

汇总附件：`.annual.csv` 为四组逐年完整明细，`.annual_matrix.csv` 为年×窗口盈亏矩阵，`.summary.csv` 为区间指标，`.summary.json` 保存日期和各组结果，`fixed_starts` 记录实际起点，`fixed_start_policy` 记录默认起始年和五年步长。CSV以 `window_id/window_label/window_kind` 区分窗口，固定起点的 `window_years` 留空。四份 `_since_20100101/_since_20150101/_since_20200101/_since_20250101` 本地报告各有完整账本、交易、FIFO和验证JSON；可通过 `python scripts/verify_portfolio.py --report reports/qlab_portfolio_windows_since_20100101.html` 等逐个核对。此前的1/3/5/10年滚动窗口及2016年起的旧报告已清理，可用自定义区间命令重新生成。

## 策略二：AlphaStep 随机选基版

只把 Fundb / Fundc 的原评分排序替换为每日随机顺序，分别选8 / 5只、每只40元。保留原as-of合格候选池（1/2/3/5年收益和评分有效，含沿用最近净值的候选），不放回抽取；不足名额时全选。仍只在入选基金有新净值的当天发信号，无新净值不下单、不递补，下一条可用净值成交。Fundd、Fundf_1、Fundg 的信号、交易及账本与同起点原策略逐值一致。

随机种子固定为 **42**，使用种子、类别、日期、基金代码的 SHA256 排序，每日重抽；同一日期的选择不受回测起点或基金遍历顺序影响。配置在 `configs/qlab_portfolio_random.json` 的 `strategy` 中，`selection_mode` 为 `random`、`random_seed` 为整数。原配置省略此项，继续按 `ranked` 执行。

```bash
# 按默认四个固定起点分别计算，独立输出，不覆盖原排名版
python -m alphatrack.window_cli --config configs/qlab_portfolio_random.json --as-of 2026-11-10
# 用已完成的相同策略/种子/日期结果重新渲染
python -m alphatrack.window_cli --config configs/qlab_portfolio_random.json --as-of 2026-11-10 --render-only
# 导出万元表格及可直接粘贴飞书的行
python scripts/export_window_tables.py --report reports/alphastep_random_windows.html
# 与原排名版独立核对
python scripts/verify_random_variant.py
```

打开 `reports/alphastep_random_windows.html` 查看四窗口逐年盈亏、累计盈亏曲线和完整指标；`reports/alphastep_random_strategy.html` / `.png` / `.svg` 是策略流程图。每个 `_since_YYYY0101` 报告带完整日账本、交易、FIFO、年度和分组明细以及 `.validation.json`。

| 区间 | 累计投入（元） | 随机版盈亏（元） | 相对原排名版盈亏差（元） | XIRR |
|---|---:|---:|---:|---:|
| 2010年至今 | 1,969,040 | 759,312.96 | -5,075.79 | 7.67% |
| 2015年至今 | 1,728,500 | 597,136.06 | -5,075.79 | 6.52% |
| 2020年至今 | 1,237,600 | 279,024.84 | -1,910.85 | 6.18% |
| 2025年至今 | 322,150 | 11,320.89 | +412.71 | 4.21% |

均截止2026-11-06，盈亏含未实现浮盈亏；这是seed 42的一条随机路径，不代表多种子的平均表现。沿用原净值观察日历，包括少量周末净值。2010/2015窗口比原版各多投入40元，来自2017-12-30抽中有新净值的700005、次日净值成交；未改变日历、金额或成交规则。差异明细见 `reports/alphastep_random_windows.variant_validation.json`。

复制数据：`.table_wan.md` / `.table_wan.tsv` 为“年份×区间”表，累计盈亏下方附XIRR、TWR、回撤、夏普及波动率；`.paste_all_wan.tsv` 是不含表头的四行横向数据，列顺序为累计盈亏、五项指标、2010—2026年盈亏；`.paste_2010_wan.tsv` 只保留2010年至今一行，配套 `.paste_header_wan.tsv` 为表头。金额按万元保留两位小数，原始CSV仍保留元口径。

## 单基金 Demo

可复现的回测软件原型：数据 → 策略 → 下一净值日成交 → 资金/持仓账本 → 逐批匹配 → 年化收益 → 离线 Plotly HTML。

首个 Demo 使用 **110020 易方达沪深300ETF联接A + QLab Fundd 宽基规则**。资金无限，按规则买50/100/150/200元；**卖出净回款立即移出，卖出后的闲置时间和收益均不纳入评价**。重点呈现每批资金买入到卖出期间的年化收益，未卖出的部分单独标为未实现。

## 快速运行

Python 3.11+：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
# 首次克隆需先恢复独立保存的数据；本机已有匹配数据时可跳过
python scripts/prepare_local_data.py --from /path/to/saved/AlphaTrack
python -m alphatrack --config configs/110020_fundd.json
pytest -q
```

打开 `reports/110020_fundd.html`；报告内置Plotly，无需联网或服务器。GitHub文件预览不会执行HTML，请下载后打开。

同时生成 `.lots.csv`（逐批年化明细）、`.daily.csv`、`.trades.csv`、`.summary.json`。明细保留全部真实数值，不截断极端年化值。

```bash
python -m alphatrack --start 2015-01-01 --end 2025-12-31 --output reports/custom.html
python scripts/update_eastmoney.py --code 110020 --output data/110020.csv
```

JSON中路径相对于配置文件，命令行路径相对于工作目录。可配置策略、日期、买卖金额/阈值/费用、资金模式。日期外历史只用于历史位置特征，不回放期初持仓。省略日期使用完整数据覆盖期，不虚构最新日期净值。

## 逐批年化：当前报告的主要口径

默认 `report_mode="lots"`，`lot_analysis={"matching":"fifo","year_days":365.0}`。

1. 每次成功买入建立一批份额，保留原始日期和含申购费总成本。
2. 卖出按**先进先出FIFO**匹配；一次卖出跨越多批买入，就拆成多条匹配记录。一次买入被分次卖出，也会对应多条记录。这是归属约定，不是声称份额存在唯一物理对应。
3. 成本按本次匹配份额占该批剩余份额的比例分摊；卖出手续费按匹配份额分摊。回款为扣除赎回费后的金额。
4. 每条已实现记录：`持有收益率=净回款/分摊买入成本−1`；`年化收益率=(净回款/分摊买入成本)^(365/持有自然日)−1`。
5. 剩余份额按回测末日净值估值，使用买入至估值日的天数计算**未实现年化**。卖出日期留空、估值日期另列，不伪造卖出、不扣假设赎回费。
6. 红利再投资份额归属原买入批次，成本不增加；拆分保持成本不变，按比例改变份额。本次110020快照没有分红/拆分事件。
7. 持有0天时年化无定义，显示“—”；浮点溢出显示计算限制，不能当成0%收益。不足30天的年化额外提示，短期收益外推并不代表能够持续一年。

**主图纵轴是每条匹配记录的年化收益率，横轴是卖出日或期末估值日。每个点是一段资金持有区间，不是账户累计收益曲线。** 悬浮可见买入/卖出日期、金额、份额、持有天数、盈亏及年化。未实现记录与已实现记录分开展示。

报告中的已实现/未实现年化中位数只是各匹配片段等权分布统计，**不是整体投资年化，也不是账户XIRR**。不可简单平均逐笔年化后称之为整体回报。FIFO会影响逐批收益归属和分布，不改变账户总盈亏。拆分频率会影响等权分布，所以完整成本、份额和日期一起提供。

## Fundd 规则

来源 `https://github.com/yqyang733/QLab` 的计算器、分析器、交易配置/引擎/服务和基金分类表；已审计提交 `1c1eb37f45a298b0aec3fabb673d269fa95892d5`。

- 在分红再投资总收益曲线上计算近5年、近10年、成立以来 `(当前−最低)/(最高−最低)`，取三者最小值x。不是统计排名。窗口按日历年、含当天、只使用当时已有历史；不足两条不交易、平盘窗口为50%。
- 110020宽基基数50：`10%≤x≤20%`买50；`5%≤x<10%`买100；`0<x<5%`买150；`x≤0%`买200。每个符合条件的观察日都会产生信号。
- 卖出阈值50%至100%等距50档，每档为本轮初始份额的 `0.9×i/(1+…+50)`，转换成当时剩余份额比例执行。累计卖90%，余10%。跨档依次执行、每轮每档一次、进入买入区间重置、买入优先。
- QLab“结合持有收益判断”只是备注，没有正收益才卖出的硬条件，此处不添加。

## 执行与资金记账

- 成立首日2009-08-26只有一条净值，因此HOLD。信号由已公布净值生成，在**下一条可用净值**成交。期末信号无下一净值则保留待成交，不强制清仓。
- 默认本金0、`funding_mode="topup"`、`withdraw_on_sell=true`：每次买入不足金额由外部补入，卖出净额立即提走。后续每笔投入重新计入本金，提走的资金不再留在账户、不产生后续收益。
- `每日盈亏 = 期末账户资产 − 前日资产 − 当日入金 + 当日提款`。
- `累计盈亏 = 期末账户资产 + 累计提款 − 累计入金`，等于所有每日盈亏之和，也等于已卖出净回款+剩余市值−全部买入成本。
- 默认交易费0。申购份额=`总额/(1+申购费率)/净值`，卖出费用=`成交总额×赎回费率`。费用率小数，如0.0015。基金管理费已体现在净值中，不重复扣除。
- 允许小数份额、即时入账；未还原成立封闭期、暂停申赎、限额、到账延迟、税及历史阶梯费率。因此是理论净值成交模型，不是完全可执行历史实盘。
- 只记录真实净值观察日，不伪造周末/缺失日。成立初期净值较稀疏，下一可用净值可能相隔数天。

当前 `.summary.json` 的 `lot_summary` 使用FIFO，是逐批报告口径；`metrics` 中保留的账户已实现/未实现划分采用平均成本，作旧模式兼容。两种划分的**总盈亏一致**，但已实现与未实现的分配不能混用。`net_contributions` 是兼容旧版的累计入金字段；新增 `total_contributions`、`cumulative_withdrawals`、`net_external_contributions` 明确入金、提款、净流入。

## 可选旧账户报告与扩展

需要保留卖出现金时，将 `engine.withdraw_on_sell=false`；需要旧账户图表时，再将 `report_mode="account"`。旧账户报告含时间加权净值、XIRR和风险指标；XIRR仅用于无中途提款的原始现金流类型，提款模式不调用该求解器。旧年化口径ACT/365.25，逐批口径ACT/365，均显式记录，避免混用。

- `data.py`：统一CSV与验证，`date,nav`必需，`dividend`每份现金分红、`split`份额倍数可选。
- `strategies/`：注册策略，因果特征`prepare(frame)`及逐日决策`decide(row)`。
- `engine.py`：下一净值成交、资金流、持仓及平均成本账本。
- `lots.py`：独立FIFO批次拆分、公司行为分配及逐段年化。
- `lot_report.py`：逐批年化Plotly报告；`report.py`保留旧账户视角。
- `metrics.py`：账户指标；`cli.py`组织配置与导出。

单基金入口保留原有接口；新增组合入口支持多基金独立执行后统一记账和评价。尚不支持共享有限预算下的多资产订单分配、任意提取计划、杠杆做空或浏览器内重算。

## 可复核验证

```bash
python scripts/verify_qlab.py --qlab /path/to/QLab
pytest -q
```

110020的4,179条净值与QLab快照规则逐日一致：687 BUY / 61 SELL / 3,431 HOLD，位置误差0。卖出提款不会改变无限资金模式下的买卖份额。测试覆盖次净值成交、无前视、补资费用、提款、分红拆分、FIFO分摊、逐笔年化、未实现估值和账目守恒；数据及规则版本写入元信息便于复现。

单基金初版45项测试、748条逐批年化独立公式核对通过，桌面和手机图表已验证。加入随机策略后全套96项测试及32项子测试通过；原排名版及随机版各四个固定区间均通过 `verify_portfolio.py` 的年度、类别、基金、FIFO及现金流核对。随机版另通过 `verify_random_variant.py` 的非目标类别全量一致、原评分资格一致、随机顺序及跨窗口一致审计。测试覆盖五年梯度、跨入2030年、数据截止滞后、历史锚点、自定义覆盖、随机种子和缓存隔离。63基金评分原有抽查20个日期、524个排名和2144个收益输入，均与QLab原函数一致。
