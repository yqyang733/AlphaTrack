# 本地回测结果

此目录只跟踪这份说明。全部 HTML、PNG/SVG、CSV/TSV、汇总 JSON 和校验结果都在本地生成，不上传 GitHub。仓库保留计算代码、配置、测试和复现说明；数据准备步骤见根目录 README。

当前本地快照截至2026-11-06。两套策略均独立计算2010、2015、2020、2025年起的四个区间。

| 本地文件前缀 | 策略 |
|---|---|
| `qlab_portfolio_windows` | 原排名版 |
| `alphastep_random_windows` | 每日随机选基版，seed 42 |

各前缀的 `.html` 是完整交互报告，`.table_wan.md` 是万元表格，`.paste_2010_wan.tsv` 是2010年起的一行飞书粘贴数据，`.summary.csv` 是区间指标，`.annual_matrix.csv` 是年度盈亏矩阵。每个 `_since_YYYY0101` 前缀另有完整账本、逐笔交易和校验文件。随机策略图为 `alphastep_random_strategy.png` / `.svg` / `.html`。

旧的1/3/5/10年滚动窗口和2016年起的报告已清理，默认四个固定起点的本地结果保留。此次精简当前版本及后续提交，未改写已发布的 Git 历史，因此历史中的大文件仍占用仓库历史空间。

## 重建本地报告

在仓库根目录激活 Python 环境，先恢复并核验单独保管的数据快照：

```bash
python scripts/prepare_local_data.py --from /path/to/saved/AlphaTrack
python scripts/prepare_local_data.py --check
python -m alphatrack.window_cli --config configs/qlab_portfolio.json --as-of 2026-11-10
python -m alphatrack.window_cli --config configs/qlab_portfolio_random.json --as-of 2026-11-10
python scripts/export_window_tables.py --report reports/qlab_portfolio_windows.html
python scripts/export_window_tables.py --report reports/alphastep_random_windows.html
python scripts/verify_random_variant.py
```

随后打开本地 `reports/qlab_portfolio_windows.html` 或 `reports/alphastep_random_windows.html`。若更新数据并要运行到最新日期，可省略 `--as-of`。单基金 Demo 用 `python -m alphatrack --config configs/110020_fundd.json` 生成。

`--render-only` 需要本地各窗口的 `.summary.json` 和 `.daily.csv`；新克隆必须先准备数据并完整运行。需要重新核对账目时，对相应窗口执行，例如：

```bash
python scripts/verify_portfolio.py --report reports/alphastep_random_windows_since_20100101.html
```

金额包含未实现浮盈亏；四个窗口不能相加。随机版是固定种子的一条路径，并非多种子平均表现。
