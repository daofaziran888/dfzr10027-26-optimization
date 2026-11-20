# 滚动窗口拟合质量子系统

优化统计模型连续观测数据的滚动窗口拟合质量计算，把**实际拟合优度与数据版本**
固化为窗口拟合数据集，供预测任务按质量规则选择已达门槛的模型。

## 模块结构

| 文件 | 职责 |
| --- | --- |
| `regression.py` | 精确分数（`Fraction`）一元线性回归与充分统计量累加器（加入/扣减/修订） |
| `observations.py` | 带修订版本号的连续观测仓库，保留每次更正的完整历史 |
| `dataset.py` | 只追加的窗口拟合数据集：实际系数、R²、数据版本快照、VALID/复核中/未满状态 |
| `engine_rolling.py` | 增量引擎：滑动 O(1) 更新；更正后短窗立即复核、长窗排队，整窗原子重建 |
| `engine_full_scan.py` | 优化前的全量重算基准实现，带样本触碰计数用于计算量对照 |
| `quality.py` | R² 质量门槛，复核中/版本过期一律不通过 |
| `prediction.py` | 预测入口：只消费 VALID + 版本最新 + 过门槛的窗口，无合格窗口拒绝出值并留痕 |

外层还有 `services/rolling_fit_service.py`（应用服务单例）、
`routers/rolling_fit.py`（HTTP 接口，前缀 `/api/rolling-fit`）、
`scripts/demo_rolling_fit.py`（场景演示）与两个 pytest 文件。

## 关键设计约束（对应业务规则）

1. **新旧实现结果一致**：所有矩用 `Fraction` 维护，增量引擎与全量重算对同一
   完整窗口的 intercept/slope/R²/SSE/预测值精确相等（测试按分数相等断言，
   不是浮点近似）。窗口内序号平移只换算截距，不改变斜率与拟合优度。
2. **滑动不重扫**：新营业日只做「扣 1 条滑出样本 + 加 1 条新样本」，
   4 日/8 日两窗各触碰 2 条样本；旧实现分别重扫 4/8 条，合计 12 条。
3. **更正不混用新旧分子**：更正先把受影响窗口的现有记录置「复核中」，
   再以观测仓库当前版本对整窗原子重建累加器后一次替换，绝不在含旧值的
   累加器上热补。复核完成前不产出新的 VALID 记录。
4. **拟合未完成不能冒充通过阈值**：复核中记录不写系数与 R²；
   `QualityGate` 只认真实计算值，并校验记录的数据版本与仓库当前版本一致。
5. **可追溯**：数据集只追加，被取代的 VALID 记录转为复核中后仍保留，
   报表可查到当时实际采用的窗口；预测入口的每次成功/失败均写入留痕。
6. **滑出自动销账**：触发复核的日期移出窗口后，待复核任务作废，
   窗口按当前数据自动恢复有效。

## 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/rolling-fit/observations/feed` | 批量灌入营业日观测，增量推进窗口 |
| POST | `/api/rolling-fit/observations/correct` | 更正某日金额（短窗立即复核、长窗排队） |
| POST | `/api/rolling-fit/recheck` | 执行复核队列中的长窗重算 |
| GET | `/api/rolling-fit/dataset` | 窗口拟合数据集（实际拟合优度+版本+状态） |
| GET | `/api/rolling-fit/dataset/export` | 导出 JSONL |
| GET | `/api/rolling-fit/status` | 两档窗口质量状态与复核队列 |
| GET | `/api/rolling-fit/predict?horizon=n` | 预测入口，无合格窗口返回 `ok=false` |
| GET | `/api/rolling-fit/parity` | 新旧实现系数/拟合优度/预测值一致性核对 |
