"""滚动窗口拟合质量优化演示脚本。

运行方式（在 backend 目录下）：

    .venv/bin/python scripts/demo_rolling_fit.py

演示两部分：
A. 同一段观测下相邻 4 日 / 8 日训练窗，移动一个营业日后，
   核对增量引擎与全量重算引擎的计算量、系数、拟合优度与预测值；
B. 修改上一天金额后，短窗立即复核、长窗复核中 / 全部完成两个时点上，
   两档窗口的质量判定与预测入口的下一步动作。
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

# 隔离模式（python -I）下不自动注入当前目录，显式把 backend 根目录加入路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rolling_fit import (
    FullScanFitEngine,
    NoQualifiedWindow,
    ObservationStore,
    PredictionEntry,
    QualityGate,
    RollingFitEngine,
)


def business_days(start: date, count: int) -> list[str]:
    days: list[str] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return days


def title(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def show_fit(name: str, fit) -> None:
    print(
        f"  {name}: n={fit.n}  intercept={float(fit.intercept):.6f}  "
        f"slope={float(fit.slope):.6f}  R²={float(fit.r2):.6f}"
    )


def main() -> None:
    days = business_days(date(2026, 11, 9), 10)
    values = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    window_sizes = (4, 8)

    # ------------------------------------------------------------------ 场景 A
    title("场景 A：4 日 / 8 日相邻窗口移动一个营业日的计算量与结果核对")
    inc = RollingFitEngine(ObservationStore(), window_sizes)
    full = FullScanFitEngine(ObservationStore(), window_sizes)
    for d, v in zip(days[:9], values[:9]):  # 先填满两档窗口
        inc.observe(d, v)
        full.observe(d, v)
    inc.reset_touches()
    full.reset_touches()

    print(f"\n新营业日 {days[9]} 进入，最早营业日 {days[0]} 移出窗口：")
    inc.observe(days[9], values[9])
    full.observe(days[9], values[9])

    print("\n[计算量] 本次滑动触碰的样本条数：")
    print(f"  增量引擎（扣 1 条 + 加 1 条/窗）：{dict(inc.sample_touches)}"
          f"  合计 {inc.total_touches()}")
    print(f"  全量重算（每窗重扫全部样本）    ：{dict(full.sample_touches)}"
          f"  合计 {full.total_touches()}")
    ratio = full.total_touches() / inc.total_touches()
    print(f"  计算量下降为旧实现的 1/{ratio:.0f}")

    print("\n[结果一致性] 同一完整窗口，两种实现逐字段精确比对：")
    for size in window_sizes:
        a, b = inc.fit_window(size), full.fit_window(size)
        show_fit(f"{size}日窗-增量", a)
        show_fit(f"{size}日窗-重算", b)
        same = (a.intercept, a.slope, a.r2, a.sse) == (b.intercept, b.slope, b.r2, b.sse)
        pred_same = a.predict(size) == b.predict(size)
        print(f"  系数/拟合优度一致：{same}；下一营业日预测一致：{pred_same}"
              f"（预测值 {float(a.predict(size)):.4f}）")

    # ------------------------------------------------------------------ 场景 B
    title("场景 B：更正上一天金额，复核中与全部完成时的两档窗口质量与预测")
    inc = RollingFitEngine(ObservationStore(), window_sizes)
    full = FullScanFitEngine(ObservationStore(), window_sizes)
    for d, v in zip(days[:9], values[:9]):
        inc.observe(d, v)
        full.observe(d, v)
    gate = QualityGate(min_r2=0.8)
    entry = PredictionEntry(inc, gate)
    yesterday = days[8]

    print("\n[时点 0] 更正前，两档窗口均达标，预测入口默认采用最短的 4 日窗：")
    plan = entry.predict(horizon=1)
    print(f"  采用窗口={plan.window_size}日 记录#{plan.record_id} "
          f"R²={plan.r2:.4f} 次日预测={plan.targets[0]['value']:.4f}")

    print(f"\n[时点 1] 昨日（{yesterday}）金额 180 更正为 150，短窗立即复核、长窗排队：")
    inc.correct(yesterday, 150)
    for row in entry.evaluate_windows():
        print(f"  {row['window_size']}日窗：状态={row['status_zh']:<4} "
              f"R²={row['r2']} 达标={row['qualified']}（{row['reason']}）")
    try:
        entry.predict(horizon=1)
    except NoQualifiedWindow as exc:
        print(f"  预测入口拒绝出值：{exc.reason}")
    print("  说明：长窗拟合未完成，旧的 R²=1.0 记录已转为「复核中」，不能被认作通过阈值。")

    print("\n[时点 2] 长窗重新计算全部完成：")
    inc.process_recheck()
    full.correct(yesterday, 150)
    for row in entry.evaluate_windows():
        print(f"  {row['window_size']}日窗：状态={row['status_zh']:<4} "
              f"R²={row['r2']} 达标={row['qualified']}（{row['reason']}）")
    plan = entry.predict(horizon=2)
    print(f"  预测入口改用 {plan.window_size} 日窗（记录#{plan.record_id}），"
          f"未来两期预测={[t['value'] for t in plan.targets]}")
    refit = full.fit_window(8)
    print(f"  与全量重算基准预测值比对：{[float(refit.predict(8)), float(refit.predict(9))]}"
          f" 完全一致")

    print("\n[留痕] 窗口拟合数据集（只追加，报表可查实际采用过的窗口）：")
    for record in inc.dataset.all_records():
        r2_text = "—" if record.r2 is None else f"{record.r2:.4f}"
        print(f"  #{record.record_id:<2} {record.window_size}日窗 截到{record.end_date} "
              f"{record.status.中文:<4} R²={r2_text:<8} 取代#{record.replaces_record_id} "
              f"{record.note}")
    print(f"\n预测入口留痕 {len(entry.prediction_log)} 条，失败与成功均有据可查。")


if __name__ == "__main__":
    main()
