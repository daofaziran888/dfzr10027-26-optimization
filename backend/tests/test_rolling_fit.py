"""滚动窗口拟合质量子系统测试。

覆盖题目要求的四个场景：
1. 新旧实现对同一完整窗口的系数/拟合优度/预测值完全一致；
2. 相邻 4 日、8 日窗口移动一个营业日的计算量与重算结果；
3. 修改上一天金额后，复核中与全部完成时两档窗口的质量与下一步预测；
4. 更正日期滑出窗口时旧贡献扣除、复核任务作废，不混用新旧分子。
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from app.services.rolling_fit import (
    FitStatus,
    FullScanFitEngine,
    NoQualifiedWindow,
    ObservationStore,
    PredictionEntry,
    QualityGate,
    RollingFitEngine,
    fit_linear,
)
from app.services.rolling_fit.regression import RegressionAccumulator


def business_days(start: date, count: int) -> list[str]:
    days: list[str] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return days


def make_engines(
    window_sizes: tuple[int, ...] = (4, 8),
) -> tuple[RollingFitEngine, FullScanFitEngine]:
    inc = RollingFitEngine(ObservationStore(), window_sizes)
    full = FullScanFitEngine(ObservationStore(), window_sizes)
    return inc, full


def feed(engines, days, values):
    for d, v in zip(days, values):
        for engine in engines:
            engine.observe(d, v)


class TestRegressionMath:
    def test_perfect_line_known_values(self):
        fit = fit_linear([(0, 1), (1, 2), (2, 3), (3, 4)])
        assert fit.slope == 1
        assert fit.intercept == 1
        assert fit.r2 == 1
        assert fit.predict(4) == 5

    def test_constant_y_has_unit_r2(self):
        fit = fit_linear([(0, 7), (1, 7), (2, 7), (3, 7)])
        assert fit.slope == 0
        assert fit.intercept == 7
        assert fit.r2 == 1

    def test_accumulator_revise_equals_fresh_rebuild(self):
        """修订样本：扣旧贡献+加新贡献的结果必须与按新值整窗重建完全相等。"""
        points = [(0, 10), (1, 12), (2, 9), (3, 15), (4, 11)]
        acc = RegressionAccumulator()
        for x, y in points:
            acc.add(x, y)
        # 把第 4 个样本（x=3）从 15 修订为 30
        acc.revise(3, 15, 30)
        corrected = [(0, 10), (1, 12), (2, 9), (3, 30), (4, 11)]
        fresh = RegressionAccumulator()
        for x, y in corrected:
            fresh.add(x, y)
        got, expected = acc.fit(), fresh.fit()
        assert got.intercept == expected.intercept
        assert got.slope == expected.slope
        assert got.r2 == expected.r2
        assert got.sse == expected.sse

    def test_remove_below_empty_rejected(self):
        acc = RegressionAccumulator()
        acc.add(0, 1)
        acc.remove(0, 1)
        with pytest.raises(ValueError):
            acc.remove(0, 1)


class TestNewOldParity:
    def test_same_coefficients_r2_and_predictions_on_every_window(self):
        days = business_days(date(2026, 11, 9), 9)
        values = [100, 108, 121, 117, 133, 129, 146, 141, 155]
        inc, full = make_engines()
        feed((inc, full), days, values)

        for size in (4, 8):
            fit_inc = inc.fit_window(size)
            fit_full = full.fit_window(size)
            # 精确分数逐字段一致，不只是浮点近似
            assert fit_inc.intercept == fit_full.intercept
            assert fit_inc.slope == fit_full.slope
            assert fit_inc.r2 == fit_full.r2
            assert fit_inc.sse == fit_full.sse
            assert fit_inc.sst == fit_full.sst
            for x in (4, 5, 6):
                assert fit_inc.predict(x) == fit_full.predict(x)

        # 数据集里固化的实际拟合优度也一致
        for size in (4, 8):
            r_inc = inc.dataset.latest_valid(size)
            r_full = full.dataset.latest_valid(size)
            assert r_inc is not None and r_full is not None
            assert r_inc.r2 == r_full.r2
            assert r_inc.intercept == r_full.intercept
            assert r_inc.slope == r_full.slope
            assert r_inc.data_revision == {d: 1 for d in days[-size:]}

    def test_parity_after_correction_and_recheck(self):
        days = business_days(date(2026, 11, 9), 9)
        values = [100, 108, 121, 117, 133, 129, 146, 141, 155]
        inc, full = make_engines()
        feed((inc, full), days, values)
        target_day = days[-1]
        inc.correct(target_day, 132)
        full.correct(target_day, 132)
        inc.process_recheck()

        for size in (4, 8):
            fit_inc = inc.fit_window(size)
            fit_full = full.fit_window(size)
            assert fit_inc.intercept == fit_full.intercept
            assert fit_inc.slope == fit_full.slope
            assert fit_inc.r2 == fit_full.r2
            assert fit_inc.predict(4) == fit_full.predict(4)
            record = inc.dataset.latest_valid(size)
            assert record.data_revision[target_day] == 2
            assert record.r2 == round(fit_full.r2_float, 6)


class TestSlideComputation:
    def test_one_business_day_slide_touches_and_results(self):
        days = business_days(date(2026, 11, 9), 10)
        values = [100, 108, 121, 117, 133, 129, 146, 141, 155, 149]
        inc, full = make_engines()
        # 先喂前 9 天把两档窗口填满
        feed((inc, full), days[:9], values[:9])
        inc.reset_touches()
        full.reset_touches()

        # 移动一个营业日（第 10 天进入，最早一天移出）
        inc.observe(days[9], values[9])
        full.observe(days[9], values[9])

        # 增量引擎每档窗口只触碰「扣 1 条 + 加 1 条」= 2 条样本
        assert inc.sample_touches == {4: 2, 8: 2}
        # 旧实现每档窗口都重扫整窗
        assert full.sample_touches == {4: 4, 8: 8}
        assert inc.total_touches() == 4
        assert full.total_touches() == 12  # 计算量降到原来的 1/3

        # 窗口成员确实只平移了一天
        state4 = inc.state_of(4)
        assert state4.member_dates() == days[-4:]
        state8 = inc.state_of(8)
        assert state8.member_dates() == days[-8:]

        # 计算量下降但结果不变
        for size in (4, 8):
            a, b = inc.fit_window(size), full.fit_window(size)
            assert (a.intercept, a.slope, a.r2, a.sse) == (b.intercept, b.slope, b.r2, b.sse)

    def test_filling_phase_incomplete_records(self):
        days = business_days(date(2026, 11, 9), 3)
        inc = RollingFitEngine(ObservationStore(), (4, 8))
        records = []
        for d, v in zip(days, [1, 2, 3]):
            records.extend(inc.observe(d, v))
        assert all(r.status is FitStatus.INCOMPLETE for r in records)
        assert inc.dataset.latest_valid(4) is None


class TestCorrectionWorkflow:
    """场景：两档窗口共用同一批观测，昨日金额更正后的分流处理。"""

    def _setup(self):
        days = business_days(date(2026, 11, 9), 9)
        # 完美线性数据，初始两档窗口 R² 都为 1
        values = [100, 110, 120, 130, 140, 150, 160, 170, 180]
        inc, full = make_engines()
        feed((inc, full), days, values)
        return days, values, inc, full

    def test_short_immediate_long_pending_after_correction(self):
        days, values, inc, full = self._setup()
        entry = PredictionEntry(inc, QualityGate(min_r2=0.8))
        yesterday = days[-1]

        outcome = inc.correct(yesterday, 150)  # 180 -> 150
        assert outcome["pending_windows"][0]["window_size"] == 8

        short_record = inc.dataset.latest_for_window(4)
        long_record = inc.dataset.latest_for_window(8)
        # 短窗立即复核完成：新 VALID 记录，基于 v2
        assert short_record.status is FitStatus.VALID
        assert short_record.data_revision[yesterday] == 2
        # 长窗仍在复核：不能被认作已通过阈值
        assert long_record.status is FitStatus.RECHECKING
        assert inc.is_pending(8) is True
        assert inc.dataset.latest_valid(8) is None

        evaluation = entry.evaluate_windows()
        by_size = {row["window_size"]: row for row in evaluation}
        assert by_size[8]["qualified"] is False
        assert "复核中" in by_size[8]["reason"]
        # 复核中记录没有实际系数，无法冒充拟合结果
        assert long_record.r2 is None and long_record.slope is None

    def test_short_fails_gate_long_blocked_then_long_takes_over(self):
        days, values, inc, full = self._setup()
        yesterday = days[-1]
        entry = PredictionEntry(inc, QualityGate(min_r2=0.8))

        # 更正前两档都达标，可以预测
        before = entry.predict(horizon=1)
        assert before.window_size == 4  # 默认优先短窗

        inc.correct(yesterday, 150)
        # 短窗复核后 R²≈0.018 远低于门槛，长窗复核中 → 此时没有合格窗口
        assert inc.dataset.latest_valid(4).r2 == pytest.approx(0.018182, abs=1e-6)
        with pytest.raises(NoQualifiedWindow) as exc:
            entry.predict(horizon=1)
        assert exc.value.evaluation  # 各档判定仍可查
        failed = entry.try_predict(horizon=1)
        assert failed["ok"] is False

        # 长窗重新计算完成：R²≈0.8182 仍达标，预测入口自动改用 8 日窗
        inc.process_recheck()
        assert inc.is_pending(8) is False
        long_record = inc.dataset.latest_valid(8)
        assert long_record.r2 == pytest.approx(0.818182, abs=1e-6)

        result = entry.predict(horizon=2)
        assert result.window_size == 8
        assert result.record_id == long_record.record_id
        # 与全量重算基准的预测值完全一致
        fit_full = full.fit_window(8)
        # 基准引擎未更正，先同步更正再比对
        full.correct(yesterday, 150)
        fit_full = full.fit_window(8)
        assert result.targets[0]["value"] == round(float(fit_full.predict(8)), 6)
        assert result.targets[1]["value"] == round(float(fit_full.predict(9)), 6)
        assert inc.fit_window(8).r2 == fit_full.r2

        # 预测留痕：失败与成功各一条，报表可查实际采用窗口
        outcomes = [log["outcome"] for log in entry.prediction_log]
        assert outcomes.count("ok") == 2
        assert any(log["outcome"] != "ok" for log in entry.prediction_log)

    def test_history_keeps_actual_window_used(self):
        days, values, inc, _ = self._setup()
        yesterday = days[-1]
        entry = PredictionEntry(inc, QualityGate(min_r2=0.8))
        entry.predict(horizon=1)
        used_record_id = entry.dataset.latest_valid(4).record_id
        inc.correct(yesterday, 150)
        inc.process_recheck()

        # 旧记录仍在数据集里可查（状态已转为复核中），保留了当时实际采用值
        old = inc.dataset.get(used_record_id)
        assert old.status is FitStatus.RECHECKING
        assert old.r2 == 1.0
        all_ids = [r.record_id for r in inc.dataset.all_records()]
        assert len(all_ids) == len(set(all_ids))  # 只追加、不覆盖

    def test_recheck_uses_no_mixed_numerators(self):
        """复核完成值必须等于按修订后数据整窗重算，而不是新旧拼接。"""
        days, values, inc, full = self._setup()
        yesterday = days[-1]
        inc.correct(yesterday, 150)
        inc.process_recheck()
        full.correct(yesterday, 150)
        for size in (4, 8):
            assert inc.fit_window(size).intercept == full.fit_window(size).intercept
            assert inc.fit_window(size).slope == full.fit_window(size).slope


class TestSlideOutDuringRecheck:
    def test_pending_job_discarded_and_old_contribution_removed(self):
        days = business_days(date(2026, 11, 9), 10)
        values = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
        inc, full = make_engines()
        feed((inc, full), days[:9], values[:9])

        # 更正第 2 个营业日：只落在 8 日窗内，4 日窗不受影响
        inc.correct(days[1], 999)
        assert inc.is_pending(8) is True
        assert inc.is_pending(4) is False

        # 新营业日进入，第 2 天从 8 日窗滑出，待复核任务随之作废，窗口自动恢复有效
        records = inc.observe(days[9], values[9])
        full.observe(days[9], values[9])
        assert inc.is_pending(8) is False
        assert inc.state_of(8).pending_jobs == []
        latest = inc.dataset.latest_valid(8)
        assert latest is not None and latest.status is FitStatus.VALID
        assert days[1] not in latest.data_revision  # 已滑出，更正与该窗口无关
        # 结果与全量重算一致
        assert inc.fit_window(8).slope == full.fit_window(8).slope
        assert inc.fit_window(8).r2 == full.fit_window(8).r2
        assert all(r.window_size != 8 or r.status is not FitStatus.RECHECKING for r in records)

    def test_correction_outside_windows_is_ignored(self):
        days = business_days(date(2026, 11, 9), 9)
        inc = RollingFitEngine(ObservationStore(), (4,))
        for d, v in zip(days, range(100, 109)):
            inc.observe(d, v)
        # 最早一天已不在 4 日窗内
        outcome = inc.correct(days[0], 1)
        assert outcome["pending_windows"] == []
        assert inc.is_pending(4) is False


class TestDatasetExport:
    def test_jsonl_export_roundtrip(self, tmp_path):
        days = business_days(date(2026, 11, 9), 9)
        inc = RollingFitEngine(ObservationStore(), (4, 8))
        for d, v in zip(days, range(100, 109)):
            inc.observe(d, v)
        inc.correct(days[-1], 200)
        target = tmp_path / "fit_dataset.jsonl"
        inc.dataset.export_jsonl(target)
        lines = target.read_text(encoding="utf-8").strip().split("\n")
        payload = [json.loads(line) for line in lines]
        statuses = {row["status"] for row in payload}
        assert "VALID" in statuses and "RECHECKING" in statuses
        # 最新的复核中记录不含伪装的拟合优度；被顶替的旧 VALID 记录转为
        # RECHECKING 后保留历史实际值，供报表追溯当时采用的窗口
        rechecking = [row for row in payload if row["status"] == "RECHECKING"]
        newest = max(rechecking, key=lambda row: row["record_id"])
        assert newest["r2"] is None and newest["slope"] is None
