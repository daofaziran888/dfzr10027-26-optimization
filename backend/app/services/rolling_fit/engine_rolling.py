"""增量滚动窗口拟合引擎。

与"每次重算"的旧实现（:mod:`engine_full_scan`）不同，本引擎为每档窗口维护
一个 :class:`RegressionAccumulator` 充分统计量：

- 新营业日到达：只做 1 次加入（窗口满时再 1 次扣减滑出样本），O(1) 更新，
  不再重扫窗口内全部样本；
- 观测更正：**不在原累加器上热修补丁**，而是先把受影响窗口的现有拟合记录
  置为「复核中」，短窗立即原子重算，长窗进入复核队列，复核完成前绝不产出
  新的有效记录，杜绝修订前分子与修订后样本混用；
- 复核完成：以观测仓库当前版本对整窗原子重建累加器，一次替换，写入新记录。

样本触碰计数（``sample_touches``）如实记录 add/remove/rebuild 触碰的样本条数，
用于和全量重算实现对比计算量。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from app.services.rolling_fit.dataset import FitDataset, FitRecord, FitStatus
from app.services.rolling_fit.observations import ObservationStore
from app.services.rolling_fit.regression import (
    FitResult,
    RegressionAccumulator,
    to_fraction,
)


@dataclass
class CorrectionJob:
    """一次落在窗口内、等待复核的更正。"""

    date: str
    from_revision: int
    to_revision: int

    def as_dict(self) -> dict[str, object]:
        return {
            "date": self.date,
            "from_revision": self.from_revision,
            "to_revision": self.to_revision,
        }


@dataclass
class WindowState:
    """一档窗口的运行时状态。"""

    size: int
    acc: RegressionAccumulator = field(default_factory=RegressionAccumulator)
    members: list[tuple[str, Fraction, int]] = field(default_factory=list)
    # (日期, 金额, 版本) —— 金额始终与累加器内实际包含的贡献保持一致
    record_id: int | None = None
    pending: bool = False
    pending_jobs: list[CorrectionJob] = field(default_factory=list)

    def contains(self, date: str) -> bool:
        return any(d == date for d, _, _ in self.members)

    def member_dates(self) -> list[str]:
        return [d for d, _, _ in self.members]

    def revision_snapshot(self) -> dict[str, int]:
        return {d: rev for d, _, rev in self.members}


class RollingFitEngine:
    """增量维护多档滚动窗口拟合质量的引擎。"""

    def __init__(
        self,
        store: ObservationStore,
        window_sizes: tuple[int, ...] = (4, 8),
        *,
        dataset: FitDataset | None = None,
        immediate_recheck_sizes: tuple[int, ...] | None = None,
    ) -> None:
        if not window_sizes or len(set(window_sizes)) != len(window_sizes):
            raise ValueError("window_sizes 不能为空且不能重复")
        self.store = store
        self.window_sizes = tuple(sorted(window_sizes))
        self.dataset = dataset or FitDataset()
        # 默认最短窗口立即复核，其余窗口排队等待重新计算
        self.immediate_recheck_sizes = (
            tuple(immediate_recheck_sizes)
            if immediate_recheck_sizes is not None
            else (min(self.window_sizes),)
        )
        self._states: dict[int, WindowState] = {size: WindowState(size) for size in self.window_sizes}
        # 样本触碰计数：增量路径每次 add/remove 计 1，整窗重建按成员数计
        self.sample_touches: dict[int, int] = {size: 0 for size in self.window_sizes}

    # ================================================================ 营业日
    def observe(self, date: str, value: object) -> list[FitRecord]:
        """登记一个新营业日的观测，并增量推进所有窗口。"""
        self.store.append(date, value)
        global_x = len(self.store) - 1  # 全局序号，累加器内部用它做 x
        new_value = self.store.get(date).value
        records: list[FitRecord] = []
        for size in self.window_sizes:
            state = self._states[size]
            # 1) 窗口已满：先把最老样本的旧贡献完整扣除
            if len(state.members) >= size:
                old_date, old_value, _ = state.members.pop(0)
                state.acc.remove(global_x - len(state.members) - 1, old_value)
                self.sample_touches[size] += 1
            # 2) 加入新样本贡献
            state.acc.add(global_x, new_value)
            state.members.append((date, new_value, 1))
            self.sample_touches[size] += 1

            # 3) 清理已随滑动移出窗口的待复核更正
            state.pending_jobs = [job for job in state.pending_jobs if state.contains(job.date)]
            if state.pending and not state.pending_jobs:
                # 触发复核的日期已全部滑出，窗口内容与仓库当前版本重新一致
                records.append(self._complete_recheck(state, reason="触发复核的日期已移出窗口"))
                continue

            if len(state.members) < size:
                if state.record_id is None:
                    record = self.dataset.add_incomplete(size, date)
                    state.record_id = record.record_id
                    records.append(record)
                continue

            if state.pending:
                # 累加器仍含修订前贡献，本营业日不能产出有效拟合，只刷新复核中记录
                record = self._add_rechecking_record(state, date, "观测更正后重新计算尚未完成")
                records.append(record)
            else:
                fit = self._fit_state(state)
                record = self.dataset.add_fit(
                    fit,
                    window_size=size,
                    end_date=date,
                    data_revision=state.revision_snapshot(),
                    replaces_record_id=state.record_id,
                )
                state.record_id = record.record_id
                records.append(record)
        return records

    # ================================================================ 更正
    def correct(self, date: str, new_value: object) -> dict[str, object]:
        """更正某营业日金额：短窗立即复核，长窗转入复核中。"""
        if not self.store.has(date):
            raise KeyError(f"业务日期 {date} 不存在")
        old = self.store.get(date)
        revised = self.store.correct(date, new_value)
        affected: list[FitRecord] = []
        for size in self.window_sizes:
            state = self._states[size]
            if not state.contains(date):
                continue  # 该日期已不在这档窗口内，更正对它无影响
            job = CorrectionJob(date, old.revision, revised.revision)
            state.pending_jobs.append(job)
            state.pending = True
            # 现有 VALID 记录立即失效：拟合未完成就不能再被当作通过阈值
            if state.record_id is not None:
                self.dataset.mark_rechecking(
                    state.record_id,
                    f"日期 {date} 金额已更正（v{old.revision}→v{revised.revision}），等待重新计算",
                )
            if size in self.immediate_recheck_sizes:
                affected.append(self._complete_recheck(state, reason="短窗立即复核"))
            else:
                affected.append(
                    self._add_rechecking_record(
                        state, revised.date, f"长窗重新计算排队中：{date} v{revised.revision}"
                    )
                )
        return {
            "date": date,
            "revision": revised.revision,
            "records": [r.as_dict() for r in affected],
            "pending_windows": self.pending_windows(),
        }

    def process_recheck(self, window_size: int | None = None) -> list[FitRecord]:
        """执行复核（长窗的重新计算）。默认处理所有排队中的窗口。"""
        done: list[FitRecord] = []
        sizes = [window_size] if window_size is not None else list(self.window_sizes)
        for size in sizes:
            state = self._states[size]
            if state.pending and state.pending_jobs:
                done.append(self._complete_recheck(state, reason="复核队列处理完成"))
        return done

    def _complete_recheck(self, state: WindowState, *, reason: str) -> FitRecord:
        """以仓库当前版本对整窗原子重建并拟合，一次替换旧累加器。"""
        size = state.size
        end_date = state.members[-1][0]
        rebuilt = RegressionAccumulator()
        start_index = self._global_index_of(state.members[0][0])
        current_dates = self.store.window_dates(size)
        current_revision = {d: self.store.revision_of(d) for d in current_dates}
        for offset, d in enumerate(current_dates):
            rebuilt.add(start_index + offset, self.store.get(d).value)
            self.sample_touches[size] += 1
        state.acc = rebuilt
        state.members = [
            (d, self.store.get(d).value, self.store.revision_of(d)) for d in current_dates
        ]
        fit = self._fit_state(state)
        record = self.dataset.add_fit(
            fit,
            window_size=size,
            end_date=end_date,
            data_revision=current_revision,
            replaces_record_id=state.record_id,
        )
        record.note = reason
        state.record_id = record.record_id
        state.pending = False
        state.pending_jobs = []
        return record

    def _add_rechecking_record(self, state: WindowState, end_date: str, note: str) -> FitRecord:
        """追加一条复核中记录（不写系数，避免被误读为当前有效拟合）。"""
        record = self.dataset.add_incomplete(state.size, end_date)
        record.status = FitStatus.RECHECKING
        record.data_revision = state.revision_snapshot()
        record.note = note
        record.replaces_record_id = state.record_id
        state.record_id = record.record_id
        return record

    # ================================================================ 读取
    def state_of(self, window_size: int) -> WindowState:
        return self._states[window_size]

    def pending_windows(self) -> list[dict[str, object]]:
        result = []
        for size in self.window_sizes:
            state = self._states[size]
            if state.pending:
                result.append({
                    "window_size": size,
                    "end_date": state.members[-1][0] if state.members else None,
                    "jobs": [j.as_dict() for j in state.pending_jobs],
                })
        return result

    def is_pending(self, window_size: int) -> bool:
        return self._states[window_size].pending

    def fit_window(self, window_size: int) -> FitResult:
        """取窗口当前拟合（复核中也可用于诊断，但预测入口不会消费它）。"""
        return self._fit_state(self._states[window_size])

    def total_touches(self) -> int:
        return sum(self.sample_touches.values())

    def reset_touches(self) -> None:
        for size in self.window_sizes:
            self.sample_touches[size] = 0

    # ================================================================ 内部
    def _global_index_of(self, date: str) -> int:
        return self.store.dates.index(date)

    def _fit_state(self, state: WindowState) -> FitResult:
        """把全局序号空间的拟合结果换算成窗口内 x=0..n-1 的口径。

        平移不改变斜率与 R²/SSE/SST，只换算截距，保证与全量重算实现
        （直接对窗口值以 0..n-1 为 x）逐字段相等。
        """
        fit = state.acc.fit()
        start_index = self._global_index_of(state.members[0][0])
        intercept_rel = fit.intercept + fit.slope * start_index
        return FitResult(
            n=fit.n,
            intercept=intercept_rel,
            slope=fit.slope,
            r2=fit.r2,
            sse=fit.sse,
            sst=fit.sst,
        )
