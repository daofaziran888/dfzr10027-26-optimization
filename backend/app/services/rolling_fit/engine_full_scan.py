"""全量重算引擎（优化前的旧实现）。

每来一个营业日，每档窗口都重新遍历窗口内全部样本计算拟合优度；
观测更正后同样整窗重扫。它是结果正确性的基准：增量引擎对同一完整窗口
给出的系数、拟合优度与预测值必须与它**完全一致**（精确分数口径）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.services.rolling_fit.dataset import FitDataset, FitRecord
from app.services.rolling_fit.observations import ObservationStore
from app.services.rolling_fit.regression import FitResult, fit_linear


@dataclass
class FullScanFitEngine:
    """每次都扫描完整窗口的朴素实现，带样本触碰计数用于对照。"""

    store: ObservationStore
    window_sizes: tuple[int, ...] = (4, 8)
    dataset: FitDataset = field(default_factory=FitDataset)
    sample_touches: dict[int, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.window_sizes = tuple(sorted(self.window_sizes))
        self.sample_touches = {size: 0 for size in self.window_sizes}

    def observe(self, date: str, value: object) -> list[FitRecord]:
        self.store.append(date, value)
        return self._recompute_all(date)

    def correct(self, date: str, new_value: object) -> list[FitRecord]:
        self.store.correct(date, new_value)
        end_date = self.store.dates[-1]
        return self._recompute_all(end_date)

    def _recompute_all(self, end_date: str) -> list[FitRecord]:
        records: list[FitRecord] = []
        for size in self.window_sizes:
            values = self.store.window_values(size)
            if len(values) < size:
                records.append(self.dataset.add_incomplete(size, end_date))
                continue
            # 旧实现：重新遍历窗口内全部样本
            self.sample_touches[size] += len(values)
            fit = fit_linear(list(enumerate(values)))
            revision = {d: self.store.revision_of(d) for d in self.store.window_dates(size)}
            records.append(
                self.dataset.add_fit(
                    fit, window_size=size, end_date=end_date, data_revision=revision
                )
            )
        return records

    def fit_window(self, window_size: int) -> FitResult:
        values = self.store.window_values(window_size)
        if len(values) != window_size:
            raise ValueError(f"窗口 {window_size} 尚未填满，当前样本 {len(values)} 个")
        self.sample_touches[window_size] += len(values)
        return fit_linear(list(enumerate(values)))

    def total_touches(self) -> int:
        return sum(self.sample_touches.values())

    def reset_touches(self) -> None:
        for size in self.window_sizes:
            self.sample_touches[size] = 0
