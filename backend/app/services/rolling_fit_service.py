"""滚动窗口拟合服务：管理引擎单例的生命周期与平台状态流转。

路由层不直接操作引擎，所有业务判断（更正分流、复核队列、门槛判定）
都在 :mod:`app.services.rolling_fit` 与本服务内完成。
"""
from __future__ import annotations

from typing import Any

from app.services.rolling_fit import (
    FitDataset,
    FullScanFitEngine,
    ObservationStore,
    PredictionEntry,
    QualityGate,
    RollingFitEngine,
)

DEFAULT_WINDOW_SIZES = (4, 8)
DEFAULT_MIN_R2 = 0.8


class RollingFitService:
    """封装一套滚动窗口拟合流水线的应用服务。"""

    def __init__(self, window_sizes: tuple[int, ...] = DEFAULT_WINDOW_SIZES,
                 min_r2: float = DEFAULT_MIN_R2) -> None:
        self.window_sizes = tuple(sorted(window_sizes))
        self.min_r2 = min_r2
        self.store = ObservationStore()
        self.engine = RollingFitEngine(self.store, self.window_sizes)
        self.entry = PredictionEntry(self.engine, QualityGate(min_r2))
        # 全量重算基准引擎，供接口侧随时做新旧实现一致性核对
        self.baseline = FullScanFitEngine(ObservationStore(), self.window_sizes)

    # ------------------------------------------------------------------ 数据
    def feed(self, points: list[dict[str, Any]]) -> dict[str, Any]:
        """批量灌入营业日观测。"""
        added = []
        for point in points:
            date, value = str(point["date"]), point["value"]
            records = self.engine.observe(date, value)
            self.baseline.observe(date, value)
            added.append({"date": date, "records": [r.record_id for r in records]})
        return {"added": added, "total": len(self.store)}

    def correct(self, date: str, new_value: Any) -> dict[str, Any]:
        """更正某日金额：短窗立即复核，长窗进入复核队列。"""
        result = self.engine.correct(date, new_value)
        self.baseline.correct(date, new_value)
        return result

    def process_recheck(self, window_size: int | None = None) -> dict[str, Any]:
        """执行排队中的长窗复核。"""
        records = self.engine.process_recheck(window_size)
        return {
            "completed": [r.as_dict() for r in records],
            "pending_windows": self.engine.pending_windows(),
        }

    # ------------------------------------------------------------------ 查询
    def dataset_rows(self) -> list[dict[str, Any]]:
        return self.engine.dataset.as_dicts()

    def status(self) -> dict[str, Any]:
        """两档窗口当前状态一览：哪档达标可预测、哪档等待复核。"""
        return {
            "window_sizes": list(self.window_sizes),
            "min_r2": self.min_r2,
            "observations": self.store.as_dicts(),
            "evaluation": self.entry.evaluate_windows(),
            "pending_windows": self.engine.pending_windows(),
            "sample_touches": self.engine.sample_touches,
        }

    def predict(self, horizon: int = 1) -> dict[str, Any]:
        """预测入口：只依据实际有效且过门槛的窗口出值。"""
        return self.entry.try_predict(horizon=horizon)

    def parity_check(self) -> dict[str, Any]:
        """新旧实现一致性核对：系数、拟合优度、预测值逐档精确比对。"""
        checks = []
        for size in self.window_sizes:
            if self.engine.state_of(size).pending or len(self.store) < size:
                checks.append({"window_size": size, "comparable": False,
                               "reason": "窗口复核中或尚未填满，跳过比对"})
                continue
            a, b = self.engine.fit_window(size), self.baseline.fit_window(size)
            checks.append({
                "window_size": size,
                "comparable": True,
                "intercept_equal": a.intercept == b.intercept,
                "slope_equal": a.slope == b.slope,
                "r2_equal": a.r2 == b.r2,
                "sse_equal": a.sse == b.sse,
                "next_prediction_equal": a.predict(size) == b.predict(size),
                "incremental_touches": self.engine.sample_touches[size],
                "full_scan_touches": self.baseline.sample_touches[size],
            })
        return {"checks": checks}

    def export_dataset(self) -> FitDataset:
        return self.engine.dataset


# 模块级单例，路由共享同一套流水线
service = RollingFitService()


def reset_service() -> None:
    """重置单例内部状态，用于测试隔离（保持对象身份不变，路由引用不失效）。"""
    fresh = RollingFitService(service.window_sizes, service.min_r2)
    service.__dict__.update(fresh.__dict__)
