"""预测入口：只依据实际有效的训练窗口与预测数据出值。

准入顺序：

1. 窗口记录必须是 VALID（复核中、窗口未满一律排除，即使历史 R² 曾达标）；
2. 记录的数据版本必须与观测仓库当前版本一致；
3. 实际 R² 通过质量门槛；
4. 在全部合格窗口里按偏好选择（默认优先短窗，响应更快、更正后恢复更早）。

无合格窗口时抛出 :class:`NoQualifiedWindow`，并在预测报告里留痕，
报表仍可查到各档窗口当时的实际状态与实际采用过的窗口。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Sequence

from app.services.rolling_fit.dataset import FitDataset, FitRecord
from app.services.rolling_fit.engine_rolling import RollingFitEngine
from app.services.rolling_fit.observations import ObservationStore
from app.services.rolling_fit.quality import QualityGate
from app.services.rolling_fit.regression import FitResult

# 选择偏好：传入按 window_size 排序后的合格记录，返回选中的一条
WindowPreference = Callable[[Sequence[FitRecord]], FitRecord]


def prefer_shortest(records: Sequence[FitRecord]) -> FitRecord:
    """默认偏好：同等达标时优先最短窗口（更正后最早恢复有效）。"""
    return min(records, key=lambda r: r.window_size)


def prefer_longest(records: Sequence[FitRecord]) -> FitRecord:
    """可选偏好：优先最长窗口（样本更多、估计更稳）。"""
    return max(records, key=lambda r: r.window_size)


class NoQualifiedWindow(RuntimeError):
    """没有任何窗口同时满足有效、版本最新、达到质量门槛。"""

    def __init__(self, reason: str, evaluation: list[dict[str, object]]):
        super().__init__(reason)
        self.reason = reason
        self.evaluation = evaluation


@dataclass(frozen=True)
class PredictionResult:
    """一次预测的完整留痕信息。"""

    window_size: int
    record_id: int
    end_date: str
    r2: float
    intercept: float
    slope: float
    targets: list[dict[str, object]]
    evaluation: list[dict[str, object]]

    def as_dict(self) -> dict[str, object]:
        return {
            "window_size": self.window_size,
            "record_id": self.record_id,
            "end_date": self.end_date,
            "r2": self.r2,
            "intercept": self.intercept,
            "slope": self.slope,
            "targets": self.targets,
            "evaluation": self.evaluation,
        }


@dataclass
class PredictionEntry:
    """预测入口：封装门槛判断、窗口选择与出值。"""

    engine: RollingFitEngine
    gate: QualityGate = field(default_factory=QualityGate)
    preference: WindowPreference = prefer_shortest
    prediction_log: list[dict[str, object]] = field(default_factory=list)

    @property
    def dataset(self) -> FitDataset:
        return self.engine.dataset

    @property
    def store(self) -> ObservationStore:
        return self.engine.store

    def evaluate_windows(self) -> list[dict[str, object]]:
        """逐档给出判定，明确哪档达标可预测、哪档仍在等待复核。"""
        result: list[dict[str, object]] = []
        for size in self.engine.window_sizes:
            record = self.dataset.latest_for_window(size)
            revision_now = {
                d: self.store.revision_of(d) for d in self.store.window_dates(size)
            }
            ok, reason = self.gate.evaluate_with_store(record, revision_now)
            result.append({
                "window_size": size,
                "record_id": record.record_id if record else None,
                "status": record.status.value if record else "NONE",
                "status_zh": record.status.中文 if record else "无记录",
                "r2": record.r2 if record else None,
                "qualified": ok,
                "reason": reason,
            })
        return result

    def _qualified_records(self, evaluation: list[dict[str, object]]) -> list[FitRecord]:
        qualified_ids = {row["record_id"] for row in evaluation if row["qualified"]}
        return [r for r in self.dataset.valid_windows() if r.record_id in qualified_ids]

    def predict(
        self,
        horizon: int = 1,
        *,
        forecast_dates: Sequence[str] | None = None,
    ) -> PredictionResult:
        """按质量规则选择已达门槛的窗口并出预测；无合格窗口则抛错并留痕。"""
        evaluation = self.evaluate_windows()
        candidates = self._qualified_records(evaluation)
        if not candidates:
            reason = "没有达到拟合质量阈值且数据版本有效的窗口，暂不预测"
            self._log(None, evaluation, reason, forecast_dates)
            raise NoQualifiedWindow(reason, evaluation)

        chosen_record = self.preference(candidates)
        size = chosen_record.window_size
        # 窗口非复核中时直接取引擎精确拟合结果，保证与基准实现逐值一致
        fit: FitResult = self.engine.fit_window(size)
        n = fit.n
        targets: list[dict[str, object]] = []
        for step in range(1, horizon + 1):
            x = n - 1 + step  # 窗口内序号口径，最后一个样本是 n-1
            value = float(fit.predict(x))
            target: dict[str, object] = {"step": step, "x": x, "value": round(value, 6)}
            if forecast_dates is not None and step - 1 < len(forecast_dates):
                target["date"] = forecast_dates[step - 1]
            targets.append(target)

        result = PredictionResult(
            window_size=size,
            record_id=chosen_record.record_id,
            end_date=chosen_record.end_date,
            r2=chosen_record.r2 if chosen_record.r2 is not None else fit.r2_float,
            intercept=round(fit.intercept_float, 6),
            slope=round(fit.slope_float, 6),
            targets=targets,
            evaluation=evaluation,
        )
        self._log(result, evaluation, "ok", forecast_dates)
        return result

    def try_predict(
        self,
        horizon: int = 1,
        *,
        forecast_dates: Sequence[str] | None = None,
    ) -> dict[str, object]:
        """不抛错版本：失败时返回各档判定，方便上游直接展示等待原因。"""
        try:
            result = self.predict(horizon, forecast_dates=forecast_dates)
        except NoQualifiedWindow as exc:
            return {"ok": False, "reason": exc.reason, "evaluation": exc.evaluation}
        return {"ok": True, **result.as_dict()}

    def _log(
        self,
        result: PredictionResult | None,
        evaluation: list[dict[str, object]],
        outcome: str,
        forecast_dates: Sequence[str] | None,
    ) -> None:
        self.prediction_log.append({
            "at": datetime.now().isoformat(timespec="seconds"),
            "outcome": outcome,
            "window_size": result.window_size if result else None,
            "record_id": result.record_id if result else None,
            "forecast_dates": list(forecast_dates) if forecast_dates else None,
            "evaluation": evaluation,
        })
