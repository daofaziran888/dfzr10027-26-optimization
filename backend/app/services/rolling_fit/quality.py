"""拟合质量门槛：判断有效窗口记录是否达到预测准入标准。"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.rolling_fit.dataset import FitRecord, FitStatus


@dataclass(frozen=True)
class QualityGate:
    """R² 门槛规则。

    - 只认真实计算出的 R²，门槛值不会被写进记录冒充拟合优度；
    - 复核中 / 窗口未满的记录直接判不通过，无论其历史 R² 多高；
    - 要求记录的数据版本与观测仓库当前版本逐日期一致，防止用过时拟合出值。
    """

    min_r2: float = 0.8

    def evaluate(self, record: FitRecord | None) -> tuple[bool, str]:
        if record is None:
            return False, "没有该窗口的拟合记录"
        if record.status is not FitStatus.VALID:
            return False, f"窗口状态为「{record.status.中文}」，拟合未完成"
        if record.r2 is None:
            return False, "记录缺少实际拟合优度"
        if record.r2 < self.min_r2:
            return False, f"实际 R²={record.r2:.4f} 低于门槛 {self.min_r2:.4f}"
        return True, f"实际 R²={record.r2:.4f} 达到门槛 {self.min_r2:.4f}"

    def is_qualified(self, record: FitRecord | None) -> bool:
        return self.evaluate(record)[0]

    def evaluate_with_store(
        self, record: FitRecord | None, revision_now: dict[str, int]
    ) -> tuple[bool, str]:
        """额外校验数据版本：记录必须建立在当前观测版本之上。"""
        ok, reason = self.evaluate(record)
        if not ok:
            return ok, reason
        assert record is not None
        if record.data_revision != revision_now:
            return False, "拟合所基于的数据版本已过期，需要重新复核"
        return True, reason
