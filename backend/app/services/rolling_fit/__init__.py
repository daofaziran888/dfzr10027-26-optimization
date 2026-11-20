"""滚动窗口拟合质量子系统。

对外暴露：

- :class:`RollingFitEngine` 增量滚动引擎（滑动/更正按贡献增量维护）
- :class:`FullScanFitEngine` 每次重算的旧实现（基准对照用）
- :class:`ObservationStore` 带版本号的连续观测仓库
- :class:`FitDataset` 窗口拟合数据集（实际拟合优度 + 版本 + 状态）
- :class:`FitRecord` / :class:`FitStatus` 拟合记录与状态
- :class:`QualityGate` 拟合质量门槛
- :class:`PredictionEntry` 预测入口
"""
from app.services.rolling_fit.dataset import FitDataset, FitRecord, FitStatus
from app.services.rolling_fit.engine_full_scan import FullScanFitEngine
from app.services.rolling_fit.engine_rolling import RollingFitEngine
from app.services.rolling_fit.observations import ObservationStore
from app.services.rolling_fit.prediction import NoQualifiedWindow, PredictionEntry, PredictionResult
from app.services.rolling_fit.quality import QualityGate
from app.services.rolling_fit.regression import FitResult, fit_linear

__all__ = [
    "RollingFitEngine",
    "FullScanFitEngine",
    "ObservationStore",
    "FitDataset",
    "FitRecord",
    "FitStatus",
    "QualityGate",
    "PredictionEntry",
    "PredictionResult",
    "NoQualifiedWindow",
    "FitResult",
    "fit_linear",
]
