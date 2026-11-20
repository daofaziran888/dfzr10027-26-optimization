"""窗口拟合数据集：把每次拟合的实际拟合优度与数据版本固化下来。

每条 :class:`FitRecord` 回答四个问题：

1. 哪一档窗口（window_size）、截到哪个业务日期（end_date）；
2. 拟合基于观测的哪个版本（data_revision，窗口内各日期版本号的快照）；
3. 实际算出的系数、R²、样本数是多少（actual_*，绝不写门槛值冒充）；
4. 这条记录当前是什么状态：有效 / 复核中（旧值已失效）/ 窗口未满。

数据集**只追加、不改写历史**：更正发生时旧记录转为 ``RECHECKING`` 并保留，
复核完成后写入新的 ``VALID`` 记录。预测入口只能消费 VALID 记录，
而报表仍可通过历史记录查到当时实际采用的窗口。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.services.rolling_fit.regression import FitResult


class FitStatus(str, Enum):
    """拟合记录状态。"""

    VALID = "VALID"            # 已完成拟合，且基于当前数据版本
    RECHECKING = "RECHECKING"  # 观测已更正，新拟合尚未完成；记录数值已过期
    INCOMPLETE = "INCOMPLETE"  # 观测数不足窗口长度，无法拟合

    @property
    def 中文(self) -> str:
        return {
            FitStatus.VALID: "有效",
            FitStatus.RECHECKING: "复核中",
            FitStatus.INCOMPLETE: "窗口未满",
        }[self]


@dataclass
class FitRecord:
    """窗口拟合数据集里的一条记录。"""

    window_size: int
    end_date: str
    status: FitStatus
    data_revision: dict[str, int]          # {业务日期: 拟合时该日期的版本号}
    n: int
    intercept: float | None
    slope: float | None
    r2: float | None
    sse: float | None
    sst: float | None
    record_id: int
    replaces_record_id: int | None = None  # 复核场景下被本条取代的旧记录
    note: str = ""

    @classmethod
    def from_fit(
        cls,
        fit: FitResult,
        *,
        window_size: int,
        end_date: str,
        data_revision: dict[str, int],
        record_id: int,
        replaces_record_id: int | None = None,
    ) -> "FitRecord":
        return cls(
            window_size=window_size,
            end_date=end_date,
            status=FitStatus.VALID,
            data_revision=dict(data_revision),
            n=fit.n,
            intercept=round(fit.intercept_float, 6),
            slope=round(fit.slope_float, 6),
            r2=round(fit.r2_float, 6),
            sse=round(float(fit.sse), 6),
            sst=round(float(fit.sst), 6),
            record_id=record_id,
            replaces_record_id=replaces_record_id,
        )

    @classmethod
    def incomplete(cls, window_size: int, end_date: str, record_id: int) -> "FitRecord":
        return cls(
            window_size=window_size,
            end_date=end_date,
            status=FitStatus.INCOMPLETE,
            data_revision={},
            n=0,
            intercept=None,
            slope=None,
            r2=None,
            sse=None,
            sst=None,
            record_id=record_id,
            note="观测样本不足，窗口尚未填满",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "window_size": self.window_size,
            "end_date": self.end_date,
            "status": self.status.value,
            "status_zh": self.status.中文,
            "data_revision": dict(self.data_revision),
            "n": self.n,
            "intercept": self.intercept,
            "slope": self.slope,
            "r2": self.r2,
            "sse": self.sse,
            "sst": self.sst,
            "replaces_record_id": self.replaces_record_id,
            "note": self.note,
        }


@dataclass
class FitDataset:
    """只追加的窗口拟合数据集。"""

    _records: list[FitRecord] = field(default_factory=list)

    def _next_id(self) -> int:
        return len(self._records) + 1

    def add_fit(
        self,
        fit: FitResult,
        *,
        window_size: int,
        end_date: str,
        data_revision: dict[str, int],
        replaces_record_id: int | None = None,
    ) -> FitRecord:
        record = FitRecord.from_fit(
            fit,
            window_size=window_size,
            end_date=end_date,
            data_revision=data_revision,
            record_id=self._next_id(),
            replaces_record_id=replaces_record_id,
        )
        self._records.append(record)
        return record

    def mark_rechecking(self, record_id: int, note: str) -> FitRecord | None:
        """把一条 VALID 记录标记为复核中（数值保留但不再允许预测消费）。"""
        record = self.get(record_id)
        if record is not None and record.status is FitStatus.VALID:
            record.status = FitStatus.RECHECKING
            record.note = note
        return record

    def add_incomplete(self, window_size: int, end_date: str) -> FitRecord:
        record = FitRecord.incomplete(window_size, end_date, self._next_id())
        self._records.append(record)
        return record

    # ------------------------------------------------------------------ 查询
    def all_records(self) -> list[FitRecord]:
        return list(self._records)

    def get(self, record_id: int) -> FitRecord | None:
        for record in self._records:
            if record.record_id == record_id:
                return record
        return None

    def records_for_window(self, window_size: int) -> list[FitRecord]:
        return [r for r in self._records if r.window_size == window_size]

    def latest_for_window(self, window_size: int) -> FitRecord | None:
        for record in reversed(self._records):
            if record.window_size == window_size:
                return record
        return None

    def latest_valid(self, window_size: int) -> FitRecord | None:
        """某档窗口当前是否有效：只有该窗口**最新一条**记录为 VALID 才返回。

        更正发生后最新记录是 RECHECKING，历史 VALID 记录仍保留在数据集里
        供报表追溯，但不能再被预测入口当作当前有效拟合消费。
        """
        latest = self.latest_for_window(window_size)
        if latest is not None and latest.status is FitStatus.VALID:
            return latest
        return None

    def valid_windows(self) -> list[FitRecord]:
        """所有档位当前有效的最新记录（预测候选池）。"""
        result: list[FitRecord] = []
        for size in sorted({r.window_size for r in self._records}):
            record = self.latest_valid(size)
            if record is not None:
                result.append(record)
        return result

    def export_jsonl(self, path: str | Path) -> Path:
        """把数据集落盘为 JSONL，供下游预测任务/报表按质量规则挑选。"""
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            for record in self._records:
                fh.write(json.dumps(record.as_dict(), ensure_ascii=False))
                fh.write("\n")
        return path

    def as_dicts(self) -> list[dict[str, Any]]:
        return [r.as_dict() for r in self._records]
