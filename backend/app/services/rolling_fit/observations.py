"""连续观测数据仓库：按业务日期保存观测值与修订版本。

设计要点：

- 观测按业务日期有序保存，日期在窗口内的位置（x 序号）由它在窗口中的
  相对顺序决定，而不是写死的全局下标，窗口滑动后序号自动平移；
- 每次更正都产生新的数据版本（revision），并完整保留修订历史，
  引擎据此知道"旧贡献"具体是哪个值，避免把修订前的分子和修订后的样本混用；
- 仓库只负责数据与版本，不做拟合计算。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterator

from app.services.rolling_fit.regression import Number, to_fraction


@dataclass(frozen=True)
class Observation:
    """单个业务日期上的一条观测（含当前版本信息）。"""

    date: str
    value: Fraction
    revision: int  # 当前版本号，首版为 1，每更正一次 +1

    @property
    def value_float(self) -> float:
        return float(self.value)

    def as_dict(self) -> dict[str, object]:
        return {"date": self.date, "value": float(self.value), "revision": self.revision}


@dataclass
class ObservationRevision:
    """修订历史中的一个版本。"""

    date: str
    value: Fraction
    revision: int


@dataclass
class ObservationStore:
    """按日期有序的观测仓库，支持追加新日期与更正历史日期。"""

    _points: list[Observation] = field(default_factory=list)
    _history: dict[str, list[ObservationRevision]] = field(default_factory=dict)
    _index: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ 写入
    def append(self, date: str, value: Number) -> Observation:
        """追加一个新业务日期的观测；日期必须晚于已有日期且不重复。"""
        if self._points and date <= self._points[-1].date:
            raise ValueError(f"观测必须按业务日期递增追加：{date} 晚于 {self._points[-1].date} 才能加入")
        if date in self._index:
            raise ValueError(f"业务日期 {date} 已存在，更正请使用 correct()")
        obs = Observation(date=date, value=to_fraction(value), revision=1)
        self._index[date] = len(self._points)
        self._points.append(obs)
        self._history[date] = [ObservationRevision(date, obs.value, 1)]
        return obs

    def correct(self, date: str, new_value: Number) -> Observation:
        """更正指定日期的金额，版本号 +1，旧版本进入历史。"""
        pos = self._index.get(date)
        if pos is None:
            raise KeyError(f"业务日期 {date} 不存在，无法更正")
        old = self._points[pos]
        new_fraction = to_fraction(new_value)
        if new_fraction == old.value:
            return old  # 金额未变化，不制造新版本
        revised = Observation(date=date, value=new_fraction, revision=old.revision + 1)
        self._points[pos] = revised
        self._history[date].append(ObservationRevision(date, new_fraction, revised.revision))
        return revised

    # ------------------------------------------------------------------ 读取
    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self) -> Iterator[Observation]:
        return iter(self._points)

    @property
    def dates(self) -> list[str]:
        return [p.date for p in self._points]

    @property
    def values(self) -> list[Fraction]:
        return [p.value for p in self._points]

    def get(self, date: str) -> Observation:
        return self._points[self._index[date]]

    def has(self, date: str) -> bool:
        return date in self._index

    def value_at(self, date: str, revision: int) -> Fraction:
        """取某日期在指定版本上的金额（用于扣除旧贡献）。"""
        for item in self._history[date]:
            if item.revision == revision:
                return item.value
        raise KeyError(f"日期 {date} 不存在版本 {revision}")

    def revision_of(self, date: str) -> int:
        return self._points[self._index[date]].revision

    def window_values(self, size: int) -> list[Fraction]:
        """最近 size 个观测的金额（不足 size 时返回全部已有观测）。"""
        return [p.value for p in self._points[-size:]]

    def window_dates(self, size: int) -> list[str]:
        """最近 size 个观测的业务日期。"""
        return [p.date for p in self._points[-size:]]

    def as_dicts(self) -> list[dict[str, object]]:
        return [p.as_dict() for p in self._points]
