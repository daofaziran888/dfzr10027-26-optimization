"""带截距一元线性回归的精确实现与增量累加器。

所有统计量都用 :class:`fractions.Fraction` 维护，因此：

1. 增量引擎（加入新样本、扣减滑出样本、修订样本）与每次全量重算的旧实现
   对同一窗口给出**完全相同**的系数、拟合优度与预测值，不存在浮点漂移；
2. 对外输出（JSON/CSV/接口）时再转成 float 或按精度取整。

模型为 ``y = slope * x + intercept``，x 为窗口内从 0 开始的整数序号。
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

Number = int | float | str | Fraction


def to_fraction(value: Number) -> Fraction:
    """把外部数值（int/float/字符串/Fraction）统一转成精确分数。"""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(value)
    return Fraction(str(value))


@dataclass(frozen=True)
class FitResult:
    """一次线性拟合的完整结果（精确分数 + 便捷浮点视图）。"""

    n: int
    intercept: Fraction
    slope: Fraction
    r2: Fraction
    sse: Fraction
    sst: Fraction

    @property
    def intercept_float(self) -> float:
        return float(self.intercept)

    @property
    def slope_float(self) -> float:
        return float(self.slope)

    @property
    def r2_float(self) -> float:
        return float(self.r2)

    def predict(self, x: Number) -> Fraction:
        """在给定 x 上的点预测（精确值）。"""
        return self.intercept + self.slope * to_fraction(x)

    def as_dict(self) -> dict[str, object]:
        """转成可序列化字典，分数转 float 并保留六位小数。"""
        return {
            "n": self.n,
            "intercept": round(self.intercept_float, 6),
            "slope": round(self.slope_float, 6),
            "r2": round(self.r2_float, 6),
            "sse": round(float(self.sse), 6),
            "sst": round(float(self.sst), 6),
        }


class RegressionAccumulator:
    """窗口级一阶/二阶矩累加器，支持加入、扣减与修订单个样本。

    维护 n、Σx、Σy、Σx²、Σy²、Σxy 六个充分统计量，
    任意子集窗口的线性回归都能由它们还原，因此滑动时无需重扫历史样本。
    """

    __slots__ = ("n", "sx", "sy", "sxx", "syy", "sxy")

    def __init__(self) -> None:
        self.n: int = 0
        self.sx: Fraction = Fraction(0)
        self.sy: Fraction = Fraction(0)
        self.sxx: Fraction = Fraction(0)
        self.syy: Fraction = Fraction(0)
        self.sxy: Fraction = Fraction(0)

    def add(self, x: Number, y: Number) -> None:
        xf, yf = to_fraction(x), to_fraction(y)
        self.n += 1
        self.sx += xf
        self.sy += yf
        self.sxx += xf * xf
        self.syy += yf * yf
        self.sxy += xf * yf

    def remove(self, x: Number, y: Number) -> None:
        """扣减一个样本的贡献。调用方必须保证 (x, y) 确实在窗口内。"""
        xf, yf = to_fraction(x), to_fraction(y)
        if self.n <= 0:
            raise ValueError("累加器已为空，不能继续扣减样本")
        self.n -= 1
        self.sx -= xf
        self.sy -= yf
        self.sxx -= xf * xf
        self.syy -= yf * yf
        self.sxy -= xf * yf

    def revise(self, x: Number, old_y: Number, new_y: Number) -> None:
        """修订样本：先扣除旧值贡献，再加入新值贡献，杜绝新旧分子混用。"""
        self.remove(x, old_y)
        self.add(x, new_y)

    def fit(self) -> FitResult:
        """由当前充分统计量计算系数与拟合优度。"""
        return _fit_from_moments(self.n, self.sx, self.sy, self.sxx, self.syy, self.sxy)

    def copy(self) -> "RegressionAccumulator":
        clone = RegressionAccumulator()
        clone.n = self.n
        clone.sx, clone.sy = self.sx, self.sy
        clone.sxx, clone.syy = self.sxx, self.syy
        clone.sxy = self.sxy
        return clone


def _fit_from_moments(
    n: int,
    sx: Fraction,
    sy: Fraction,
    sxx: Fraction,
    syy: Fraction,
    sxy: Fraction,
) -> FitResult:
    if n < 2:
        raise ValueError("至少需要 2 个样本才能拟合线性模型")
    # Sxx = Σ(x-x̄)²，Sxy = Σ(x-x̄)(y-ȳ)，Syy = Σ(y-ȳ)²
    sxx_centered = sxx - sx * sx / n
    sxy_centered = sxy - sx * sy / n
    syy_centered = syy - sy * sy / n

    if sxx_centered == 0:
        # x 无变异（理论上整数序号窗口不会出现），退化为均值模型
        slope = Fraction(0)
        intercept = sy / n
    else:
        slope = sxy_centered / sxx_centered
        intercept = sy / n - slope * (sx / n)

    # SSE = Syy - Sxy² / Sxx（最小二乘恒等式，保持精确分数）
    if sxx_centered == 0:
        sse = syy_centered
    else:
        sse = syy_centered - sxy_centered * sxy_centered / sxx_centered
    if sse < 0:  # 精确运算下只可能为 0，防御性兜底
        sse = Fraction(0)

    if syy_centered == 0:
        # y 为常数：模型完全解释，约定 R² = 1
        r2 = Fraction(1)
    else:
        r2 = Fraction(1) - sse / syy_centered
    return FitResult(n=n, intercept=intercept, slope=slope, r2=r2, sse=sse, sst=syy_centered)


def fit_linear(points: Sequence[tuple[Number, Number]]) -> FitResult:
    """旧实现口径：对传入的完整 (x, y) 样本集合一次性全量拟合。"""
    acc = RegressionAccumulator()
    for x, y in points:
        acc.add(x, y)
    return acc.fit()


def fit_linear_values(values: Iterable[Number]) -> FitResult:
    """对纯 y 序列拟合，x 自动取 0..n-1（与 forecast_engine 的口径一致）。"""
    return fit_linear(list(enumerate(values)))
