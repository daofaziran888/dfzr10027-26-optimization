"""滚动窗口拟合质量接口。

把「观测灌入 / 金额更正 / 复核处理 / 拟合数据集 / 质量状态 / 预测入口 /
新旧实现核对」暴露为 HTTP 接口；路由层不做业务判断，全部委托给
:class:`RollingFitService`。
"""
from __future__ import annotations

import io
import json

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.rolling_fit_service import service

router = APIRouter(prefix="/api/rolling-fit", tags=["滚动窗口拟合"])


class FeedPoint(BaseModel):
    date: str
    value: float | int | str


class FeedPayload(BaseModel):
    points: list[FeedPoint] = Field(default_factory=list)


class CorrectPayload(BaseModel):
    date: str
    value: float | int | str


class RecheckPayload(BaseModel):
    window_size: int | None = None


@router.post("/observations/feed")
def feed_observations(payload: FeedPayload) -> dict[str, object]:
    """批量灌入按营业日递增的连续观测，增量推进所有窗口。"""
    result = service.feed([p.model_dump() for p in payload.points])
    return {"ok": True, **result}


@router.post("/observations/correct")
def correct_observation(payload: CorrectPayload) -> dict[str, object]:
    """更正某营业日金额：短窗立即复核，长窗转复核中并排队。"""
    return {"ok": True, **service.correct(payload.date, payload.value)}


@router.post("/recheck")
def process_recheck(payload: RecheckPayload) -> dict[str, object]:
    """执行复核队列里的长窗重新计算。"""
    return {"ok": True, **service.process_recheck(payload.window_size)}


@router.get("/dataset")
def get_dataset() -> dict[str, object]:
    """窗口拟合数据集：实际拟合优度、数据版本与状态（只追加留痕）。"""
    rows = service.dataset_rows()
    return {"items": rows, "total": len(rows)}


@router.get("/dataset/export")
def export_dataset() -> StreamingResponse:
    """下载 JSONL 格式的窗口拟合数据集，供下游预测任务按质量规则挑选。"""
    buffer = io.StringIO()
    for row in service.dataset_rows():
        buffer.write(json.dumps(row, ensure_ascii=False))
        buffer.write("\n")
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="rolling_fit_dataset.jsonl"'},
    )


@router.get("/status")
def get_status() -> dict[str, object]:
    """两档窗口当前质量状态：哪档达标可预测、哪档仍在等待复核。"""
    return service.status()


@router.get("/predict")
def predict(horizon: int = Query(default=1, ge=1, le=30)) -> dict[str, object]:
    """预测入口：无合格窗口时 ok=false 并返回逐档判定，而不是硬出值。"""
    return service.predict(horizon)


@router.get("/parity")
def parity_check() -> dict[str, object]:
    """新旧实现对同一完整窗口的系数/拟合优度/预测值一致性核对。"""
    return service.parity_check()
