"""滚动窗口拟合接口层测试：直接调用路由函数，验证编排与状态流转。"""
from __future__ import annotations

import asyncio
from datetime import date, timedelta

import pytest

from app.routers import rolling_fit as router
from app.services.rolling_fit_service import reset_service


@pytest.fixture(autouse=True)
def clean_service():
    reset_service()
    yield
    reset_service()


def business_days(start: date, count: int) -> list[str]:
    days: list[str] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return days


async def _collect(iterator) -> list[str]:
    return [chunk async for chunk in iterator]


class TestRollingFitAPI:
    def test_feed_correct_recheck_predict_flow(self):
        days = business_days(date(2026, 11, 9), 9)
        values = [100, 110, 120, 130, 140, 150, 160, 170, 180]
        resp = router.feed_observations(router.FeedPayload(points=[
            router.FeedPoint(date=d, value=v) for d, v in zip(days, values)
        ]))
        assert resp["ok"] is True and resp["total"] == 9

        # 初始状态：两档窗口均达标，预测采用短窗
        status = router.get_status()
        assert {row["window_size"] for row in status["evaluation"]} == {4, 8}
        assert all(row["qualified"] for row in status["evaluation"])
        plan = router.predict(horizon=2)
        assert plan["ok"] is True and plan["window_size"] == 4
        assert len(plan["targets"]) == 2

        # 更正昨日金额：短窗复核后失效、长窗复核中 → 预测被拒绝
        router.correct_observation(router.CorrectPayload(date=days[-1], value=150))
        blocked = router.predict(horizon=1)
        assert blocked["ok"] is False
        statuses = {row["window_size"]: row for row in blocked["evaluation"]}
        assert statuses[8]["status"] == "RECHECKING"
        assert statuses[4]["qualified"] is False

        # 复核队列处理完成：长窗达标，预测入口自动切换到 8 日窗
        done = router.process_recheck(router.RecheckPayload())
        assert done["ok"] is True and done["pending_windows"] == []
        plan = router.predict(horizon=1)
        assert plan["ok"] is True and plan["window_size"] == 8

        # 新旧实现一致性核对全部通过
        parity = router.parity_check()
        assert all(check.get("comparable") and check["r2_equal"]
                   and check["next_prediction_equal"] for check in parity["checks"])

    def test_dataset_export_jsonl(self):
        days = business_days(date(2026, 11, 9), 9)
        router.feed_observations(router.FeedPayload(points=[
            router.FeedPoint(date=d, value=v) for d, v in zip(days, range(100, 109))
        ]))
        router.correct_observation(router.CorrectPayload(date=days[-1], value=200))
        response = router.export_dataset()
        chunks = asyncio.run(_collect(response.body_iterator))
        body = "".join(chunks)
        lines = [line for line in body.strip().split("\n") if line]
        assert len(lines) == router.get_dataset()["total"]
        assert all('"window_size"' in line for line in lines)
        assert "RECHECKING" in body

    def test_unknown_date_correction_rejected(self):
        days = business_days(date(2026, 11, 9), 2)
        router.feed_observations(router.FeedPayload(points=[
            router.FeedPoint(date=d, value=v) for d, v in zip(days, [1, 2])
        ]))
        with pytest.raises(KeyError):
            router.correct_observation(router.CorrectPayload(date="2099-01-01", value=1))
