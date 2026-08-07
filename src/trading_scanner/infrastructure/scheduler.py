from collections.abc import Awaitable, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler


def create_hourly_scheduler(job: Callable[[], Awaitable[None]], timezone: str) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=timezone)
    scheduler.add_job(job, trigger="interval", hours=1, id="hourly-scan", replace_existing=True)
    return scheduler
