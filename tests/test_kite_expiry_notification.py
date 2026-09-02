"""Tests for the 2026-09-02 fix: the "Kite session expired" Telegram alert
re-sends every 15 minutes instead of once per calendar day -- see
application/pipeline/market_data.py's _notify_kite_expired_periodically and
docs/decisions/009-kite-expiry-renotify.md. The old once-per-day version let
a single missed alert sit unattended for hours; in production this blocked
real trading for 30-104 minutes at market open on three consecutive days."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_scanner.application.pipeline.market_data import (
    _EXPIRY_RENOTIFY_INTERVAL,
    _notify_kite_expired_periodically,
)
from trading_scanner.infrastructure.db import TursoKiteSessionRepository, create_turso_client


def _local_url(tmp_path: Path) -> str:
    return f"file:{tmp_path / 'test.db'}"


class _FakeNotifier:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def send_text(self, message: str) -> None:
        self.texts.append(message)


@pytest.mark.asyncio
async def test_first_call_sends_and_records_the_timestamp(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoKiteSessionRepository(client)
        await repository.ensure_schema()
        notifier = _FakeNotifier()

        await _notify_kite_expired_periodically(repository, notifier)

        assert len(notifier.texts) == 1
        assert await repository.get_expiry_notified_at() is not None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_second_call_soon_after_does_not_resend(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoKiteSessionRepository(client)
        await repository.ensure_schema()
        notifier = _FakeNotifier()

        await _notify_kite_expired_periodically(repository, notifier)
        await _notify_kite_expired_periodically(repository, notifier)

        # This is exactly the old once-per-day bug's opposite failure mode
        # to guard against: it must not spam every single poll (~60s).
        assert len(notifier.texts) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_call_after_the_renotify_interval_sends_again(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoKiteSessionRepository(client)
        await repository.ensure_schema()
        notifier = _FakeNotifier()
        stale = datetime.now(UTC) - _EXPIRY_RENOTIFY_INTERVAL - timedelta(seconds=1)
        await repository.set_expiry_notified_at(stale.isoformat())

        await _notify_kite_expired_periodically(repository, notifier)

        assert len(notifier.texts) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_call_within_the_renotify_interval_stays_silent(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoKiteSessionRepository(client)
        await repository.ensure_schema()
        notifier = _FakeNotifier()
        recent = datetime.now(UTC) - timedelta(minutes=5)
        await repository.set_expiry_notified_at(recent.isoformat())

        await _notify_kite_expired_periodically(repository, notifier)

        assert notifier.texts == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_no_notifier_is_a_silent_no_op(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoKiteSessionRepository(client)
        await repository.ensure_schema()

        await _notify_kite_expired_periodically(repository, None)  # must not raise

        assert await repository.get_expiry_notified_at() is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_expiry_notified_at_round_trips(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoKiteSessionRepository(client)
        await repository.ensure_schema()
        assert await repository.get_expiry_notified_at() is None

        now = datetime.now(UTC).isoformat()
        await repository.set_expiry_notified_at(now)

        assert await repository.get_expiry_notified_at() == now
    finally:
        await client.close()
