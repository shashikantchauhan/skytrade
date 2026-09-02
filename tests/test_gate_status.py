"""Tests for the 2026-09-02 Gates dashboard tab: GateStatusSnapshot's
persistence (TursoGateStatusRepository) and its computation
(_record_gate_status, application/pipeline/orchestrator.py) -- see
docs/decisions/008-gate-status-snapshot.md."""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from trading_scanner.application.fast_predict import ExitState, FastPredictResult, QueueState
from trading_scanner.application.pipeline.orchestrator import _record_gate_status
from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import Candle, GateStatusSnapshot, SignalSide, Trade
from trading_scanner.infrastructure.db import TursoGateStatusRepository, create_turso_client


def _local_url(tmp_path: Path) -> str:
    return f"file:{tmp_path / 'test.db'}"


def _snapshot(**overrides) -> GateStatusSnapshot:
    defaults = dict(
        symbol="RELIANCE.NS", interval="1h", signal="BUY", adx=25.0,
        regime_normalized=2.0, volatility_margin=10.0,
        track_record_passed=True, quality_passed=True, conviction_passed=True,
        evaluated_at=datetime(2026, 9, 2, 10, 0, tzinfo=UTC),
        updated_at=datetime(2026, 9, 2, 10, 0, 5, tzinfo=UTC),
    )
    defaults.update(overrides)
    return GateStatusSnapshot(**defaults)


@pytest.mark.asyncio
async def test_set_and_get_round_trips(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoGateStatusRepository(client)
        await repository.ensure_schema()

        await repository.set_snapshot(_snapshot())
        snapshots = await repository.get_all_snapshots("1h")

        assert len(snapshots) == 1
        assert snapshots[0] == _snapshot()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_second_write_for_the_same_symbol_overwrites_not_appends(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoGateStatusRepository(client)
        await repository.ensure_schema()

        await repository.set_snapshot(_snapshot(signal="NEUTRAL"))
        await repository.set_snapshot(_snapshot(signal="BUY"))
        snapshots = await repository.get_all_snapshots("1h")

        assert len(snapshots) == 1
        assert snapshots[0].signal == "BUY"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_all_snapshots_only_returns_the_requested_interval(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoGateStatusRepository(client)
        await repository.ensure_schema()
        await repository.set_snapshot(_snapshot(interval="1h"))
        await repository.set_snapshot(_snapshot(symbol="TCS.NS", interval="15m"))

        snapshots = await repository.get_all_snapshots("1h")

        assert [s.symbol for s in snapshots] == ["RELIANCE.NS"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_record_event_and_get_events_since(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoGateStatusRepository(client)
        await repository.ensure_schema()
        await repository.record_event(_snapshot(
            evaluated_at=datetime(2026, 9, 2, 4, 45, tzinfo=UTC),
        ))
        await repository.record_event(_snapshot(
            symbol="AXISBANK.NS", signal="SELL",
            evaluated_at=datetime(2026, 9, 2, 8, 45, tzinfo=UTC),
        ))

        events = await repository.get_events_since("1h", datetime(2026, 9, 2, tzinfo=UTC))

        assert len(events) == 2
        assert events[0].symbol == "AXISBANK.NS"  # most recent first
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_events_since_excludes_events_before_the_cutoff(tmp_path: Path) -> None:
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoGateStatusRepository(client)
        await repository.ensure_schema()
        await repository.record_event(_snapshot(
            evaluated_at=datetime(2026, 9, 1, 10, 0, tzinfo=UTC),  # yesterday
        ))

        events = await repository.get_events_since("1h", datetime(2026, 9, 2, tzinfo=UTC))

        assert events == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_replaying_the_same_bar_does_not_duplicate_the_event(tmp_path: Path) -> None:
    # Catch-up replay (or any re-processing) hitting an already-recorded
    # bar must not duplicate or clobber the original event.
    client = create_turso_client(_local_url(tmp_path), None)
    try:
        repository = TursoGateStatusRepository(client)
        await repository.ensure_schema()
        snapshot = _snapshot(evaluated_at=datetime(2026, 9, 2, 4, 45, tzinfo=UTC))
        await repository.record_event(snapshot)
        await repository.record_event(snapshot)

        events = await repository.get_events_since("1h", datetime(2026, 9, 2, tzinfo=UTC))

        assert len(events) == 1
    finally:
        await client.close()


class _FakeTradeRepository:
    """Scripts is_eligible's outcome directly via enough closed trades."""

    def __init__(self, eligible: bool) -> None:
        self._eligible = eligible

    async def get_trades(self, symbol, interval):
        if not self._eligible:
            return []
        return [
            Trade(
                symbol=symbol, side=SignalSide.BUY,
                entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC), entry_price=Decimal("100"),
                prediction_at_entry=1, is_early_signal_flip=False,
                exit_timestamp=datetime(2026, 1, 2, tzinfo=UTC), exit_price=Decimal("110"),
                pnl_percent=Decimal("10"), status="closed",
            )
            for _ in range(5)
        ]


class _FakeGateStatusRepository:
    def __init__(self) -> None:
        self.recorded: list[GateStatusSnapshot] = []
        self.events: list[GateStatusSnapshot] = []

    async def set_snapshot(self, snapshot: GateStatusSnapshot) -> None:
        self.recorded.append(snapshot)

    async def record_event(self, snapshot: GateStatusSnapshot) -> None:
        self.events.append(snapshot)


def _config() -> AppConfig:
    return AppConfig(
        scan_interval_hours=1, candle_interval="1h", candle_history=300,
        symbols_file=None, logging_level=20, turso_database_url=None, turso_auth_token=None,
        telegram_bot_token=None, telegram_chat_id=None, index_symbol=None,
        kite_api_key=None, kite_api_secret=None, live_trading_enabled=False,
        live_trading_symbols=frozenset(), live_trading_max_lots=1,
        futures_paper_symbols_file=None,
    )


def _result(**overrides) -> FastPredictResult:
    defaults = dict(
        signal="BUY", prediction=5, end_long=False, end_short=False,
        is_early_signal_flip=False, signal_previous=1, queue_state=QueueState(),
        exit_state=ExitState(), adx=25.0, regime_normalized=2.0, volatility_margin=10.0,
    )
    defaults.update(overrides)
    return FastPredictResult(**defaults)


def _candle() -> Candle:
    return Candle(
        symbol="RELIANCE.NS", timestamp=datetime(2026, 9, 2, 10, 0, tzinfo=UTC),
        open=Decimal("2900"), high=Decimal("2950"), low=Decimal("2895"), close=Decimal("2940"),
        volume=1000,
    )


@pytest.mark.asyncio
async def test_record_gate_status_writes_a_snapshot_for_a_neutral_symbol() -> None:
    # The whole point: computed regardless of whether the symbol has an
    # active BUY/SELL signal right now.
    gate_status_repository = _FakeGateStatusRepository()

    await _record_gate_status(
        "RELIANCE.NS", _config(), _result(signal="NEUTRAL"), _candle(),
        _FakeTradeRepository(eligible=True), gate_status_repository,
    )

    assert len(gate_status_repository.recorded) == 1
    snapshot = gate_status_repository.recorded[0]
    assert snapshot.signal == "NEUTRAL"
    assert gate_status_repository.events == []  # NEUTRAL never becomes an event


@pytest.mark.asyncio
async def test_record_gate_status_logs_a_permanent_event_for_a_buy_signal() -> None:
    gate_status_repository = _FakeGateStatusRepository()

    await _record_gate_status(
        "RELIANCE.NS", _config(), _result(signal="BUY"), _candle(),
        _FakeTradeRepository(eligible=True), gate_status_repository,
    )

    assert len(gate_status_repository.events) == 1
    assert gate_status_repository.events[0].signal == "BUY"


@pytest.mark.asyncio
async def test_record_gate_status_logs_a_permanent_event_for_a_sell_signal() -> None:
    # SELL never reaches entry_decisions (BUY-only, see capital_allocation.
    # py) -- this is the only permanent record a SELL signal gets at all.
    gate_status_repository = _FakeGateStatusRepository()

    await _record_gate_status(
        "AXISBANK.NS", _config(), _result(signal="SELL"), _candle(),
        _FakeTradeRepository(eligible=True), gate_status_repository,
    )

    assert len(gate_status_repository.events) == 1
    assert gate_status_repository.events[0].signal == "SELL"


@pytest.mark.asyncio
async def test_record_gate_status_reflects_a_failed_track_record() -> None:
    gate_status_repository = _FakeGateStatusRepository()

    await _record_gate_status(
        "RELIANCE.NS", _config(), _result(), _candle(),
        _FakeTradeRepository(eligible=False), gate_status_repository,
    )

    assert gate_status_repository.recorded[0].track_record_passed is False


@pytest.mark.asyncio
async def test_record_gate_status_reflects_a_failed_quality_filter() -> None:
    gate_status_repository = _FakeGateStatusRepository()

    await _record_gate_status(
        # Deliberately below entry_quality_filter's floors.
        "RELIANCE.NS", _config(), _result(volatility_margin=0.0, regime_normalized=0.0), _candle(),
        _FakeTradeRepository(eligible=True), gate_status_repository,
    )

    assert gate_status_repository.recorded[0].quality_passed is False


@pytest.mark.asyncio
async def test_record_gate_status_is_best_effort_and_never_raises() -> None:
    class _RaisingTradeRepository:
        async def get_trades(self, symbol, interval):
            raise RuntimeError("DB unavailable")

    gate_status_repository = _FakeGateStatusRepository()

    # Must not raise.
    await _record_gate_status(
        "RELIANCE.NS", _config(), _result(), _candle(),
        _RaisingTradeRepository(), gate_status_repository,
    )

    assert gate_status_repository.recorded == []
