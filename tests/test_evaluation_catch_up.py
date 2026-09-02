"""Tests for the 2026-09-02 fix: _evaluate_from_stored_candles used to
evaluate only the single newest stored candle, silently skipping any other
candle that closed in a gap between two calls (Kite session expired,
process down). In production this meant no candle closed during an outage
was ever evaluated, before or after logging back in. See
application/pipeline/evaluation.py's own docstring and docs/decisions/
010-outage-catch-up-replay.md."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from trading_scanner.alpha_engine import AlphaEngine
from trading_scanner.application.fast_predict import ExitState, FastPredictResult, QueueState
from trading_scanner.application.pipeline.evaluation import (
    _evaluate_from_stored_candles,
    _find_new_bar_indices,
    _notify_stale_catch_up_signals,
)
from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import Candle
from trading_scanner.domain.ports import EngineState


class _FakeCandleRepository:
    def __init__(self, candles: list[Candle]) -> None:
        self._candles = list(candles)

    async def get_candles(self, symbol, interval, limit=None):
        return self._candles[-limit:] if limit is not None else self._candles

    def add(self, candle: Candle) -> None:
        self._candles.append(candle)


class _FakeEngineStateRepository:
    def __init__(self) -> None:
        self._state = EngineState()
        self.set_calls: list[EngineState] = []

    async def get_state(self, symbol, interval) -> EngineState:
        return self._state

    async def set_state(self, symbol, interval, state: EngineState) -> None:
        self._state = state
        self.set_calls.append(state)


class _FakeNotifier:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def send_text(self, message: str) -> None:
        self.texts.append(message)


def _flat_candles(count: int, start: datetime | None = None) -> list[Candle]:
    """Flat, unchanging OHLCV -- fine for testing catch-up *mechanics*
    (bar count, order, state persistence), not for provoking a specific
    BUY/SELL signal out of AlphaEngine."""
    start = start or datetime(2026, 1, 1, tzinfo=UTC)
    return [
        Candle(
            symbol="TEST.NS",
            timestamp=start + pd.Timedelta(hours=index),
            open=Decimal("100"), high=Decimal("101"), low=Decimal("99"), close=Decimal("100"),
            volume=1_000,
        )
        for index in range(count)
    ]


def _config() -> AppConfig:
    return AppConfig(
        scan_interval_hours=1, candle_interval="1h", candle_history=300,
        symbols_file=None, logging_level=20, turso_database_url=None, turso_auth_token=None,
        telegram_bot_token=None, telegram_chat_id=None, index_symbol=None,
        kite_api_key=None, kite_api_secret=None, live_trading_enabled=False,
        live_trading_symbols=frozenset(), live_trading_max_lots=1,
        futures_paper_symbols_file=None,
    )


@pytest.mark.asyncio
async def test_the_normal_no_gap_case_returns_exactly_one_result() -> None:
    candles = _flat_candles(200)
    candle_repository = _FakeCandleRepository(candles)
    engine_state_repository = _FakeEngineStateRepository()
    engine = AlphaEngine(include_full_history=True)

    results = await _evaluate_from_stored_candles(
        "TEST.NS", _config(), engine, candle_repository, engine_state_repository
    )

    assert len(results) == 1
    result, candle = results[0]
    assert candle.timestamp == candles[-1].timestamp
    assert isinstance(result, FastPredictResult)


@pytest.mark.asyncio
async def test_a_second_call_with_no_new_candle_returns_nothing() -> None:
    candles = _flat_candles(200)
    candle_repository = _FakeCandleRepository(candles)
    engine_state_repository = _FakeEngineStateRepository()
    engine = AlphaEngine(include_full_history=True)
    await _evaluate_from_stored_candles(
        "TEST.NS", _config(), engine, candle_repository, engine_state_repository
    )

    results = await _evaluate_from_stored_candles(
        "TEST.NS", _config(), engine, candle_repository, engine_state_repository
    )

    assert results == []


@pytest.mark.asyncio
async def test_a_gap_of_three_candles_replays_all_three_in_order() -> None:
    candles = _flat_candles(200)
    candle_repository = _FakeCandleRepository(candles)
    engine_state_repository = _FakeEngineStateRepository()
    engine = AlphaEngine(include_full_history=True)
    # Bootstrap on candle #200 (the normal first run).
    await _evaluate_from_stored_candles(
        "TEST.NS", _config(), engine, candle_repository, engine_state_repository
    )
    # Simulate an outage: 3 more candles closed with nobody evaluating them.
    for candle in _flat_candles(3, start=candles[-1].timestamp + timedelta(hours=1)):
        candle_repository.add(candle)
    engine_state_repository.set_calls.clear()  # drop the bootstrap call above

    results = await _evaluate_from_stored_candles(
        "TEST.NS", _config(), engine, candle_repository, engine_state_repository
    )

    assert len(results) == 3
    timestamps = [candle.timestamp for _, candle in results]
    assert timestamps == sorted(timestamps)  # chronological, oldest first
    assert timestamps[-1] == candle_repository._candles[-1].timestamp
    # engine_state persisted once per replayed bar, not just once at the end.
    assert len(engine_state_repository.set_calls) == 3
    assert engine_state_repository.set_calls[-1].last_bar_timestamp == timestamps[-1].isoformat()


@pytest.mark.asyncio
async def test_a_gap_wider_than_the_cap_falls_back_to_only_the_newest_bar() -> None:
    candles = _flat_candles(200)
    candle_repository = _FakeCandleRepository(candles)
    engine_state_repository = _FakeEngineStateRepository()
    engine = AlphaEngine(include_full_history=True)
    await _evaluate_from_stored_candles(
        "TEST.NS", _config(), engine, candle_repository, engine_state_repository
    )
    for candle in _flat_candles(60, start=candles[-1].timestamp + timedelta(hours=1)):
        candle_repository.add(candle)

    results = await _evaluate_from_stored_candles(
        "TEST.NS", _config(), engine, candle_repository, engine_state_repository
    )

    assert len(results) == 1
    assert results[0][1].timestamp == candle_repository._candles[-1].timestamp


def test_find_new_bar_indices_normal_single_bar_advance() -> None:
    candles = _flat_candles(5)
    indices = _find_new_bar_indices(candles, candles[3].timestamp.isoformat())
    assert indices == [4]


def test_find_new_bar_indices_multi_bar_gap() -> None:
    candles = _flat_candles(5)
    indices = _find_new_bar_indices(candles, candles[1].timestamp.isoformat())
    assert indices == [2, 3, 4]


def test_find_new_bar_indices_none_timestamp_falls_back_to_last() -> None:
    candles = _flat_candles(5)
    assert _find_new_bar_indices(candles, None) == [4]


def test_find_new_bar_indices_unknown_timestamp_falls_back_to_last() -> None:
    # Defensive case: history was rewritten/pruned and the old timestamp
    # simply isn't there anymore -- must never replay from an arbitrary
    # unbounded starting point.
    candles = _flat_candles(5)
    indices = _find_new_bar_indices(candles, "2020-01-01T00:00:00+00:00")
    assert indices == [4]


def _result(signal: str) -> FastPredictResult:
    return FastPredictResult(
        signal=signal, prediction=0, end_long=False, end_short=False,
        is_early_signal_flip=False, signal_previous=0, queue_state=QueueState(),
        exit_state=ExitState(),
    )


@pytest.mark.asyncio
async def test_stale_catch_up_notifies_only_for_buy_or_sell() -> None:
    candle = _flat_candles(1)[0]
    notifier = _FakeNotifier()
    stale = [
        (_result("NEUTRAL"), candle),
        (_result("BUY"), candle),
        (_result("SELL"), candle),
    ]

    await _notify_stale_catch_up_signals("TEST.NS", stale, notifier)

    assert len(notifier.texts) == 2
    assert "BUY" in notifier.texts[0]
    assert "SELL" in notifier.texts[1]


@pytest.mark.asyncio
async def test_stale_catch_up_is_silent_with_no_notifier() -> None:
    candle = _flat_candles(1)[0]
    # Must not raise.
    await _notify_stale_catch_up_signals("TEST.NS", [(_result("BUY"), candle)], None)


@pytest.mark.asyncio
async def test_stale_catch_up_is_silent_when_nothing_is_stale() -> None:
    notifier = _FakeNotifier()
    await _notify_stale_catch_up_signals("TEST.NS", [], notifier)
    assert notifier.texts == []
