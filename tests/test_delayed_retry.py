"""Tests for the 2026-09-02 delayed re-entry window: a real BUY candidate
that cleared every gate but didn't get a real order (capacity/cutoff/
execution) gets retried for up to 2 trading days if price comes back near
the original signal price and the strategy's own exit hasn't fired since.
See docs/decisions/011-delayed-reentry-window.md.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_scanner.application.fast_predict import ExitState, FastPredictResult, QueueState
from trading_scanner.application.pipeline.capital_allocation import (
    _RETRY_PRICE_TOLERANCE,
    _collect_delayed_retry_candidates,
    _notify_filled_delayed_retries,
    _trading_days_ago,
)
from trading_scanner.application.ranking import RankedCandidate
from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import Candle, EntryDecisionRecord, SignalSide, Trade
from trading_scanner.infrastructure.db import LiveCashToggleState


def _config() -> AppConfig:
    return AppConfig(
        scan_interval_hours=1, candle_interval="1h", candle_history=300,
        symbols_file=None, logging_level=20, turso_database_url=None, turso_auth_token=None,
        telegram_bot_token=None, telegram_chat_id=None, index_symbol=None,
        kite_api_key=None, kite_api_secret=None, live_trading_enabled=False,
        live_trading_symbols=frozenset(), live_trading_max_lots=1,
        futures_paper_symbols_file=None,
    )


def _cash_state(**overrides) -> LiveCashToggleState:
    defaults = dict(
        enabled=True, symbols=frozenset({"UPL.NS"}), notional=Decimal("55000"),
        max_positions=8, delayed_retry_enabled=True,
    )
    defaults.update(overrides)
    return LiveCashToggleState(**defaults)


def _decision(**overrides) -> EntryDecisionRecord:
    defaults = dict(
        symbol="UPL.NS", strategy="alpha_engine",
        signal_timestamp=datetime(2026, 9, 1, 8, 45, tzinfo=UTC),
        signal_side=SignalSide.BUY, signal_price=Decimal("100"),
        track_record_passed=True, quality_passed=True, conviction_passed=True,
        ranking_score=Decimal("72.5"), ranking_passed=True,
        capital_passed=None, position_limit_passed=None, cutoff_passed=None,
        final_decision="skipped",
        blocked_reason="cash: SKIPPED (no free slot, past cutoff, or execution failed -- "
        "see logs)",
        created_at=datetime(2026, 9, 1, 8, 46, tzinfo=UTC),
    )
    defaults.update(overrides)
    return EntryDecisionRecord(**defaults)


def _result(**overrides) -> FastPredictResult:
    # NEUTRAL by default -- the "thesis still alive, just not re-firing"
    # baseline for a retry candidate (transition-only signal, see
    # docs/decisions/008-gate-status-snapshot.md's addendum).
    defaults = dict(
        signal="NEUTRAL", prediction=5, end_long=False, end_short=False,
        is_early_signal_flip=False, signal_previous=1, queue_state=QueueState(),
        exit_state=ExitState(), adx=25.0, regime_normalized=2.0, volatility_margin=10.0,
    )
    defaults.update(overrides)
    return FastPredictResult(**defaults)


def _candle(**overrides) -> Candle:
    # market_price = (H+L+O+O)/4 = (101+99+100+100)/4 = 100 -- exactly the
    # default _decision()'s signal_price, 0% drift. close=100.8 clears
    # conviction's CLV>=0.7 floor: (100.8-99)/(101-99) = 0.9.
    defaults = dict(
        symbol="UPL.NS", timestamp=datetime(2026, 9, 2, 10, 0, tzinfo=UTC),
        open=Decimal("100"), high=Decimal("101"), low=Decimal("99"), close=Decimal("100.8"),
        volume=1000,
    )
    defaults.update(overrides)
    return Candle(**defaults)


class _FakeTradeRepository:
    """Scripts is_eligible's outcome directly via enough closed trades --
    same shape as tests/test_gate_status.py's fake."""

    def __init__(self, eligible: bool = True) -> None:
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


class _FakeLiveOrderRepository:
    def __init__(self, open_symbols: set[str] | None = None) -> None:
        self._open_symbols = open_symbols or set()

    async def get_unclosed_cash_legs(self, symbol: str) -> list:
        return ["leg"] if symbol in self._open_symbols else []


class _FakeEntryDecisionRepository:
    def __init__(self, pending: list[EntryDecisionRecord]) -> None:
        self._pending = pending
        self.since_calls: list[datetime] = []

    async def get_pending_cash_retries(self, since: datetime) -> list[EntryDecisionRecord]:
        self.since_calls.append(since)
        return self._pending


class _FakeNotifier:
    def __init__(self, raise_error: bool = False) -> None:
        self.sent: list[str] = []
        self._raise_error = raise_error

    async def send_text(self, message: str) -> None:
        if self._raise_error:
            raise RuntimeError("Telegram unavailable")
        self.sent.append(message)


# --- _trading_days_ago ------------------------------------------------------


def test_trading_days_ago_skips_the_weekend():
    now = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    monday = now - timedelta(days=now.weekday())  # this week's Monday, whatever today is
    assert monday.weekday() == 0

    result = _trading_days_ago(monday, 2)

    assert result.weekday() == 3  # Thursday -- Sun/Sat don't count against the 2-day budget
    assert result == monday - timedelta(days=4)


# --- _collect_delayed_retry_candidates: on/off + missing repos -------------


@pytest.mark.asyncio
async def test_disabled_toggle_returns_nothing():
    entry_decision_repository = _FakeEntryDecisionRepository([_decision()])

    candidates = await _collect_delayed_retry_candidates(
        {"UPL.NS": [(_result(), _candle())]}, _config(),
        _cash_state(delayed_retry_enabled=False), _FakeTradeRepository(),
        _FakeLiveOrderRepository(), entry_decision_repository,
    )

    assert candidates == []
    assert entry_decision_repository.since_calls == []  # short-circuits before any DB read


@pytest.mark.asyncio
async def test_missing_cash_state_returns_nothing():
    candidates = await _collect_delayed_retry_candidates(
        {"UPL.NS": [(_result(), _candle())]}, _config(), None, _FakeTradeRepository(),
        _FakeLiveOrderRepository(), _FakeEntryDecisionRepository([_decision()]),
    )
    assert candidates == []


@pytest.mark.asyncio
async def test_missing_live_order_repository_returns_nothing():
    candidates = await _collect_delayed_retry_candidates(
        {"UPL.NS": [(_result(), _candle())]}, _config(), _cash_state(), _FakeTradeRepository(),
        None, _FakeEntryDecisionRepository([_decision()]),
    )
    assert candidates == []


@pytest.mark.asyncio
async def test_missing_entry_decision_repository_returns_nothing():
    candidates = await _collect_delayed_retry_candidates(
        {"UPL.NS": [(_result(), _candle())]}, _config(), _cash_state(), _FakeTradeRepository(),
        _FakeLiveOrderRepository(), None,
    )
    assert candidates == []


# --- the happy path and its traceability field ------------------------------


@pytest.mark.asyncio
async def test_a_qualifying_pending_skip_becomes_a_candidate():
    candidates = await _collect_delayed_retry_candidates(
        {"UPL.NS": [(_result(), _candle())]}, _config(), _cash_state(), _FakeTradeRepository(),
        _FakeLiveOrderRepository(), _FakeEntryDecisionRepository([_decision()]),
    )

    assert len(candidates) == 1
    symbol, candidate = candidates[0]
    assert symbol == "UPL.NS"
    assert isinstance(candidate, RankedCandidate)
    assert candidate.retry_of_signal_timestamp == datetime(2026, 9, 1, 8, 45, tzinfo=UTC)
    # Priced off TODAY's candle, not the stale original signal_price.
    assert candidate.entry_price == Decimal("100")
    assert candidate.entry_timestamp == datetime(2026, 9, 2, 10, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_the_cutoff_passed_to_the_repository_is_two_trading_days_back():
    entry_decision_repository = _FakeEntryDecisionRepository([])

    await _collect_delayed_retry_candidates(
        {}, _config(), _cash_state(), _FakeTradeRepository(),
        _FakeLiveOrderRepository(), entry_decision_repository,
    )

    assert len(entry_decision_repository.since_calls) == 1
    since = entry_decision_repository.since_calls[0]
    assert since <= datetime.now(UTC) - timedelta(days=2)


# --- exclusions --------------------------------------------------------------


@pytest.mark.asyncio
async def test_excludes_a_symbol_with_no_fresh_evaluation_this_cycle():
    candidates = await _collect_delayed_retry_candidates(
        {}, _config(), _cash_state(), _FakeTradeRepository(),
        _FakeLiveOrderRepository(), _FakeEntryDecisionRepository([_decision()]),
    )
    assert candidates == []


@pytest.mark.asyncio
async def test_excludes_a_symbol_with_an_empty_evaluation_list():
    candidates = await _collect_delayed_retry_candidates(
        {"UPL.NS": []}, _config(), _cash_state(), _FakeTradeRepository(),
        _FakeLiveOrderRepository(), _FakeEntryDecisionRepository([_decision()]),
    )
    assert candidates == []


@pytest.mark.asyncio
async def test_excludes_a_symbol_that_already_has_a_fresh_buy_signal_this_cycle():
    # The main candidate loop already handles this one -- don't double-enter it.
    candidates = await _collect_delayed_retry_candidates(
        {"UPL.NS": [(_result(signal="BUY"), _candle())]}, _config(), _cash_state(),
        _FakeTradeRepository(), _FakeLiveOrderRepository(),
        _FakeEntryDecisionRepository([_decision()]),
    )
    assert candidates == []


@pytest.mark.asyncio
async def test_excludes_a_symbol_whose_dynamic_exit_fired_since():
    candidates = await _collect_delayed_retry_candidates(
        {"UPL.NS": [(_result(end_long=True), _candle())]}, _config(), _cash_state(),
        _FakeTradeRepository(), _FakeLiveOrderRepository(),
        _FakeEntryDecisionRepository([_decision()]),
    )
    assert candidates == []


@pytest.mark.asyncio
async def test_excludes_a_symbol_whose_signal_flipped_to_sell_since():
    candidates = await _collect_delayed_retry_candidates(
        {"UPL.NS": [(_result(signal="SELL"), _candle())]}, _config(), _cash_state(),
        _FakeTradeRepository(), _FakeLiveOrderRepository(),
        _FakeEntryDecisionRepository([_decision()]),
    )
    assert candidates == []


@pytest.mark.asyncio
async def test_a_merely_neutral_signal_does_not_invalidate_the_retry():
    # The whole point: NEUTRAL alone must NOT be read as "thesis dead" --
    # signal is transition-only (see docs/decisions/008's addendum).
    candidates = await _collect_delayed_retry_candidates(
        {"UPL.NS": [(_result(signal="NEUTRAL", end_long=False), _candle())]}, _config(),
        _cash_state(), _FakeTradeRepository(), _FakeLiveOrderRepository(),
        _FakeEntryDecisionRepository([_decision()]),
    )
    assert len(candidates) == 1


@pytest.mark.asyncio
async def test_excludes_a_symbol_already_holding_an_open_position():
    candidates = await _collect_delayed_retry_candidates(
        {"UPL.NS": [(_result(), _candle())]}, _config(), _cash_state(), _FakeTradeRepository(),
        _FakeLiveOrderRepository(open_symbols={"UPL.NS"}),
        _FakeEntryDecisionRepository([_decision()]),
    )
    assert candidates == []


@pytest.mark.asyncio
async def test_excludes_a_symbol_whose_price_moved_outside_tolerance():
    # market_price = (102+100+101+101)/4 = 101 -- 1% above the Rs100
    # original signal price, well past the 0.5% floor.
    far_candle = _candle(open=Decimal("101"), high=Decimal("102"), low=Decimal("100"))
    candidates = await _collect_delayed_retry_candidates(
        {"UPL.NS": [(_result(), far_candle)]}, _config(), _cash_state(), _FakeTradeRepository(),
        _FakeLiveOrderRepository(), _FakeEntryDecisionRepository([_decision()]),
    )
    assert candidates == []


@pytest.mark.asyncio
async def test_includes_a_symbol_exactly_at_the_tolerance_boundary():
    # market_price = (101+100+100.5+100.5)/4 = 100.5 -- exactly 0.5% above
    # Rs100, the inclusive edge (`> tolerance` excludes, `== tolerance`
    # doesn't).
    boundary_candle = _candle(
        open=Decimal("100.5"), high=Decimal("101"), low=Decimal("100"), close=Decimal("100.9")
    )
    assert abs(
        (Decimal("101") + Decimal("100") + Decimal("100.5") + Decimal("100.5")) / 4
        - Decimal("100")
    ) / Decimal("100") == _RETRY_PRICE_TOLERANCE

    candidates = await _collect_delayed_retry_candidates(
        {"UPL.NS": [(_result(), boundary_candle)]}, _config(), _cash_state(),
        _FakeTradeRepository(), _FakeLiveOrderRepository(),
        _FakeEntryDecisionRepository([_decision()]),
    )
    assert len(candidates) == 1


@pytest.mark.asyncio
async def test_excludes_when_track_record_gate_fails_on_recheck():
    candidates = await _collect_delayed_retry_candidates(
        {"UPL.NS": [(_result(), _candle())]}, _config(), _cash_state(),
        _FakeTradeRepository(eligible=False), _FakeLiveOrderRepository(),
        _FakeEntryDecisionRepository([_decision()]),
    )
    assert candidates == []


@pytest.mark.asyncio
async def test_excludes_when_quality_gate_fails_on_the_current_candle():
    # Deliberately below entry_quality_filter's floors, even though the
    # ORIGINAL decision row says quality_passed=True -- proves this is a
    # genuine re-check against today's values, not trusting the stale row.
    weak_result = _result(volatility_margin=0.0, regime_normalized=0.0)
    candidates = await _collect_delayed_retry_candidates(
        {"UPL.NS": [(weak_result, _candle())]}, _config(), _cash_state(),
        _FakeTradeRepository(), _FakeLiveOrderRepository(),
        _FakeEntryDecisionRepository([_decision()]),
    )
    assert candidates == []


@pytest.mark.asyncio
async def test_excludes_when_conviction_gate_fails_on_the_current_candle():
    # close sits near the low of its own bar -- CLV well under 0.7.
    weak_candle = _candle(open=Decimal("100"), high=Decimal("101"), low=Decimal("99"),
                           close=Decimal("99.1"))
    candidates = await _collect_delayed_retry_candidates(
        {"UPL.NS": [(_result(), weak_candle)]}, _config(), _cash_state(), _FakeTradeRepository(),
        _FakeLiveOrderRepository(), _FakeEntryDecisionRepository([_decision()]),
    )
    assert candidates == []


# --- _notify_filled_delayed_retries -----------------------------------------


def _retry_candidate(symbol: str = "UPL.NS") -> tuple[str, RankedCandidate]:
    return (
        symbol,
        RankedCandidate(
            symbol=symbol, entry_timestamp=datetime(2026, 9, 2, 10, 0, tzinfo=UTC),
            entry_price=Decimal("100"), prediction_at_entry=5, adx=25.0,
            regime_normalized=2.0, volatility_margin=10.0,
            retry_of_signal_timestamp=datetime(2026, 9, 1, 8, 45, tzinfo=UTC),
        ),
    )


@pytest.mark.asyncio
async def test_notifies_only_for_a_symbol_that_actually_opened():
    notifier = _FakeNotifier()

    await _notify_filled_delayed_retries(
        [_retry_candidate()], {"UPL.NS": "cash: opened 10 qty (₹5,000)"}, notifier
    )

    assert len(notifier.sent) == 1
    assert "UPL.NS" in notifier.sent[0]
    assert "DELAYED ENTRY FILLED" in notifier.sent[0]


@pytest.mark.asyncio
async def test_does_not_notify_a_retry_that_did_not_fill():
    notifier = _FakeNotifier()

    await _notify_filled_delayed_retries(
        [_retry_candidate()],
        {"UPL.NS": "cash: SKIPPED (no free slot, past cutoff, or execution failed -- see logs)"},
        notifier,
    )

    assert notifier.sent == []


@pytest.mark.asyncio
async def test_notify_is_best_effort_and_never_raises():
    notifier = _FakeNotifier(raise_error=True)

    # Must not raise -- a Telegram failure can't be allowed to look like
    # the real entry itself failed.
    await _notify_filled_delayed_retries(
        [_retry_candidate()], {"UPL.NS": "cash: opened 10 qty (₹5,000)"}, notifier
    )
