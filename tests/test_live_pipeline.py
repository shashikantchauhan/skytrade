"""Characterization tests for live_pipeline.py -- Phase 9-10 of
`projectedPlann.md` (see docs/architecture/000-audit.md). No test here
opens a real KiteTicker/Kite connection or touches a real broker.

Scope, stated honestly: this covers the pieces of ``LiveTickerPipeline``
testable today without a broker-connection seam (tick callbacks, the
tick-level stop-loss/trailing-stop check, and the concurrent-loop
first-exit-wins helper -- the exact mechanism behind the 2026-08-18
orphaned-task incident documented in ``_run_until_first_exit``'s own
docstring). Full connection/reconnect/stale-hang scenarios through
``run_forever`` need a real dependency-injection seam for ``KiteTicker``/
``KiteConnect`` that doesn't exist yet -- that's the ``KiteTickerAdapter``
extraction this stage's own docstring describes, deliberately not rushed
in the same pass as this harness. Not claiming full Phase 10 coverage;
this is what's safely testable against the current structure.
"""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import PaperPosition
from trading_scanner.live_pipeline import LiveTickerPipeline, _run_until_first_exit


def _config() -> AppConfig:
    return AppConfig(
        scan_interval_hours=1,
        candle_interval="1h",
        candle_history=300,
        symbols_file=None,
        logging_level=20,
        turso_database_url=None,
        turso_auth_token=None,
        telegram_bot_token=None,
        telegram_chat_id=None,
        index_symbol=None,
        kite_api_key=None,
        kite_api_secret=None,
        live_trading_enabled=False,
        live_trading_symbols=frozenset(),
        live_trading_max_lots=1,
        futures_paper_symbols_file=None,
    )


class FakeSignalRepository:
    def __init__(self) -> None:
        self.recorded: set[str] = set()

    async def contains(self, fingerprint: str) -> bool:
        return fingerprint in self.recorded

    async def record(self, fingerprint: str, created_at) -> None:
        self.recorded.add(fingerprint)


class FakePaperAccountRepository:
    """In-memory PaperAccountRepository fake -- only what
    _close_paper_position/_check_stop_loss actually call."""

    def __init__(self) -> None:
        self.opened: list[PaperPosition] = []
        self.closed: list[PaperPosition] = []
        self.peak_updates: list[tuple[str, Decimal]] = []

    async def close_position(self, symbol: str, exit_timestamp, exit_price: Decimal):
        matching = [p for p in self.opened if p.symbol == symbol]
        if not matching:
            return None
        position = matching[-1]
        pnl_amount = (exit_price - position.entry_price) * position.quantity
        closed = replace(
            position, exit_timestamp=exit_timestamp, exit_price=exit_price,
            pnl_amount=pnl_amount, status="closed",
        )
        self.closed.append(closed)
        return closed

    async def update_peak_price(self, symbol: str, peak_price: Decimal) -> None:
        self.peak_updates.append((symbol, peak_price))


class FakeNotifier:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def send_text(self, message: str) -> None:
        self.texts.append(message)


def _position(symbol: str, entry_price: Decimal, quantity: int = 10) -> PaperPosition:
    return PaperPosition(
        symbol=symbol, entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        entry_price=entry_price, quantity=quantity,
        capital_allocated=entry_price * quantity,
    )


def _pipeline_with_fakes() -> tuple[LiveTickerPipeline, FakePaperAccountRepository, FakeNotifier]:
    pipeline = LiveTickerPipeline(_config(), ["RELIANCE.NS"])
    paper_account_repository = FakePaperAccountRepository()
    pipeline._repos = {
        "paper_account": paper_account_repository,
        "signal": FakeSignalRepository(),
    }
    notifier = FakeNotifier()
    pipeline._notifier = notifier
    return pipeline, paper_account_repository, notifier


# --- tick callbacks --------------------------------------------------------


def test_on_ticks_updates_last_tick_at_and_enqueues():
    pipeline = LiveTickerPipeline(_config(), ["RELIANCE.NS"])
    before = pipeline._last_tick_at
    ticks = [{"instrument_token": 1, "last_price": 100.0}]

    pipeline._on_ticks(None, ticks)

    assert pipeline._last_tick_at >= before
    assert pipeline._tick_queue.get_nowait() == ticks


def test_on_connect_resets_last_tick_at_and_subscribes():
    pipeline = LiveTickerPipeline(_config(), ["RELIANCE.NS"])
    pipeline._token_to_symbol = {123: "RELIANCE.NS"}
    before = datetime(2020, 1, 1, tzinfo=UTC)
    pipeline._last_tick_at = before

    class _FakeWs:
        MODE_FULL = "full"

        def __init__(self) -> None:
            self.subscribed: list[int] | None = None
            self.mode_set: tuple[str, list[int]] | None = None

        def subscribe(self, tokens):
            self.subscribed = tokens

        def set_mode(self, mode, tokens):
            self.mode_set = (mode, tokens)

    ws = _FakeWs()
    pipeline._on_connect(ws, {})

    assert pipeline._last_tick_at > before
    assert ws.subscribed == [123]
    assert ws.mode_set == ("full", [123])


# --- _close_symbol_caches ---------------------------------------------------


def test_close_symbol_caches_drops_both_caches_for_the_symbol_only():
    pipeline = LiveTickerPipeline(_config(), ["RELIANCE.NS"])
    pipeline._open_positions_cache = {"RELIANCE.NS": Decimal("100"), "TCS.NS": Decimal("200")}
    pipeline._peak_price_cache = {"RELIANCE.NS": Decimal("110"), "TCS.NS": Decimal("210")}

    pipeline._close_symbol_caches("RELIANCE.NS")

    assert "RELIANCE.NS" not in pipeline._open_positions_cache
    assert "RELIANCE.NS" not in pipeline._peak_price_cache
    assert pipeline._open_positions_cache["TCS.NS"] == Decimal("200")
    assert pipeline._peak_price_cache["TCS.NS"] == Decimal("210")


def test_close_symbol_caches_is_a_no_op_for_a_symbol_not_cached():
    pipeline = LiveTickerPipeline(_config(), ["RELIANCE.NS"])
    pipeline._open_positions_cache = {}
    pipeline._peak_price_cache = {}

    pipeline._close_symbol_caches("RELIANCE.NS")  # must not raise

    assert pipeline._open_positions_cache == {}


# --- _check_stop_loss --------------------------------------------------------


async def test_check_stop_loss_ignores_a_token_with_no_symbol_mapping():
    pipeline, paper_account_repository, notifier = _pipeline_with_fakes()
    pipeline._token_to_symbol = {}

    await pipeline._check_stop_loss(999, Decimal("100"))

    assert paper_account_repository.closed == []


async def test_check_stop_loss_ignores_a_symbol_with_no_open_position():
    pipeline, paper_account_repository, notifier = _pipeline_with_fakes()
    pipeline._token_to_symbol = {1: "RELIANCE.NS"}
    pipeline._open_positions_cache = {}

    await pipeline._check_stop_loss(1, Decimal("100"))

    assert paper_account_repository.closed == []


async def test_check_stop_loss_force_closes_below_the_hard_stop():
    pipeline, paper_account_repository, notifier = _pipeline_with_fakes()
    entry = Decimal("100")
    pipeline._token_to_symbol = {1: "RELIANCE.NS"}
    pipeline._open_positions_cache = {"RELIANCE.NS": entry}
    paper_account_repository.opened.append(_position("RELIANCE.NS", entry))

    # Default STOP_LOSS_PCT is 3% -- 96 is comfortably below the 97 stop.
    await pipeline._check_stop_loss(1, Decimal("96"))

    assert len(paper_account_repository.closed) == 1
    assert paper_account_repository.closed[0].symbol == "RELIANCE.NS"
    # Force-closing clears both caches for the symbol immediately (see
    # _close_symbol_caches's docstring -- a burst of ticks must not
    # trigger a second close attempt before the next cache refresh).
    assert "RELIANCE.NS" not in pipeline._open_positions_cache


async def test_check_stop_loss_does_not_close_above_the_stop_and_below_trail_activation():
    pipeline, paper_account_repository, notifier = _pipeline_with_fakes()
    entry = Decimal("100")
    pipeline._token_to_symbol = {1: "RELIANCE.NS"}
    pipeline._open_positions_cache = {"RELIANCE.NS": entry}
    paper_account_repository.opened.append(_position("RELIANCE.NS", entry))

    # 105 is above the 97 hard stop and well below the 115 trail-activation
    # threshold (default 15%) -- nothing should fire.
    await pipeline._check_stop_loss(1, Decimal("105"))

    assert paper_account_repository.closed == []
    assert "RELIANCE.NS" in pipeline._open_positions_cache


async def test_check_stop_loss_tracks_a_new_peak_without_closing():
    pipeline, paper_account_repository, notifier = _pipeline_with_fakes()
    entry = Decimal("100")
    pipeline._token_to_symbol = {1: "RELIANCE.NS"}
    pipeline._open_positions_cache = {"RELIANCE.NS": entry}
    paper_account_repository.opened.append(_position("RELIANCE.NS", entry))

    await pipeline._check_stop_loss(1, Decimal("108"))

    assert pipeline._peak_price_cache["RELIANCE.NS"] == Decimal("108")
    assert paper_account_repository.peak_updates == [("RELIANCE.NS", Decimal("108"))]
    assert paper_account_repository.closed == []


async def test_check_stop_loss_force_closes_below_the_trailing_stop_once_activated():
    pipeline, paper_account_repository, notifier = _pipeline_with_fakes()
    entry = Decimal("100")
    pipeline._token_to_symbol = {1: "RELIANCE.NS"}
    # Peak already at 120 (well past the 15% activation threshold) -- the
    # trail sits at 120 * (1 - 3%) = 116.4.
    pipeline._open_positions_cache = {"RELIANCE.NS": entry}
    pipeline._peak_price_cache = {"RELIANCE.NS": Decimal("120")}
    paper_account_repository.opened.append(_position("RELIANCE.NS", entry))

    await pipeline._check_stop_loss(1, Decimal("116"))

    assert len(paper_account_repository.closed) == 1
    assert "RELIANCE.NS" not in pipeline._open_positions_cache


async def test_check_stop_loss_hard_stop_takes_priority_over_trailing_stop():
    # A price that breaches BOTH the hard stop and would-be trail must
    # still only close once, via the hard-stop branch -- checked first,
    # returns immediately (see _check_stop_loss's own docstring on order).
    pipeline, paper_account_repository, notifier = _pipeline_with_fakes()
    entry = Decimal("100")
    pipeline._token_to_symbol = {1: "RELIANCE.NS"}
    pipeline._open_positions_cache = {"RELIANCE.NS": entry}
    pipeline._peak_price_cache = {"RELIANCE.NS": Decimal("150")}
    paper_account_repository.opened.append(_position("RELIANCE.NS", entry))

    await pipeline._check_stop_loss(1, Decimal("50"))  # crashes well below both

    assert len(paper_account_repository.closed) == 1


# --- _run_until_first_exit ---------------------------------------------------
# 2026-08-18 incident (see this function's own docstring): a plain
# asyncio.gather() here leaked every other still-running loop as an
# orphaned task whenever one of them raised, instead of cancelling them --
# these tests are the regression net for that fix.


async def test_first_exit_cancels_the_rest_when_one_completes():
    cancelled = asyncio.Event()

    async def _finishes_first():
        return "done"

    async def _hangs_forever():
        try:
            await asyncio.sleep(1000)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    await _run_until_first_exit((_finishes_first(), _hangs_forever()))

    assert cancelled.is_set()


async def test_first_exit_reraises_the_first_coroutines_exception():
    cancelled = asyncio.Event()

    async def _raises_first():
        raise RuntimeError("boom")

    async def _hangs_forever():
        try:
            await asyncio.sleep(1000)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with pytest.raises(RuntimeError, match="boom"):
        await _run_until_first_exit((_raises_first(), _hangs_forever()))

    assert cancelled.is_set()


async def test_first_exit_propagates_cancellation_of_the_whole_group():
    async def _hangs_forever():
        await asyncio.sleep(1000)

    task = asyncio.ensure_future(
        _run_until_first_exit((_hangs_forever(), _hangs_forever()))
    )
    await asyncio.sleep(0)  # let it start
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
