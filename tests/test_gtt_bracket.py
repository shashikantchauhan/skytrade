"""Tests for the real target/stop-loss OCO GTT bracket lifecycle (see
application/gtt_bracket.py). Everything here uses fakes -- no real Kite
connection, no real money.

Focus: the kill switch gates, the extension only fires once and moves the
stop to breakeven, and reconcile_before_exit correctly distinguishes "GTT
still live, safe to cancel + market-exit" from "GTT already fired, do NOT
place a redundant/wrong market order."
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_scanner.application import gtt_bracket
from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import GttBracket


def _config(
    *, enabled: bool = True, symbols: frozenset[str] = frozenset({"RELIANCE.NS"})
) -> AppConfig:
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
        live_cash_trading_enabled=enabled,
        live_cash_trading_symbols=symbols,
        live_cash_trading_notional=Decimal("5000"),
    )


class FakeOrderExecutor:
    def __init__(
        self,
        gtt_status: str = "active",
        tick_size: Decimal = Decimal("0.05"),
        holding_quantity: int | None = None,
        raise_on_gtt_status: bool = False,
    ) -> None:
        self.placed: list[tuple] = []
        self.modified: list[tuple] = []
        self.deleted: list[int] = []
        self._gtt_status = gtt_status
        self._next_trigger_id = 100
        self._tick_size = tick_size
        # None means "not configured for this fake" -- reconcile_before_exit
        # treats a raised exception the same as a real API failure, i.e.
        # falls back to GTT-status-only reasoning (see its docstring).
        self._holding_quantity = holding_quantity
        self._raise_on_gtt_status = raise_on_gtt_status

    def tick_size(self, tradingsymbol: str) -> Decimal:
        return self._tick_size

    def holding_quantity(self, tradingsymbol: str) -> int:
        if self._holding_quantity is None:
            raise RuntimeError("holding_quantity not configured for this fake")
        return self._holding_quantity

    def place_cash_bracket_gtt(self, tradingsymbol, quantity, last_price, stop_price, target_price):
        self.placed.append((tradingsymbol, quantity, last_price, stop_price, target_price))
        self._next_trigger_id += 1
        return self._next_trigger_id

    def modify_cash_bracket_gtt(
        self, trigger_id, tradingsymbol, quantity, last_price, stop_price, target_price
    ):
        self.modified.append(
            (trigger_id, tradingsymbol, quantity, last_price, stop_price, target_price)
        )

    def gtt_status(self, trigger_id):
        if self._raise_on_gtt_status:
            raise RuntimeError("Error during internal conversion.")
        return self._gtt_status

    def delete_gtt(self, trigger_id):
        self.deleted.append(trigger_id)


class FakeGttRepository:
    def __init__(self, active: GttBracket | None = None) -> None:
        self.recorded: list[GttBracket] = []
        self._active = active
        self.status_updates: list[tuple] = []

    async def record(self, bracket: GttBracket) -> None:
        self.recorded.append(bracket)
        self._active = bracket

    async def get_active(self, symbol: str):
        return self._active

    async def update_status(self, trigger_id, status, stop_price=None, target_price=None):
        self.status_updates.append((trigger_id, status, stop_price, target_price))
        if self._active is not None and self._active.trigger_id == trigger_id:
            self._active.status = status
            if stop_price is not None:
                self._active.stop_price = stop_price
            if target_price is not None:
                self._active.target_price = target_price


class FakeNotifier:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def send_text(self, message: str) -> None:
        self.texts.append(message)


def _bracket(**overrides) -> GttBracket:
    defaults = dict(
        symbol="RELIANCE.NS", trigger_id=101, tradingsymbol="RELIANCE", quantity=1,
        entry_price=Decimal("1400"), stop_price=Decimal("1358"), target_price=Decimal("1540"),
        created_at=datetime.now(UTC), status="active",
    )
    defaults.update(overrides)
    return GttBracket(**defaults)


@pytest.mark.asyncio
async def test_place_bracket_noop_when_gate_closed():
    executor = FakeOrderExecutor()
    repo = FakeGttRepository()
    await gtt_bracket.place_bracket(
        "RELIANCE.NS", "RELIANCE", 1, Decimal("1400"),
        _config(enabled=False), executor, repo, FakeNotifier(),
    )
    assert executor.placed == []
    assert repo.recorded == []


@pytest.mark.asyncio
async def test_place_bracket_computes_10pct_target_and_3pct_stop():
    executor = FakeOrderExecutor()
    repo = FakeGttRepository()
    notifier = FakeNotifier()

    await gtt_bracket.place_bracket(
        "RELIANCE.NS", "RELIANCE", 1, Decimal("1400"), _config(), executor, repo, notifier,
    )

    assert len(executor.placed) == 1
    tradingsymbol, quantity, last_price, stop_price, target_price = executor.placed[0]
    assert stop_price == Decimal("1358.00")  # 1400 * 0.97
    assert target_price == Decimal("1540.00")  # 1400 * 1.10
    assert len(repo.recorded) == 1
    assert any("GTT BRACKET PLACED" in t for t in notifier.texts)


@pytest.mark.asyncio
async def test_place_bracket_rounds_to_the_instruments_real_tick_size():
    # 2026-08-25 regression: a flat 0.05 quantize rejected stocks whose real
    # tick size is 0.10 (InputException: "Stoploss trigger price should be
    # a multiple of tick size 0.10") -- must round to whatever tick_size()
    # actually returns, and the DB-recorded bracket must match exactly what
    # was sent to Kite.
    executor = FakeOrderExecutor(tick_size=Decimal("0.10"))
    repo = FakeGttRepository()
    notifier = FakeNotifier()

    await gtt_bracket.place_bracket(
        "BDL.NS", "BDL", 3, Decimal("1380.5"), _config(symbols=frozenset({"BDL.NS"})),
        executor, repo, notifier,
    )

    assert len(executor.placed) == 1
    _, _, _, stop_price, target_price = executor.placed[0]
    assert stop_price == Decimal("1339.10")  # 1380.5 * 0.97, on a 0.10 tick
    assert target_price == Decimal("1518.60")  # 1380.5 * 1.10, on a 0.10 tick
    assert repo.recorded[0].stop_price == stop_price
    assert repo.recorded[0].target_price == target_price


@pytest.mark.asyncio
async def test_extend_noop_below_threshold():
    executor = FakeOrderExecutor()
    repo = FakeGttRepository(active=_bracket())
    # 1400 -> 1500 is ~7.1% gain, below the 8% extension trigger.
    await gtt_bracket.check_and_extend(
        "RELIANCE.NS", Decimal("1500"), _config(), executor, repo, FakeNotifier(),
    )
    assert executor.modified == []


@pytest.mark.asyncio
async def test_extend_fires_at_8pct_gain_moves_stop_to_breakeven():
    executor = FakeOrderExecutor()
    bracket = _bracket()
    repo = FakeGttRepository(active=bracket)
    notifier = FakeNotifier()

    await gtt_bracket.check_and_extend(
        "RELIANCE.NS", Decimal("1520"), _config(), executor, repo, notifier,  # ~8.6% gain
    )

    assert len(executor.modified) == 1
    _, _, _, _, new_stop, new_target = executor.modified[0]
    assert new_stop == Decimal("1400")  # breakeven, i.e. the original entry price
    assert new_target == Decimal("1610.00")  # 1400 * 1.15
    assert bracket.status == "extended"
    assert any("GTT EXTENDED" in t for t in notifier.texts)


@pytest.mark.asyncio
async def test_extend_only_fires_once():
    executor = FakeOrderExecutor()
    repo = FakeGttRepository(active=_bracket(status="extended"))
    await gtt_bracket.check_and_extend(
        "RELIANCE.NS", Decimal("1700"), _config(), executor, repo, FakeNotifier(),
    )
    assert executor.modified == []  # already extended -- left alone


@pytest.mark.asyncio
async def test_reconcile_cancels_a_still_live_gtt_and_allows_market_exit():
    executor = FakeOrderExecutor(gtt_status="active")
    bracket = _bracket()
    repo = FakeGttRepository(active=bracket)

    should_exit = await gtt_bracket.reconcile_before_exit(
        "RELIANCE.NS", "RELIANCE", _config(), executor, repo,
    )

    assert should_exit is True
    assert executor.deleted == [101]


@pytest.mark.asyncio
async def test_reconcile_still_fires_when_toggle_is_disabled():
    # A "Stop" click (enabled=False) must never leave a still-live GTT
    # unmanaged -- reconcile is gated on "is a bracket actually active,"
    # not on the current toggle state.
    executor = FakeOrderExecutor(gtt_status="active")
    bracket = _bracket()
    repo = FakeGttRepository(active=bracket)

    should_exit = await gtt_bracket.reconcile_before_exit(
        "RELIANCE.NS", "RELIANCE", _config(enabled=False), executor, repo,
    )

    assert should_exit is True
    assert executor.deleted == [101]
    assert bracket.status == "cancelled"
    assert bracket.status == "cancelled"


@pytest.mark.asyncio
async def test_reconcile_skips_market_exit_when_gtt_already_triggered():
    executor = FakeOrderExecutor(gtt_status="triggered")
    bracket = _bracket()
    repo = FakeGttRepository(active=bracket)

    should_exit = await gtt_bracket.reconcile_before_exit(
        "RELIANCE.NS", "RELIANCE", _config(), executor, repo,
    )

    assert should_exit is False  # real position already flat -- don't sell again
    assert executor.deleted == []  # nothing to delete, it already fired
    assert bracket.status == "closed"


@pytest.mark.asyncio
async def test_reconcile_noop_when_no_bracket_exists():
    executor = FakeOrderExecutor()
    repo = FakeGttRepository(active=None)

    should_exit = await gtt_bracket.reconcile_before_exit(
        "RELIANCE.NS", "RELIANCE", _config(), executor, repo,
    )

    assert should_exit is True  # nothing to reconcile, normal exit proceeds
    assert executor.deleted == []


@pytest.mark.asyncio
async def test_reconcile_trusts_real_holding_over_a_stale_active_gtt_status():
    # 2026-08-28 regression: COCHINSHIP.NS's GTT still reported "active" at
    # Kite days after the real position was already flat -- gtt_status
    # alone said "still live," which would have sent a real SELL against
    # zero real shares. Real holding must win.
    executor = FakeOrderExecutor(gtt_status="active", holding_quantity=0)
    bracket = _bracket()
    repo = FakeGttRepository(active=bracket)

    should_exit = await gtt_bracket.reconcile_before_exit(
        "COCHINSHIP.NS", "COCHINSHIP", _config(), executor, repo,
    )

    assert should_exit is False  # real position already flat -- don't sell
    assert executor.deleted == [101]  # still clean up the dangling live GTT
    assert bracket.status == "cancelled"


@pytest.mark.asyncio
async def test_reconcile_trusts_real_holding_when_gtt_status_check_raises():
    # 2026-08-28 regression: VMM.NS's gtt_status() raised a bare
    # kiteconnect GeneralException ("Error during internal conversion")
    # instead of returning a clean status -- the exception fallback used
    # to assume "active" and proceed to a real (rejected) SELL. Real
    # holding must still be trusted even when the GTT status check itself
    # is broken.
    executor = FakeOrderExecutor(holding_quantity=0, raise_on_gtt_status=True)
    bracket = _bracket()
    repo = FakeGttRepository(active=bracket)

    should_exit = await gtt_bracket.reconcile_before_exit(
        "VMM.NS", "VMM", _config(), executor, repo,
    )

    assert should_exit is False
    assert bracket.status == "cancelled"


@pytest.mark.asyncio
async def test_reconcile_allows_exit_when_real_holding_confirms_still_open():
    executor = FakeOrderExecutor(gtt_status="active", holding_quantity=1)
    bracket = _bracket()
    repo = FakeGttRepository(active=bracket)

    should_exit = await gtt_bracket.reconcile_before_exit(
        "RELIANCE.NS", "RELIANCE", _config(), executor, repo,
    )

    assert should_exit is True
    assert executor.deleted == [101]
