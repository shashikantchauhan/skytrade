import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
import pytest

import trading_scanner.application.signal_pipeline as signal_pipeline_module
from trading_scanner.application.fast_predict import ExitState, FastPredictResult, QueueState
from trading_scanner.application.ranking import RankedCandidate
from trading_scanner.application.signal_pipeline import (
    _BACKFILL_WINDOW_DAYS,
    _RECENT_WINDOW_DAYS,
    _close_futures_paper,
    _collect_and_open_ranked_positions,
    _notify_missed_cash_entry,
    _open_futures_paper,
    _rank_and_open_cash_positions,
    _rank_and_open_futures_positions,
    _rank_and_open_paper_positions,
    run_signal_pipeline,
)
from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import (
    Candle,
    FuturesPaperPosition,
    LiveOrderLeg,
    PaperPosition,
    SignalSide,
    Trade,
)
from trading_scanner.domain.ports import EngineState
from trading_scanner.infrastructure.db import LiveCashToggleState
from trading_scanner.infrastructure.yahoo import YahooProvider


class FakeCandleRepository:
    """In-memory CandleRepository fake that accumulates candles per symbol."""

    def __init__(self, seed: dict[str, list[Candle]] | None = None) -> None:
        self._store: dict[tuple[str, str], list[Candle]] = {}
        for symbol, candles in (seed or {}).items():
            self._store[(symbol, "1h")] = list(candles)
        self.upserted: list[tuple[str, str, list[Candle]]] = []

    async def upsert_candles(self, symbol, interval, candles) -> None:
        self.upserted.append((symbol, interval, list(candles)))
        existing = {candle.timestamp: candle for candle in self._store.get((symbol, interval), [])}
        for candle in candles:
            existing[candle.timestamp] = candle
        self._store[(symbol, interval)] = sorted(existing.values(), key=lambda c: c.timestamp)

    async def get_candles(self, symbol, interval, limit=None):
        candles = self._store.get((symbol, interval), [])
        return candles[-limit:] if limit is not None else candles


class FakeSignalRepository:
    def __init__(self) -> None:
        self.recorded: set[str] = set()

    async def contains(self, fingerprint: str) -> bool:
        return fingerprint in self.recorded

    async def record(self, fingerprint: str, created_at) -> None:
        self.recorded.add(fingerprint)


class FakeEngineStateRepository:
    """In-memory EngineStateRepository fake -- returns EngineState() defaults
    (signaling "never seen, bootstrap") for any symbol not yet stored."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], EngineState] = {}

    async def get_state(self, symbol, interval) -> EngineState:
        return self._store.get((symbol, interval), EngineState())

    async def set_state(self, symbol, interval, state: EngineState) -> None:
        self._store[(symbol, interval)] = state


class FakeTradeRepository:
    """In-memory TradeRepository fake."""

    def __init__(self) -> None:
        self.opened: list[Trade] = []
        self.closed: list[tuple[str, str, object, object, object]] = []
        self.abandoned: list[tuple[str, str, object]] = []

    async def open_trade(self, interval, trade: Trade) -> None:
        self.opened.append(trade)

    async def close_open_trade(self, symbol, interval, side, exit_timestamp, exit_price) -> None:
        self.closed.append((symbol, interval, side, exit_timestamp, exit_price))

    async def abandon_open_trade(self, symbol, interval, side) -> None:
        self.abandoned.append((symbol, interval, side))

    async def get_trades(self, symbol, interval):
        return self.opened


class FakePaperAccountRepository:
    """In-memory PaperAccountRepository fake."""

    def __init__(self, cash_balance=Decimal("500000")) -> None:
        self._cash_balance = cash_balance
        self.opened = []
        self.closed = []

    async def get_cash_balance(self) -> Decimal:
        return self._cash_balance

    async def open_position(self, position) -> None:
        self.opened.append(position)
        self._cash_balance -= position.capital_allocated

    async def close_position(self, symbol, exit_timestamp, exit_price):
        matching = [p for p in self.opened if p.symbol == symbol]
        if not matching:
            return None
        position = matching[-1]
        pnl_amount = (exit_price - position.entry_price) * position.quantity
        self._cash_balance += position.capital_allocated + pnl_amount
        closed = replace(
            position,
            exit_timestamp=exit_timestamp,
            exit_price=exit_price,
            pnl_amount=pnl_amount,
            status="closed",
        )
        self.closed.append(closed)
        return closed

    async def get_open_positions(self):
        return [p for p in self.opened if p.status == "open"]

    async def update_peak_price(self, symbol, peak_price) -> None:
        for index, position in enumerate(self.opened):
            if position.symbol == symbol and position.status == "open":
                self.opened[index] = replace(position, peak_price=peak_price)


class FakeNotifier:
    def __init__(self) -> None:
        self.texts = []

    async def send_text(self, message: str) -> None:
        self.texts.append(message)


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


def _small_recent_download() -> pd.DataFrame:
    """A tiny download -- just the newest candle, as a real hourly run would fetch."""
    return pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Volume": [1_000],
        },
        index=pd.DatetimeIndex([datetime(2026, 8, 6, 15, 15, tzinfo=UTC)]),
    )


def _seed_candles(symbol: str, count: int) -> list[Candle]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        Candle(
            symbol=symbol,
            timestamp=start + pd.Timedelta(hours=index),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=1_000,
        )
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_symbol_below_minimum_history_is_skipped_without_analysis(monkeypatch) -> None:
    monkeypatch.setattr(
        YahooProvider,
        "get_recent_history",
        lambda self, symbol, interval, days: _small_recent_download(),
    )
    seed = {"AARTIIND.NS": _seed_candles("AARTIIND.NS", 50)}
    candle_repository = FakeCandleRepository(seed=seed)
    signal_repository = FakeSignalRepository()
    engine_state_repository = FakeEngineStateRepository()
    trade_repository = FakeTradeRepository()
    paper_account_repository = FakePaperAccountRepository()
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        paper_account_repository,
        notifier,
        market_data_provider=YahooProvider(),
    )

    assert candle_repository.upserted  # the recent window is still ingested while warming up


@pytest.mark.asyncio
async def test_candles_are_upserted_every_run(monkeypatch) -> None:
    monkeypatch.setattr(
        YahooProvider,
        "get_recent_history",
        lambda self, symbol, interval, days: _small_recent_download(),
    )
    seed = {"AARTIIND.NS": _seed_candles("AARTIIND.NS", 199)}
    candle_repository = FakeCandleRepository(seed=seed)
    signal_repository = FakeSignalRepository()
    engine_state_repository = FakeEngineStateRepository()
    trade_repository = FakeTradeRepository()
    paper_account_repository = FakePaperAccountRepository()
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        paper_account_repository,
        notifier,
        market_data_provider=YahooProvider(),
    )

    symbol, interval, upserted_candles = candle_repository.upserted[0]
    assert symbol == "AARTIIND.NS"
    assert interval == "1h"
    assert len(upserted_candles) == 1


@pytest.mark.asyncio
async def test_new_symbol_triggers_a_full_backfill_window(monkeypatch) -> None:
    requested_days = []

    def fake_download(self, symbol, interval, days):
        requested_days.append(days)
        return _small_recent_download()

    monkeypatch.setattr(YahooProvider, "get_recent_history", fake_download)
    candle_repository = FakeCandleRepository()  # no seed -- brand-new symbol
    signal_repository = FakeSignalRepository()
    engine_state_repository = FakeEngineStateRepository()
    trade_repository = FakeTradeRepository()
    paper_account_repository = FakePaperAccountRepository()
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        paper_account_repository,
        notifier,
        market_data_provider=YahooProvider(),
    )

    assert requested_days == [_BACKFILL_WINDOW_DAYS]


@pytest.mark.asyncio
async def test_known_symbol_only_requests_the_recent_window(monkeypatch) -> None:
    requested_days = []

    def fake_download(self, symbol, interval, days):
        requested_days.append(days)
        return _small_recent_download()

    monkeypatch.setattr(YahooProvider, "get_recent_history", fake_download)
    seed = {"AARTIIND.NS": _seed_candles("AARTIIND.NS", 200)}  # already cleared warm-up
    candle_repository = FakeCandleRepository(seed=seed)
    signal_repository = FakeSignalRepository()
    engine_state_repository = FakeEngineStateRepository()
    trade_repository = FakeTradeRepository()
    paper_account_repository = FakePaperAccountRepository()
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        paper_account_repository,
        notifier,
        market_data_provider=YahooProvider(),
    )

    assert requested_days == [_RECENT_WINDOW_DAYS]


@pytest.mark.asyncio
async def test_one_failing_symbol_does_not_stop_the_batch(monkeypatch) -> None:
    def fake_download(self, symbol, interval, days):
        if symbol == "BROKEN.NS":
            raise RuntimeError("boom")
        return _small_recent_download()

    monkeypatch.setattr(YahooProvider, "get_recent_history", fake_download)
    seed = {
        "BROKEN.NS": _seed_candles("BROKEN.NS", 50),
        "AARTIIND.NS": _seed_candles("AARTIIND.NS", 50),
    }
    candle_repository = FakeCandleRepository(seed=seed)
    signal_repository = FakeSignalRepository()
    engine_state_repository = FakeEngineStateRepository()
    trade_repository = FakeTradeRepository()
    paper_account_repository = FakePaperAccountRepository()
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["BROKEN.NS", "AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        paper_account_repository,
        notifier,
        market_data_provider=YahooProvider(),
    )

    # BROKEN.NS raised during download; AARTIIND.NS should still be processed.
    assert any(symbol == "AARTIIND.NS" for symbol, _, _ in candle_repository.upserted)


@pytest.mark.asyncio
async def test_repeated_run_with_same_newest_candle_does_not_reprocess(monkeypatch) -> None:
    """A second run that sees no new candle (download returns the same latest
    bar) must not re-advance the persisted ANN queue -- doing so would
    double-count that bar's contribution and corrupt future predictions."""
    monkeypatch.setattr(
        YahooProvider,
        "get_recent_history",
        lambda self, symbol, interval, days: _small_recent_download(),
    )
    seed = {"AARTIIND.NS": _seed_candles("AARTIIND.NS", 200)}
    candle_repository = FakeCandleRepository(seed=seed)
    signal_repository = FakeSignalRepository()
    engine_state_repository = FakeEngineStateRepository()
    trade_repository = FakeTradeRepository()
    paper_account_repository = FakePaperAccountRepository()
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        paper_account_repository,
        notifier,
        market_data_provider=YahooProvider(),
    )
    state_after_first_run = await engine_state_repository.get_state("AARTIIND.NS", "1h")

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        paper_account_repository,
        notifier,
        market_data_provider=YahooProvider(),
    )
    state_after_second_run = await engine_state_repository.get_state("AARTIIND.NS", "1h")

    assert state_after_second_run == state_after_first_run


@pytest.mark.asyncio
async def test_buy_entry_opens_a_trade(monkeypatch) -> None:
    """A BUY signal must open a trade with the entry bar's own price/prediction,
    independent of fast_predict's own correctness (covered in test_fast_predict.py)."""
    monkeypatch.setattr(
        YahooProvider,
        "get_recent_history",
        lambda self, symbol, interval, days: _small_recent_download(),
    )
    monkeypatch.setattr(
        "trading_scanner.application.signal_pipeline.evaluate_latest_bar",
        lambda engine, history, signal_previous, queue_state, exit_state: FastPredictResult(
            signal="BUY",
            prediction=6,
            end_long=False,
            end_short=False,
            is_early_signal_flip=True,
            signal_previous=1,
            queue_state=QueueState(),
            exit_state=ExitState(),
        ),
    )
    seed = {"AARTIIND.NS": _seed_candles("AARTIIND.NS", 200)}
    candle_repository = FakeCandleRepository(seed=seed)
    signal_repository = FakeSignalRepository()
    engine_state_repository = FakeEngineStateRepository()
    trade_repository = FakeTradeRepository()
    paper_account_repository = FakePaperAccountRepository()
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        paper_account_repository,
        notifier,
        market_data_provider=YahooProvider(),
    )

    assert len(trade_repository.opened) == 1
    trade = trade_repository.opened[0]
    assert trade.symbol == "AARTIIND.NS"
    assert trade.side == SignalSide.BUY
    assert trade.prediction_at_entry == 6
    assert trade.is_early_signal_flip is True
    assert trade.entry_price == Decimal("100")  # (high+low+2*open)/4 = (101+99+200)/4
    # 2026-08-21: entry-signal Telegram notifications are off entirely --
    # follow only real cash-market order events now.
    assert len(signal_repository.recorded) == 1  # still fingerprint-recorded


@pytest.mark.asyncio
async def test_end_long_closes_the_open_buy_trade(monkeypatch) -> None:
    """An end_long result must close the open BUY trade for that symbol,
    without also treating it as a new entry (signal stays NEUTRAL)."""
    monkeypatch.setattr(
        YahooProvider,
        "get_recent_history",
        lambda self, symbol, interval, days: _small_recent_download(),
    )
    monkeypatch.setattr(
        "trading_scanner.application.signal_pipeline.evaluate_latest_bar",
        lambda engine, history, signal_previous, queue_state, exit_state: FastPredictResult(
            signal="NEUTRAL",
            prediction=-2,
            end_long=True,
            end_short=False,
            is_early_signal_flip=False,
            signal_previous=1,
            queue_state=QueueState(),
            exit_state=ExitState(),
        ),
    )
    seed = {"AARTIIND.NS": _seed_candles("AARTIIND.NS", 200)}
    candle_repository = FakeCandleRepository(seed=seed)
    signal_repository = FakeSignalRepository()
    engine_state_repository = FakeEngineStateRepository()
    trade_repository = FakeTradeRepository()
    paper_account_repository = FakePaperAccountRepository()
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        paper_account_repository,
        notifier,
        market_data_provider=YahooProvider(),
    )

    assert trade_repository.opened == []  # NEUTRAL -- no new trade
    assert len(trade_repository.closed) == 1
    symbol, interval, side, exit_timestamp, exit_price = trade_repository.closed[0]
    assert symbol == "AARTIIND.NS"
    assert side == SignalSide.BUY
    assert exit_price == Decimal("100")  # (high+low+2*open)/4 = (101+99+200)/4


@pytest.mark.asyncio
async def test_sell_entry_abandons_a_still_open_buy_trade_without_scoring_it(monkeypatch) -> None:
    """A new SELL entry must abandon any still-open BUY position rather than
    leave it dangling -- mirrors Pine's ml.backtest, which silently discards
    whatever position was open when the opposite side enters, never scoring
    it as a win or a loss."""
    monkeypatch.setattr(
        YahooProvider,
        "get_recent_history",
        lambda self, symbol, interval, days: _small_recent_download(),
    )
    monkeypatch.setattr(
        "trading_scanner.application.signal_pipeline.evaluate_latest_bar",
        lambda engine, history, signal_previous, queue_state, exit_state: FastPredictResult(
            signal="SELL",
            prediction=-6,
            end_long=False,
            end_short=False,
            is_early_signal_flip=False,
            signal_previous=-1,
            queue_state=QueueState(),
            exit_state=ExitState(),
        ),
    )
    seed = {"AARTIIND.NS": _seed_candles("AARTIIND.NS", 200)}
    candle_repository = FakeCandleRepository(seed=seed)
    signal_repository = FakeSignalRepository()
    engine_state_repository = FakeEngineStateRepository()
    trade_repository = FakeTradeRepository()
    paper_account_repository = FakePaperAccountRepository()
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        paper_account_repository,
        notifier,
        market_data_provider=YahooProvider(),
    )

    assert trade_repository.abandoned == [("AARTIIND.NS", "1h", SignalSide.BUY)]
    assert trade_repository.closed == []  # abandoned, not closed -- never scored
    assert len(trade_repository.opened) == 1
    assert trade_repository.opened[0].side == SignalSide.SELL


@pytest.mark.asyncio
async def test_end_long_is_fingerprint_recorded_but_not_notified(monkeypatch) -> None:
    """A dynamic exit must still be recorded (fingerprint distinct from any
    entry at the same symbol/side/timestamp) even though, as of 2026-08-21,
    it no longer sends a Telegram notification -- follow only real
    cash-market order events now."""
    monkeypatch.setattr(
        YahooProvider,
        "get_recent_history",
        lambda self, symbol, interval, days: _small_recent_download(),
    )
    monkeypatch.setattr(
        "trading_scanner.application.signal_pipeline.evaluate_latest_bar",
        lambda engine, history, signal_previous, queue_state, exit_state: FastPredictResult(
            signal="NEUTRAL",
            prediction=-2,
            end_long=True,
            end_short=False,
            is_early_signal_flip=False,
            signal_previous=1,
            queue_state=QueueState(),
            exit_state=ExitState(),
        ),
    )
    seed = {"AARTIIND.NS": _seed_candles("AARTIIND.NS", 200)}
    candle_repository = FakeCandleRepository(seed=seed)
    signal_repository = FakeSignalRepository()
    engine_state_repository = FakeEngineStateRepository()
    trade_repository = FakeTradeRepository()
    paper_account_repository = FakePaperAccountRepository()
    trade_repository.opened = [
        Trade(
            symbol="AARTIIND.NS", side=SignalSide.BUY,
            entry_timestamp=datetime(2026, 8, 1, tzinfo=UTC), entry_price=Decimal("80"),
            prediction_at_entry=4, is_early_signal_flip=False, status="open",
        ),
    ]
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        paper_account_repository,
        notifier,
        market_data_provider=YahooProvider(),
    )

    assert len(trade_repository.closed) == 1
    assert len(signal_repository.recorded) == 1  # still fingerprint-recorded


@pytest.mark.asyncio
async def test_win_rate_summary_is_buy_only_and_matches_eligibility_gate() -> None:
    """``_win_rate_summary`` -- BUY-only, matching the paper account's
    eligibility gate exactly, so its number is never a healthier combined
    BUY+SELL figure that would misrepresent what eligibility actually used.

    2026-08-21: tested directly rather than through the pipeline's
    notification, now that entry-signal Telegram notifications (the only
    place this summary used to surface) are off entirely -- follow only
    real cash-market order events now."""
    trade_repository = FakeTradeRepository()
    # Pre-existing closed trade history: 2 wins, 1 loss.
    trade_repository.opened = [
        Trade(
            symbol="AARTIIND.NS", side=SignalSide.BUY,
            entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC), entry_price=Decimal("100"),
            prediction_at_entry=4, is_early_signal_flip=False,
            exit_timestamp=datetime(2026, 1, 2, tzinfo=UTC), exit_price=Decimal("110"),
            pnl_percent=Decimal("10"), status="closed",
        ),
        Trade(
            symbol="AARTIIND.NS", side=SignalSide.SELL,
            entry_timestamp=datetime(2026, 1, 3, tzinfo=UTC), entry_price=Decimal("100"),
            prediction_at_entry=-4, is_early_signal_flip=False,
            exit_timestamp=datetime(2026, 1, 4, tzinfo=UTC), exit_price=Decimal("105"),
            pnl_percent=Decimal("-5"), status="closed",
        ),
        Trade(
            symbol="AARTIIND.NS", side=SignalSide.BUY,
            entry_timestamp=datetime(2026, 1, 5, tzinfo=UTC), entry_price=Decimal("100"),
            prediction_at_entry=4, is_early_signal_flip=False,
            exit_timestamp=datetime(2026, 1, 6, tzinfo=UTC), exit_price=Decimal("108"),
            pnl_percent=Decimal("8"), status="closed",
        ),
    ]

    summary = await signal_pipeline_module._win_rate_summary(
        "AARTIIND.NS", _config(), trade_repository
    )

    # BUY-only: the two seeded BUY trades (+10%, +8%) are both wins; the
    # seeded SELL loss is excluded, matching paper_trading's eligibility gate.
    assert summary == "win_rate=100.0%(2W/0L)"


@pytest.mark.asyncio
async def test_buy_entry_opens_a_paper_position_when_eligible(monkeypatch) -> None:
    """A BUY entry for a symbol with a strong (>=55%) closed BUY-only track
    record must open a real paper position, sized off the account's
    remaining cash balance."""
    monkeypatch.setattr(
        YahooProvider,
        "get_recent_history",
        lambda self, symbol, interval, days: _small_recent_download(),
    )
    monkeypatch.setattr(
        "trading_scanner.application.signal_pipeline.evaluate_latest_bar",
        lambda engine, history, signal_previous, queue_state, exit_state: FastPredictResult(
            signal="BUY",
            prediction=6,
            end_long=False,
            end_short=False,
            is_early_signal_flip=False,
            signal_previous=1,
            queue_state=QueueState(),
            exit_state=ExitState(),
        ),
    )
    seed = {"AARTIIND.NS": _seed_candles("AARTIIND.NS", 200)}
    candle_repository = FakeCandleRepository(seed=seed)
    signal_repository = FakeSignalRepository()
    engine_state_repository = FakeEngineStateRepository()
    trade_repository = FakeTradeRepository()
    # 5 closed BUY trades, 4 wins -- 80% win rate, well above the 55% bar.
    trade_repository.opened = [
        Trade(
            symbol="AARTIIND.NS", side=SignalSide.BUY,
            entry_timestamp=datetime(2026, 1, index, tzinfo=UTC), entry_price=Decimal("100"),
            prediction_at_entry=4, is_early_signal_flip=False,
            exit_timestamp=datetime(2026, 1, index, tzinfo=UTC), exit_price=Decimal("110"),
            pnl_percent=Decimal("10" if index != 1 else "-5"), status="closed",
        )
        for index in range(1, 6)
    ]
    paper_account_repository = FakePaperAccountRepository()
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        paper_account_repository,
        notifier,
        market_data_provider=YahooProvider(),
    )

    assert len(paper_account_repository.opened) == 1
    position = paper_account_repository.opened[0]
    assert position.symbol == "AARTIIND.NS"
    # total_equity(500000, no open positions)/TARGET_SLOTS(32) = 15625, floored
    # to MIN_POSITION_SIZE(25000) -> quantity = 25000/entry_price(100) = 250.
    assert position.quantity == 250
    # 2026-08-21: entry-signal Telegram notifications are off entirely --
    # follow only real cash-market order events now; the quantity/capital
    # assertions above already confirm the position opened correctly.


@pytest.mark.asyncio
async def test_open_paper_position_skips_when_not_eligible() -> None:
    """A symbol with no (or insufficient) closed BUY track record must not
    open a paper position -- tagged as not eligible in the returned note.

    2026-08-21: called directly rather than through the pipeline's
    notification, now that entry-signal Telegram notifications (the only
    place this note used to surface) are off entirely."""
    trade_repository = FakeTradeRepository()  # no closed trade history at all
    paper_account_repository = FakePaperAccountRepository()

    note = await signal_pipeline_module._open_paper_position(
        "AARTIIND.NS", _config(), datetime(2026, 1, 1, tzinfo=UTC), Decimal("100"),
        trade_repository, paper_account_repository, asyncio.Lock(),
    )

    assert paper_account_repository.opened == []
    assert note is not None and "paper: not eligible" in note


@pytest.mark.asyncio
async def test_open_paper_position_skips_when_no_capital() -> None:
    """An eligible symbol must still be skipped once the account's cash
    balance can't cover one more position slot.

    2026-08-21: called directly rather than through the pipeline's
    notification, now that entry-signal Telegram notifications (the only
    place this note used to surface) are off entirely."""
    trade_repository = FakeTradeRepository()
    trade_repository.opened = [
        Trade(
            symbol="AARTIIND.NS", side=SignalSide.BUY,
            entry_timestamp=datetime(2026, 1, index, tzinfo=UTC), entry_price=Decimal("100"),
            prediction_at_entry=4, is_early_signal_flip=False,
            exit_timestamp=datetime(2026, 1, index, tzinfo=UTC), exit_price=Decimal("110"),
            pnl_percent=Decimal("10"), status="closed",
        )
        for index in range(1, 6)
    ]
    paper_account_repository = FakePaperAccountRepository(cash_balance=Decimal("1000"))

    note = await signal_pipeline_module._open_paper_position(
        "AARTIIND.NS", _config(), datetime(2026, 1, 6, tzinfo=UTC), Decimal("100"),
        trade_repository, paper_account_repository, asyncio.Lock(),
    )

    assert paper_account_repository.opened == []
    assert note is not None and "paper: SKIPPED (no capital available)" in note


class _DelayedPaperAccountRepository(FakePaperAccountRepository):
    """FakePaperAccountRepository with a real suspension point in the
    check-then-act window try_open_position relies on paper_account_lock to
    protect.

    The plain fake's get_cash_balance/open_position never actually suspend
    (no real I/O), so asyncio never interleaves two tasks mid-check even when
    scheduled concurrently -- CPython only switches tasks at genuine
    suspension points. This subclass adds one real ``asyncio.sleep`` between
    reading the balance and the caller deciding whether to commit, so
    concurrent symbols racing for the same capital actually interleave right
    at the vulnerable window, the way real network-backed I/O naturally
    would.
    """

    async def get_cash_balance(self) -> Decimal:
        # Snapshot BEFORE suspending: every concurrently-racing task's
        # snapshot is taken while cash_balance is still unmodified, then each
        # yields control during the sleep (letting other tasks also snapshot
        # the same stale value), and returns that stale snapshot regardless
        # of what anyone else did during the wait -- reproducing exactly the
        # window paper_account_lock exists to close. (Sleeping *before*
        # snapshotting would not race: each waker would just re-read the
        # already-updated value with no real interleaving of the decision.)
        snapshot = self._cash_balance
        await asyncio.sleep(0.005)
        return snapshot


@pytest.mark.asyncio
async def test_concurrent_symbols_never_overspend_shared_paper_account(monkeypatch) -> None:
    """Many symbols processed concurrently must never collectively open more
    positions than the account's capital actually allows.

    Symbols are processed concurrently (see run_signal_pipeline's semaphore),
    so without paper_account_lock serializing the check-then-act in
    try_open_position, two symbols could both read the same "enough capital"
    balance and both commit -- overspending the shared account.
    _DelayedPaperAccountRepository forces genuine interleaving right at that
    check-then-act window (see its docstring)."""
    monkeypatch.setattr(
        YahooProvider, "get_recent_history",
        lambda self, symbol, interval, days: _small_recent_download(),
    )
    monkeypatch.setattr(
        "trading_scanner.application.signal_pipeline.evaluate_latest_bar",
        lambda engine, history, signal_previous, queue_state, exit_state: FastPredictResult(
            signal="BUY",
            prediction=6,
            end_long=False,
            end_short=False,
            is_early_signal_flip=False,
            signal_previous=1,
            queue_state=QueueState(),
            exit_state=ExitState(),
        ),
    )

    symbols = [f"SYM{i}.NS" for i in range(20)]
    seed = {symbol: _seed_candles(symbol, 200) for symbol in symbols}
    candle_repository = FakeCandleRepository(seed=seed)
    signal_repository = FakeSignalRepository()
    engine_state_repository = FakeEngineStateRepository()
    trade_repository = FakeTradeRepository()
    # Shared across all symbols (the fake's get_trades ignores its symbol
    # argument) -- every symbol clears the eligibility bar identically, so
    # all 20 genuinely compete for the same limited capital.
    trade_repository.opened = [
        Trade(
            symbol="ANY.NS", side=SignalSide.BUY,
            entry_timestamp=datetime(2026, 1, index, tzinfo=UTC), entry_price=Decimal("100"),
            prediction_at_entry=4, is_early_signal_flip=False,
            exit_timestamp=datetime(2026, 1, index, tzinfo=UTC), exit_price=Decimal("110"),
            pnl_percent=Decimal("10"), status="closed",
        )
        for index in range(1, 6)
    ]
    # cash_balance/TARGET_SLOTS(32) = 60000/32 = 1875, floored to
    # MIN_POSITION_SIZE(25000) -- so position_size is fixed at 25000, and
    # only floor(60000/25000) = 2 of the 20 symbols can actually be afforded.
    starting_cash = Decimal("60000")
    paper_account_repository = _DelayedPaperAccountRepository(cash_balance=starting_cash)
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        symbols,
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        paper_account_repository,
        notifier,
        market_data_provider=YahooProvider(),
    )

    total_allocated = sum(
        (position.capital_allocated for position in paper_account_repository.opened),
        start=Decimal("0"),
    )
    # The invariant the lock protects: never collectively commit more capital
    # than the account actually had, no matter how the concurrent symbols
    # interleaved.
    assert total_allocated <= starting_cash
    assert len(paper_account_repository.opened) == 2
    # 2026-08-21: entry-signal Telegram notifications are off entirely, so
    # the per-symbol skip-reason tagging ("SKIPPED (no capital available)"
    # vs "SKIPPED (ranked below capacity...)") is no longer independently
    # observable here -- the capital-safety invariant above (never
    # collectively commit more than the account had, exactly 2 of 20
    # symbols afforded) is this test's actual point and is unaffected.


@pytest.mark.asyncio
async def test_end_long_closes_the_paper_position_with_realized_pnl(monkeypatch) -> None:
    """A dynamic exit must close the matching open paper position (crediting
    cash back) with the realized rupee P&L.

    2026-08-21: no longer sends a distinct paper-exit Telegram notification
    -- follow only real cash-market order events now. P&L is verified via
    the closed position itself rather than a notification's rationale."""
    monkeypatch.setattr(
        YahooProvider,
        "get_recent_history",
        lambda self, symbol, interval, days: _small_recent_download(),
    )
    monkeypatch.setattr(
        "trading_scanner.application.signal_pipeline.evaluate_latest_bar",
        lambda engine, history, signal_previous, queue_state, exit_state: FastPredictResult(
            signal="NEUTRAL",
            prediction=-2,
            end_long=True,
            end_short=False,
            is_early_signal_flip=False,
            signal_previous=1,
            queue_state=QueueState(),
            exit_state=ExitState(),
        ),
    )
    seed = {"AARTIIND.NS": _seed_candles("AARTIIND.NS", 200)}
    candle_repository = FakeCandleRepository(seed=seed)
    signal_repository = FakeSignalRepository()
    engine_state_repository = FakeEngineStateRepository()
    trade_repository = FakeTradeRepository()
    paper_account_repository = FakePaperAccountRepository()
    paper_account_repository.opened = [
        PaperPosition(
            symbol="AARTIIND.NS",
            entry_timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            entry_price=Decimal("80"),
            quantity=100,
            capital_allocated=Decimal("8000"),
        )
    ]
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        paper_account_repository,
        notifier,
        market_data_provider=YahooProvider(),
    )

    assert len(paper_account_repository.closed) == 1
    closed_position = paper_account_repository.closed[0]
    assert closed_position.pnl_amount == Decimal("2000")  # (100-80)*100 qty


@pytest.mark.asyncio
async def test_sell_signal_is_tagged_informational_only(monkeypatch) -> None:
    """SELL signals must never touch the paper account -- NSE cash market
    doesn't allow short selling for multi-day holds. Long-only: a SELL
    signal is still recorded (fingerprint dedup/backfill unaffected) but
    never sent to Telegram."""
    monkeypatch.setattr(
        YahooProvider,
        "get_recent_history",
        lambda self, symbol, interval, days: _small_recent_download(),
    )
    monkeypatch.setattr(
        "trading_scanner.application.signal_pipeline.evaluate_latest_bar",
        lambda engine, history, signal_previous, queue_state, exit_state: FastPredictResult(
            signal="SELL",
            prediction=-6,
            end_long=False,
            end_short=False,
            is_early_signal_flip=False,
            signal_previous=-1,
            queue_state=QueueState(),
            exit_state=ExitState(),
        ),
    )
    seed = {"AARTIIND.NS": _seed_candles("AARTIIND.NS", 200)}
    candle_repository = FakeCandleRepository(seed=seed)
    signal_repository = FakeSignalRepository()
    engine_state_repository = FakeEngineStateRepository()
    trade_repository = FakeTradeRepository()
    paper_account_repository = FakePaperAccountRepository()
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        paper_account_repository,
        notifier,
        market_data_provider=YahooProvider(),
    )

    assert paper_account_repository.opened == []
    assert len(signal_repository.recorded) == 1  # but is still fingerprint-recorded


@pytest.mark.asyncio
async def test_index_disagreement_never_blocks_the_stock_signal(
    monkeypatch,
) -> None:
    """When index_symbol is configured, its current state must never block a
    stock signal from being processed, even when the two disagree.

    2026-08-21: no longer verified via a notification's rationale --
    entry-signal Telegram notifications are off entirely (follow only real
    cash-market order events now) -- verified instead via the trade
    actually opening and the signal being fingerprint-recorded."""
    monkeypatch.setattr(
        YahooProvider,
        "get_recent_history",
        lambda self, symbol, interval, days: _small_recent_download(),
    )

    async def fake_evaluate_symbol(
        symbol, config, provider, engine, candle_repository, engine_state_repository
    ):
        newest_candle = _seed_candles(symbol, 1)[0]
        if symbol == "^NSEI":
            # Index disagrees with the stock signal below -- must still notify.
            return (
                FastPredictResult(
                    signal="SELL",
                    prediction=-4,
                    end_long=False,
                    end_short=False,
                    is_early_signal_flip=True,
                    signal_previous=-1,
                    queue_state=QueueState(),
                    exit_state=ExitState(),
                ),
                newest_candle,
            )
        return (
            FastPredictResult(
                signal="BUY",
                prediction=6,
                end_long=False,
                end_short=False,
                is_early_signal_flip=False,
                signal_previous=1,
                queue_state=QueueState(),
                exit_state=ExitState(),
            ),
            newest_candle,
        )

    monkeypatch.setattr(
        "trading_scanner.application.signal_pipeline._evaluate_symbol", fake_evaluate_symbol
    )
    config = AppConfig(
        scan_interval_hours=1,
        candle_interval="1h",
        candle_history=300,
        symbols_file=None,
        logging_level=20,
        turso_database_url=None,
        turso_auth_token=None,
        telegram_bot_token=None,
        telegram_chat_id=None,
        index_symbol="^NSEI",
        kite_api_key=None,
        kite_api_secret=None,
        live_trading_enabled=False,
        live_trading_symbols=frozenset(),
        live_trading_max_lots=1,
        futures_paper_symbols_file=None,
    )
    candle_repository = FakeCandleRepository()
    signal_repository = FakeSignalRepository()
    engine_state_repository = FakeEngineStateRepository()
    trade_repository = FakeTradeRepository()
    paper_account_repository = FakePaperAccountRepository()
    notifier = FakeNotifier()

    await run_signal_pipeline(
        config,
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        paper_account_repository,
        notifier,
        market_data_provider=YahooProvider(),
    )

    assert len(trade_repository.opened) == 1  # stock BUY still processed despite index disagreeing
    assert trade_repository.opened[0].symbol == "AARTIIND.NS"
    assert len(signal_repository.recorded) == 1  # still fingerprint-recorded


async def test_rank_and_open_paper_positions_rejects_below_score_floor(monkeypatch) -> None:
    """A candidate scoring below MIN_SCORE is rejected outright, even with
    plenty of free capital -- distinct from a capacity-driven skip."""
    monkeypatch.setattr(signal_pipeline_module, "MIN_SCORE", 80.0)

    # Both leave expectancy unset (None), which scores as the neutral
    # median decile (50 * 1.5 = 75) -- same fixed offset on both, so it
    # doesn't affect which one wins, only the absolute floor needed here.
    strong = RankedCandidate(  # score = decile(0)*1 + decile(0)*0.5 + 8*2 + 0*1 + 75 = 91
        symbol="A", entry_timestamp=datetime(2026, 2, 1, tzinfo=UTC), entry_price=Decimal("100"),
        prediction_at_entry=8, adx=0.0, regime_normalized=0.0, volatility_margin=0.0,
    )
    weak = RankedCandidate(  # score = 1*2 + 75 (neutral expectancy) = 77, below the floor
        symbol="B", entry_timestamp=datetime(2026, 2, 1, tzinfo=UTC), entry_price=Decimal("100"),
        prediction_at_entry=1, adx=0.0, regime_normalized=0.0, volatility_margin=0.0,
    )
    paper_account_repository = FakePaperAccountRepository()

    notes = await _rank_and_open_paper_positions(
        [("A", strong), ("B", weak)], paper_account_repository, asyncio.Lock()
    )

    assert "opened" in notes["A"]
    assert "REJECTED" in notes["B"]
    assert "below minimum 80" in notes["B"]
    # Rejected by policy, not by running out of capital -- must not have
    # even attempted to open a position for it.
    assert all(position.symbol != "B" for position in paper_account_repository.opened)


async def test_rank_and_open_paper_positions_default_floor_is_a_no_op(monkeypatch) -> None:
    monkeypatch.setattr(signal_pipeline_module, "MIN_SCORE", 0.0)

    weak = RankedCandidate(
        symbol="B", entry_timestamp=datetime(2026, 2, 1, tzinfo=UTC), entry_price=Decimal("100"),
        prediction_at_entry=1, adx=0.0, regime_normalized=0.0, volatility_margin=0.0,
    )
    paper_account_repository = FakePaperAccountRepository()

    notes = await _rank_and_open_paper_positions(
        [("B", weak)], paper_account_repository, asyncio.Lock()
    )

    assert "opened" in notes["B"]


# --- _open_futures_paper / _close_futures_paper ---
# The gating wrapper _process_symbol actually calls per BUY/SELL signal --
# see application/futures_trading.py for the eligibility/margin logic these
# wrap.


class _FakeFuturesDerivativesChain:
    def ltp(self, exchange_tradingsymbol):
        return 2910.0

    def margin_benefit(self, legs):
        return {
            "primary_only_margin": 15000.0, "combined_margin": 10000.0, "margin_benefit": 5000.0,
        }

    def nearest_future(self, symbol):
        return {
            "tradingsymbol": f"{symbol.removesuffix('.NS')}26AUGFUT", "lot_size": 500,
            "instrument_token": 111, "expiry": "2026-08-25",
        }

    def nearest_atm_option(self, symbol, option_type, underlying_price):
        return {
            "tradingsymbol": f"{symbol.removesuffix('.NS')}26AUG{option_type}", "lot_size": 500,
            "instrument_token": 222, "strike": underlying_price,
        }


class _FakeFuturesPaperAccountRepository:
    def __init__(self) -> None:
        self.opened = []

    async def get_cash_balance(self):
        return Decimal("400000")

    async def open_position(self, position) -> None:
        self.opened.append(position)

    async def close_position(self, symbol, exit_timestamp, futures_exit_price):
        matching = [p for p in self.opened if p.symbol == symbol]
        return matching[-1] if matching else None

    async def get_open_positions(self):
        return self.opened


def _eligible_trade_repo() -> FakeTradeRepository:
    repo = FakeTradeRepository()
    repo.opened = [
        Trade(
            symbol="RELIANCE.NS", side=SignalSide.BUY,
            entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC), entry_price=Decimal("100"),
            prediction_at_entry=1, is_early_signal_flip=False,
            exit_timestamp=datetime(2026, 1, 2, tzinfo=UTC), exit_price=Decimal("110"),
            pnl_percent=Decimal("10"), status="closed",
        )
        for _ in range(5)
    ]
    return repo


@pytest.mark.asyncio
async def test_open_futures_paper_noop_when_symbol_not_in_allowlist():
    account = _FakeFuturesPaperAccountRepository()

    note = await _open_futures_paper(
        "RELIANCE.NS", SignalSide.BUY, datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "1h", _eligible_trade_repo(), _FakeFuturesDerivativesChain(), account,
        frozenset(),  # empty allowlist -- RELIANCE.NS isn't on it
    )

    assert note is None
    assert account.opened == []


@pytest.mark.asyncio
async def test_open_futures_paper_noop_when_no_account_repository():
    note = await _open_futures_paper(
        "RELIANCE.NS", SignalSide.BUY, datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "1h", _eligible_trade_repo(), _FakeFuturesDerivativesChain(), None,
        frozenset({"RELIANCE.NS"}),
    )

    assert note is None


@pytest.mark.asyncio
async def test_open_futures_paper_opens_when_allowlisted_and_eligible():
    account = _FakeFuturesPaperAccountRepository()

    note = await _open_futures_paper(
        "RELIANCE.NS", SignalSide.BUY, datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "1h", _eligible_trade_repo(), _FakeFuturesDerivativesChain(), account,
        frozenset({"RELIANCE.NS"}),
    )

    assert note is not None
    assert len(account.opened) == 1


@pytest.mark.asyncio
async def test_open_futures_paper_swallows_exceptions():
    class _ExplodingChain:
        def nearest_future(self, symbol):
            raise RuntimeError("Kite API hiccup")

    note = await _open_futures_paper(
        "RELIANCE.NS", SignalSide.BUY, datetime(2026, 2, 1, tzinfo=UTC), Decimal("2900"),
        "1h", _eligible_trade_repo(), _ExplodingChain(), _FakeFuturesPaperAccountRepository(),
        frozenset({"RELIANCE.NS"}),
    )

    assert note is None  # never raises into the caller


@pytest.mark.asyncio
async def test_close_futures_paper_noop_when_symbol_not_in_allowlist():
    account = _FakeFuturesPaperAccountRepository()
    account.opened.append(
        FuturesPaperPosition(
            symbol="RELIANCE.NS", side="long", entry_timestamp=datetime(2026, 2, 1, tzinfo=UTC),
            futures_entry_price=Decimal("2900"), futures_tradingsymbol="RELIANCE26AUGFUT",
            hedge_tradingsymbol="RELIANCE26AUGPE", lot_size=500, margin_allocated=Decimal("10000"),
        )
    )

    await _close_futures_paper(
        "RELIANCE.NS", SignalSide.BUY, datetime(2026, 2, 5, tzinfo=UTC), Decimal("2950"),
        Decimal("2950"), account, frozenset(), FakeSignalRepository(), FakeNotifier(),
    )

    # close_position was never even called -- position stays "open" (no
    # status flip in this fake, since close_position wasn't invoked).
    assert account.opened[0].status == "open"


# --- _rank_and_open_futures_positions / _collect_and_open_ranked_positions ---
# 2026-08-14: these close the gap where live_pipeline.py (the actual
# production driver) never ranked either book at all -- see
# _collect_and_open_ranked_positions's own docstring.


def _eligible_multi_symbol_trade_repo(entries: list[tuple[str, SignalSide]]) -> FakeTradeRepository:
    """Like _eligible_trade_repo, but for arbitrary (symbol, side) pairs --
    each gets 5 winning closed trades, clearing the 55% eligibility bar."""
    repo = FakeTradeRepository()
    for symbol, side in entries:
        repo.opened.extend(
            Trade(
                symbol=symbol, side=side,
                entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC), entry_price=Decimal("100"),
                prediction_at_entry=1, is_early_signal_flip=False,
                exit_timestamp=datetime(2026, 1, 2, tzinfo=UTC), exit_price=Decimal("110"),
                pnl_percent=Decimal("10"), status="closed",
            )
            for _ in range(5)
        )
    return repo


@pytest.mark.asyncio
async def test_rank_and_open_futures_positions_opens_strongest_first():
    strong = RankedCandidate(  # prediction=8 -> highest score
        symbol="STRONG.NS", entry_timestamp=datetime(2026, 2, 1, tzinfo=UTC),
        entry_price=Decimal("2900"), prediction_at_entry=8, adx=0.0, regime_normalized=0.0,
        volatility_margin=0.0, direction=SignalSide.BUY,
    )
    weak = RankedCandidate(  # prediction=1 -> lowest score, opposite direction
        symbol="WEAK.NS", entry_timestamp=datetime(2026, 2, 1, tzinfo=UTC),
        entry_price=Decimal("2900"), prediction_at_entry=-1, adx=0.0, regime_normalized=0.0,
        volatility_margin=0.0, direction=SignalSide.SELL,
    )
    chain = _FakeFuturesDerivativesChain()
    account = _FakeFuturesPaperAccountRepository()
    trade_repository = _eligible_multi_symbol_trade_repo(
        [("STRONG.NS", SignalSide.BUY), ("WEAK.NS", SignalSide.SELL)]
    )

    notes = await _rank_and_open_futures_positions(
        [("WEAK.NS", weak), ("STRONG.NS", strong)],
        "1h", trade_repository, chain, account,
    )

    assert "STRONG.NS" in notes["STRONG.NS"] or "opened" in notes["STRONG.NS"]
    # Both actually open here (no capital constraint in this fake), but the
    # opening order itself is strongest-first -- confirm STRONG went first.
    assert account.opened[0].symbol == "STRONG.NS"
    assert account.opened[1].symbol == "WEAK.NS"


def _fast_predict_result(
    signal: str, prediction: int, *, volatility_margin: float = 0.0, regime_normalized: float = 0.0
) -> FastPredictResult:
    return FastPredictResult(
        signal=signal, prediction=prediction, end_long=False, end_short=False,
        is_early_signal_flip=False, signal_previous=0,
        queue_state=QueueState(), exit_state=ExitState(),
        adx=0.0, regime_normalized=regime_normalized, volatility_margin=volatility_margin,
    )


def _fake_candle(symbol: str) -> Candle:
    return Candle(
        symbol=symbol, timestamp=datetime(2026, 2, 1, tzinfo=UTC),
        open=Decimal("2900"), high=Decimal("2905"), low=Decimal("2895"), close=Decimal("2900"),
        volume=1000,
    )


@pytest.mark.asyncio
async def test_collect_and_open_ranked_positions_populates_both_books():
    trade_repository = _eligible_trade_repo()  # RELIANCE.NS has an eligible BUY track record
    paper_account_repository = FakePaperAccountRepository()
    futures_account_repository = _FakeFuturesPaperAccountRepository()
    evaluated_by_symbol = {
        "RELIANCE.NS": (_fast_predict_result("BUY", 5), _fake_candle("RELIANCE.NS")),
    }

    paper_notes, futures_notes, cash_notes = await _collect_and_open_ranked_positions(
        evaluated_by_symbol, _config(), trade_repository, paper_account_repository,
        asyncio.Lock(), _FakeFuturesDerivativesChain(), futures_account_repository,
        frozenset({"RELIANCE.NS"}), FakeNotifier(),
    )

    assert "opened" in paper_notes["RELIANCE.NS"]
    assert "opened" in futures_notes["RELIANCE.NS"]
    assert len(paper_account_repository.opened) == 1
    assert len(futures_account_repository.opened) == 1
    assert cash_notes == {}  # no order_executor/live_order_repository given -- cash lane untouched


@pytest.mark.asyncio
async def test_collect_and_open_ranked_positions_skips_futures_when_not_allowlisted():
    trade_repository = _eligible_trade_repo()
    paper_account_repository = FakePaperAccountRepository()
    futures_account_repository = _FakeFuturesPaperAccountRepository()
    evaluated_by_symbol = {
        "RELIANCE.NS": (_fast_predict_result("BUY", 5), _fake_candle("RELIANCE.NS")),
    }

    paper_notes, futures_notes, cash_notes = await _collect_and_open_ranked_positions(
        evaluated_by_symbol, _config(), trade_repository, paper_account_repository,
        asyncio.Lock(), _FakeFuturesDerivativesChain(), futures_account_repository,
        frozenset(), FakeNotifier(),  # empty allowlist
    )

    assert "opened" in paper_notes["RELIANCE.NS"]  # cash unaffected
    assert "RELIANCE.NS" not in futures_notes  # never even collected as a futures candidate
    assert futures_account_repository.opened == []
    assert cash_notes == {}


# --- _rank_and_open_cash_positions / cash lane of _collect_and_open_ranked_positions ---
# Real money, so held to the exact same "strongest-ranked-first, capacity
# genuinely scarce" bar as the paper/futures ranking tests above, plus its
# own extra gates (entry_quality_filter, conviction_filter) that paper
# doesn't have.


class _FakeCashOrderExecutor:
    """Always fills COMPLETE -- mirrors test_live_cash_execution.py's
    FakeOrderExecutor, trimmed to what execute_cash_entry actually calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self._order_counter = 0

    def place_cash_market_order(self, tradingsymbol, transaction_type, quantity, reference_price):
        self.calls.append((tradingsymbol, transaction_type, quantity))
        self._order_counter += 1
        return f"order-{self._order_counter}"

    def wait_for_fill(self, order_id, timeout_seconds, poll_interval=1.0):
        return {"status": "COMPLETE", "average_price": 1000.0, "status_message": None}


class _StatefulFakeLiveOrderRepository:
    """Unlike test_live_cash_execution.py's FakeLiveOrderRepository (fixed
    scripted lists), this one tracks legs as execute_cash_entry actually
    records them -- so a real capacity cap (max_positions) genuinely fills
    up across sequential calls within one test, proving
    _rank_and_open_cash_positions attempts ranked candidates one at a time
    and a later one can really lose a scarce slot, not just a scripted
    outcome."""

    def __init__(self) -> None:
        self.recorded: list[LiveOrderLeg] = []

    async def record_leg(self, leg: LiveOrderLeg) -> None:
        self.recorded.append(leg)

    def _open_symbols(self) -> set[str]:
        buys = {leg.symbol for leg in self.recorded if leg.transaction_type == "BUY"}
        sells = {leg.symbol for leg in self.recorded if leg.transaction_type == "SELL"}
        return buys - sells

    async def get_unclosed_cash_legs(self, symbol: str):
        if symbol not in self._open_symbols():
            return []
        return [leg for leg in self.recorded if leg.symbol == symbol]

    async def get_open_cash_legs(self, symbol: str):
        return await self.get_unclosed_cash_legs(symbol)

    async def get_all_open_cash_legs(self):
        open_symbols = self._open_symbols()
        return [leg for leg in self.recorded if leg.symbol in open_symbols]

    async def get_all_unclosed_cash_legs(self):
        return await self.get_all_open_cash_legs()

    async def get_legs_by_intent(self, intent_id: str):
        return [leg for leg in self.recorded if leg.intent_id == intent_id]


def _cash_config(*, max_positions: int = 8, symbols: frozenset[str] = frozenset()) -> AppConfig:
    return replace(
        _config(),
        live_cash_trading_enabled=True,
        live_cash_trading_symbols=symbols,
        live_cash_trading_notional=Decimal("5000"),
        live_cash_trading_max_positions=max_positions,
        live_cash_entry_cutoff_ist=None,  # unrelated to these tests, avoid wall-clock flakiness
    )


def _cash_state(
    *, max_positions: int = 8, symbols: frozenset[str] = frozenset()
) -> LiveCashToggleState:
    """Mirrors ``_cash_config``'s kwargs -- the functions below take this
    explicitly instead of reading enabled/symbols/notional/max_positions
    off ``AppConfig`` (see live_cash_execution.py's module docstring)."""
    return LiveCashToggleState(
        enabled=True, symbols=symbols, notional=Decimal("5000"), max_positions=max_positions
    )


def _cash_candidate(symbol: str, prediction: int) -> RankedCandidate:
    return RankedCandidate(
        symbol=symbol, entry_timestamp=datetime(2026, 2, 1, tzinfo=UTC),
        entry_price=Decimal("1000"), prediction_at_entry=prediction, adx=0.0,
        regime_normalized=0.0, volatility_margin=0.0, direction=SignalSide.BUY,
    )


@pytest.mark.asyncio
async def test_rank_and_open_cash_positions_opens_strongest_first_when_capacity_scarce():
    strong = _cash_candidate("STRONG.NS", prediction=8)  # highest score
    weak = _cash_candidate("WEAK.NS", prediction=1)  # lowest score
    config = _cash_config(max_positions=1, symbols=frozenset({"STRONG.NS", "WEAK.NS"}))
    cash_state = _cash_state(max_positions=1, symbols=frozenset({"STRONG.NS", "WEAK.NS"}))
    live_order_repository = _StatefulFakeLiveOrderRepository()
    notifier = FakeNotifier()

    notes = await _rank_and_open_cash_positions(
        [("WEAK.NS", weak), ("STRONG.NS", strong)],  # listed weak-first -- ranking must reorder
        config, cash_state, _FakeCashOrderExecutor(), live_order_repository, notifier, None, None,
    )

    assert "opened" in notes["STRONG.NS"]
    assert "SKIPPED" in notes["WEAK.NS"]
    assert [leg.symbol for leg in live_order_repository.recorded] == ["STRONG.NS"]
    # Losing a real slot to a stronger candidate is the specific costly
    # case _notify_missed_cash_entry exists for -- confirm it actually fired.
    assert any("MISSED BUY SIGNAL" in text and "WEAK.NS" in text for text in notifier.texts)


@pytest.mark.asyncio
async def test_rank_and_open_cash_positions_rejects_below_score_floor(monkeypatch):
    monkeypatch.setattr(signal_pipeline_module, "MIN_SCORE", 10_000.0)  # nothing can clear this
    candidate = _cash_candidate("RELIANCE.NS", prediction=5)
    config = _cash_config(max_positions=8, symbols=frozenset({"RELIANCE.NS"}))
    cash_state = _cash_state(max_positions=8, symbols=frozenset({"RELIANCE.NS"}))
    live_order_repository = _StatefulFakeLiveOrderRepository()

    notes = await _rank_and_open_cash_positions(
        [("RELIANCE.NS", candidate)], config, cash_state, _FakeCashOrderExecutor(),
        live_order_repository, FakeNotifier(), None, None,
    )

    assert "REJECTED" in notes["RELIANCE.NS"]
    assert live_order_repository.recorded == []


def _strong_conviction_candle(symbol: str) -> Candle:
    """Closes at its own high -- CLV 1.0, clears the 0.7 conviction floor."""
    return Candle(
        symbol=symbol, timestamp=datetime(2026, 2, 1, tzinfo=UTC),
        open=Decimal("990"), high=Decimal("1000"), low=Decimal("990"), close=Decimal("1000"),
        volume=1000,
    )


def _weak_conviction_candle(symbol: str) -> Candle:
    """Closes at its own low -- CLV 0.0, fails the 0.7 conviction floor."""
    return Candle(
        symbol=symbol, timestamp=datetime(2026, 2, 1, tzinfo=UTC),
        open=Decimal("1000"), high=Decimal("1000"), low=Decimal("990"), close=Decimal("990"),
        volume=1000,
    )


@pytest.mark.asyncio
async def test_collect_and_open_ranked_positions_rejects_cash_on_weak_conviction_candle():
    # Same symbol, same eligible track record, same quality-clearing
    # prediction/regime/volatility -- ONLY the entry candle's own shape
    # differs. Cash must reject it; paper (not gated by conviction) must
    # not even notice.
    trade_repository = _eligible_trade_repo()  # RELIANCE.NS
    paper_account_repository = FakePaperAccountRepository()
    futures_account_repository = _FakeFuturesPaperAccountRepository()
    live_order_repository = _StatefulFakeLiveOrderRepository()
    evaluated_by_symbol = {
        "RELIANCE.NS": (
            # volatility_margin/regime_normalized comfortably clear
            # entry_quality_filter's floor -- isolates conviction as the
            # only variable this test is about.
            _fast_predict_result("BUY", 5, volatility_margin=10.0, regime_normalized=2.0),
            _weak_conviction_candle("RELIANCE.NS"),
        ),
    }
    config = _cash_config(max_positions=8, symbols=frozenset({"RELIANCE.NS"}))
    cash_state = _cash_state(max_positions=8, symbols=frozenset({"RELIANCE.NS"}))

    paper_notes, _, cash_notes = await _collect_and_open_ranked_positions(
        evaluated_by_symbol, config, trade_repository, paper_account_repository,
        asyncio.Lock(), _FakeFuturesDerivativesChain(), futures_account_repository, frozenset(),
        FakeNotifier(), _FakeCashOrderExecutor(), live_order_repository, None, None, cash_state,
    )

    assert "opened" in paper_notes["RELIANCE.NS"]  # paper unaffected by conviction
    assert "REJECTED (conviction filter" in cash_notes["RELIANCE.NS"]
    assert live_order_repository.recorded == []  # never reached execute_cash_entry


@pytest.mark.asyncio
async def test_collect_and_open_ranked_positions_opens_cash_on_strong_conviction_candle():
    trade_repository = _eligible_trade_repo()  # RELIANCE.NS
    paper_account_repository = FakePaperAccountRepository()
    futures_account_repository = _FakeFuturesPaperAccountRepository()
    live_order_repository = _StatefulFakeLiveOrderRepository()
    evaluated_by_symbol = {
        "RELIANCE.NS": (
            _fast_predict_result("BUY", 5, volatility_margin=10.0, regime_normalized=2.0),
            _strong_conviction_candle("RELIANCE.NS"),
        ),
    }
    config = _cash_config(max_positions=8, symbols=frozenset({"RELIANCE.NS"}))
    cash_state = _cash_state(max_positions=8, symbols=frozenset({"RELIANCE.NS"}))

    _, _, cash_notes = await _collect_and_open_ranked_positions(
        evaluated_by_symbol, config, trade_repository, paper_account_repository,
        asyncio.Lock(), _FakeFuturesDerivativesChain(), futures_account_repository, frozenset(),
        FakeNotifier(), _FakeCashOrderExecutor(), live_order_repository, None, None, cash_state,
    )

    assert "opened" in cash_notes["RELIANCE.NS"]
    assert live_order_repository.recorded[0].symbol == "RELIANCE.NS"


class _FakeEntryDecisionRepository:
    """In-memory stand-in for TursoEntryDecisionRepository -- records
    every EntryDecisionRecord passed to it, nothing else."""

    def __init__(self) -> None:
        self.recorded: list = []

    async def record(self, decision) -> None:
        self.recorded.append(decision)


@pytest.mark.asyncio
async def test_collect_and_open_ranked_positions_persists_a_rejected_decision():
    trade_repository = _eligible_trade_repo()  # RELIANCE.NS
    paper_account_repository = FakePaperAccountRepository()
    futures_account_repository = _FakeFuturesPaperAccountRepository()
    live_order_repository = _StatefulFakeLiveOrderRepository()
    entry_decision_repository = _FakeEntryDecisionRepository()
    evaluated_by_symbol = {
        "RELIANCE.NS": (
            _fast_predict_result("BUY", 5, volatility_margin=10.0, regime_normalized=2.0),
            _weak_conviction_candle("RELIANCE.NS"),
        ),
    }
    config = _cash_config(max_positions=8, symbols=frozenset({"RELIANCE.NS"}))
    cash_state = _cash_state(max_positions=8, symbols=frozenset({"RELIANCE.NS"}))

    await _collect_and_open_ranked_positions(
        evaluated_by_symbol, config, trade_repository, paper_account_repository,
        asyncio.Lock(), _FakeFuturesDerivativesChain(), futures_account_repository, frozenset(),
        FakeNotifier(), _FakeCashOrderExecutor(), live_order_repository, None, None, cash_state,
        entry_decision_repository,
    )

    assert len(entry_decision_repository.recorded) == 1
    decision = entry_decision_repository.recorded[0]
    assert decision.symbol == "RELIANCE.NS"
    assert decision.track_record_passed is True
    assert decision.quality_passed is True
    assert decision.conviction_passed is False
    assert decision.final_decision == "rejected"
    assert decision.blocked_reason == "conviction filter -- weak entry candle"


@pytest.mark.asyncio
async def test_collect_and_open_ranked_positions_persists_an_opened_decision_with_ranking():
    trade_repository = _eligible_trade_repo()  # RELIANCE.NS
    paper_account_repository = FakePaperAccountRepository()
    futures_account_repository = _FakeFuturesPaperAccountRepository()
    live_order_repository = _StatefulFakeLiveOrderRepository()
    entry_decision_repository = _FakeEntryDecisionRepository()
    evaluated_by_symbol = {
        "RELIANCE.NS": (
            _fast_predict_result("BUY", 5, volatility_margin=10.0, regime_normalized=2.0),
            _strong_conviction_candle("RELIANCE.NS"),
        ),
    }
    config = _cash_config(max_positions=8, symbols=frozenset({"RELIANCE.NS"}))
    cash_state = _cash_state(max_positions=8, symbols=frozenset({"RELIANCE.NS"}))

    await _collect_and_open_ranked_positions(
        evaluated_by_symbol, config, trade_repository, paper_account_repository,
        asyncio.Lock(), _FakeFuturesDerivativesChain(), futures_account_repository, frozenset(),
        FakeNotifier(), _FakeCashOrderExecutor(), live_order_repository, None, None, cash_state,
        entry_decision_repository,
    )

    assert len(entry_decision_repository.recorded) == 1
    decision = entry_decision_repository.recorded[0]
    assert decision.final_decision == "opened"
    assert decision.blocked_reason is None
    assert decision.ranking_passed is True
    assert decision.ranking_score is not None


class _FakeLiveOrderRepositoryForMissedNotify:
    """Only what ``_notify_missed_cash_entry`` reads -- the current real
    open-position count, to tell a full-capacity miss from any other."""

    def __init__(self, open_count: int) -> None:
        self._open_count = open_count

    async def get_all_unclosed_cash_legs(self):
        return [object()] * self._open_count


@pytest.mark.asyncio
async def test_missed_cash_entry_notification_names_a_full_capacity_miss():
    notifier = FakeNotifier()
    repo = _FakeLiveOrderRepositoryForMissedNotify(open_count=8)  # matches the 8-slot cap
    await _notify_missed_cash_entry(
        "RELIANCE.NS", Decimal("2500"), _cash_state(max_positions=8), repo, notifier
    )
    assert len(notifier.texts) == 1
    assert "MISSED BUY SIGNAL" in notifier.texts[0]
    assert "RELIANCE.NS" in notifier.texts[0]
    assert "8 real slots are already full" in notifier.texts[0]


@pytest.mark.asyncio
async def test_missed_cash_entry_notification_falls_back_when_slots_are_free():
    notifier = FakeNotifier()
    repo = _FakeLiveOrderRepositoryForMissedNotify(open_count=3)  # capacity wasn't the reason
    await _notify_missed_cash_entry(
        "RELIANCE.NS", Decimal("2500"), _cash_state(max_positions=8), repo, notifier
    )
    assert "entry cutoff" in notifier.texts[0]
