from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
import pytest

from trading_scanner.application.fast_predict import ExitState, FastPredictResult, QueueState
from trading_scanner.application.signal_pipeline import (
    _BACKFILL_WINDOW_DAYS,
    _RECENT_WINDOW_DAYS,
    run_signal_pipeline,
)
from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import Candle, SignalSide, Trade
from trading_scanner.domain.ports import EngineState
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


class FakeNotifier:
    def __init__(self) -> None:
        self.sent = []

    async def send_signal(self, signal) -> None:
        self.sent.append(signal)


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
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        notifier,
    )

    assert notifier.sent == []
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
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        notifier,
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
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        notifier,
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
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        notifier,
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
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["BROKEN.NS", "AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        notifier,
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
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        notifier,
    )
    state_after_first_run = await engine_state_repository.get_state("AARTIIND.NS", "1h")

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        notifier,
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
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        notifier,
    )

    assert len(trade_repository.opened) == 1
    trade = trade_repository.opened[0]
    assert trade.symbol == "AARTIIND.NS"
    assert trade.side == SignalSide.BUY
    assert trade.prediction_at_entry == 6
    assert trade.is_early_signal_flip is True
    assert trade.entry_price == Decimal("100")  # (high+low+2*open)/4 = (101+99+200)/4
    assert notifier.sent  # entry still notifies as before


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
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        notifier,
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
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        notifier,
    )

    assert trade_repository.abandoned == [("AARTIIND.NS", "1h", SignalSide.BUY)]
    assert trade_repository.closed == []  # abandoned, not closed -- never scored
    assert len(trade_repository.opened) == 1
    assert trade_repository.opened[0].side == SignalSide.SELL


@pytest.mark.asyncio
async def test_end_long_sends_an_exit_notification_with_pnl(monkeypatch) -> None:
    """A dynamic exit must notify too (not just silently close the trade),
    showing the realized pnl_percent, using a fingerprint distinct from any
    entry notification at the same symbol/side/timestamp."""
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
        notifier,
    )

    assert len(notifier.sent) == 1
    exit_signal = notifier.sent[0]
    assert exit_signal.strategy == "lorentzian-exit"
    assert exit_signal.side == SignalSide.BUY
    assert "pnl=25.00%" in exit_signal.rationale  # (100-80)/80*100, market_price=100


@pytest.mark.asyncio
async def test_win_rate_summary_is_attached_to_notification(monkeypatch) -> None:
    """Prior closed trades for this symbol must be summarized in the
    notification's rationale, so a signal is never sent without context on
    how this symbol has actually performed historically."""
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
    notifier = FakeNotifier()

    await run_signal_pipeline(
        _config(),
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        notifier,
    )

    assert notifier.sent
    assert "win_rate=66.7%(2W/1L)" in notifier.sent[0].rationale


@pytest.mark.asyncio
async def test_index_context_is_attached_to_notification_but_never_suppresses_it(
    monkeypatch,
) -> None:
    """When index_symbol is configured, its current state is appended to the
    notified signal's rationale purely for the user's own judgment -- it must
    never block a stock signal from notifying, even when the two disagree."""
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
    )
    candle_repository = FakeCandleRepository()
    signal_repository = FakeSignalRepository()
    engine_state_repository = FakeEngineStateRepository()
    trade_repository = FakeTradeRepository()
    notifier = FakeNotifier()

    await run_signal_pipeline(
        config,
        ["AARTIIND.NS"],
        candle_repository,
        signal_repository,
        engine_state_repository,
        trade_repository,
        notifier,
    )

    assert len(notifier.sent) == 1  # stock BUY still notifies despite index disagreeing
    assert "index(^NSEI)=SELL" in notifier.sent[0].rationale
    assert "pred=-4" in notifier.sent[0].rationale
    assert "early_flip=True" in notifier.sent[0].rationale
