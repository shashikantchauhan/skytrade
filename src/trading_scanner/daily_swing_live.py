"""Always-on 30-minute live runner for the frozen liquidity-sweep strategy."""

from __future__ import annotations

import asyncio
import logging
import queue
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pandas as pd
from kiteconnect import KiteConnect, KiteTicker
from kiteconnect.exceptions import TokenException as KiteTokenException

from trading_scanner.application.daily_swing import (
    EntryCandidate,
    SwingSetup,
    build_live_context_from_aggregates,
    confirm_entry,
    trailed_stop,
)
from trading_scanner.application.live_execution import (
    defined_max_loss,
    execute_basket_entry,
    execute_basket_exit,
)
from trading_scanner.application.symbols import SymbolLoader, SymbolLoadError
from trading_scanner.config.settings import AppConfig, load_config
from trading_scanner.domain.models import Candle, DailySwingPosition, SignalSide
from trading_scanner.infrastructure.db import (
    DailySwingRepository,
    TursoCandleRepository,
    TursoKiteSessionRepository,
    TursoLiveOrderRepository,
    create_turso_client,
)
from trading_scanner.infrastructure.kite import (
    KiteDerivativesChain,
    KiteInstrumentMap,
    KiteOrderExecutor,
    KiteProvider,
)
from trading_scanner.infrastructure.kite_ticker import (
    IST,
    CandleAggregator,
    bucket_start,
    is_market_hours,
)
from trading_scanner.infrastructure.telegram import LoggingNotifier, TelegramNotifier

logger = logging.getLogger(__name__)
_INTERVAL = "30m"
_BUCKET_MINUTES = 30
_HISTORY_DAYS = 120
_BACKFILL_DAYS = 10
_API_DELAY_SECONDS = 0.36


def _notifier(config: AppConfig):
    if config.telegram_bot_token and config.telegram_chat_id:
        return TelegramNotifier(config.telegram_bot_token, config.telegram_chat_id, "Daily Swing")
    return LoggingNotifier()


def _side(value: int) -> SignalSide:
    return SignalSide.BUY if value == 1 else SignalSide.SELL


class DailySwingLive:
    def __init__(self, config: AppConfig, universe: list[str], signal_symbols: frozenset[str]):
        self.config = config
        self.universe = universe
        self.signal_symbols = signal_symbols
        self.client = create_turso_client(config.turso_database_url, config.turso_auth_token)
        self.candles = TursoCandleRepository(self.client)
        self.positions = DailySwingRepository(self.client)
        self.orders = TursoLiveOrderRepository(self.client)
        self.sessions = TursoKiteSessionRepository(self.client)
        self.notifier = _notifier(config)
        self.queue: queue.Queue = queue.Queue()
        self.aggregators: dict[int, CandleAggregator] = {}
        self.token_to_symbol: dict[int, str] = {}
        self.current_bucket: datetime | None = None
        self.ticker: KiteTicker | None = None
        self.access_token: str | None = None
        self.setups: dict[str, SwingSetup] = {}
        self.slot_volume: dict[tuple[str, int], float] = {}
        self.context_session = None
        self.context_for_date = None
        self.session_opens: dict[str, Decimal] = {}
        self.session_open_date = None
        self.active: DailySwingPosition | None = None
        self.exit_lock = asyncio.Lock()
        self.exit_failed = False
        self.bucket_lock = asyncio.Lock()
        self.last_tick_at = datetime.now(UTC)

    async def setup(self) -> None:
        for repository in (self.candles, self.positions, self.orders, self.sessions):
            await repository.ensure_schema()
        active = list(await self.positions.get_active())
        if len(active) > 1:
            raise RuntimeError(f"More than one active daily-swing position: {len(active)}")
        self.active = active[0] if active else None

    async def token(self) -> str | None:
        row = await self.sessions.get_token()
        return row[0] if row else None

    async def backfill(self, kite: KiteConnect) -> None:
        latest = await self.candles.get_latest_interval_timestamp(_INTERVAL)
        today = datetime.now(UTC).astimezone(IST).date()
        latest_date = latest.astimezone(IST).date() if latest is not None else None
        complete_symbols = (
            await self.candles.get_complete_symbol_count(_INTERVAL, latest_date.isoformat())
            if latest_date is not None
            else 0
        )
        current_enough = latest_date == today or (
            latest_date is not None
            and latest_date >= today - timedelta(days=3)
            and complete_symbols >= 450
        )
        if latest is not None and current_enough:
            logger.info("30m history is current through %s; full backfill skipped.", latest)
            return
        provider = KiteProvider(kite, KiteInstrumentMap(kite))
        logger.info("Backfilling %d symbols (%d days).", len(self.universe), _BACKFILL_DAYS)
        completed_before = bucket_start(datetime.now(UTC), _BUCKET_MINUTES)
        for index, symbol in enumerate(self.universe, start=1):
            try:
                frame = await asyncio.to_thread(
                    provider.get_recent_history, symbol, _INTERVAL, _BACKFILL_DAYS
                )
                rows = []
                for timestamp, row in frame.iterrows():
                    when = pd.Timestamp(timestamp).to_pydatetime().astimezone(UTC)
                    if when >= completed_before:
                        continue
                    rows.append(
                        Candle(
                            symbol=symbol,
                            timestamp=when,
                            open=Decimal(str(row.Open)),
                            high=Decimal(str(row.High)),
                            low=Decimal(str(row.Low)),
                            close=Decimal(str(row.Close)),
                            volume=int(row.Volume),
                        )
                    )
                await self.candles.upsert_candles(symbol, _INTERVAL, rows)
            except Exception:
                logger.warning("Backfill failed for %s", symbol, exc_info=True)
            if index % 50 == 0:
                logger.info("Backfill progress: %d/%d", index, len(self.universe))
            await asyncio.sleep(_API_DELAY_SECONDS)
        logger.info("Backfill complete.")

    async def rebuild_context(self, now: datetime) -> None:
        today_ist = now.astimezone(IST).date()
        since = now - timedelta(days=_HISTORY_DAYS)
        before = datetime.combine(today_ist, datetime.min.time(), tzinfo=IST).astimezone(UTC)
        daily_rows = await self.candles.get_daily_aggregates(_INTERVAL, since, before)
        volume_rows = await self.candles.get_volume_rows_since(
            _INTERVAL, now - timedelta(days=35), before
        )
        daily = pd.DataFrame(
            daily_rows,
            columns=["symbol", "date", "open", "high", "low", "close", "volume", "bars"],
        )
        volume = pd.DataFrame(volume_rows, columns=["symbol", "timestamp", "volume"])
        if daily.empty or volume.empty:
            raise RuntimeError("No 30-minute history available for daily-swing context.")
        self.setups, self.slot_volume, self.context_session = await asyncio.to_thread(
            build_live_context_from_aggregates, daily, volume, self.signal_symbols
        )
        self.context_for_date = today_ist
        logger.info(
            "Daily-swing context from %s: %d setups, %d volume baselines.",
            self.context_session,
            len(self.setups),
            len(self.slot_volume),
        )

    async def resolve_tokens(self, kite: KiteConnect) -> None:
        resolver = KiteInstrumentMap(kite)
        self.aggregators.clear()
        self.token_to_symbol.clear()
        for symbol in self.universe:
            try:
                token = await asyncio.to_thread(resolver.resolve, symbol)
            except Exception:
                logger.warning("No Kite token for %s; skipping.", symbol)
                continue
            self.token_to_symbol[token] = symbol
            self.aggregators[token] = CandleAggregator()
        logger.info(
            "Resolved %d/%d daily-swing instruments.", len(self.aggregators), len(self.universe)
        )

    def on_ticks(self, ws, ticks) -> None:  # noqa: ANN001
        self.last_tick_at = datetime.now(UTC)
        self.queue.put(ticks)

    def on_connect(self, ws, response) -> None:  # noqa: ANN001
        tokens = list(self.token_to_symbol)
        ws.subscribe(tokens)
        # Quote packets include LTP and cumulative volume, which are all the
        # candle builder needs. Full mode also streams five-level market depth
        # for every symbol and needlessly increases queue/memory pressure.
        ws.set_mode(ws.MODE_QUOTE, tokens)
        self.last_tick_at = datetime.now(UTC)
        logger.info("Daily-swing ticker connected to %d instruments.", len(tokens))

    def connect(self, token: str) -> None:
        ticker = KiteTicker(api_key=self.config.kite_api_key, access_token=token)
        ticker.on_ticks = self.on_ticks
        ticker.on_connect = self.on_connect
        ticker.on_error = lambda ws, code, reason: logger.error("Ticker error: %s %s", code, reason)
        ticker.on_close = lambda ws, code, reason: logger.warning(
            "Ticker closed: %s %s", code, reason
        )
        ticker.connect(threaded=True)
        self.ticker = ticker
        self.access_token = token

    def disconnect(self) -> None:
        if self.ticker:
            try:
                self.ticker.close()
            except Exception:
                logger.warning("Ticker close failed.", exc_info=True)
        self.ticker = None

    async def drain(self, kite: KiteConnect) -> None:
        while True:
            ticks = await asyncio.to_thread(self.queue.get)
            for tick in ticks:
                aggregator = self.aggregators.get(tick.get("instrument_token"))
                price = tick.get("last_price")
                if aggregator is None or price is None:
                    continue
                value = Decimal(str(price))
                now = datetime.now(UTC)
                async with self.bucket_lock:
                    await self.roll_to(bucket_start(now, _BUCKET_MINUTES), kite)
                    aggregator.add_tick(now, value, int(tick.get("volume_traded", 0) or 0))
                symbol = self.token_to_symbol[tick["instrument_token"]]
                await self.check_live_exit(symbol, value, kite)

    async def check_live_exit(self, symbol: str, price: Decimal, kite: KiteConnect) -> None:
        position = self.active
        if position is None or position.status != "open" or position.symbol != symbol:
            return
        if self.exit_failed:
            return
        if position.side == 1:
            reason = (
                "stop"
                if price <= position.active_stop
                else "target"
                if price >= position.target
                else None
            )
        else:
            reason = (
                "stop"
                if price >= position.active_stop
                else "target"
                if price <= position.target
                else None
            )
        if reason:
            await self.exit_position(price, reason, kite)

    async def exit_position(self, price: Decimal, reason: str, kite: KiteConnect) -> None:
        async with self.exit_lock:
            position = self.active
            if position is None or position.status != "open":
                return
            basket = await execute_basket_exit(
                position.symbol,
                self.config,
                KiteOrderExecutor(kite),
                self.orders,
                self.notifier,
            )
            remaining = await self.orders.get_open_primary_legs(position.symbol)
            if basket is None or remaining:
                logger.error("Exit incomplete for %s; position remains active.", position.symbol)
                self.exit_failed = True
                return
            await self.positions.close(position.symbol, datetime.now(UTC), price, reason)
            await self.notifier.send_text(
                f"Daily swing {position.symbol} closed: {reason} at {price}"
            )
            self.active = None

    async def boundary(self, kite: KiteConnect) -> None:
        while True:
            await asyncio.sleep(2)
            async with self.bucket_lock:
                await self.roll_to(bucket_start(datetime.now(UTC), _BUCKET_MINUTES), kite)

    async def roll_to(self, new_bucket: datetime, kite: KiteConnect) -> None:
        if self.current_bucket is None:
            self.current_bucket = new_bucket
            for aggregator in self.aggregators.values():
                aggregator.start(new_bucket)
            return
        if new_bucket <= self.current_bucket:
            return
        closed = self.current_bucket
        candles = self.finalize(closed)
        self.current_bucket = new_bucket
        for aggregator in self.aggregators.values():
            aggregator.start(new_bucket)
        if candles:
            await self.process_bucket(closed, candles, kite)

    def finalize(self, bucket: datetime) -> dict[str, Candle]:
        result = {}
        for token, aggregator in self.aggregators.items():
            values = aggregator.finalize()
            if values is None:
                continue
            open_, high, low, close, volume = values
            symbol = self.token_to_symbol[token]
            result[symbol] = Candle(symbol, bucket, open_, high, low, close, volume)
        return result

    async def process_bucket(
        self, bucket: datetime, candles: dict[str, Candle], kite: KiteConnect
    ) -> None:
        await self.candles.upsert_many(_INTERVAL, list(candles.values()))
        logger.info("30m bucket %s stored for %d symbols.", bucket, len(candles))
        if self.context_for_date != bucket.astimezone(IST).date():
            await self.rebuild_context(bucket)

        await self.update_position_on_close(bucket, candles, kite)
        market = candles.get("^NSEI")
        if market is None:
            logger.warning("No Nifty candle in bucket; entries suppressed.")
            return
        open_dt = datetime.combine(bucket.astimezone(IST).date(), datetime.min.time(), tzinfo=IST)
        open_dt = open_dt.replace(hour=9, minute=15)
        slot = int((bucket.astimezone(IST) - open_dt).total_seconds() // 1800)
        if slot not in range(4):
            return
        session_date = bucket.astimezone(IST).date()
        if self.session_open_date != session_date:
            self.session_opens = await self.candles.get_bucket_opens(
                _INTERVAL, open_dt.astimezone(UTC)
            )
            self.session_open_date = session_date
        market_row = _candle_dict(market)
        candidates = []
        for symbol, setup in self.setups.items():
            candle = candles.get(symbol)
            if candle is None or await self.positions.has_attempt(
                symbol, setup.setup_date.isoformat()
            ):
                continue
            candidate = confirm_entry(
                setup,
                _candle_dict(candle),
                market_row,
                slot,
                self.slot_volume.get((symbol, slot)),
                float(self.session_opens[symbol]) if symbol in self.session_opens else None,
            )
            if candidate:
                candidates.append(candidate)
        if not candidates:
            return
        candidates.sort(key=lambda item: (-item.setup.score, item.setup.symbol))
        if self.active is not None or await self.orders.get_all_unclosed_primary_legs():
            for candidate in candidates:
                await self.positions.record_attempt(
                    candidate.setup.symbol,
                    candidate.setup.setup_date.isoformat(),
                    candidate.timestamp,
                    "blocked_global_position",
                )
            return
        broker_positions = await asyncio.to_thread(KiteOrderExecutor(kite).open_nfo_positions)
        if broker_positions:
            for candidate in candidates:
                await self.positions.record_attempt(
                    candidate.setup.symbol,
                    candidate.setup.setup_date.isoformat(),
                    candidate.timestamp,
                    "blocked_unknown_broker_nfo_position",
                )
            await self.notifier.send_text(
                "Daily swing entry blocked: Kite has an existing NFO position. "
                "Verify it before this strategy opens a new basket."
            )
            return
        await self.enter(candidates[0], kite)
        for candidate in candidates[1:]:
            await self.positions.record_attempt(
                candidate.setup.symbol,
                candidate.setup.setup_date.isoformat(),
                candidate.timestamp,
                "lower_rank_same_bucket",
            )

    async def enter(self, candidate: EntryCandidate, kite: KiteConnect) -> None:
        setup = candidate.setup
        attempted = await self.positions.record_attempt(
            setup.symbol, setup.setup_date.isoformat(), candidate.timestamp, "evaluating"
        )
        if not attempted:
            return
        chain = KiteDerivativesChain(kite)
        option_type = "PE" if setup.side == 1 else "CE"
        minimum_expiry = date.today() + timedelta(days=16)
        future = await asyncio.to_thread(chain.future_for_horizon, setup.symbol, minimum_expiry)
        option = (
            await asyncio.to_thread(
                chain.nearest_atm_option,
                setup.symbol,
                option_type,
                setup.stop,
                future["expiry"],
            )
            if future is not None
            else None
        )
        if option is None or future is None:
            logger.info("No current derivative contracts for %s.", setup.symbol)
            return
        option_price = await asyncio.to_thread(chain.ltp, f"NFO:{option['tradingsymbol']}")
        future_price = await asyncio.to_thread(chain.ltp, f"NFO:{future['tradingsymbol']}")
        if option_price is None or future_price is None:
            logger.warning("No live derivative quote for %s.", setup.symbol)
            return
        quantity = int(future["lot_size"]) * self.config.live_trading_max_lots
        risk = defined_max_loss(
            _side(setup.side),
            Decimal(str(future_price)),
            Decimal(str(option["strike"])),
            Decimal(str(option_price)),
            quantity,
        )
        if risk > self.config.daily_swing_max_defined_risk:
            await self.notifier.send_text(
                f"Daily swing {setup.symbol} rejected: defined loss ₹{risk:,.0f} "
                f"> ₹{self.config.daily_swing_max_defined_risk:,.0f}"
            )
            return
        position = DailySwingPosition(
            symbol=setup.symbol,
            side=setup.side,
            setup_date=setup.setup_date.isoformat(),
            entry_timestamp=candidate.timestamp,
            entry_price=Decimal(str(candidate.entry)),
            initial_stop=Decimal(str(setup.stop)),
            active_stop=Decimal(str(setup.stop)),
            target=Decimal(str(candidate.target)),
            atr=Decimal(str(setup.atr)),
            risk=Decimal(str(candidate.risk)),
            best_close=Decimal(str(candidate.entry)),
            basket_id=None,
        )
        await self.positions.create_entering(position)
        basket = await execute_basket_entry(
            setup.symbol,
            _side(setup.side),
            option_type,
            Decimal(str(setup.stop)),
            self.config,
            chain,
            KiteOrderExecutor(kite),
            self.orders,
            self.notifier,
            contract_expiry=future["expiry"],
        )
        primary = await self.orders.get_open_primary_legs(setup.symbol)
        if basket is None or not primary:
            await self.positions.reject_entering(
                setup.symbol, setup.setup_date.isoformat(), "basket_not_open"
            )
            return
        await self.positions.mark_open(setup.symbol, setup.setup_date.isoformat(), basket)
        self.active = replace(position, basket_id=basket, status="open")
        legs = await self.orders.get_legs(basket)
        hedge_fill = next(
            (
                leg.average_price
                for leg in legs
                if leg.purpose == "hedge" and leg.status == "COMPLETE"
            ),
            None,
        )
        primary_fill = next(
            (
                leg.average_price
                for leg in legs
                if leg.purpose == "primary" and leg.status == "COMPLETE"
            ),
            None,
        )
        if hedge_fill is not None and primary_fill is not None:
            actual_risk = defined_max_loss(
                _side(setup.side),
                primary_fill,
                Decimal(str(option["strike"])),
                hedge_fill,
                quantity,
            )
            if actual_risk > self.config.daily_swing_max_defined_risk:
                await self.notifier.send_text(
                    f"Daily swing {setup.symbol}: filled risk ₹{actual_risk:,.0f} exceeds cap; "
                    "closing immediately."
                )
                await self.exit_position(Decimal(str(candidate.entry)), "risk_cap_after_fill", kite)
                return
        await self.notifier.send_text(
            f"Daily swing {setup.symbol} opened {'LONG' if setup.side == 1 else 'SHORT'}; "
            f"spot close {candidate.entry:.2f}, stop {setup.stop:.2f}, "
            f"target {candidate.target:.2f}, "
            f"defined risk estimate ₹{risk:,.0f}."
        )

    async def update_position_on_close(
        self, bucket: datetime, candles: dict[str, Candle], kite: KiteConnect
    ) -> None:
        position = self.active
        if position is None or position.status != "open":
            return
        candle = candles.get(position.symbol)
        if candle is None or candle.timestamp <= position.entry_timestamp:
            return
        close = candle.close
        best = (
            max(position.best_close, close)
            if position.side == 1
            else min(position.best_close, close)
        )
        stop = Decimal(
            str(
                trailed_stop(
                    position.side,
                    float(position.entry_price),
                    float(position.initial_stop),
                    float(position.risk),
                    float(position.atr),
                    float(best),
                )
            )
        )
        await self.positions.update_trail(position.symbol, stop, best)
        self.active = replace(position, active_stop=stop, best_close=best)
        if bucket.astimezone(IST).hour == 15 and bucket.astimezone(IST).minute == 15:
            sessions = await self.candles.count_sessions(
                position.symbol, _INTERVAL, position.entry_timestamp, bucket
            )
            if sessions >= 10:
                await self.exit_position(close, "time", kite)

    async def heartbeat(self) -> None:
        while True:
            await asyncio.sleep(60)
            if is_market_hours(datetime.now(UTC)):
                logger.info(
                    "Daily-swing heartbeat: last tick %.0fs ago; active=%s",
                    (datetime.now(UTC) - self.last_tick_at).total_seconds(),
                    self.active.symbol if self.active else "none",
                )

    async def market_close(self, kite: KiteConnect) -> None:
        while is_market_hours(datetime.now(UTC)):
            await asyncio.sleep(15)
        # NSE's final 15:15 candle is a short 15-minute bucket. At 15:30,
        # normal 30-minute bucketing still maps to 15:15, so no rollover
        # occurs naturally; flush it explicitly after the last ticks land.
        await asyncio.sleep(2)
        async with self.bucket_lock:
            if self.current_bucket is not None:
                closed = self.current_bucket
                candles = self.finalize(closed)
                self.current_bucket = None
                if candles:
                    await self.process_bucket(closed, candles, kite)

    async def token_changed(self, current: str) -> None:
        while True:
            await asyncio.sleep(60)
            latest = await self.token()
            if latest != current:
                return

    async def run_session(self, kite: KiteConnect, token: str) -> None:
        tasks = [
            asyncio.create_task(self.drain(kite)),
            asyncio.create_task(self.boundary(kite)),
            asyncio.create_task(self.heartbeat()),
            asyncio.create_task(self.market_close(kite)),
            asyncio.create_task(self.token_changed(token)),
        ]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run(self) -> None:
        await self.setup()
        initialized_token = None
        try:
            while True:
                token = await self.token()
                if not token:
                    await asyncio.sleep(60)
                    continue
                kite = KiteConnect(api_key=self.config.kite_api_key)
                kite.set_access_token(token)
                try:
                    await asyncio.to_thread(kite.profile)
                except KiteTokenException:
                    logger.warning("Daily swing waiting for a fresh Kite login.")
                    await asyncio.sleep(60)
                    continue
                if self.active is not None and self.active.status == "entering":
                    ledger_open = await self.orders.get_open_primary_legs(self.active.symbol)
                    broker_open = await asyncio.to_thread(
                        KiteOrderExecutor(kite).open_nfo_positions
                    )
                    if ledger_open:
                        basket_id = ledger_open[0].basket_id
                        await self.positions.mark_open(
                            self.active.symbol, self.active.setup_date, basket_id
                        )
                        self.active = replace(self.active, basket_id=basket_id, status="open")
                    elif broker_open:
                        await self.notifier.send_text(
                            f"Daily swing reconciliation required for {self.active.symbol}: "
                            "entry was interrupted and Kite still has an NFO position."
                        )
                    else:
                        await self.positions.reject_entering(
                            self.active.symbol, self.active.setup_date, "startup_no_broker_position"
                        )
                        self.active = None
                if initialized_token != token:
                    await self.backfill(kite)
                    await self.rebuild_context(datetime.now(UTC))
                    initialized_token = token
                if not is_market_hours(datetime.now(UTC)):
                    await asyncio.sleep(30)
                    continue
                await self.resolve_tokens(kite)
                self.current_bucket = bucket_start(datetime.now(UTC), _BUCKET_MINUTES)
                for aggregator in self.aggregators.values():
                    aggregator.start(self.current_bucket)
                self.connect(token)
                try:
                    await self.run_session(kite, token)
                finally:
                    self.disconnect()
        finally:
            await self.client.close()


def _candle_dict(candle: Candle) -> dict:
    return {
        "timestamp": candle.timestamp,
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
    }


async def _run(config: AppConfig) -> None:
    if not config.daily_swing_enabled:
        logger.warning("Daily swing is disabled; set TRADING_SCANNER_DAILY_SWING_ENABLED=true.")
        return
    universe = SymbolLoader().load(config.daily_swing_universe_file)
    signals = frozenset(SymbolLoader().load(config.daily_swing_signal_symbols_file))
    if "^NSEI" not in universe:
        universe.append("^NSEI")
    while True:
        runner = DailySwingLive(config, universe, signals)
        try:
            await runner.run()
        except Exception:
            logger.exception("Daily-swing runner crashed; restarting in 30 seconds.")
            await asyncio.sleep(30)


def main() -> None:
    config = load_config()
    logging.basicConfig(level=config.logging_level, format="%(asctime)s %(levelname)s: %(message)s")
    if not config.kite_api_key or not config.turso_database_url:
        logger.error("Kite API key and local database URL are required.")
        return
    try:
        asyncio.run(_run(config))
    except (KeyboardInterrupt, SymbolLoadError):
        logger.info("Daily-swing runner stopped.")


if __name__ == "__main__":
    main()
