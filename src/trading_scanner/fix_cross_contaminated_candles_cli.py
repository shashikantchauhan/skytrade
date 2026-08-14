"""One-time CLI: remove cross-symbol-contaminated candles and rebuild trades
for just the affected symbols.

Root cause found by hand on 2026-08-13: unrelated symbols (e.g. POLICYBZR.NS,
PNBHOUSING.NS, POWERGRID.NS -- insurance, housing finance, and power utility,
with no real connection) were found sharing byte-for-byte identical
open/high/low/close/volume at the same timestamps, across a full trading
day. Real, independent stock prices cannot coincidentally match to full
float precision -- this is the signature of one symbol's fetched data being
written into another symbol's candle rows during a bulk historical fetch,
not a bad exchange print (see fix_corrupted_candles_cli.py for that,
separate, class of bug) and not a stock split/bonus issue (those are
permanent, single-symbol, one-way price steps -- not a value shared
identically across multiple unrelated companies, and not one that reverts
within a couple of hours the way these candles do).

Traced to application/signal_pipeline.py's now-removed Yahoo fallback
(``_select_provider``): yfinance's ``yf.download`` has a documented failure
mode under concurrent request load where a response silently belongs to the
wrong ticker entirely. That fallback only ever triggered when the Kite
session was stale/not yet logged in for the day -- consistent with the
corrupted window found here. The fallback itself has been removed
(2026-08-13); this script is the one-time cleanup of what it already wrote.

Detection: group every candle by its exact (timestamp, open, high, low,
close, volume) tuple; any group spanning more than one distinct symbol is
corrupted by definition -- no price-swing/percentage threshold needed or
used, this is a direct fingerprint of the bug, not a heuristic.

Scoped, not a full re-backtest: only symbols with at least one contaminated
candle get their candles cleaned and trades rebuilt. Every other symbol's
trades are left untouched.

After deleting the bad candles, this also refills that exact gap with real
data from Kite Connect (not a re-derivation from whatever's left) before
rebuilding trades -- requires a valid Kite login (see /kite/login) to run.
``upsert_candles`` is keyed by (symbol, interval, timestamp), so refilling
a multi-day window is safe even where it overlaps already-good candles on
either side of the gap -- it just re-writes them to the same values.
"""

import asyncio
import logging
from datetime import UTC, datetime

from kiteconnect import KiteConnect

from trading_scanner.alpha_engine import AlphaEngine
from trading_scanner.application.backtest import compute_historical_events, replay_pine_backtest
from trading_scanner.application.signal_pipeline import _dataframe_to_candles
from trading_scanner.config.settings import load_config
from trading_scanner.infrastructure.db import (
    TursoCandleRepository,
    TursoKiteSessionRepository,
    TursoTradeRepository,
    create_turso_client,
)
from trading_scanner.infrastructure.kite import KiteInstrumentMap, KiteProvider

_ENGINE_SETTINGS = {"include_full_history": True, "use_dynamic_exits": True}
_MINIMUM_CANDLES = 200

# The contaminated window found by hand (2026-07-30 through 2026-08-10),
# with a one-day buffer on each side. Scoped deliberately rather than
# scanning the full history -- the detection query is O(candles) and a
# multi-year, 220-symbol table is not free to group-by on this VPS.
_WINDOW_START = "2026-07-29T00:00:00"
_WINDOW_END = "2026-08-12T00:00:00"
_WINDOW_START_DT = datetime.fromisoformat(_WINDOW_START).replace(tzinfo=UTC)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    config = load_config()
    if not config.turso_database_url:
        raise RuntimeError("TRADING_SCANNER_TURSO_URL must be set.")

    client = create_turso_client(config.turso_database_url, config.turso_auth_token)
    candle_repository = TursoCandleRepository(client)
    trade_repository = TursoTradeRepository(client)
    kite_session_repository = TursoKiteSessionRepository(client)
    await candle_repository.ensure_schema()
    await trade_repository.ensure_schema()
    await kite_session_repository.ensure_schema()

    # A valid Kite login is required up front -- the whole point of this
    # cleanup is replacing bad data with real Kite data, not leaving a gap
    # or (worse) falling back to anything else. Fails loudly rather than
    # silently skipping the refill if no session is available.
    token_row = await kite_session_repository.get_token()
    if token_row is None:
        raise RuntimeError(
            "No Kite session on file -- log in at /kite/login before running this cleanup, "
            "so the gap left by deleted candles can be refilled with real data."
        )
    access_token, _obtained_at = token_row
    kite = KiteConnect(api_key=config.kite_api_key)
    kite.set_access_token(access_token)
    instrument_map = KiteInstrumentMap(kite)
    kite_provider = KiteProvider(kite, instrument_map)
    refill_days = (datetime.now(UTC) - _WINDOW_START_DT).days + 2

    result = await client.execute(
        """
        SELECT timestamp, open, high, low, close, volume, GROUP_CONCAT(symbol) AS symbols
        FROM candles
        WHERE interval = ? AND timestamp BETWEEN ? AND ?
        GROUP BY timestamp, open, high, low, close, volume
        HAVING COUNT(DISTINCT symbol) > 1
        """,
        [config.candle_interval, _WINDOW_START, _WINDOW_END],
    )
    groups = result.rows
    logger.info("Found %d cross-symbol-contaminated candle groups.", len(groups))

    affected_symbols: set[str] = set()
    deleted_count = 0
    for timestamp, open_, high, low, close, volume, symbols_csv in groups:
        symbols = symbols_csv.split(",")
        affected_symbols.update(symbols)
        placeholders = ",".join("?" for _ in symbols)
        deleted = await client.execute(
            f"""
            DELETE FROM candles
            WHERE interval = ? AND timestamp = ? AND open = ? AND high = ? AND low = ?
              AND close = ? AND volume = ? AND symbol IN ({placeholders})
            """,
            [config.candle_interval, timestamp, open_, high, low, close, volume, *symbols],
        )
        deleted_count += deleted.rows_affected

    affected_symbols_sorted = sorted(affected_symbols)
    logger.info(
        "Deleted %d contaminated candle rows across %d symbols.",
        deleted_count, len(affected_symbols_sorted),
    )

    engine = AlphaEngine(**_ENGINE_SETTINGS)
    for symbol in affected_symbols_sorted:
        try:
            # Refill the gap with real Kite data before rebuilding trades --
            # upsert_candles is keyed by (symbol, interval, timestamp), so
            # this only touches the deleted rows plus whatever's most
            # recent; it can't corrupt or duplicate the good candles either
            # side of the gap.
            fresh = await asyncio.to_thread(
                kite_provider.get_recent_history, symbol, config.candle_interval, refill_days
            )
            fresh_candles = _dataframe_to_candles(symbol, fresh)
            await candle_repository.upsert_candles(symbol, config.candle_interval, fresh_candles)
            logger.info("Refilled %s: %d candles from Kite.", symbol, len(fresh_candles))

            await client.execute(
                "DELETE FROM trades WHERE symbol = ? AND interval = ?",
                [symbol, config.candle_interval],
            )
            candles = await candle_repository.get_candles(
                symbol, config.candle_interval, limit=None
            )
            if len(candles) < _MINIMUM_CANDLES:
                logger.info(
                    "Skipping %s: only %d candles remain after cleanup.", symbol, len(candles)
                )
                continue
            events = compute_historical_events(engine, candles)
            pine_trades = replay_pine_backtest(events)
            from trading_scanner.backtest import _store_trade  # reuse the exact storage logic

            for pine_trade in pine_trades:
                await _store_trade(symbol, config, pine_trade, trade_repository)
            logger.info(
                "Rebuilt %s: %d trades from %d clean candles.",
                symbol, len(pine_trades), len(candles),
            )
        except Exception:
            logger.exception("Unexpected exception while rebuilding %s", symbol)

    await client.close()
    logger.info("Cross-symbol-contamination cleanup finished.")


if __name__ == "__main__":
    asyncio.run(main())
