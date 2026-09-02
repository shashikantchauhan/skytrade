"""Daily candle ground-truth resync -- scheduled via cron (see deploy notes
below), not run by the live pipeline itself.

2026-09-02: closes a gap the same day's outage investigation surfaced
(docs/decisions/010-outage-catch-up-replay.md) that the catch-up-replay fix
does NOT cover -- that fix replays any *stored* candle the engine hasn't
evaluated yet, but if the live-ticker path never stored a candle at all
(the WebSocket was disconnected for an entire hourly bucket -- Kite session
expired, process down -- so the aggregator saw zero ticks and
``_finalize_bucket`` skipped it outright, see ``live_pipeline.py``), there
is nothing in ``candle_repository`` to replay in the first place. The live-
ticker path has no automatic backfill of its own; only a manual dashboard
"Run pipeline now" click (or this script) ever calls the download-based
path that can actually re-fetch what Kite's own historical record says
really happened.

Re-fetches the last ``_RESYNC_WINDOW_DAYS`` days of real candles from Kite
for every tracked symbol and upserts them -- correcting any candle a
ticker gap left missing or wrong, refreshing an already-correct one to a
no-op. Read-only against Kite, upsert-only against ``candle_repository`` --
never touches trades, positions, engine_state, or orders, so it cannot by
itself place or affect a real order.

Scope note: this restores *data* correctness (what's stored matches Kite's
own record), not retroactive *evaluation* -- a candle this script adds
that lands chronologically before what the engine has already evaluated
is not automatically replayed (the catch-up logic only looks forward from
its own last-processed timestamp, see evaluation.py's own docstring for
why). If a fully-missing candle needs to actually be scored (not just
correctly stored), that's still a manual follow-up today.

Deploy: added to the VPS crontab (not a systemd timer -- matches this
deployment's existing scheduled-job pattern, see backup_db.py's crontab
entry) at 11:00 UTC / 16:30 IST daily -- an hour after market close
(15:30 IST), well before the 18:30 UTC DB backup so the backup captures
the corrected data:
    0 11 * * 1-5 cd /opt/p-trade && /opt/p-trade/.venv/bin/python3 -m \\
        trading_scanner.resync_daily_candles_cli >> /var/log/p-trade/resync.log 2>&1
"""

import asyncio
import logging

from kiteconnect import KiteConnect

from trading_scanner.application.pipeline.market_data import _dataframe_to_candles
from trading_scanner.application.symbols import SymbolLoader
from trading_scanner.config.settings import load_config
from trading_scanner.infrastructure.db import (
    TursoCandleRepository,
    TursoKiteSessionRepository,
    create_turso_client,
)
from trading_scanner.infrastructure.kite import KiteInstrumentMap, KiteProvider

# 2 days, not 1 -- safety margin for a late-running job crossing a date
# boundary, or a symbol whose most recent real session was yesterday
# (a market holiday today). upsert_candles is idempotent per (symbol,
# interval, timestamp), so re-fetching an already-correct extra day is
# just a no-op, not a risk.
_RESYNC_WINDOW_DAYS = 2


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    config = load_config()
    if not config.turso_database_url:
        raise RuntimeError("TRADING_SCANNER_TURSO_URL must be set.")
    if not config.kite_api_key:
        raise RuntimeError("TRADING_SCANNER_KITE_API_KEY must be set.")

    client = create_turso_client(config.turso_database_url, config.turso_auth_token)
    try:
        candle_repository = TursoCandleRepository(client)
        await candle_repository.ensure_schema()
        kite_session_repository = TursoKiteSessionRepository(client)
        await kite_session_repository.ensure_schema()
        token_row = await kite_session_repository.get_token()
        if token_row is None:
            logger.warning("No Kite session -- skipping today's resync (will retry tomorrow).")
            return
        access_token, _obtained_at = token_row
        kite = KiteConnect(api_key=config.kite_api_key)
        kite.set_access_token(access_token)
        instrument_map = KiteInstrumentMap(kite)
        provider = KiteProvider(kite, instrument_map)

        symbols = SymbolLoader().load(config.symbols_file)
        logger.info("Resyncing %d symbols' candles against Kite ground truth.", len(symbols))
        fixed = 0
        failed = 0
        for symbol in symbols:
            try:
                downloaded = await asyncio.to_thread(
                    provider.get_recent_history, symbol, config.candle_interval,
                    _RESYNC_WINDOW_DAYS,
                )
                candles = _dataframe_to_candles(symbol, downloaded)
                await candle_repository.upsert_candles(symbol, config.candle_interval, candles)
                fixed += 1
            except Exception:
                failed += 1
                logger.exception("Failed to resync %s -- left as-is, will retry tomorrow.", symbol)
        logger.info(
            "Resync complete: %d/%d symbols refreshed, %d failed.", fixed, len(symbols), failed
        )
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
