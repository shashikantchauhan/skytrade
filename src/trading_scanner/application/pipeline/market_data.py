"""Market-data provider selection, candle<->DataFrame conversion, and Kite
session/token-expiry helpers -- split out of ``signal_pipeline.py`` (Phase
8, see ``application/pipeline/__init__.py``). No behavior changed; every
function's body moved as-is.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
from kiteconnect import KiteConnect
from kiteconnect.exceptions import TokenException as KiteTokenException

from trading_scanner.config.settings import AppConfig
from trading_scanner.domain.models import Candle
from trading_scanner.domain.ports import Notifier
from trading_scanner.infrastructure.db import TursoKiteSessionRepository
from trading_scanner.infrastructure.kite import KiteInstrumentMap, KiteProvider
from trading_scanner.infrastructure.yahoo import YahooProvider

# Both providers implement the same duck-typed get_recent_history(symbol,
# interval, days) -> pd.DataFrame interface; nothing below needs to know
# which one it got.
MarketDataProvider = YahooProvider | KiteProvider

_STRATEGY_NAME = "lorentzian"


def _is_kite_token_error(error: BaseException) -> bool:
    """Walks the exception chain -- ``KiteProvider.get_recent_history``
    wraps the underlying ``TokenException`` in a ``RuntimeError`` (see
    ``infrastructure/kite.py``), so a plain ``isinstance`` check on the
    caught exception alone would miss it."""
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, KiteTokenException):
            return True
        current = current.__cause__ or current.__context__
    return False


class NoValidKiteSession(RuntimeError):
    """Raised by ``_select_provider`` when no usable Kite session exists.

    There is deliberately no Yahoo fallback here anymore (see this
    function's own docstring) -- the caller's job is to notify and skip
    this cycle cleanly, not to substitute a different data source.
    """


async def _select_provider(
    config: AppConfig,
    kite_session_repository: TursoKiteSessionRepository | None,
    notifier: Notifier | None = None,
) -> tuple[MarketDataProvider, str, KiteConnect | None]:
    """Require a valid Kite session; never silently substitute Yahoo.

    Used to fall back to Yahoo Finance when no Kite session was available.
    Removed 2026-08-13: Yahoo's ``yf.download`` has a documented failure
    mode under concurrent request load where a response silently belongs
    to the *wrong* ticker entirely (not just a malformed/partial one --
    see ``infrastructure/yahoo.py``'s ``_clean_data`` docstring, which
    already guarded the partial case). Root-caused to exactly this: 176
    symbols' candles were found byte-for-byte identical across unrelated
    stocks over a ~2-week window, matching runs where Kite's daily token
    hadn't been refreshed yet and this fallback silently kicked in. Kite
    Connect's Historical Data API is a paid, official data source with no
    such issue -- there is no good reason to ever substitute a free,
    unofficial scrape for real trading data again. A stale/missing Kite
    session now means this cycle is skipped entirely (see
    ``NoValidKiteSession``) rather than degrading to a worse, riskier data
    source.

    "Valid" is checked by actually calling Kite's instruments endpoint and
    confirming the hardcoded index mapping still resolves (see
    ``KiteInstrumentMap.validate_index_mapping``) rather than trusting a
    stored token blindly -- catches both an expired/revoked token and a
    stale index mapping in one check.

    Also re-sends a "please log in again" notification every 15 minutes
    for as long as the session stays broken (deduped via
    ``kite_session_repository``'s ``expiry_notified_at``, see
    ``TursoKiteSessionRepository`` and ``_notify_kite_expired_
    periodically``'s own docstring for why this isn't once-per-day
    anymore) -- Kite tokens expire daily with no documented exact time, so
    this piggybacks on the pipeline's own hourly cron rather than needing
    separate infrastructure to detect expiry.
    """
    logger = logging.getLogger(__name__)
    if config.kite_api_key and kite_session_repository is not None:
        token_row = await kite_session_repository.get_token()
        if token_row is not None:
            access_token, obtained_at = token_row
            try:
                kite = KiteConnect(api_key=config.kite_api_key)
                kite.set_access_token(access_token)
                instrument_map = KiteInstrumentMap(kite)
                await asyncio.to_thread(instrument_map.validate_index_mapping)
                return KiteProvider(kite, instrument_map), "kite", kite
            except Exception:
                logger.warning(
                    "Kite session unusable (obtained_at=%s); skipping this run.",
                    obtained_at,
                    exc_info=True,
                )
        await _notify_kite_expired_periodically(kite_session_repository, notifier)
    elif kite_session_repository is not None:
        await _notify_kite_expired_periodically(kite_session_repository, notifier)
    raise NoValidKiteSession("No valid Kite session -- skipping this run rather than using Yahoo.")


# 2026-09-02: was once-per-calendar-day -- a single missed/late-seen alert
# could then sit unattended for the rest of the day. In production, an
# expired token blocked real trading for 30-104 minutes at market open on
# three consecutive trading days (2026-08-31, 2026-09-01, 2026-09-02)
# because that one alert came and went unnoticed each time. Re-nudging
# every 15 minutes instead can't guarantee a human sees it any faster, but
# it stops one missed ping from silently costing the rest of the morning.
_EXPIRY_RENOTIFY_INTERVAL = timedelta(minutes=15)


async def _notify_kite_expired_periodically(
    kite_session_repository: TursoKiteSessionRepository, notifier: Notifier | None
) -> None:
    """Re-sends the "Kite session expired, please re-login" alert every
    ``_EXPIRY_RENOTIFY_INTERVAL`` for as long as the session stays broken --
    see that constant's own comment for why this replaced the old once-per-
    day version. Callers already retry this whole check every ~60s (the
    live-ticker path's poll loop, or the next hourly run), so this function
    only needs to decide whether *enough time has passed* to re-send, not
    to run its own timer."""
    if notifier is None:
        return
    now = datetime.now(UTC)
    try:
        last_notified_raw = await kite_session_repository.get_expiry_notified_at()
        if last_notified_raw is not None:
            last_notified = datetime.fromisoformat(last_notified_raw)
            if now - last_notified < _EXPIRY_RENOTIFY_INTERVAL:
                return
        await notifier.send_text(
            "⚠️ <b>SYSTEM ALERT</b>\n"
            "Kite session expired/missing -- this run is being skipped (no Yahoo fallback).\n"
            "Log in again: https://skytrade.oneatem.com/kite/login"
        )
        await kite_session_repository.set_expiry_notified_at(now.isoformat())
    except Exception:
        logging.getLogger(__name__).exception("Failed to send Kite-expiry notification")


def _market_price(candle: Candle) -> Decimal:
    """Pine's ``ml.backtest`` scoring price: (high + low + open + open) / 4.

    Not the close -- matches ``application/backtest.py``'s historical
    replay so live and backtested trades use the same price convention.
    """
    return (candle.high + candle.low + candle.open + candle.open) / 4


def _dataframe_to_candles(symbol: str, data: pd.DataFrame) -> list[Candle]:
    """Convert a downloaded OHLCV DataFrame into domain Candle objects.

    Normalized to UTC here -- Yahoo returns NSE timestamps tz-aware in IST,
    and storing that offset as-is produces an ISO string ("+05:30") that
    sorts incorrectly against UTC-stored rows ("+00:00") under a plain text
    ORDER BY, scrambling chronological order at every day boundary. Always
    normalizing to UTC before it ever reaches the repository keeps every
    row's timestamp column in the same offset, so text sort order matches
    real chronological order.
    """
    return [
        Candle(
            symbol=symbol,
            timestamp=timestamp.to_pydatetime().astimezone(UTC),
            open=Decimal(str(row["Open"])),
            high=Decimal(str(row["High"])),
            low=Decimal(str(row["Low"])),
            close=Decimal(str(row["Close"])),
            volume=int(row["Volume"]),
        )
        for timestamp, row in data.iterrows()
    ]


def _candles_to_dataframe(candles) -> pd.DataFrame:
    """Convert chronological Candle objects into the OHLCV DataFrame AlphaEngine expects.

    Normalizes every timestamp to UTC before building the index -- candles
    stored across different runs can carry equivalent-offset but distinct
    tzinfo objects (e.g. a fixed +05:30 offset vs. a zoneinfo-based one),
    which pandas refuses to unify into one DatetimeIndex without this
    (raises "Tz-aware datetime.datetime cannot be converted to datetime64
    unless utc=True"). AlphaEngine only depends on chronological order and
    OHLCV values, never the displayed hour, so this is safe -- the original
    Candle objects (with their real tzinfo) are still used everywhere else.
    """
    return pd.DataFrame(
        {
            "Open": [float(candle.open) for candle in candles],
            "High": [float(candle.high) for candle in candles],
            "Low": [float(candle.low) for candle in candles],
            "Close": [float(candle.close) for candle in candles],
            "Volume": [candle.volume for candle in candles],
        },
        index=pd.DatetimeIndex([candle.timestamp.astimezone(UTC) for candle in candles]),
    )
