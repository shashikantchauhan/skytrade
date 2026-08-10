"""Kite Connect market-data provider and session/auth helpers.

Zerodha's Historical Data API is the same feed NSE brokers use -- more
reliable than Yahoo Finance's unofficial scrape, which is why this replaces
YahooProvider as the pipeline's primary data source (see
``application/signal_pipeline.py``'s provider selection, which falls back to
Yahoo automatically if no valid Kite session exists).

Kite access tokens expire daily (undocumented exact time, commonly reported
around 6 AM IST) -- there is no way around a fresh login each trading day.
``webapp.py``'s ``/kite/login``/``/kite/callback`` routes handle this via
Kite's own OAuth-style login page, so the user's Zerodha password never
touches this server; only the resulting access token is stored (in Turso,
see ``TursoKiteSessionRepository``).
"""

from datetime import date, datetime, timedelta

import pandas as pd
from kiteconnect import KiteConnect

_INTERVAL_MAP = {
    "1h": "60minute",
    "60m": "60minute",
    "1d": "day",
    "day": "day",
    "5m": "5minute",
    "15m": "15minute",
    "30m": "30minute",
}

# Kite's Historical Data API caps how many days can be requested in a single
# call, and the cap depends on the interval -- exceeding it raises
# InputException("interval exceeds max limit"). This deployment's backfill
# window (729 days, for a symbol with no accumulated history yet) exceeds
# 60minute's limit, so requests wider than this are split into chunks and
# concatenated rather than sent as one call.
_MAX_DAYS_PER_REQUEST = {
    "minute": 60,
    "3minute": 100,
    "5minute": 100,
    "10minute": 100,
    "15minute": 200,
    "30minute": 200,
    "60minute": 400,
    "day": 2000,
}

# Manually curated -- Kite's NSE:INDICES segment uses different trading
# symbols than Yahoo Finance's index tickers (e.g. "NIFTY BANK" vs
# "^NSEBANK"), and there's no reliable automatic way to derive one from the
# other. Validated at startup against Kite's own instrument dump (see
# ``KiteInstrumentMap.validate_index_mapping``) rather than trusted blindly
# -- a silent mismatch here would corrupt index-context data for every
# symbol, not just fail loudly for the index itself.
INDEX_SYMBOL_MAP = {
    "^NSEI": "NIFTY 50",
    "^NSEBANK": "NIFTY BANK",
    "NIFTY_FIN_SERVICE.NS": "NIFTY FIN SERVICE",
    "^CNXIT": "NIFTY IT",
    "^CNXAUTO": "NIFTY AUTO",
    "^CNXFMCG": "NIFTY FMCG",
    "^CNXMETAL": "NIFTY METAL",
    "^CNXPHARMA": "NIFTY PHARMA",
    "^CNXREALTY": "NIFTY REALTY",
    "^CNXENERGY": "NIFTY ENERGY",
    "^CNXMEDIA": "NIFTY MEDIA",
    "^NSMIDCP": "NIFTY MIDCAP 50",
}


def _to_kite_tradingsymbol(symbol: str) -> str:
    """Yahoo-style symbol -> Kite tradingsymbol, without needing an
    instrument token (``kite.ltp`` accepts ``"NSE:TRADINGSYMBOL"`` strings
    directly) -- shared by ``get_last_prices`` below."""
    if symbol in INDEX_SYMBOL_MAP:
        return INDEX_SYMBOL_MAP[symbol]
    return symbol.removesuffix(".NS")


def get_last_prices(kite: KiteConnect, symbols: list[str]) -> dict[str, float]:
    """Live last-traded prices for equities/indices via Kite -- for the
    dashboard's mark-to-market display, which previously used
    ``YahooProvider.get_last_prices``' daily-close download and could lag
    the real intraday price by a full session. Best-effort: a symbol Kite
    can't currently price is simply left out, matching Yahoo's version."""
    if not symbols:
        return {}
    keys = {symbol: f"NSE:{_to_kite_tradingsymbol(symbol)}" for symbol in symbols}
    try:
        quote = kite.ltp(list(keys.values()))
    except Exception:
        return {}
    prices: dict[str, float] = {}
    for symbol, key in keys.items():
        row = quote.get(key)
        if row:
            prices[symbol] = row["last_price"]
    return prices


class InstrumentLookupError(RuntimeError):
    """A symbol couldn't be confidently mapped to a Kite instrument token."""


class KiteInstrumentMap:
    """Resolves Yahoo-style symbols (RELIANCE.NS, ^NSEI, ...) to Kite instrument tokens."""

    def __init__(self, kite: KiteConnect) -> None:
        self._kite = kite
        self._by_key: dict[tuple[str, str], int] | None = None  # (exchange, tradingsymbol) -> token

    def _ensure_loaded(self) -> None:
        if self._by_key is not None:
            return
        instruments = self._kite.instruments("NSE")
        self._by_key = {
            (row["exchange"], row["tradingsymbol"]): row["instrument_token"] for row in instruments
        }

    def validate_index_mapping(self) -> None:
        """Confirm every hardcoded index tradingsymbol actually exists in
        Kite's live instrument dump right now -- fails loudly rather than
        silently mismatching if Zerodha ever renames one."""
        self._ensure_loaded()
        missing = [
            (yahoo_symbol, kite_symbol)
            for yahoo_symbol, kite_symbol in INDEX_SYMBOL_MAP.items()
            if ("NSE", kite_symbol) not in self._by_key
        ]
        if missing:
            raise InstrumentLookupError(
                f"Index symbol mapping stale/wrong for: {missing} -- "
                "update INDEX_SYMBOL_MAP in infrastructure/kite.py."
            )

    def resolve(self, symbol: str) -> int:
        self._ensure_loaded()
        if symbol in INDEX_SYMBOL_MAP:
            trading_symbol = INDEX_SYMBOL_MAP[symbol]
        elif symbol.endswith(".NS"):
            trading_symbol = symbol.removesuffix(".NS")
        else:
            trading_symbol = symbol
        assert self._by_key is not None
        token = self._by_key.get(("NSE", trading_symbol))
        if token is None:
            raise InstrumentLookupError(
                f"No Kite NSE instrument found for {symbol!r} "
                f"(tried tradingsymbol={trading_symbol!r})."
            )
        return token


class KiteProvider:
    """Drop-in replacement for YahooProvider's get_recent_history, backed by
    Kite Connect's Historical Data API (the same feed NSE brokers use)."""

    def __init__(self, kite: KiteConnect, instrument_map: KiteInstrumentMap) -> None:
        self._kite = kite
        self._instrument_map = instrument_map

    def get_recent_history(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        if days <= 0:
            raise ValueError("Days must be greater than zero.")
        kite_interval = _INTERVAL_MAP.get(interval)
        if kite_interval is None:
            raise ValueError(f"Unsupported interval for Kite: {interval!r}")
        token = self._instrument_map.resolve(symbol)
        max_days = _MAX_DAYS_PER_REQUEST.get(kite_interval, days)

        rows: list[dict] = []
        window_end = datetime.now()
        remaining_days = days
        try:
            # Walk backwards in max_days-sized windows until the full
            # requested range is covered -- Kite has no built-in pagination
            # for this, chunking client-side is the documented workaround.
            while remaining_days > 0:
                chunk_days = min(remaining_days, max_days)
                window_start = window_end - timedelta(days=chunk_days)
                rows = (
                    self._kite.historical_data(token, window_start, window_end, kite_interval)
                    + rows
                )
                window_end = window_start
                remaining_days -= chunk_days
        except Exception as error:
            raise RuntimeError(f"Failed to download Kite history for {symbol}: {error}") from error
        if not rows:
            raise RuntimeError(f"No usable history returned for {symbol}.")
        data = pd.DataFrame(rows)
        data = data.drop_duplicates(subset="date")
        data = data.rename(
            columns={
                "date": "Datetime",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        data = data.set_index("Datetime").sort_index()
        return data[["Open", "High", "Low", "Close", "Volume"]]


class KiteDerivativesChain:
    """Looks up the nearest at-the-money option contract or nearest-expiry
    futures contract for a symbol, and fetches live premiums/prices --
    backs the derivatives shadow-tracking feature (``application/
    options_shadow.py``, ``application/futures_shadow.py``), never the live
    pipeline's actual equity signal generation. Yahoo has no Indian
    derivatives data, so this is only ever available when Kite is the
    active data source.
    """

    def __init__(self, kite: KiteConnect) -> None:
        self._kite = kite
        self._nfo: list[dict] | None = None

    def _ensure_loaded(self) -> None:
        if self._nfo is None:
            self._nfo = self._kite.instruments("NFO")

    def nearest_atm_option(
        self, symbol: str, option_type: str, underlying_price: float
    ) -> dict | None:
        """The nearest-expiry, nearest-strike contract of ``option_type``
        ("CE" or "PE") for ``symbol`` (Yahoo-style, e.g. RELIANCE.NS), or
        None if it has no listed options chain -- not every NSE stock does.
        """
        self._ensure_loaded()
        name = symbol.removesuffix(".NS")
        assert self._nfo is not None
        today = date.today()
        candidates = [
            row
            for row in self._nfo
            if row["name"] == name
            and row["instrument_type"] == option_type
            and row["expiry"] >= today
        ]
        if not candidates:
            return None
        nearest_expiry = min(row["expiry"] for row in candidates)
        same_expiry = [row for row in candidates if row["expiry"] == nearest_expiry]
        return min(same_expiry, key=lambda row: abs(row["strike"] - underlying_price))

    def nearest_future(self, symbol: str) -> dict | None:
        """The nearest-expiry FUT contract for ``symbol``, or None if it has
        no listed futures contract."""
        self._ensure_loaded()
        name = symbol.removesuffix(".NS")
        assert self._nfo is not None
        today = date.today()
        candidates = [
            row
            for row in self._nfo
            if row["name"] == name and row["instrument_type"] == "FUT" and row["expiry"] >= today
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda row: row["expiry"])

    def ltp(self, exchange_tradingsymbol: str) -> float | None:
        """Live last-traded price for one contract, e.g.
        'NFO:RELIANCE25AUG1400PE' or 'NFO:RELIANCE25AUGFUT'. None if Kite
        has no quote for it."""
        quote = self._kite.ltp([exchange_tradingsymbol])
        row = quote.get(exchange_tradingsymbol)
        return row["last_price"] if row else None

    def historical_premium(self, instrument_token: int, when: datetime) -> float | None:
        """Historical close nearest to ``when`` for one contract -- backs
        the current-month backtest (``application/derivatives_backtest.py``)
        rather than the live shadow-tracking flow, which always uses
        ``ltp``. Only works for contracts whose ``instrument_token`` is
        still resolvable today, i.e. not yet expired -- see this class's
        docstring on why that limits backtesting to the current month.
        None on any failure (unlisted token, no candles near that time,
        Kite API hiccup) -- best-effort, matching the rest of this
        feature's error handling.
        """
        window_start = when - timedelta(days=3)
        window_end = when + timedelta(days=1)
        try:
            candles = self._kite.historical_data(
                instrument_token, window_start, window_end, "60minute"
            )
        except Exception:
            return None
        if not candles:
            return None
        # Compare by epoch seconds, not naive datetime subtraction -- ``when``
        # and Kite's candle timestamps may carry different (but both valid)
        # tzinfo offsets, and this codebase has already been bitten once by
        # comparing timestamps without normalizing them first (see
        # ``turso.py``'s candle-timestamp corruption fix).
        target = when.timestamp()
        nearest = min(candles, key=lambda candle: abs(candle["date"].timestamp() - target))
        return nearest["close"]


def build_login_url(api_key: str) -> str:
    """The URL to send a user to for Kite's own login page (never our server)."""
    return f"https://kite.zerodha.com/connect/login?api_key={api_key}&v=3"


def exchange_request_token(api_key: str, api_secret: str, request_token: str) -> tuple[str, str]:
    """Exchange a one-time request_token (from the login redirect) for a
    day-long access token. Returns (access_token, login_time_iso)."""
    kite = KiteConnect(api_key=api_key)
    session = kite.generate_session(request_token, api_secret=api_secret)
    return session["access_token"], str(session["login_time"])
