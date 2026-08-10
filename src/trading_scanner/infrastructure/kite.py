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

from datetime import datetime, timedelta

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


def build_login_url(api_key: str) -> str:
    """The URL to send a user to for Kite's own login page (never our server)."""
    return f"https://kite.zerodha.com/connect/login?api_key={api_key}&v=3"


def exchange_request_token(api_key: str, api_secret: str, request_token: str) -> tuple[str, str]:
    """Exchange a one-time request_token (from the login redirect) for a
    day-long access token. Returns (access_token, login_time_iso)."""
    kite = KiteConnect(api_key=api_key)
    session = kite.generate_session(request_token, api_secret=api_secret)
    return session["access_token"], str(session["login_time"])
