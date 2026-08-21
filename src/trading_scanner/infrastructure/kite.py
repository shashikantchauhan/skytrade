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

import time
from datetime import date, datetime, timedelta
from decimal import Decimal

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


def to_kite_tradingsymbol(symbol: str) -> str:
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
    keys = {symbol: f"NSE:{to_kite_tradingsymbol(symbol)}" for symbol in symbols}
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

    def margin_benefit(
        self, hedged_tradingsymbols: list[tuple[str, str, int]]
    ) -> dict | None:
        """Live margin required for a combo of legs vs. holding the first
        leg alone, using Kite's own basket-margin API (the same SPAN/hedge
        netting the actual Kite app's basket screen shows) -- not a guessed
        percentage.

        ``hedged_tradingsymbols`` is ``[(tradingsymbol, transaction_type,
        quantity), ...]``, first leg is the "primary" position the rest are
        hedging.

        The API returns two totals per basket: ``initial`` (gross, before
        any hedge/combo netting) and ``final`` (what's actually blocked
        after netting) -- the real benefit only shows up in ``final``;
        ``initial`` looks identical whether or not a hedge helps, which is
        the mistake this function exists to not repeat (see this project's
        own history: a naive first pass compared ``initial`` totals and
        wrongly concluded hedging never reduces margin here).

        ``consider_positions=True`` (Kite's default) so the calculation
        reflects real portfolio netting against the account's existing
        positions, not just this basket in isolation. None on any failure.
        """
        if not hedged_tradingsymbols:
            return None

        def _leg(tradingsymbol: str, transaction_type: str, quantity: int) -> dict:
            return {
                "exchange": "NFO",
                "tradingsymbol": tradingsymbol,
                "transaction_type": transaction_type,
                "variety": "regular",
                "product": "NRML",
                "order_type": "MARKET",
                "quantity": quantity,
            }

        try:
            primary_symbol, primary_txn, primary_qty = hedged_tradingsymbols[0]
            primary_only = self._kite.basket_order_margins(
                [_leg(primary_symbol, primary_txn, primary_qty)], consider_positions=True
            )
            combined = self._kite.basket_order_margins(
                [_leg(sym, txn, qty) for sym, txn, qty in hedged_tradingsymbols],
                consider_positions=True,
            )
        except Exception:
            return None
        primary_only_margin = primary_only["final"]["total"]
        combined_margin = combined["final"]["total"]
        return {
            "primary_only_margin": primary_only_margin,
            "combined_margin": combined_margin,
            "margin_benefit": primary_only_margin - combined_margin,
        }

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


class KiteOrderExecutor:
    """Places and polls *real* orders on Zerodha -- deliberately a separate
    class from ``KiteDerivativesChain`` (which is analysis-only per its own
    docstring: "never a real order"). Nothing in this class is wired into
    the signal pipeline unless ``AppConfig.live_trading_enabled`` is set --
    see ``application/live_execution.py`` for the gated basket-entry/exit
    flow that actually calls this.
    """

    def __init__(self, kite: KiteConnect) -> None:
        self._kite = kite

    def place_market_order(self, tradingsymbol: str, transaction_type: str, quantity: int) -> str:
        """Places a real NFO market order, product NRML (carries positions
        overnight -- appropriate for a swing strategy, not an intraday
        one). Returns Kite's order_id; does not wait for a fill -- see
        ``poll_order_status``/``wait_for_fill`` for that."""
        return self._kite.place_order(
            variety=self._kite.VARIETY_REGULAR,
            exchange="NFO",
            tradingsymbol=tradingsymbol,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=self._kite.ORDER_TYPE_MARKET,
            product=self._kite.PRODUCT_NRML,
        )

    def place_cash_market_order(
        self, tradingsymbol: str, transaction_type: str, quantity: int
    ) -> str:
        """Places a real NSE cash-equity market order, product CNC
        (delivery -- carries the position overnight, matching this
        strategy's multi-day swing holds; MIS would auto-square-off the
        same day, which is wrong here). ``tradingsymbol`` must already be
        Kite's own form (no ``.NS`` suffix -- see ``_kite_symbol``).
        Returns Kite's order_id; does not wait for a fill -- see
        ``poll_order_status``/``wait_for_fill`` for that."""
        return self._kite.place_order(
            variety=self._kite.VARIETY_REGULAR,
            exchange="NSE",
            tradingsymbol=tradingsymbol,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=self._kite.ORDER_TYPE_MARKET,
            product=self._kite.PRODUCT_CNC,
        )

    def place_cash_bracket_gtt(
        self,
        tradingsymbol: str,
        quantity: int,
        last_price: Decimal,
        stop_price: Decimal,
        target_price: Decimal,
    ) -> int:
        """Places a two-leg OCO GTT (Good-Till-Triggered) on NSE cash: a
        SELL LIMIT at ``target_price`` and a SELL LIMIT at ``stop_price``.
        Whichever triggers first cancels the other automatically at the
        exchange -- see ``application/gtt_bracket.py`` for the entry/exit
        lifecycle this belongs to. Kite requires ``trigger_values`` sorted
        ascending; the order list itself doesn't need to match that order,
        each leg carries its own ``trigger_price``/``price``. Returns the
        GTT's ``trigger_id`` for later ``modify``/``delete``."""
        response = self._kite.place_gtt(
            trigger_type=self._kite.GTT_TYPE_OCO,
            tradingsymbol=tradingsymbol,
            exchange="NSE",
            trigger_values=[float(stop_price), float(target_price)],
            last_price=float(last_price),
            orders=[
                {
                    "transaction_type": self._kite.TRANSACTION_TYPE_SELL,
                    "quantity": quantity,
                    "order_type": self._kite.ORDER_TYPE_LIMIT,
                    "product": self._kite.PRODUCT_CNC,
                    "price": float(stop_price),
                },
                {
                    "transaction_type": self._kite.TRANSACTION_TYPE_SELL,
                    "quantity": quantity,
                    "order_type": self._kite.ORDER_TYPE_LIMIT,
                    "product": self._kite.PRODUCT_CNC,
                    "price": float(target_price),
                },
            ],
        )
        return int(response["trigger_id"])

    def modify_cash_bracket_gtt(
        self,
        trigger_id: int,
        tradingsymbol: str,
        quantity: int,
        last_price: Decimal,
        stop_price: Decimal,
        target_price: Decimal,
    ) -> None:
        """Replaces both trigger prices on an existing OCO GTT (e.g.
        extending the target and trailing the stop-loss up -- see
        ``application/gtt_bracket.py``). Same leg shape as
        ``place_cash_bracket_gtt``, just re-sent against ``trigger_id``."""
        self._kite.modify_gtt(
            trigger_id=trigger_id,
            trigger_type=self._kite.GTT_TYPE_OCO,
            tradingsymbol=tradingsymbol,
            exchange="NSE",
            trigger_values=[float(stop_price), float(target_price)],
            last_price=float(last_price),
            orders=[
                {
                    "transaction_type": self._kite.TRANSACTION_TYPE_SELL,
                    "quantity": quantity,
                    "order_type": self._kite.ORDER_TYPE_LIMIT,
                    "product": self._kite.PRODUCT_CNC,
                    "price": float(stop_price),
                },
                {
                    "transaction_type": self._kite.TRANSACTION_TYPE_SELL,
                    "quantity": quantity,
                    "order_type": self._kite.ORDER_TYPE_LIMIT,
                    "product": self._kite.PRODUCT_CNC,
                    "price": float(target_price),
                },
            ],
        )

    def gtt_status(self, trigger_id: int) -> str:
        """Kite's own current status string for a GTT ("active",
        "triggered", "deleted", "expired", ...) -- used to tell "still
        live, safe to cancel + market-exit" apart from "already fired, the
        real position is already flat, a market SELL now would just get
        rejected/be wrong" (see ``application/gtt_bracket.py``'s
        reconcile-before-exit flow)."""
        return self._kite.get_gtt(trigger_id)["status"]

    def delete_gtt(self, trigger_id: int) -> None:
        """Cancels a GTT outright -- used when a strategy exit signal fires
        while the bracket is still open (see ``application/
        gtt_bracket.py``'s cancel-before-market-exit flow). Best-effort at
        the call site: Kite raises if the trigger already fired/was
        deleted, which the caller treats the same as "already gone,"
        never as a reason to skip the market exit."""
        self._kite.delete_gtt(trigger_id)

    def order_status(self, order_id: str) -> dict:
        """The most recent status entry for ``order_id`` -- Kite's
        ``order_history`` returns every state transition the order has gone
        through (OPEN -> COMPLETE, or OPEN -> REJECTED, etc.); the last
        entry is always the current state."""
        history = self._kite.order_history(order_id)
        return history[-1]

    def wait_for_fill(self, order_id: str, timeout_seconds: float, poll_interval: float = 1.0):
        """Blocks (synchronously -- callers run this via ``asyncio.
        to_thread``) polling ``order_status`` until it reaches a terminal
        state (COMPLETE/REJECTED/CANCELLED) or ``timeout_seconds`` elapses.

        Returns the final status dict. On timeout, returns the last-seen
        status as-is (still likely OPEN/TRIGGER PENDING) rather than
        raising -- the caller decides what "still pending after our
        timeout" means for basket sequencing (see ``live_execution.py``),
        this function's only job is to stop polling and hand back what it
        last saw.
        """
        deadline = time.monotonic() + timeout_seconds
        status = self.order_status(order_id)
        while status["status"] not in ("COMPLETE", "REJECTED", "CANCELLED"):
            if time.monotonic() >= deadline:
                break
            time.sleep(poll_interval)
            status = self.order_status(order_id)
        return status


def build_login_url(api_key: str) -> str:
    """The URL to send a user to for Kite's own login page (never our server)."""
    return f"https://kite.zerodha.com/connect/login?api_key={api_key}&v=3"


def exchange_request_token(api_key: str, api_secret: str, request_token: str) -> tuple[str, str]:
    """Exchange a one-time request_token (from the login redirect) for a
    day-long access token. Returns (access_token, login_time_iso)."""
    kite = KiteConnect(api_key=api_key)
    session = kite.generate_session(request_token, api_secret=api_secret)
    return session["access_token"], str(session["login_time"])
