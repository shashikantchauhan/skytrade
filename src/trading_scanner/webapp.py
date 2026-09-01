# ruff: noqa: E501 -- long lines below are almost all inside embedded
# HTML/CSS/JS template strings, which read worse wrapped than over-length.
"""Web dashboard: view the paper account live and control the pipeline.

Single-file FastAPI app, protected by a cookie-based login (one shared
password -- this is a personal tool, not a multi-user product). Sessions are
kept in an in-memory dict, so a dashboard restart logs everyone out -- fine
for a single-user tool. Reads directly from the same Turso database the
hourly pipeline writes to; no separate data layer.

Run with: `trading-scanner-dashboard` (see pyproject.toml), or directly:
    PYTHONPATH=src python -m trading_scanner.webapp
"""

import asyncio
import logging
import os
import secrets
import subprocess
import sys
import time
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace

from fastapi import Cookie, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from kiteconnect import KiteConnect
from pydantic import BaseModel

from trading_scanner.application import (
    broker_reconciliation,
    futures_trading,
    manual_exit,
    paper_trading,
)
from trading_scanner.application.options_analytics import enrich_trade
from trading_scanner.config.settings import load_config
from trading_scanner.domain.models import PaperPosition
from trading_scanner.infrastructure.db import (
    LiveCashToggleState,
    TursoFuturesPaperAccountRepository,
    TursoFuturesTradeRepository,
    TursoGttRepository,
    TursoKiteSessionRepository,
    TursoLiveCashToggleRepository,
    TursoLiveOrderRepository,
    TursoOptionsTradeRepository,
    TursoPaperAccountRepository,
    TursoPaperBenchmarkRepository,
    TursoTradeRepository,
    create_turso_client,
)
from trading_scanner.infrastructure.kite import (
    KiteDerivativesChain,
    KiteOrderExecutor,
    build_login_url,
    exchange_request_token,
)
from trading_scanner.infrastructure.kite import (
    get_last_prices as kite_get_last_prices,
)
from trading_scanner.infrastructure.telegram import LoggingNotifier, TelegramNotifier
from trading_scanner.infrastructure.yahoo import YahooProvider
from trading_scanner.web.services.auth import (
    _SESSION_COOKIE,
    _SESSION_TTL_SECONDS,
    _authenticate,
    _require_admin,
    _require_session,
    _sessions,
)

_yahoo = YahooProvider()

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _REPO_ROOT / ".env"
_LOG_PATH = Path(os.getenv("TRADING_SCANNER_LOG_PATH", "/var/log/p-trade/signals.log"))
_BACKTEST_LOG_PATH = _LOG_PATH.with_name("derivatives-backtest.log")

# 2026-08-16: skytrade-smallcap (Nifty Smallcap 250, weekly signals) is a
# separate fork deployed as a subfolder alongside this app (/opt/p-trade/
# smallcap), with its own local SQLite file, own daily crontab, own paper
# account -- entirely independent of the cash/futures books above. This
# dashboard just reads its DB read-only to fold it into one URL/login
# rather than standing up a second dashboard. If that fork's own .env ever
# changes PAPER_CAPITAL/_SLOTS/_MIN_POSITION, these three must be updated
# to match by hand -- they're read from a different process's environment,
# so they can't be imported/shared automatically.
_SMALLCAP_DB_PATH = os.getenv(
    "TRADING_SCANNER_SMALLCAP_DB_PATH", "/opt/p-trade/smallcap/data/skytrade-smallcap.db"
)
_SMALLCAP_INITIAL_CAPITAL = Decimal("500000")
_SMALLCAP_TARGET_SLOTS = 10
_SMALLCAP_MIN_POSITION_SIZE = Decimal("50000")

app = FastAPI(title="SkyTrade dashboard")


def _client():
    config = load_config()
    if not config.turso_database_url:
        raise HTTPException(status_code=500, detail="TRADING_SCANNER_TURSO_URL is not set.")
    return create_turso_client(config.turso_database_url, config.turso_auth_token), config


def _decimal(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


async def _last_prices(positions: list[PaperPosition], client, config) -> dict[str, float]:
    """Fetch current market prices for open positions' symbols.

    Prefers a live Kite quote when a session is active -- Yahoo's
    ``get_last_prices`` downloads the last *daily close*, which during a
    trading session can lag the real price by a full day (the bug reported
    against the dashboard: showing yesterday's close while the live price
    had already moved). Falls back to Yahoo if Kite isn't configured or has
    no active session today. Both calls are blocking, so run off the event
    loop."""
    symbols = [p.symbol for p in positions]
    if not symbols:
        return {}
    if config.kite_api_key:
        repository = TursoKiteSessionRepository(client)
        await repository.ensure_schema()
        token_row = await repository.get_token()
        if token_row is not None:
            access_token, _obtained_at = token_row
            kite = KiteConnect(api_key=config.kite_api_key)
            kite.set_access_token(access_token)
            prices = await asyncio.to_thread(kite_get_last_prices, kite, symbols)
            if prices:
                return prices
    return await asyncio.to_thread(_yahoo.get_last_prices, symbols)


def _unrealized_pnl(position: PaperPosition, last_prices: dict[str, float]) -> dict:
    current_price = last_prices.get(position.symbol)
    if current_price is None:
        return {"current_price": None, "unrealized_pnl": None, "unrealized_pnl_pct": None}
    pnl = (Decimal(str(current_price)) - position.entry_price) * position.quantity
    return {
        "current_price": current_price,
        "unrealized_pnl": _decimal(pnl),
        "unrealized_pnl_pct": _decimal(pnl / position.capital_allocated * 100),
    }


@app.get("/", response_model=None)
async def index(request: Request) -> HTMLResponse | RedirectResponse:
    session = _sessions.get(request.cookies.get(_SESSION_COOKIE, ""))
    if session is None or session["expiry"] < time.time():
        return RedirectResponse("/login")
    return HTMLResponse(_PAGE)


@app.get("/login", response_class=HTMLResponse)
async def login_page() -> str:
    return _LOGIN_PAGE


class LoginRequest(BaseModel):
    password: str


@app.post("/login")
async def login(body: LoginRequest) -> JSONResponse:
    authenticated = _authenticate(body.password)
    if authenticated is None:
        raise HTTPException(status_code=401, detail="Wrong password.")
    role, name = authenticated
    token = secrets.token_urlsafe(32)
    _sessions[token] = {"expiry": time.time() + _SESSION_TTL_SECONDS, "role": role, "name": name}
    response = JSONResponse({"ok": True, "role": role})
    response.set_cookie(
        _SESSION_COOKIE,
        token,
        max_age=_SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
    )
    return response


@app.get("/api/me")
async def me(
    ptrade_session: str | None = Cookie(default=None), _: None = Depends(_require_session)
) -> JSONResponse:
    """Lets the dashboard's own JS know whether to show admin-only controls
    (Kite login/status, pipeline trigger, backtest trigger, config) --
    those routes are enforced server-side too via ``_require_admin``, this
    is just so a viewer's UI doesn't show buttons that would 403."""
    session = _sessions[ptrade_session]
    return JSONResponse({"role": session["role"], "name": session["name"]})


@app.post("/logout")
async def logout(ptrade_session: str | None = Cookie(default=None)) -> JSONResponse:
    _sessions.pop(ptrade_session or "", None)
    response = JSONResponse({"ok": True})
    response.delete_cookie(_SESSION_COOKIE)
    return response


@app.get("/kite/login")
async def kite_login(_: None = Depends(_require_admin)) -> RedirectResponse:
    """Send the user to Kite's own login page -- their Zerodha password is
    entered there, never on this server. Requires being logged into this
    dashboard first (so a stranger can't hijack the Kite session)."""
    config = load_config()
    if not config.kite_api_key:
        raise HTTPException(status_code=500, detail="TRADING_SCANNER_KITE_API_KEY is not set.")
    return RedirectResponse(build_login_url(config.kite_api_key))


@app.get("/kite/callback", response_class=HTMLResponse)
async def kite_callback(request: Request) -> str:
    """Kite redirects here after login with a one-time request_token, which
    is exchanged immediately for the day's access token and stored. Not
    behind _require_session -- Kite itself is the auth gate for this step,
    and the request_token is single-use/short-lived, so there's nothing
    sensitive to protect on this specific hop."""
    request_token = request.query_params.get("request_token")
    status_param = request.query_params.get("status")
    if status_param != "success" or not request_token:
        return "<p>Kite login failed or was cancelled. You can close this tab and try again.</p>"
    config = load_config()
    if not config.kite_api_key or not config.kite_api_secret:
        return "<p>Kite API key/secret not configured on the server.</p>"
    try:
        access_token, obtained_at = exchange_request_token(
            config.kite_api_key, config.kite_api_secret, request_token
        )
    except Exception as error:
        return f"<p>Failed to exchange Kite token: {error}</p>"
    client = create_turso_client(config.turso_database_url, config.turso_auth_token)
    try:
        repository = TursoKiteSessionRepository(client)
        await repository.ensure_schema()
        await repository.set_token(access_token, obtained_at)
    finally:
        await client.close()
    return (
        "<p>Kite login successful -- today's session is active. "
        '<a href="/">Back to dashboard</a></p>'
    )


@app.get("/api/kite-status")
async def kite_status(_: None = Depends(_require_admin)) -> JSONResponse:
    config = load_config()
    if not config.kite_api_key:
        return JSONResponse({"configured": False})
    client = create_turso_client(config.turso_database_url, config.turso_auth_token)
    try:
        repository = TursoKiteSessionRepository(client)
        await repository.ensure_schema()
        token_row = await repository.get_token()
    finally:
        await client.close()
    if token_row is None:
        return JSONResponse({"configured": True, "logged_in": False})
    return JSONResponse({"configured": True, "logged_in": True, "obtained_at": token_row[1]})


def _merge_real_cash_summary(positions: dict, holdings: list[dict]) -> dict:
    """Combine same-day CNC entries with prior-day CNC holdings into one
    real, currently-held-position summary.

    Kite splits a CNC (delivery) position's visible lifetime across two
    separate endpoints: the day it's bought, it's in
    ``positions()['net']`` with a nonzero ``quantity``; every day after
    that it nets to 0 there and shows up in ``holdings()`` instead, as
    ``quantity`` (fully settled) or ``t1_quantity`` (bought yesterday,
    settling) -- either way, real shares actually owned. Reading only one
    endpoint makes a real, currently-open position disappear the moment it
    survives past its entry day. Scoped to CNC everywhere, matching what
    ``live_cash_execution.py`` always trades, so nothing else on this Kite
    account (manual trades, another product) skews these numbers.

    2026-08-28: ``positions()['net']`` quantity can also be *negative* --
    that's Kite's way of showing a same-day SELL of shares that came from
    yesterday's holdings (it tracks the day's net buy/sell activity, not a
    running total), not an open position. Confirmed live right after this
    app sold UNIONBANK.NS: ``holdings()`` had already dropped to 0 by the
    time it was read, but the -26 in ``net`` was still being counted as an
    "open position bought today," inflating the count and pulling its
    ``unrealised`` figure into ``unrealized_pnl`` for a position that was
    actually already closed.
    """
    net_cnc = [p for p in positions.get("net", []) if p.get("product") == "CNC"]
    open_today = [p for p in net_cnc if p.get("quantity", 0) > 0]
    day_cnc = [p for p in positions.get("day", []) if p.get("product") == "CNC"]

    open_holdings = []
    for holding in holdings:
        if holding.get("product") != "CNC":
            continue
        total_qty = holding.get("quantity", 0) + holding.get("t1_quantity", 0)
        if total_qty != 0:
            open_holdings.append(holding)

    unrealized_pnl = sum(Decimal(str(p.get("unrealised", 0))) for p in open_today) + sum(
        Decimal(str(h.get("pnl", 0))) for h in open_holdings
    )
    # Today's P&L: same-day entries contribute their whole P&L (all of it
    # happened today); a prior-day holding contributes only today's price
    # move -- day_change is Kite's per-share rupee move since yesterday's
    # close, so it needs multiplying by the shares actually held.
    today_pnl = sum(Decimal(str(p.get("pnl", 0))) for p in day_cnc) + sum(
        Decimal(str(h.get("day_change", 0)))
        * Decimal(str(h.get("quantity", 0) + h.get("t1_quantity", 0)))
        for h in open_holdings
    )
    return {
        "open_position_count": len(open_today) + len(open_holdings),
        "unrealized_pnl": unrealized_pnl,
        "today_pnl": today_pnl,
    }


@app.get("/api/status")
async def status(_: None = Depends(_require_session)) -> JSONResponse:
    """Real-money summary, straight from Kite -- not this app's own
    bookkeeping. 2026-08-25: this used to read the old, retired paper-
    trading account (paper_account/paper_positions), which meant every
    number here -- cash balance, open count, P&L, everything -- was stale
    simulator data with no relationship to real cash trading at all.

    2026-08-26: reading only ``kite.positions()`` (as this endpoint did the
    day before) is itself incomplete -- Kite only keeps a CNC position in
    ``positions()['net']`` on the day it was bought. The very next trading
    day it settles out of ``positions()`` entirely (nets to 0 there) and
    moves into ``kite.holdings()`` instead. A position bought yesterday and
    still open today is therefore real, still open, still at risk -- but
    invisible to this endpoint unless holdings are read too. See
    ``_merge_real_cash_summary`` for the actual merge.

    Deliberately has no "P&L since start" figure: Kite's own APIs only
    reflect today's activity plus currently-held positions (a fully closed
    position drops out entirely the next trading day), so there's no clean
    multi-day cumulative number to pull from the broker directly.
    Reconstructing one from this app's own live_order_legs ledger would
    undercount -- a GTT bracket that closes a position directly on the
    exchange never creates a matching closing leg here, only a strategy
    exit does (see application/gtt_bracket.py's reconciliation, which only
    notices after the fact). ``today_pnl``/``unrealized_pnl`` avoid both
    problems: always exactly what Kite itself would show you right now.
    """
    client, config = _client()
    try:
        if not config.kite_api_key:
            return JSONResponse({"kite_connected": False, "last_run": _last_run_summary()})
        repository = TursoKiteSessionRepository(client)
        await repository.ensure_schema()
        token_row = await repository.get_token()
        if token_row is None:
            return JSONResponse({"kite_connected": False, "last_run": _last_run_summary()})
        access_token, _obtained_at = token_row
        kite = KiteConnect(api_key=config.kite_api_key)
        kite.set_access_token(access_token)

        def _fetch() -> tuple[dict, dict, list]:
            return kite.margins(), kite.positions(), kite.holdings()

        try:
            margins, positions, holdings = await asyncio.to_thread(_fetch)
        except Exception:
            # Stale/invalid token, network hiccup, etc. -- same "just show
            # not-connected" fallback as no session at all, never a 500.
            return JSONResponse({"kite_connected": False, "last_run": _last_run_summary()})

        summary = _merge_real_cash_summary(positions, holdings)
        available_cash = margins.get("equity", {}).get("available", {}).get("live_balance")

        return JSONResponse(
            {
                "kite_connected": True,
                "available_cash": _decimal(Decimal(str(available_cash)))
                if available_cash is not None
                else None,
                "open_position_count": summary["open_position_count"],
                "unrealized_pnl": _decimal(summary["unrealized_pnl"]),
                "today_pnl": _decimal(summary["today_pnl"]),
                "last_run": _last_run_summary(),
            }
        )
    finally:
        await client.close()


@app.get("/api/positions")
async def positions(_: None = Depends(_require_session)) -> JSONResponse:
    client, config = _client()
    try:
        repository = TursoPaperAccountRepository(client, paper_trading.INITIAL_CAPITAL)
        open_positions = await repository.get_open_positions()
        last_prices = await _last_prices(list(open_positions), client, config)
        return JSONResponse(
            [
                {
                    "symbol": p.symbol,
                    "entry_timestamp": p.entry_timestamp.isoformat(),
                    "entry_price": _decimal(p.entry_price),
                    "quantity": p.quantity,
                    "capital_allocated": _decimal(p.capital_allocated),
                    **_unrealized_pnl(p, last_prices),
                }
                for p in sorted(open_positions, key=lambda p: p.entry_timestamp, reverse=True)
            ]
        )
    finally:
        await client.close()


@app.get("/api/live-cash-positions")
async def live_cash_positions(_: None = Depends(_require_session)) -> JSONResponse:
    """Real open cash-equity positions (application/live_cash_execution.py
    + gtt_bracket.py) -- what /api/positions used to show before the paper
    simulator was retired (see PAPER_TRADING_ENABLED). Shows the GTT's
    current target/stop alongside live mark-to-market, since those move
    (extension) independently of the entry itself."""
    client, config = _client()
    try:
        live_order_repository = TursoLiveOrderRepository(client)
        await live_order_repository.ensure_schema()
        gtt_repository = TursoGttRepository(client)
        await gtt_repository.ensure_schema()

        # broker_reconciliation, not get_all_open_cash_legs directly -- the
        # latter is COMPLETE-only and would hide a real position stuck at
        # status=UNKNOWN from this view entirely. See that module's own
        # docstring.
        legs = await broker_reconciliation.get_all_unclosed_positions(live_order_repository)
        brackets = {leg.symbol: await gtt_repository.get_active(leg.symbol) for leg in legs}
        last_prices = await _last_prices(
            [SimpleNamespace(symbol=leg.symbol) for leg in legs], client, config
        )

        rows = []
        for leg in sorted(legs, key=lambda leg: leg.placed_at, reverse=True):
            entry_price = leg.average_price
            current_price = last_prices.get(leg.symbol)
            unrealized = None
            unrealized_pct = None
            if entry_price is not None and current_price is not None:
                unrealized = (Decimal(str(current_price)) - entry_price) * leg.quantity
                unrealized_pct = (Decimal(str(current_price)) / entry_price - 1) * 100
            bracket = brackets.get(leg.symbol)
            rows.append(
                {
                    "symbol": leg.symbol,
                    "tradingsymbol": leg.tradingsymbol,
                    "entry_timestamp": leg.placed_at.isoformat(),
                    "entry_price": _decimal(entry_price),
                    "quantity": leg.quantity,
                    "current_price": current_price,
                    "unrealized_pnl": _decimal(unrealized),
                    "unrealized_pnl_pct": _decimal(unrealized_pct),
                    "target_price": _decimal(bracket.target_price) if bracket else None,
                    "stop_price": _decimal(bracket.stop_price) if bracket else None,
                    "gtt_status": bracket.status if bracket else None,
                }
            )
        return JSONResponse(rows)
    finally:
        await client.close()


@app.post("/api/live-cash-positions/{symbol}/exit")
async def exit_live_cash_position(
    symbol: str, _: None = Depends(_require_admin)
) -> JSONResponse:
    """Manually close one real open cash position right now -- cancels its
    GTT bracket first, then places a real market sell, exactly the same
    path a strategy-driven exit takes (application/manual_exit.py). Admin-
    only: this places a real order, unlike every ``_require_session`` read
    route above."""
    client, config = _client()
    try:
        if not config.kite_api_key:
            raise HTTPException(status_code=400, detail="Kite is not configured.")
        kite_session_repository = TursoKiteSessionRepository(client)
        await kite_session_repository.ensure_schema()
        token_row = await kite_session_repository.get_token()
        if token_row is None:
            raise HTTPException(status_code=400, detail="No active Kite session.")
        access_token, _obtained_at = token_row

        live_order_repository = TursoLiveOrderRepository(client)
        await live_order_repository.ensure_schema()
        # broker_reconciliation, not get_open_cash_legs directly -- this is
        # exactly the pre-check that used to 404 before manual_exit.
        # exit_position ever got a chance to run its own (already-fixed)
        # reconciliation for a real position stuck at status=UNKNOWN. See
        # that module's own docstring.
        entry_leg = await broker_reconciliation.get_unclosed_entry_leg(
            symbol, live_order_repository
        )
        if entry_leg is None:
            raise HTTPException(status_code=404, detail=f"No real open position for {symbol}.")

        kite = KiteConnect(api_key=config.kite_api_key)
        kite.set_access_token(access_token)
        last_prices = await _last_prices([SimpleNamespace(symbol=symbol)], client, config)
        current_price = last_prices.get(symbol)
        if current_price is None:
            raise HTTPException(status_code=502, detail=f"No live quote available for {symbol}.")

        gtt_repository = TursoGttRepository(client)
        await gtt_repository.ensure_schema()
        paper_benchmark_repository = TursoPaperBenchmarkRepository(client)
        await paper_benchmark_repository.ensure_schema()
        notifier = (
            TelegramNotifier(config.telegram_bot_token, config.telegram_chat_id, config.notification_label)
            if config.telegram_bot_token and config.telegram_chat_id
            else LoggingNotifier()
        )

        result = await manual_exit.exit_position(
            symbol, Decimal(str(current_price)), config, KiteOrderExecutor(kite),
            gtt_repository, live_order_repository, paper_benchmark_repository, notifier,
        )
        return JSONResponse({"ok": result.ok, "message": result.message})
    finally:
        await client.close()


def _paper_benchmark_summary(closed: list) -> dict:
    """Slippage/PnL rollup across closed paper-benchmark pairs -- same
    "compute in Python over the closed list" pattern as
    _futures_monthly_summary. Entry slippage is always computable for a
    closed pair (a paper-benchmark row is only ever opened once a real fill
    happened); exit slippage is skipped for the rare row with no
    paper_exit_price recorded."""
    if not closed:
        return {
            "count": 0,
            "avg_entry_slippage_pct": None,
            "avg_exit_slippage_pct": None,
            "total_paper_pnl": None,
            "total_real_pnl": None,
        }
    entry_slippages = [
        float((p.real_entry_price - p.paper_entry_price) / p.paper_entry_price * 100)
        for p in closed
    ]
    exit_slippages = [
        float((p.real_exit_price - p.paper_exit_price) / p.paper_exit_price * 100)
        for p in closed
        if p.paper_exit_price is not None and p.real_exit_price is not None
    ]
    total_paper = sum((p.paper_pnl_amount or Decimal("0") for p in closed), start=Decimal("0"))
    total_real = sum((p.real_pnl_amount or Decimal("0") for p in closed), start=Decimal("0"))
    return {
        "count": len(closed),
        "avg_entry_slippage_pct": sum(entry_slippages) / len(entry_slippages),
        "avg_exit_slippage_pct": (
            sum(exit_slippages) / len(exit_slippages) if exit_slippages else None
        ),
        "total_paper_pnl": _decimal(total_paper),
        "total_real_pnl": _decimal(total_real),
    }


@app.get("/api/paper-benchmark")
async def paper_benchmark_positions(_: None = Depends(_require_session)) -> JSONResponse:
    """Paper-simulated benchmark run 1:1 alongside every real live-cash
    trade (see application/paper_benchmark.py) -- measures execution
    quality/slippage: paper "fill" = decision price (no friction) vs. real
    fill = Kite's actual average_price, same symbol/qty/signal. Strictly
    paired with real trades, not a general paper-trading system -- fresh
    table, unrelated to the retired paper_trading.py/paper_account system."""
    client, _config = _client()
    try:
        repository = TursoPaperBenchmarkRepository(client)
        await repository.ensure_schema()
        open_positions = list(await repository.get_open_positions())
        recent_closed = list(await repository.get_recent_closed_positions(200))

        def _row(p) -> dict:
            return {
                "symbol": p.symbol,
                "basket_id": p.basket_id,
                "quantity": p.quantity,
                "entry_timestamp": p.entry_timestamp.isoformat(),
                "paper_entry_price": _decimal(p.paper_entry_price),
                "real_entry_price": _decimal(p.real_entry_price),
                "exit_timestamp": p.exit_timestamp.isoformat() if p.exit_timestamp else None,
                "paper_exit_price": _decimal(p.paper_exit_price),
                "real_exit_price": _decimal(p.real_exit_price),
                "paper_pnl_amount": _decimal(p.paper_pnl_amount),
                "real_pnl_amount": _decimal(p.real_pnl_amount),
                "status": p.status,
            }

        return JSONResponse(
            {
                "open_positions": [_row(p) for p in open_positions],
                "recent_closed": [_row(p) for p in recent_closed],
                "summary": _paper_benchmark_summary(recent_closed),
            }
        )
    finally:
        await client.close()


async def _futures_last_prices(positions: list, client, config) -> dict[str, float]:
    """Live LTP for the futures leg of each open combo (NFO segment) --
    separate from ``_last_prices`` above, which fetches equity/index quotes.
    Kite-only (no Yahoo fallback for NFO); returns {} if no active Kite
    session, matching this dashboard's other best-effort quote fetches."""
    if not positions or not config.kite_api_key:
        return {}
    repository = TursoKiteSessionRepository(client)
    await repository.ensure_schema()
    token_row = await repository.get_token()
    if token_row is None:
        return {}
    access_token, _obtained_at = token_row
    kite = KiteConnect(api_key=config.kite_api_key)
    kite.set_access_token(access_token)
    keys = [f"NFO:{p.futures_tradingsymbol}" for p in positions]

    def _fetch() -> dict:
        try:
            return kite.ltp(keys)
        except Exception:
            return {}

    data = await asyncio.to_thread(_fetch)
    return {
        p.symbol: data[f"NFO:{p.futures_tradingsymbol}"]["last_price"]
        for p in positions
        if f"NFO:{p.futures_tradingsymbol}" in data
    }


def _futures_unrealized_pnl(position, last_prices: dict[str, float]) -> dict:
    current_price = last_prices.get(position.symbol)
    if current_price is None:
        return {"current_price": None, "unrealized_pnl": None, "unrealized_pnl_pct": None}
    current = Decimal(str(current_price))
    pnl = (
        (current - position.futures_entry_price) * position.lot_size
        if position.side == "long"
        else (position.futures_entry_price - current) * position.lot_size
    )
    return {
        "current_price": current_price,
        "unrealized_pnl": _decimal(pnl),
        "unrealized_pnl_pct": _decimal(pnl / position.margin_allocated * 100),
    }


def _futures_monthly_summary(open_positions: list, closed_positions: list, window_days: int = 30) -> dict:
    """Trailing-window (default 30 days) performance summary for the
    futures paper account -- opened/closed counts, win rate, total P&L,
    average margin per trade.

    2026-08-17: built at the user's explicit request to track a full
    month of real paper performance before deciding whether to fund this
    for real trading next month -- the dashboard only ever showed
    "right now" state (open positions, last 50 closed) with no rollup a
    non-technical read could use to decide "was this month good enough."

    ``trades_opened`` counts by entry_timestamp (still-open or already
    closed, whichever) so a trade that opens AND closes inside the window
    counts once, not twice. ``trades_closed``/win rate/P&L are scoped by
    exit_timestamp instead, since that's when a trade's outcome is
    actually known -- a trade opened just before the window and closed
    inside it counts toward the outcome stats even though its own entry
    falls outside ``trades_opened``'s count, which is intentional: it's
    real P&L realized inside this window either way.
    """
    window_start = datetime.now(UTC) - timedelta(days=window_days)

    def _aware(ts: datetime) -> datetime:
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)

    opened = [
        p
        for p in (list(open_positions) + list(closed_positions))
        if _aware(p.entry_timestamp) >= window_start
    ]
    closed_in_window = [
        p
        for p in closed_positions
        if p.exit_timestamp is not None and _aware(p.exit_timestamp) >= window_start
    ]
    wins = [p for p in closed_in_window if p.pnl_amount is not None and p.pnl_amount > 0]
    total_pnl = sum(
        (p.pnl_amount for p in closed_in_window if p.pnl_amount is not None), start=Decimal("0")
    )
    total_margin_opened = sum((p.margin_allocated for p in opened), start=Decimal("0"))
    return {
        "window_days": window_days,
        "window_start": window_start.isoformat(),
        "trades_opened": len(opened),
        "trades_closed": len(closed_in_window),
        "trades_still_open": sum(1 for p in opened if p.status == "open"),
        "wins": len(wins),
        "losses": len(closed_in_window) - len(wins),
        "win_rate_pct": (
            _decimal(Decimal(100 * len(wins)) / len(closed_in_window)) if closed_in_window else None
        ),
        "total_pnl": _decimal(total_pnl),
        "avg_margin_per_trade": (
            _decimal(total_margin_opened / len(opened)) if opened else None
        ),
    }


@app.get("/api/futures-paper")
async def futures_paper(_: None = Depends(_require_session)) -> JSONResponse:
    """The real, capital-gated Nifty50 futures paper account (see
    application/futures_trading.py) -- separate book from the cash paper
    account above, own margin-based capital pool, own eligibility track
    record. Not shadow-tracking (that's /api/derivatives-shadow, uncapped,
    every symbol); this is only the trades that actually cleared the
    55%-win-rate bar and a real Kite margin check."""
    client, config = _client()
    try:
        repository = TursoFuturesPaperAccountRepository(client, futures_trading.FUTURES_INITIAL_CAPITAL)
        await repository.ensure_schema()
        cash_balance = await repository.get_cash_balance()
        open_positions = list(await repository.get_open_positions())
        # 500, not 50 -- large enough to cover several months at current
        # real trade volume (see _futures_monthly_summary, which needs
        # every closed trade inside the trailing window, not just the
        # most recent handful the plain "recent closed" list below shows).
        recent_closed = list(await repository.get_recent_closed_positions(500))
        last_prices = await _futures_last_prices(open_positions, client, config)
        total_margin_allocated = sum(
            (p.margin_allocated for p in open_positions), start=Decimal("0")
        )
        return JSONResponse(
            {
                "cash_balance": _decimal(cash_balance),
                "total_equity": _decimal(cash_balance + total_margin_allocated),
                "monthly_summary": _futures_monthly_summary(open_positions, recent_closed),
                "open_positions": [
                    {
                        "symbol": p.symbol,
                        "side": p.side,
                        "entry_timestamp": p.entry_timestamp.isoformat(),
                        "futures_entry_price": _decimal(p.futures_entry_price),
                        "futures_tradingsymbol": p.futures_tradingsymbol,
                        "hedge_tradingsymbol": p.hedge_tradingsymbol,
                        "lot_size": p.lot_size,
                        "margin_allocated": _decimal(p.margin_allocated),
                        **_futures_unrealized_pnl(p, last_prices),
                    }
                    for p in sorted(open_positions, key=lambda p: p.entry_timestamp, reverse=True)
                ],
                "recent_closed": [
                    {
                        "symbol": p.symbol,
                        "side": p.side,
                        "entry_timestamp": p.entry_timestamp.isoformat(),
                        "futures_entry_price": _decimal(p.futures_entry_price),
                        "exit_timestamp": p.exit_timestamp.isoformat() if p.exit_timestamp else None,
                        "futures_exit_price": _decimal(p.futures_exit_price),
                        "margin_allocated": _decimal(p.margin_allocated),
                        "pnl_amount": _decimal(p.pnl_amount),
                    }
                    # Display list stays capped at 50 like before -- the
                    # full 500-row fetch above is only for the monthly
                    # summary's window math, not meant to render as a table.
                    for p in recent_closed[:50]
                ],
            }
        )
    finally:
        await client.close()


def _smallcap_client():
    """Separate client for the skytrade-smallcap fork's own local SQLite
    file -- not the main app's Turso database. Returns None if the fork
    hasn't been deployed yet (e.g. local dev), so these endpoints degrade
    to an empty/absent state instead of throwing."""
    if not Path(_SMALLCAP_DB_PATH).exists():
        return None
    return create_turso_client(f"file:{_SMALLCAP_DB_PATH}", None)


@app.get("/api/smallcap-status")
async def smallcap_status(_: None = Depends(_require_session)) -> JSONResponse:
    """Nifty Smallcap 250 weekly-signal paper account (skytrade-smallcap
    fork, see application/signal_pipeline.py's docstring there) -- its own
    Rs 5L/10-slot capital pool, entirely separate from the cash and futures
    books above."""
    client = _smallcap_client()
    if client is None:
        return JSONResponse({"deployed": False})
    try:
        repository = TursoPaperAccountRepository(client, _SMALLCAP_INITIAL_CAPITAL)
        await repository.ensure_schema()
        cash_balance = await repository.get_cash_balance()
        open_positions = list(await repository.get_open_positions())
        total_equity = cash_balance + sum(
            (p.capital_allocated for p in open_positions), start=Decimal("0")
        )
        position_size = max(total_equity / _SMALLCAP_TARGET_SLOTS, _SMALLCAP_MIN_POSITION_SIZE)
        return JSONResponse(
            {
                "deployed": True,
                "cash_balance": _decimal(cash_balance),
                "total_equity": _decimal(total_equity),
                "open_position_count": len(open_positions),
                "target_slots": _SMALLCAP_TARGET_SLOTS,
                "current_slot_size": _decimal(position_size),
                "pnl_since_start": _decimal(total_equity - _SMALLCAP_INITIAL_CAPITAL),
                "pnl_since_start_pct": _decimal(
                    (total_equity - _SMALLCAP_INITIAL_CAPITAL) / _SMALLCAP_INITIAL_CAPITAL * 100
                ),
            }
        )
    finally:
        await client.close()


async def _smallcap_last_prices(symbols: list[str]) -> dict[str, float]:
    """Live quotes for smallcap symbols via the *parent* app's own Kite
    session -- same Zerodha account/api_key (see run_daily.sh's docstring),
    so no separate login is needed just to price a dashboard column. No
    Yahoo fallback here on purpose, unlike the main dashboard's
    ``_last_prices``: a day-old Yahoo close next to a smallcap table is more
    likely to mislead than a blank cell, and this is exactly the data-source
    boundary the 2026-08-13 cross-contamination bug taught us to keep
    strict about -- Kite or nothing, never silently something else."""
    if not symbols:
        return {}
    client, config = _client()
    try:
        if not config.kite_api_key:
            return {}
        repository = TursoKiteSessionRepository(client)
        await repository.ensure_schema()
        token_row = await repository.get_token()
        if token_row is None:
            return {}
        access_token, _obtained_at = token_row
        kite = KiteConnect(api_key=config.kite_api_key)
        kite.set_access_token(access_token)
        return await asyncio.to_thread(kite_get_last_prices, kite, symbols) or {}
    finally:
        await client.close()


@app.get("/api/smallcap-positions")
async def smallcap_positions(_: None = Depends(_require_session)) -> JSONResponse:
    client = _smallcap_client()
    if client is None:
        return JSONResponse([])
    try:
        repository = TursoPaperAccountRepository(client, _SMALLCAP_INITIAL_CAPITAL)
        open_positions = list(await repository.get_open_positions())
        symbols = [p.symbol for p in open_positions]
        last_prices = await _smallcap_last_prices(symbols)
        return JSONResponse(
            [
                {
                    "symbol": p.symbol,
                    "entry_timestamp": p.entry_timestamp.isoformat(),
                    "entry_price": _decimal(p.entry_price),
                    "quantity": p.quantity,
                    "capital_allocated": _decimal(p.capital_allocated),
                    **_unrealized_pnl(p, last_prices),
                }
                for p in sorted(open_positions, key=lambda p: p.entry_timestamp, reverse=True)
            ]
        )
    finally:
        await client.close()


@app.get("/api/smallcap-trades")
async def smallcap_trades(_: None = Depends(_require_session)) -> JSONResponse:
    """Full closed trade history (no recent-N cap, unlike the main /api/trades)
    -- this fork's total trade count is small enough (low hundreds, weekly
    cadence) that there's no need to hide anything behind a limit."""
    client = _smallcap_client()
    if client is None:
        return JSONResponse({"overall_win_rate": None, "closed_buy_count": 0, "recent": []})
    try:
        repository = TursoTradeRepository(client)
        all_trades = await repository.get_trades(None, "week")
        closed_buys = [t for t in all_trades if t.side.value == "buy" and t.status == "closed"]
        closed_all = [t for t in all_trades if t.status == "closed"]
        closed_all.sort(key=lambda t: t.exit_timestamp or t.entry_timestamp, reverse=True)
        wins = sum(1 for t in closed_buys if t.pnl_percent is not None and t.pnl_percent > 0)
        return JSONResponse(
            {
                "overall_win_rate": _decimal(
                    Decimal(100 * wins) / len(closed_buys) if closed_buys else None
                ),
                "closed_buy_count": len(closed_buys),
                "recent": [
                    {
                        "symbol": t.symbol,
                        "side": t.side.value,
                        "entry_timestamp": t.entry_timestamp.isoformat(),
                        "entry_price": _decimal(t.entry_price),
                        "exit_timestamp": t.exit_timestamp.isoformat() if t.exit_timestamp else None,
                        "exit_price": _decimal(t.exit_price),
                        "pnl_percent": _decimal(t.pnl_percent),
                    }
                    for t in closed_all
                ],
            }
        )
    finally:
        await client.close()


@app.get("/api/smallcap-open-signals")
async def smallcap_open_signals(_: None = Depends(_require_session)) -> JSONResponse:
    """Symbols the model currently has an open BUY signal on (entered, no
    SELL reversal yet) -- separate from /api/smallcap-positions, which only
    shows *real* paper positions. A symbol can show here with no matching
    real position if its entry happened before this fork went live (no
    retroactive catch-up, see application/signal_pipeline.py) -- this list
    is what lets you judge those cases and decide whether to enter by hand.
    SELL-side opens are never real trade candidates in the NSE cash market
    (see is_eligible's own BUY-only framing), so this only ever shows buys."""
    client = _smallcap_client()
    if client is None:
        return JSONResponse([])
    try:
        repository = TursoTradeRepository(client)
        all_trades = await repository.get_trades(None, "week")
        open_buys = [t for t in all_trades if t.side.value == "buy" and t.status == "open"]
        open_buys.sort(key=lambda t: t.entry_timestamp, reverse=True)
        symbols = [t.symbol for t in open_buys]
        last_prices = await _smallcap_last_prices(symbols)
        return JSONResponse(
            [
                {
                    "symbol": t.symbol,
                    "entry_timestamp": t.entry_timestamp.isoformat(),
                    "entry_price": _decimal(t.entry_price),
                    "current_price": last_prices.get(t.symbol),
                    "price_change_pct": (
                        _decimal(
                            (Decimal(str(last_prices[t.symbol])) - t.entry_price)
                            / t.entry_price
                            * 100
                        )
                        if t.symbol in last_prices
                        else None
                    ),
                }
                for t in open_buys
            ]
        )
    finally:
        await client.close()


@app.get("/api/symbols")
async def symbols(_: None = Depends(_require_session)) -> JSONResponse:
    """Symbols with at least one closed BUY trade, for the dashboard's filter
    dropdown -- alongside each one's own win rate/trade count so the dropdown
    can show something useful without a second round trip per symbol."""
    client, config = _client()
    try:
        repository = TursoTradeRepository(client)
        all_trades = await repository.get_trades(None, config.candle_interval)
        by_symbol: dict[str, list] = {}
        for t in all_trades:
            if t.side.value == "buy" and t.status == "closed":
                by_symbol.setdefault(t.symbol, []).append(t)
        rows = []
        for symbol, trades_ in sorted(by_symbol.items()):
            wins = sum(1 for t in trades_ if t.pnl_percent is not None and t.pnl_percent > 0)
            rows.append(
                {
                    "symbol": symbol,
                    "trade_count": len(trades_),
                    "win_rate": _decimal(Decimal(100 * wins) / len(trades_)),
                }
            )
        return JSONResponse(rows)
    finally:
        await client.close()


@app.get("/api/trades")
async def trades(
    limit: int = 50, symbol: str | None = None, _: None = Depends(_require_session)
) -> JSONResponse:
    client, config = _client()
    try:
        repository = TursoTradeRepository(client)
        all_trades = await repository.get_trades(symbol, config.candle_interval)
        # Win rate stays BUY-only -- matches the paper-trading eligibility
        # gate exactly (see application/paper_trading.py). SELL trades are
        # never tradeable in the NSE cash market, so folding them into this
        # number would make it not match what eligibility actually uses.
        closed_buys = [
            t for t in all_trades if t.side.value == "buy" and t.status == "closed"
        ]
        # The visible table, though, shows both sides -- SELL rows are real
        # backtest results too (see application/backtest.py), just never
        # became real/paper positions; hiding them was the actual bug.
        closed_all = [t for t in all_trades if t.status == "closed"]
        closed_all.sort(key=lambda t: t.exit_timestamp or t.entry_timestamp, reverse=True)
        # A symbol-filtered view shows its full backtested history rather
        # than just the dashboard's default recent-N window.
        recent = closed_all if symbol else closed_all[:limit]
        wins = sum(1 for t in closed_buys if t.pnl_percent is not None and t.pnl_percent > 0)
        return JSONResponse(
            {
                "overall_win_rate": _decimal(
                    Decimal(100 * wins) / len(closed_buys) if closed_buys else None
                ),
                "closed_buy_count": len(closed_buys),
                "recent": [
                    {
                        "symbol": t.symbol,
                        "side": t.side.value,
                        "entry_timestamp": t.entry_timestamp.isoformat(),
                        "entry_price": _decimal(t.entry_price),
                        "exit_timestamp": t.exit_timestamp.isoformat() if t.exit_timestamp else None,
                        "exit_price": _decimal(t.exit_price),
                        "pnl_percent": _decimal(t.pnl_percent),
                    }
                    for t in recent
                ],
            }
        )
    finally:
        await client.close()


def _moneyness(option_type: str, strike: Decimal, underlying_price: Decimal) -> str:
    """ATM/ITM/OTM label for display -- the strike chosen is always the
    *nearest* one to the underlying's price at entry (see
    ``KiteDerivativesChain.nearest_atm_option``), so it's close to ATM by
    construction, but listed strikes are spaced in fixed increments (e.g.
    every 50 or 100 rupees), so the nearest one can still land a step into
    ITM or OTM territory -- this makes that visible instead of implying
    every trade is exactly at-the-money."""
    if strike == underlying_price:
        return "ATM"
    if option_type == "CE":
        return "ITM" if strike < underlying_price else "OTM"
    return "ITM" if strike > underlying_price else "OTM"


def _options_greeks_payload(trade) -> dict | None:
    """Implied volatility + delta/theta/gamma/vega at entry (and exit, if
    closed) for one options trade -- see ``application/
    options_analytics.py``. None if the underlying computation couldn't
    resolve (e.g. a stale/implausible stored premium) -- the row still
    renders, just without this extra detail, rather than breaking the
    whole derivatives tab over one bad historical row.
    """
    try:
        result = enrich_trade(
            trade.option_type,
            trade.strike,
            trade.expiry,
            trade.entry_timestamp,
            trade.underlying_price_at_entry,
            trade.entry_premium,
            trade.exit_timestamp,
            trade.underlying_price_at_exit,
            trade.exit_premium,
        )
    except Exception:
        logging.getLogger(__name__).warning(
            "Greeks computation failed for %s -- omitting from this row.",
            trade.option_tradingsymbol, exc_info=True,
        )
        return None
    return result


def _derivatives_summary_payload(options_trades: list, futures_trades: list) -> dict:
    """Shared by the live shadow-tracking endpoint and the current-month
    backtest endpoint -- same shape, different ``source`` filter upstream.

    One leg per signal (see ``application/signal_pipeline.py``'s
    ``_open_derivatives_shadow``): a futures position hedged by an option
    at the opposite delta (``primary_futures`` + ``hedge_options``).
    ``directional_options`` and ``hedge_futures`` are legacy/always empty
    going forward -- earlier versions of this feature also tracked a naked
    directional option (dropped after review) and, briefly, hedged that
    option with a future too (a mistake, reverted). Kept here only so any
    rows already written under those schemes don't silently vanish from the
    API shape.
    """

    def _options_summary(purpose: str) -> dict:
        closed = [t for t in options_trades if t.purpose == purpose and t.status == "closed"]
        wins = sum(1 for t in closed if t.pnl_percent is not None and t.pnl_percent > 0)
        return {
            "closed_count": len(closed),
            "win_rate": _decimal(Decimal(100 * wins) / len(closed)) if closed else None,
            "total_pnl": _decimal(sum((t.pnl_amount or Decimal("0") for t in closed), Decimal("0"))),
        }

    def _futures_summary(purpose: str) -> dict:
        closed = [t for t in futures_trades if t.purpose == purpose and t.status == "closed"]
        wins = sum(1 for t in closed if t.pnl_percent is not None and t.pnl_percent > 0)
        return {
            "closed_count": len(closed),
            "win_rate": _decimal(Decimal(100 * wins) / len(closed)) if closed else None,
            "total_pnl": _decimal(sum((t.pnl_amount or Decimal("0") for t in closed), Decimal("0"))),
        }

    return {
        "directional_options": _options_summary("directional"),
        "hedge_futures": _futures_summary("hedge"),
        "primary_futures": _futures_summary("primary"),
        "hedge_options": _options_summary("hedge"),
        "recent_options": [
            {
                "symbol": t.symbol,
                "option_type": t.option_type,
                "purpose": t.purpose,
                "tradingsymbol": t.option_tradingsymbol,
                "entry_timestamp": t.entry_timestamp.isoformat(),
                "underlying_price_at_entry": _decimal(t.underlying_price_at_entry),
                "moneyness": _moneyness(t.option_type, t.strike, t.underlying_price_at_entry),
                "strike": _decimal(t.strike),
                "entry_premium": _decimal(t.entry_premium),
                "exit_timestamp": t.exit_timestamp.isoformat() if t.exit_timestamp else None,
                "exit_premium": _decimal(t.exit_premium),
                "lot_size": t.lot_size,
                "pnl_percent": _decimal(t.pnl_percent),
                "pnl_amount": _decimal(t.pnl_amount),
                "status": t.status,
                "greeks": _options_greeks_payload(t),
            }
            for t in sorted(options_trades, key=lambda t: t.entry_timestamp, reverse=True)[:30]
        ],
        "recent_futures": [
            {
                "symbol": t.symbol,
                "side": t.side,
                "purpose": t.purpose,
                "tradingsymbol": t.futures_tradingsymbol,
                "entry_timestamp": t.entry_timestamp.isoformat(),
                "entry_price": _decimal(t.entry_price),
                "exit_timestamp": t.exit_timestamp.isoformat() if t.exit_timestamp else None,
                "exit_price": _decimal(t.exit_price),
                "lot_size": t.lot_size,
                "pnl_percent": _decimal(t.pnl_percent),
                "pnl_amount": _decimal(t.pnl_amount),
                "status": t.status,
            }
            for t in sorted(futures_trades, key=lambda t: t.entry_timestamp, reverse=True)[:30]
        ],
    }


@app.get("/api/derivatives-shadow")
async def derivatives_shadow(symbol: str | None = None, _: None = Depends(_require_session)) -> JSONResponse:
    """Live forward shadow-tracking summary (source='live') -- analysis
    only, never a real order (see application/options_shadow.py,
    futures_shadow.py). Current-month backtest results live separately at
    /api/derivatives-backtest so they never dilute this live win rate."""
    client, _config = _client()
    try:
        options_repository = TursoOptionsTradeRepository(client)
        futures_repository = TursoFuturesTradeRepository(client)
        options_trades = await options_repository.get_trades(symbol, source="live")
        futures_trades = await futures_repository.get_trades(symbol, source="live")
        return JSONResponse(_derivatives_summary_payload(options_trades, futures_trades))
    finally:
        await client.close()


@app.get("/api/derivatives-backtest")
async def derivatives_backtest(
    symbol: str | None = None, _: None = Depends(_require_session)
) -> JSONResponse:
    """Current-month options/futures backtest summary (source='backtest')
    -- see application/derivatives_backtest.py. Trigger a run via POST
    /api/trigger-backtest."""
    client, _config = _client()
    try:
        options_repository = TursoOptionsTradeRepository(client)
        futures_repository = TursoFuturesTradeRepository(client)
        options_trades = await options_repository.get_trades(symbol, source="backtest")
        futures_trades = await futures_repository.get_trades(symbol, source="backtest")
        return JSONResponse(_derivatives_summary_payload(options_trades, futures_trades))
    finally:
        await client.close()


@app.get("/api/margin-benefit")
async def margin_benefit(
    symbol: str, _: None = Depends(_require_admin)
) -> JSONResponse:
    """Live margin required for the symbol's open futures position + its
    hedge option vs. holding the future alone, using Kite's own
    basket-margin API -- not a guessed percentage (see
    ``KiteDerivativesChain.margin_benefit``'s docstring for why the naive
    version of this got the wrong answer at first). Requires an active
    Kite session and an open primary-future position for the symbol."""
    config = load_config()
    if not config.kite_api_key:
        raise HTTPException(status_code=400, detail="Kite is not configured.")
    client = create_turso_client(config.turso_database_url, config.turso_auth_token)
    try:
        kite_session_repository = TursoKiteSessionRepository(client)
        await kite_session_repository.ensure_schema()
        token_row = await kite_session_repository.get_token()
        if token_row is None:
            raise HTTPException(status_code=400, detail="No active Kite session.")
        access_token, _obtained_at = token_row

        futures_repository = TursoFuturesTradeRepository(client)
        options_repository = TursoOptionsTradeRepository(client)
        await futures_repository.ensure_schema()
        await options_repository.ensure_schema()
        primary_future = await futures_repository.get_open_trade(symbol, purpose="primary")
        if primary_future is None:
            raise HTTPException(
                status_code=404, detail=f"No open primary futures position for {symbol}."
            )
        hedge_option_type = "PE" if primary_future.side == "long" else "CE"
        hedge_option = await options_repository.get_open_trade(symbol, hedge_option_type, "hedge")

        kite = KiteConnect(api_key=config.kite_api_key)
        kite.set_access_token(access_token)
        derivatives_chain = KiteDerivativesChain(kite)
        legs = [(primary_future.futures_tradingsymbol, "BUY", primary_future.lot_size)]
        if hedge_option is not None:
            legs.append((hedge_option.option_tradingsymbol, "BUY", hedge_option.lot_size))
        result = await asyncio.to_thread(derivatives_chain.margin_benefit, legs)
        if result is None:
            raise HTTPException(status_code=502, detail="Kite margin lookup failed.")
        return JSONResponse(
            {
                "symbol": symbol,
                "has_hedge": hedge_option is not None,
                **result,
            }
        )
    finally:
        await client.close()


class ConfigUpdate(BaseModel):
    capital: str | None = None
    slots: str | None = None
    min_position: str | None = None


@app.get("/api/config")
async def get_config(_: None = Depends(_require_admin)) -> JSONResponse:
    return JSONResponse(
        {
            "capital": str(paper_trading.INITIAL_CAPITAL),
            "slots": paper_trading.TARGET_SLOTS,
            "min_position": str(paper_trading.MIN_POSITION_SIZE),
        }
    )


@app.post("/api/config")
async def update_config(update: ConfigUpdate, _: None = Depends(_require_admin)) -> JSONResponse:
    """Rewrite the relevant lines in .env. Takes effect on the *next* pipeline
    run/dashboard restart -- this process's own already-imported constants
    are not changed live, since paper_trading.py reads them once at import."""
    fields = {
        "capital": "TRADING_SCANNER_PAPER_CAPITAL",
        "slots": "TRADING_SCANNER_PAPER_SLOTS",
        "min_position": "TRADING_SCANNER_PAPER_MIN_POSITION",
    }
    updates: dict[str, str] = {}
    for field, env_key in fields.items():
        value = getattr(update, field)
        if value is None:
            continue
        try:
            Decimal(value)
        except InvalidOperation as error:
            raise HTTPException(status_code=400, detail=f"{field} must be numeric.") from error
        updates[env_key] = value
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update.")
    _write_env_updates(updates)
    return JSONResponse({"updated": updates, "note": "Takes effect on the next pipeline run."})


class LiveCashTradingUpdate(BaseModel):
    enabled: bool
    symbols: list[str]
    notional: str
    max_positions: int = 8


@app.get("/api/live-cash-trading")
async def get_live_cash_trading(_: None = Depends(_require_admin)) -> JSONResponse:
    """Current "Go Live" state -- DB-backed (not .env), so this reflects
    exactly what the running live_pipeline.py will see next cycle. See
    infrastructure/db/live_cash_toggle.py."""
    client, config = _client()
    try:
        repository = TursoLiveCashToggleRepository(client)
        await repository.ensure_schema()
        state = await repository.get_state(
            LiveCashToggleState(
                enabled=config.live_cash_trading_enabled,
                symbols=config.live_cash_trading_symbols,
                notional=config.live_cash_trading_notional,
                max_positions=config.live_cash_trading_max_positions,
            )
        )
        universe_path = _REPO_ROOT / config.symbols_file
        universe_symbols = (
            sorted(
                line.strip()
                for line in universe_path.read_text().splitlines()
                # symbols_file also carries sector index tickers (e.g.
                # ^NSEBANK, ^CNXAUTO) for breadth/relative-strength
                # tracking -- not real equities, can't be bought as cash
                # shares on Kite. Excluded from what "Use full universe"
                # offers for real-money trading.
                if line.strip() and not line.strip().startswith("^")
            )
            if universe_path.exists()
            else []
        )
        return JSONResponse(
            {
                "enabled": state.enabled,
                "symbols": sorted(state.symbols),
                "notional": str(state.notional),
                "max_positions": state.max_positions,
                "updated_at": state.updated_at.isoformat() if state.updated_at else None,
                "todays_error_count": _todays_error_count(),
                "universe_symbols": universe_symbols,
            }
        )
    finally:
        await client.close()


@app.post("/api/live-cash-trading")
async def update_live_cash_trading(
    update: LiveCashTradingUpdate, _: None = Depends(_require_admin)
) -> JSONResponse:
    """Flips the real cash-order kill switch -- takes effect on the very
    next scan cycle of the already-running live_pipeline.py, no restart.
    Real money moves once ``enabled`` is true and ``symbols`` is non-empty
    -- see application/live_cash_execution.py / gtt_bracket.py."""
    try:
        notional = Decimal(update.notional)
    except InvalidOperation as error:
        raise HTTPException(status_code=400, detail="notional must be numeric.") from error
    if notional <= 0:
        raise HTTPException(status_code=400, detail="notional must be positive.")
    if update.max_positions <= 0:
        raise HTTPException(status_code=400, detail="max_positions must be positive.")
    symbols = frozenset(s.strip() for s in update.symbols if s.strip())
    if update.enabled and not symbols:
        raise HTTPException(
            status_code=400, detail="Cannot enable live trading with an empty symbol list."
        )

    client, _config = _client()
    try:
        repository = TursoLiveCashToggleRepository(client)
        await repository.ensure_schema()
        new_state = LiveCashToggleState(
            enabled=update.enabled,
            symbols=symbols,
            notional=notional,
            max_positions=update.max_positions,
        )
        await repository.set_state(new_state)
        return JSONResponse(
            {
                "enabled": new_state.enabled,
                "symbols": sorted(new_state.symbols),
                "notional": str(new_state.notional),
                "max_positions": new_state.max_positions,
                "note": "Takes effect on the next scan cycle -- no restart needed.",
            }
        )
    finally:
        await client.close()


@app.post("/api/trigger")
async def trigger(_: None = Depends(_require_admin)) -> JSONResponse:
    """Kick off one manual pipeline run in the background, same command cron uses."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = _LOG_PATH.open("a")
    subprocess.Popen(
        [sys.executable, "-m", "trading_scanner.signals"],
        cwd=str(_REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")},
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )
    return JSONResponse({"triggered": True})


@app.post("/api/trigger-backtest")
async def trigger_backtest(_: None = Depends(_require_admin)) -> JSONResponse:
    """Kick off one manual current-month derivatives backtest in the
    background (see application/derivatives_backtest.py). Requires an
    active Kite session (uses historical data, not live LTP)."""
    _BACKTEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = _BACKTEST_LOG_PATH.open("a")
    subprocess.Popen(
        [sys.executable, "-m", "trading_scanner.derivatives_backtest_cli"],
        cwd=str(_REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")},
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )
    return JSONResponse({"triggered": True})


@app.get("/api/logs")
async def logs(lines: int = 200, _: None = Depends(_require_session)) -> JSONResponse:
    if not _LOG_PATH.exists():
        return JSONResponse({"lines": []})
    tail = subprocess.run(
        ["tail", "-n", str(lines), str(_LOG_PATH)], capture_output=True, text=True
    )
    return JSONResponse({"lines": tail.stdout.splitlines()})


def _last_run_summary() -> dict | None:
    """Most recent pipeline-activity log line, classified into a status.

    2026-08-26: this used to grep only for "Signal pipeline (started|
    finished)" -- the legacy hourly-cron path's own log lines
    (``signals.py``). Since ``live_pipeline.py`` (the always-on WebSocket
    runner) replaced that during market hours, it never logs those exact
    words, so this silently found nothing recent and kept returning
    whatever the hourly cron last logged days earlier -- a stale "last run"
    on the dashboard even while the live pipeline was actively running.
    Now recognizes the live pipeline's own lines too, so recency reflects
    what's actually running.
    """
    if not _LOG_PATH.exists():
        return None
    pattern = (
        "Signal pipeline (started|finished)"
        "|Live ticker pipeline (starting|stopped)"
        "|Bucket .* closed"
    )
    result = subprocess.run(
        ["grep", "-nE", pattern, str(_LOG_PATH)],
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    last_line = lines[-1]
    if "finished" in last_line or "stopped" in last_line:
        status = "finished"
    elif "closed with no ticks" in last_line:
        status = "idle"
    elif "closed" in last_line:
        status = "processing"
    else:
        status = "started"
    return {
        "status": status,
        "raw": last_line.split(":", 1)[-1].strip() if ":" in last_line else last_line,
    }


def _todays_error_count() -> int:
    """Rough count of ERROR-level log lines from today (server-local date)
    -- shown next to the "Go Live" toggle so there's a quick sanity check
    of today's pipeline health before flipping real trading on. Not a
    precise error audit -- see /api/logs for the actual lines."""
    if not _LOG_PATH.exists():
        return 0
    today = date.today().isoformat()
    result = subprocess.run(
        ["grep", "-c", f"^{today}.*ERROR", str(_LOG_PATH)], capture_output=True, text=True
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


# 2026-08-27: the live pipeline logs at least once a minute even when idle
# ("Outside market hours -- sleeping"), so a genuinely healthy process is
# never quiet longer than ~60s. Matches vps_watchdog's own threshold (see
# that script) -- the cron-based watchdog force-restarts automatically at
# this same age; this just makes that same fact visible/actionable from
# the dashboard without waiting for cron's next 5-minute tick, for the
# specific failure mode that motivated both: a silent async-level hang
# where `systemctl status` keeps reporting the process as healthy while it
# has actually stopped doing anything (see the 2026-08-27 incident -- a
# Kite WebSocket 403 during a reconnect retry, then zero further log lines
# for 7+ hours).
_PIPELINE_STALE_SECONDS = 300


def _live_pipeline_health() -> dict:
    if not _LOG_PATH.exists():
        return {"healthy": False, "age_seconds": None, "last_log_at": None}
    last_modified = _LOG_PATH.stat().st_mtime
    age_seconds = int(time.time() - last_modified)
    return {
        "healthy": age_seconds < _PIPELINE_STALE_SECONDS,
        "age_seconds": age_seconds,
        "last_log_at": datetime.fromtimestamp(last_modified, tz=UTC).isoformat(),
    }


@app.get("/api/live-pipeline-health")
async def live_pipeline_health(_: None = Depends(_require_session)) -> JSONResponse:
    return JSONResponse(_live_pipeline_health())


@app.get("/api/reconciliation-status")
async def reconciliation_status(_: None = Depends(_require_session)) -> JSONResponse:
    """Symbols whose real cash position currently needs broker ground
    truth to resolve (Phase 15, observability -- see application/broker_
    reconciliation.py). Read-only: never places an order or otherwise
    acts on what it finds -- surfacing this is the whole point, since an
    UNKNOWN-status leg used to be invisible everywhere until a strategy
    exit or manual action happened to resolve it (see that module's own
    docstring for the incident)."""
    client, _config = _client()
    try:
        live_order_repository = TursoLiveOrderRepository(client)
        await live_order_repository.ensure_schema()
        flagged = await broker_reconciliation.get_reconciliation_required_symbols(
            live_order_repository
        )
        return JSONResponse({"reconciliation_required": flagged})
    finally:
        await client.close()


@app.post("/api/live-pipeline-health/restart")
async def restart_live_pipeline(_: None = Depends(_require_admin)) -> JSONResponse:
    """Force-restart the live pipeline service -- the same action the
    external cron watchdog takes automatically, exposed here so a stuck
    pipeline (or one about to look stuck) can be restarted immediately
    from the dashboard instead of waiting for cron's next check or
    reaching for SSH. Real positions are never at risk from this: their
    GTT brackets live at the broker, independent of this service being up.
    """
    result = subprocess.run(
        ["systemctl", "restart", "p-trade-live"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise HTTPException(
            status_code=500, detail=f"Restart failed: {result.stderr.strip() or 'unknown error'}"
        )
    return JSONResponse({"ok": True})


def _write_env_updates(updates: dict[str, str]) -> None:
    existing_lines = _ENV_PATH.read_text(encoding="utf-8").splitlines() if _ENV_PATH.exists() else []
    remaining = dict(updates)
    new_lines: list[str] = []
    for line in existing_lines:
        key = line.split("=", 1)[0] if "=" in line else None
        if key in remaining:
            new_lines.append(f"{key}={remaining.pop(key)}")
        else:
            new_lines.append(line)
    for key, value in remaining.items():
        new_lines.append(f"{key}={value}")
    _ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_PAGE = (_TEMPLATES_DIR / "dashboard.html").read_text(encoding="utf-8")

_LOGIN_PAGE = (_TEMPLATES_DIR / "login.html").read_text(encoding="utf-8")


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("TRADING_SCANNER_DASHBOARD_PORT", "8000")))


if __name__ == "__main__":
    main()
