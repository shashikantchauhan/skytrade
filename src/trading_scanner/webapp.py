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
from decimal import Decimal, InvalidOperation
from pathlib import Path

from fastapi import Cookie, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from trading_scanner.application import paper_trading
from trading_scanner.config.settings import load_config
from trading_scanner.domain.models import PaperPosition
from trading_scanner.infrastructure.kite import build_login_url, exchange_request_token
from trading_scanner.infrastructure.turso import (
    TursoKiteSessionRepository,
    TursoPaperAccountRepository,
    TursoTradeRepository,
    create_turso_client,
)
from trading_scanner.infrastructure.yahoo import YahooProvider

_yahoo = YahooProvider()

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _REPO_ROOT / ".env"
_LOG_PATH = Path(os.getenv("TRADING_SCANNER_LOG_PATH", "/var/log/p-trade/signals.log"))
_SESSION_COOKIE = "ptrade_session"
_SESSION_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days

app = FastAPI(title="SkyTrade dashboard")

# token -> expiry (unix time). In-memory: fine for a single-process personal
# tool; a restart just means logging in again.
_sessions: dict[str, float] = {}


def _dashboard_password() -> str:
    password = os.getenv("TRADING_SCANNER_DASHBOARD_PASSWORD")
    if not password:
        raise HTTPException(
            status_code=500,
            detail="TRADING_SCANNER_DASHBOARD_PASSWORD is not set on the server.",
        )
    return password


def _require_session(ptrade_session: str | None = Cookie(default=None)) -> None:
    """API-route auth: 401 JSON if the session cookie is missing/expired."""
    expiry = _sessions.get(ptrade_session or "")
    if expiry is None or expiry < time.time():
        raise HTTPException(status_code=401, detail="Not logged in.")


def _client():
    config = load_config()
    if not config.turso_database_url:
        raise HTTPException(status_code=500, detail="TRADING_SCANNER_TURSO_URL is not set.")
    return create_turso_client(config.turso_database_url, config.turso_auth_token), config


def _decimal(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


async def _last_prices(positions: list[PaperPosition]) -> dict[str, float]:
    """Fetch current market prices for open positions' symbols (blocking
    yfinance call, so run off the event loop). Best-effort -- a symbol
    Yahoo can't currently price is simply left out by ``get_last_prices``."""
    symbols = [p.symbol for p in positions]
    if not symbols:
        return {}
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
    expiry = _sessions.get(request.cookies.get(_SESSION_COOKIE, ""))
    if expiry is None or expiry < time.time():
        return RedirectResponse("/login")
    return HTMLResponse(_PAGE)


@app.get("/login", response_class=HTMLResponse)
async def login_page() -> str:
    return _LOGIN_PAGE


class LoginRequest(BaseModel):
    password: str


@app.post("/login")
async def login(body: LoginRequest) -> JSONResponse:
    if not secrets.compare_digest(body.password, _dashboard_password()):
        raise HTTPException(status_code=401, detail="Wrong password.")
    token = secrets.token_urlsafe(32)
    _sessions[token] = time.time() + _SESSION_TTL_SECONDS
    response = JSONResponse({"ok": True})
    response.set_cookie(
        _SESSION_COOKIE,
        token,
        max_age=_SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
    )
    return response


@app.post("/logout")
async def logout(ptrade_session: str | None = Cookie(default=None)) -> JSONResponse:
    _sessions.pop(ptrade_session or "", None)
    response = JSONResponse({"ok": True})
    response.delete_cookie(_SESSION_COOKIE)
    return response


@app.get("/kite/login")
async def kite_login(_: None = Depends(_require_session)) -> RedirectResponse:
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
async def kite_status(_: None = Depends(_require_session)) -> JSONResponse:
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


@app.get("/api/status")
async def status(_: None = Depends(_require_session)) -> JSONResponse:
    client, config = _client()
    try:
        repository = TursoPaperAccountRepository(client, paper_trading.INITIAL_CAPITAL)
        cash_balance = await repository.get_cash_balance()
        open_positions = await repository.get_open_positions()
        total_equity = cash_balance + sum(
            (p.capital_allocated for p in open_positions), start=Decimal("0")
        )
        position_size = max(total_equity / paper_trading.TARGET_SLOTS, paper_trading.MIN_POSITION_SIZE)

        last_prices = await _last_prices(list(open_positions))
        unrealized_total = sum(
            (
                (Decimal(str(last_prices[p.symbol])) - p.entry_price) * p.quantity
                for p in open_positions
                if p.symbol in last_prices
            ),
            start=Decimal("0"),
        )
        priced_count = sum(1 for p in open_positions if p.symbol in last_prices)
        total_equity_mtm = total_equity + unrealized_total

        return JSONResponse(
            {
                "cash_balance": _decimal(cash_balance),
                "total_equity": _decimal(total_equity),
                "open_position_count": len(open_positions),
                "target_slots": paper_trading.TARGET_SLOTS,
                "current_slot_size": _decimal(position_size),
                "pnl_since_start": _decimal(total_equity - paper_trading.INITIAL_CAPITAL),
                "pnl_since_start_pct": _decimal(
                    (total_equity - paper_trading.INITIAL_CAPITAL)
                    / paper_trading.INITIAL_CAPITAL
                    * 100
                ),
                # Mark-to-market: total_equity above only reflects capital
                # committed at entry, not what open positions are worth right
                # now. unrealized_pnl is None if no live price could be
                # fetched for any open symbol (market closed, Yahoo hiccup).
                "unrealized_pnl": _decimal(unrealized_total) if priced_count else None,
                "total_equity_mtm": _decimal(total_equity_mtm) if priced_count else None,
                "pnl_since_start_mtm": (
                    _decimal(total_equity_mtm - paper_trading.INITIAL_CAPITAL)
                    if priced_count
                    else None
                ),
                "pnl_since_start_mtm_pct": (
                    _decimal(
                        (total_equity_mtm - paper_trading.INITIAL_CAPITAL)
                        / paper_trading.INITIAL_CAPITAL
                        * 100
                    )
                    if priced_count
                    else None
                ),
                "last_run": _last_run_summary(),
            }
        )
    finally:
        await client.close()


@app.get("/api/positions")
async def positions(_: None = Depends(_require_session)) -> JSONResponse:
    client, _config = _client()
    try:
        repository = TursoPaperAccountRepository(client, paper_trading.INITIAL_CAPITAL)
        open_positions = await repository.get_open_positions()
        last_prices = await _last_prices(list(open_positions))
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
        closed_buys = [
            t for t in all_trades if t.side.value == "buy" and t.status == "closed"
        ]
        closed_buys.sort(key=lambda t: t.exit_timestamp or t.entry_timestamp, reverse=True)
        # A symbol-filtered view shows its full backtested history rather
        # than just the dashboard's default recent-N window.
        recent = closed_buys if symbol else closed_buys[:limit]
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


class ConfigUpdate(BaseModel):
    capital: str | None = None
    slots: str | None = None
    min_position: str | None = None


@app.get("/api/config")
async def get_config(_: None = Depends(_require_session)) -> JSONResponse:
    return JSONResponse(
        {
            "capital": str(paper_trading.INITIAL_CAPITAL),
            "slots": paper_trading.TARGET_SLOTS,
            "min_position": str(paper_trading.MIN_POSITION_SIZE),
        }
    )


@app.post("/api/config")
async def update_config(update: ConfigUpdate, _: None = Depends(_require_session)) -> JSONResponse:
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


@app.post("/api/trigger")
async def trigger(_: None = Depends(_require_session)) -> JSONResponse:
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


@app.get("/api/logs")
async def logs(lines: int = 200, _: None = Depends(_require_session)) -> JSONResponse:
    if not _LOG_PATH.exists():
        return JSONResponse({"lines": []})
    tail = subprocess.run(
        ["tail", "-n", str(lines), str(_LOG_PATH)], capture_output=True, text=True
    )
    return JSONResponse({"lines": tail.stdout.splitlines()})


def _last_run_summary() -> dict | None:
    if not _LOG_PATH.exists():
        return None
    result = subprocess.run(
        ["grep", "-n", "Signal pipeline \\(started\\|finished\\)", str(_LOG_PATH)],
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    last_line = lines[-1]
    return {
        "status": "finished" if "finished" in last_line else "started",
        "raw": last_line.split(":", 1)[-1].strip() if ":" in last_line else last_line,
    }


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


_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SkyTrade dashboard</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #f5f6f8;
    --surface: #ffffff;
    --border: #e2e4e9;
    --text: #14161a;
    --muted: #6b7280;
    --accent: #2563eb;
    --accent-soft: #eaf0fe;
    --green: #0f9d58;
    --green-soft: #e6f7ee;
    --red: #d93025;
    --red-soft: #fdeceb;
    --shadow: 0 1px 2px rgba(16,24,40,0.04), 0 1px 3px rgba(16,24,40,0.06);
    --radius: 12px;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1115;
      --surface: #191b20;
      --border: #2b2e35;
      --text: #eceef1;
      --muted: #9aa1ac;
      --accent: #5b8def;
      --accent-soft: #1c2740;
      --green: #35c07a;
      --green-soft: #103524;
      --red: #f0685f;
      --red-soft: #3a1a19;
      --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 2px 6px rgba(0,0,0,0.25);
    }
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    max-width: 1080px;
    margin: 0 auto;
    padding: 1.5rem 1.25rem 4rem;
  }
  .topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.75rem; }
  .topbar h1 { font-size: 1.15rem; font-weight: 650; margin: 0; letter-spacing: -0.01em; }
  .topbar .sub { color: var(--muted); font-size: 0.8rem; margin-top: 0.15rem; }
  h2 { font-size: 0.95rem; font-weight: 650; margin: 0 0 0.75rem 0; letter-spacing: -0.01em; }
  section { margin-bottom: 1.75rem; }
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.2rem;
    box-shadow: var(--shadow);
  }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.7rem; }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.85rem 1rem;
    box-shadow: var(--shadow);
  }
  .card .label { font-size: 0.72rem; color: var(--muted); margin-bottom: 0.3rem; font-weight: 500; }
  .card .value { font-size: 1.25rem; font-weight: 650; letter-spacing: -0.01em; }
  .green { color: var(--green); } .red { color: var(--red); }
  .pill { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.72rem; font-weight: 600; }
  .pill.green { background: var(--green-soft); color: var(--green); }
  .pill.red { background: var(--red-soft); color: var(--red); }
  table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
  th {
    text-align: left; padding: 0.5rem 0.7rem; color: var(--muted); font-weight: 600;
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.03em;
    border-bottom: 1px solid var(--border);
  }
  td { padding: 0.55rem 0.7rem; border-bottom: 1px solid var(--border); }
  tbody tr:hover { background: var(--accent-soft); }
  tbody tr:last-child td { border-bottom: none; }
  .table-wrap { overflow-x: auto; }
  button {
    padding: 0.45rem 0.95rem; border-radius: 8px; border: 1px solid var(--border);
    cursor: pointer; background: var(--surface); color: var(--text); font-size: 0.85rem; font-weight: 550;
    transition: background 0.15s;
  }
  button:hover { background: var(--accent-soft); border-color: var(--accent); }
  button.primary { background: var(--accent); color: white; border-color: var(--accent); }
  button.primary:hover { opacity: 0.9; }
  input, select {
    padding: 0.4rem 0.6rem; border-radius: 8px; border: 1px solid var(--border);
    background: var(--surface); color: var(--text); font-size: 0.85rem;
  }
  input { width: 110px; }
  select { min-width: 220px; }
  label.field { display: flex; flex-direction: column; gap: 0.3rem; font-size: 0.72rem; color: var(--muted); font-weight: 500; }
  pre {
    background: var(--bg); border: 1px solid var(--border); padding: 0.75rem;
    border-radius: 10px; max-height: 300px; overflow: auto; font-size: 0.72rem; line-height: 1.5;
  }
  .row { display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; }
  .row.between { justify-content: space-between; }
  .msg { font-size: 0.8rem; color: var(--muted); }
</style>
</head>
<body>
<div class="topbar">
  <div>
    <h1>SkyTrade</h1>
    <div class="sub">NSE signal scanner &amp; paper trading dashboard</div>
  </div>
  <button onclick="logout()">Log out</button>
</div>

<section>
  <div class="cards" id="status-cards">Loading...</div>
</section>

<section>
  <div class="row">
    <button class="primary" onclick="trigger()">Run pipeline now</button>
    <span class="msg" id="trigger-msg"></span>
    <span class="row" id="kite-status" style="margin-left: auto;"></span>
  </div>
</section>

<section>
  <h2>Open positions</h2>
  <div class="panel table-wrap">
    <table id="positions-table"><thead><tr><th>Symbol</th><th>Entry</th><th>Qty</th><th>Capital</th><th>Current price</th><th>Unrealized P&amp;L</th></tr></thead><tbody></tbody></table>
  </div>
</section>

<section>
  <div class="row between">
    <h2><span id="trades-title">Closed trades (buy-only win rate</span>: <span id="win-rate">-</span>)</h2>
    <label class="field">Filter by symbol
      <select id="symbol-filter" onchange="loadTrades()">
        <option value="">All symbols (recent 30)</option>
      </select>
    </label>
  </div>
  <div class="panel table-wrap">
    <table id="trades-table"><thead><tr><th>Symbol</th><th>Entry</th><th>Exit</th><th>PnL%</th></tr></thead><tbody></tbody></table>
  </div>
</section>

<section>
  <h2>Config</h2>
  <div class="panel row">
    <label class="field">Capital<input id="cfg-capital" /></label>
    <label class="field">Slots<input id="cfg-slots" /></label>
    <label class="field">Min position<input id="cfg-min" /></label>
    <button class="primary" onclick="saveConfig()" style="align-self: flex-end;">Save</button>
    <span class="msg" id="config-msg"></span>
  </div>
</section>

<section>
  <h2>Recent logs</h2>
  <pre id="logs"></pre>
</section>

<script>
async function api(path, opts) {
  const res = await fetch(path, opts);
  if (res.status === 401) { window.location.href = "/login"; throw new Error("401"); }
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function logout() {
  await fetch("/logout", {method: "POST"});
  window.location.href = "/login";
}

function fmt(n) { return n === null || n === undefined ? "-" : Number(n).toLocaleString("en-IN", {maximumFractionDigits: 2}); }

// Every timestamp from the API is UTC ISO -- displayed in IST throughout
// this dashboard since that's the timezone the strategy/market actually
// operates in, to avoid the exact confusion UTC timestamps caused before.
function fmtTime(iso) {
  if (!iso) return "-";
  return new Date(iso).toLocaleString("en-IN", {
    timeZone: "Asia/Kolkata", day: "2-digit", month: "short",
    hour: "2-digit", minute: "2-digit", hour12: false,
  });
}

async function loadStatus() {
  const s = await api("/api/status");
  const pnlClass = s.pnl_since_start >= 0 ? "green" : "red";
  const haveMtm = s.total_equity_mtm !== null && s.total_equity_mtm !== undefined;
  const mtmClass = haveMtm && s.pnl_since_start_mtm >= 0 ? "green" : "red";
  const mtmCard = haveMtm
    ? `<div class="card"><div class="label">P&L (live, mark-to-market)</div><div class="value ${mtmClass}">₹${fmt(s.pnl_since_start_mtm)} (${fmt(s.pnl_since_start_mtm_pct)}%)</div></div>
       <div class="card"><div class="label">Unrealized P&L (open positions)</div><div class="value ${s.unrealized_pnl >= 0 ? "green" : "red"}">₹${fmt(s.unrealized_pnl)}</div></div>`
    : "";
  document.getElementById("status-cards").innerHTML = `
    <div class="card"><div class="label">Cash balance</div><div class="value">₹${fmt(s.cash_balance)}</div></div>
    <div class="card"><div class="label">Total equity (at cost)</div><div class="value">₹${fmt(s.total_equity)}</div></div>
    <div class="card"><div class="label">P&L since start (at cost)</div><div class="value ${pnlClass}">₹${fmt(s.pnl_since_start)} (${fmt(s.pnl_since_start_pct)}%)</div></div>
    ${mtmCard}
    <div class="card"><div class="label">Open positions</div><div class="value">${s.open_position_count} / ${s.target_slots}</div></div>
    <div class="card"><div class="label">Current slot size</div><div class="value">₹${fmt(s.current_slot_size)}</div></div>
    <div class="card"><div class="label">Last run</div><div class="value" style="font-size:0.85rem">${s.last_run ? s.last_run.status : "unknown"}</div></div>
  `;
}

async function loadPositions() {
  const rows = await api("/api/positions");
  document.querySelector("#positions-table tbody").innerHTML = rows.map(p => {
    const known = p.unrealized_pnl !== null && p.unrealized_pnl !== undefined;
    const cls = known && p.unrealized_pnl >= 0 ? "green" : "red";
    const pnlCell = known
      ? `<td class="${cls}">₹${fmt(p.unrealized_pnl)} (${fmt(p.unrealized_pnl_pct)}%)</td>`
      : `<td>-</td>`;
    const priceCell = known ? `₹${fmt(p.current_price)}` : "-";
    return `<tr><td>${p.symbol}</td><td>${fmtTime(p.entry_timestamp)}</td><td>${p.quantity}</td><td>₹${fmt(p.capital_allocated)}</td><td>${priceCell}</td>${pnlCell}</tr>`;
  }).join("") || "<tr><td colspan=6>No open positions</td></tr>";
}

let symbolsLoaded = false;

async function loadSymbols() {
  const rows = await api("/api/symbols");
  const select = document.getElementById("symbol-filter");
  const current = select.value;
  select.innerHTML = '<option value="">All symbols (recent 30)</option>' + rows.map(r =>
    `<option value="${r.symbol}">${r.symbol} (${r.trade_count} trades, ${fmt(r.win_rate)}% win)</option>`
  ).join("");
  select.value = current;
  symbolsLoaded = true;
}

async function loadTrades() {
  const select = document.getElementById("symbol-filter");
  const symbol = select.value;
  const query = symbol ? `symbol=${encodeURIComponent(symbol)}` : "limit=30";
  const t = await api(`/api/trades?${query}`);
  document.getElementById("trades-title").textContent = symbol
    ? `${symbol} trades (win rate`
    : "Closed trades (buy-only win rate";
  document.getElementById("win-rate").textContent = t.overall_win_rate !== null ? fmt(t.overall_win_rate) + "%" : "-";
  document.querySelector("#trades-table tbody").innerHTML = t.recent.map(r => {
    const cls = r.pnl_percent >= 0 ? "green" : "red";
    return `<tr><td>${r.symbol}</td><td>${fmtTime(r.entry_timestamp)}</td><td>${fmtTime(r.exit_timestamp)}</td><td><span class="pill ${cls}">${fmt(r.pnl_percent)}%</span></td></tr>`;
  }).join("") || "<tr><td colspan=4>No closed trades yet</td></tr>";
}

async function loadConfig() {
  const c = await api("/api/config");
  document.getElementById("cfg-capital").value = c.capital;
  document.getElementById("cfg-slots").value = c.slots;
  document.getElementById("cfg-min").value = c.min_position;
}

async function saveConfig() {
  const body = {
    capital: document.getElementById("cfg-capital").value,
    slots: document.getElementById("cfg-slots").value,
    min_position: document.getElementById("cfg-min").value,
  };
  const opts = {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  };
  await api("/api/config", opts);
  document.getElementById("config-msg").textContent = "Saved - takes effect next pipeline run.";
}

async function trigger() {
  document.getElementById("trigger-msg").textContent = "Triggering...";
  await api("/api/trigger", {method: "POST"});
  const msg = "Triggered - check logs below in a few minutes.";
  document.getElementById("trigger-msg").textContent = msg;
}

async function loadLogs() {
  const l = await api("/api/logs?lines=150");
  document.getElementById("logs").textContent = l.lines.join("\\n");
}

async function loadKiteStatus() {
  const k = await api("/api/kite-status");
  const el = document.getElementById("kite-status");
  if (!k.configured) { el.innerHTML = ""; return; }
  if (k.logged_in) {
    // obtained_at is already IST (Kite's own login_time), not UTC -- do not
    // run it through fmtTime()'s UTC->IST conversion, that would shift it.
    el.innerHTML = `<span class="pill green">Kite: logged in (${k.obtained_at})</span> <button onclick="window.location.href='/kite/login'">Re-login</button>`;
  } else {
    el.innerHTML = `<span class="pill red">Kite: not logged in (using Yahoo)</span> <button class="primary" onclick="window.location.href='/kite/login'">Login to Kite</button>`;
  }
}

async function refreshAll() {
  if (!symbolsLoaded) await loadSymbols();
  await Promise.all([loadStatus(), loadPositions(), loadTrades(), loadConfig(), loadLogs(), loadKiteStatus()]);
}
refreshAll();
setInterval(refreshAll, 30000);
</script>
</body>
</html>
"""

_LOGIN_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SkyTrade dashboard &mdash; log in</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, system-ui, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
  form { border: 1px solid #8884; border-radius: 12px; padding: 2rem; width: 260px; display: flex; flex-direction: column; gap: 0.8rem; }
  h1 { font-size: 1.1rem; margin: 0 0 0.5rem 0; }
  input { padding: 0.5rem; border-radius: 6px; border: 1px solid #8884; font-size: 1rem; }
  button { padding: 0.5rem; border-radius: 6px; border: none; background: #2563eb; color: white; font-size: 1rem; cursor: pointer; }
  #error { color: #d33; font-size: 0.85rem; min-height: 1.2em; }
</style>
</head>
<body>
<form id="login-form">
  <h1>SkyTrade dashboard</h1>
  <input id="password" type="password" placeholder="Password" autofocus required />
  <button type="submit">Log in</button>
  <div id="error"></div>
</form>
<script>
document.getElementById("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const password = document.getElementById("password").value;
  const res = await fetch("/login", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({password}),
  });
  if (res.ok) {
    window.location.href = "/";
  } else {
    document.getElementById("error").textContent = "Wrong password.";
  }
});
</script>
</body>
</html>
"""


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("TRADING_SCANNER_DASHBOARD_PORT", "8000")))


if __name__ == "__main__":
    main()
