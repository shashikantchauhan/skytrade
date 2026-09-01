"""Cookie-based dashboard session auth -- split out of ``webapp.py`` (Phase
16, see ``trading_scanner/web/__init__.py``). No behavior changed; every
function's body moved as-is.

One shared password (personal tool, not a multi-user product), plus
optional named view-only logins. Sessions are an in-memory dict -- a
dashboard restart logs everyone out, fine for a single-process personal
tool. ``_sessions`` is deliberately module-level state, not a class: every
route in ``webapp.py`` (and any future route module) imports this same
dict object, so a session created via one route is visible to every other
route's ``_require_session``/``_require_admin`` check, exactly as it was
when all of this lived in one file.
"""

import os
import secrets
import time

from fastapi import Cookie, HTTPException

_SESSION_COOKIE = "ptrade_session"
_SESSION_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days

# token -> {expiry, role, name}. In-memory: fine for a single-process
# personal tool; a restart just means logging in again.
_sessions: dict[str, dict] = {}


def _dashboard_password() -> str:
    password = os.getenv("TRADING_SCANNER_DASHBOARD_PASSWORD")
    if not password:
        raise HTTPException(
            status_code=500,
            detail="TRADING_SCANNER_DASHBOARD_PASSWORD is not set on the server.",
        )
    return password


def _viewer_credentials() -> dict[str, str]:
    """Named view-only logins, e.g. for a spouse/friend who should see the
    dashboard but never touch Kite login or trigger a pipeline run.

    ``TRADING_SCANNER_VIEWER_LOGINS="wife:somepassword,friend:otherpassword"``
    -- each name gets its own password so access can be revoked individually
    later without changing the admin password everyone else still uses.
    """
    raw = os.getenv("TRADING_SCANNER_VIEWER_LOGINS", "")
    result: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        name, password = entry.split(":", 1)
        if name.strip() and password:
            result[name.strip()] = password
    return result


def _authenticate(password: str) -> tuple[str, str] | None:
    """Returns (role, name) on a password match, checking the admin
    password first, then each named viewer -- None if it matches nothing."""
    if secrets.compare_digest(password, _dashboard_password()):
        return "admin", "admin"
    for name, viewer_password in _viewer_credentials().items():
        if secrets.compare_digest(password, viewer_password):
            return "viewer", name
    return None


def _require_session(ptrade_session: str | None = Cookie(default=None)) -> None:
    """API-route auth: 401 JSON if the session cookie is missing/expired.
    Allows both admin and viewer roles -- use ``_require_admin`` for routes
    that touch Kite or trigger the pipeline."""
    session = _sessions.get(ptrade_session or "")
    if session is None or session["expiry"] < time.time():
        raise HTTPException(status_code=401, detail="Not logged in.")


def _require_admin(ptrade_session: str | None = Cookie(default=None)) -> None:
    """Admin-only routes: Kite login/status, triggering the pipeline or a
    backtest, and config changes -- a viewer (e.g. a spouse checking in on
    the numbers) should never be able to touch any of these, both to keep
    Kite credentials private and because a second person clicking 'Kite
    login' can stomp the one active session the pipeline depends on (this
    happened once -- see the commit that added this check)."""
    session = _sessions.get(ptrade_session or "")
    if session is None or session["expiry"] < time.time():
        raise HTTPException(status_code=401, detail="Not logged in.")
    if session["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only.")
