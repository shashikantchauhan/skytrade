"""Tests for trading_scanner/web/services/auth.py -- split out of
webapp.py (Phase 16). Only the pieces testable without a running FastAPI
app: password matching and the two session-gate functions' pure
expiry/role logic."""

import time

import pytest
from fastapi import HTTPException

from trading_scanner.web.services import auth


@pytest.fixture(autouse=True)
def _isolated_sessions(monkeypatch):
    """Every test gets its own empty _sessions dict -- these tests must
    not see state left behind by another test (or by webapp.py's own
    routes, if the whole suite runs in the same process)."""
    monkeypatch.setattr(auth, "_sessions", {})


def test_authenticate_matches_the_admin_password(monkeypatch):
    monkeypatch.setenv("TRADING_SCANNER_DASHBOARD_PASSWORD", "adminpass")
    monkeypatch.delenv("TRADING_SCANNER_VIEWER_LOGINS", raising=False)

    assert auth._authenticate("adminpass") == ("admin", "admin")


def test_authenticate_matches_a_named_viewer_password(monkeypatch):
    monkeypatch.setenv("TRADING_SCANNER_DASHBOARD_PASSWORD", "adminpass")
    monkeypatch.setenv("TRADING_SCANNER_VIEWER_LOGINS", "wife:wifepass,friend:friendpass")

    assert auth._authenticate("wifepass") == ("viewer", "wife")
    assert auth._authenticate("friendpass") == ("viewer", "friend")


def test_authenticate_returns_none_for_a_wrong_password(monkeypatch):
    monkeypatch.setenv("TRADING_SCANNER_DASHBOARD_PASSWORD", "adminpass")
    monkeypatch.delenv("TRADING_SCANNER_VIEWER_LOGINS", raising=False)

    assert auth._authenticate("wrongpass") is None


def test_require_session_rejects_a_missing_cookie():
    with pytest.raises(HTTPException) as excinfo:
        auth._require_session(None)
    assert excinfo.value.status_code == 401


def test_require_session_rejects_an_expired_session():
    auth._sessions["tok"] = {"expiry": time.time() - 1, "role": "viewer", "name": "wife"}
    with pytest.raises(HTTPException) as excinfo:
        auth._require_session("tok")
    assert excinfo.value.status_code == 401


def test_require_session_allows_a_viewer():
    auth._sessions["tok"] = {"expiry": time.time() + 3600, "role": "viewer", "name": "wife"}
    auth._require_session("tok")  # must not raise


def test_require_admin_rejects_a_viewer_role():
    auth._sessions["tok"] = {"expiry": time.time() + 3600, "role": "viewer", "name": "wife"}
    with pytest.raises(HTTPException) as excinfo:
        auth._require_admin("tok")
    assert excinfo.value.status_code == 403


def test_require_admin_allows_an_admin_role():
    auth._sessions["tok"] = {"expiry": time.time() + 3600, "role": "admin", "name": "admin"}
    auth._require_admin("tok")  # must not raise
