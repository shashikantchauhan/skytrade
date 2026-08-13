"""Turso (hosted libSQL) storage, split into one focused module per
repository -- was previously a single 1,300+ line ``turso.py``.

The same ``libsql_client`` connection works against a local ``file:``
database (no account, no network -- used for tests and local development)
or a hosted ``libsql://...`` database with an auth token (production). No
repository depends on which one is in use.

Every repository class is re-exported here so callers can still do
``from trading_scanner.infrastructure.db import TursoTradeRepository`` etc.
without needing to know which submodule it actually lives in.
"""

from trading_scanner.infrastructure.db._shared import add_column_if_missing, create_turso_client
from trading_scanner.infrastructure.db.candles import TursoCandleRepository
from trading_scanner.infrastructure.db.engine_state import TursoEngineStateRepository
from trading_scanner.infrastructure.db.futures_paper_account import (
    TursoFuturesPaperAccountRepository,
)
from trading_scanner.infrastructure.db.futures_trades import TursoFuturesTradeRepository
from trading_scanner.infrastructure.db.kite_session import TursoKiteSessionRepository
from trading_scanner.infrastructure.db.live_orders import TursoLiveOrderRepository
from trading_scanner.infrastructure.db.options_trades import TursoOptionsTradeRepository
from trading_scanner.infrastructure.db.paper_account import TursoPaperAccountRepository
from trading_scanner.infrastructure.db.signals import TursoSignalRepository
from trading_scanner.infrastructure.db.trades import TursoTradeRepository

__all__ = [
    "add_column_if_missing",
    "create_turso_client",
    "TursoCandleRepository",
    "TursoEngineStateRepository",
    "TursoFuturesPaperAccountRepository",
    "TursoFuturesTradeRepository",
    "TursoKiteSessionRepository",
    "TursoLiveOrderRepository",
    "TursoOptionsTradeRepository",
    "TursoPaperAccountRepository",
    "TursoSignalRepository",
    "TursoTradeRepository",
]
