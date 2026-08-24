"""Local SQLite storage (via ``aiosqlite``), split into one focused module
per repository -- was previously a single 1,300+ line ``turso.py``.

Hosted Turso is no longer supported (2026-08-20 decision -- production
never actually used it, see ``_shared.py``'s module docstring for the real
history and the freeze it caused). Every deployment uses a local ``file:``
database now. Class/function names below keep their historical "Turso"
naming -- a purely cosmetic mismatch with zero functional effect, not
worth the churn of renaming across this package's own 20+ call sites.

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
from trading_scanner.infrastructure.db.gtt import TursoGttRepository
from trading_scanner.infrastructure.db.kite_session import TursoKiteSessionRepository
from trading_scanner.infrastructure.db.live_cash_toggle import (
    LiveCashToggleState,
    TursoLiveCashToggleRepository,
)
from trading_scanner.infrastructure.db.live_orders import TursoLiveOrderRepository
from trading_scanner.infrastructure.db.options_trades import TursoOptionsTradeRepository
from trading_scanner.infrastructure.db.paper_account import TursoPaperAccountRepository
from trading_scanner.infrastructure.db.paper_benchmark import TursoPaperBenchmarkRepository
from trading_scanner.infrastructure.db.signals import TursoSignalRepository
from trading_scanner.infrastructure.db.trades import TursoTradeRepository

__all__ = [
    "add_column_if_missing",
    "create_turso_client",
    "TursoCandleRepository",
    "TursoEngineStateRepository",
    "TursoFuturesPaperAccountRepository",
    "TursoFuturesTradeRepository",
    "TursoGttRepository",
    "TursoKiteSessionRepository",
    "LiveCashToggleState",
    "TursoLiveCashToggleRepository",
    "TursoLiveOrderRepository",
    "TursoOptionsTradeRepository",
    "TursoPaperAccountRepository",
    "TursoPaperBenchmarkRepository",
    "TursoSignalRepository",
    "TursoTradeRepository",
]
