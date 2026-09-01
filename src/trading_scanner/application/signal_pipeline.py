"""Hourly pipeline: accumulate candles per symbol and notify on AlphaEngine
signals.

Backward-compatible facade over ``application/pipeline/*`` (Phase 8, see
that package's own docstring) -- this file used to hold every function
below directly (~1900 lines, too many responsibilities in one module); it
now just re-exports them from the submodule each actually lives in, so
every existing import of ``trading_scanner.application.signal_pipeline.X``
(``live_pipeline.py``, ``signals.py``, ``fix_cross_contaminated_candles_
cli.py``, and this module's own test suite) keeps working unchanged. New
code should import directly from the relevant ``application/pipeline/*``
submodule instead of through this facade.
"""

from trading_scanner.application.pipeline.capital_allocation import (
    MIN_SCORE,
    _collect_and_open_ranked_positions,
    _open_paper_position,
    _persist_entry_decision,
    _rank_and_open_cash_positions,
    _rank_and_open_futures_positions,
    _rank_and_open_paper_positions,
)
from trading_scanner.application.pipeline.entry_decision import (
    _finalize_cash_entry,
    _notify_missed_cash_entry,
)
from trading_scanner.application.pipeline.evaluation import (
    _BACKFILL_WINDOW_DAYS,
    _FULL_HISTORY,
    _MINIMUM_CANDLES,
    _RECENT_WINDOW_DAYS,
    _evaluate_from_stored_candles,
    _evaluate_symbol,
)
from trading_scanner.application.pipeline.lifecycle import (
    _HEDGE_OTM_PCT,
    _close_derivatives_shadow,
    _close_futures_paper,
    _close_paper_position,
    _notify_exit,
    _open_derivatives_shadow,
    _open_futures_paper,
    _win_rate_summary,
)
from trading_scanner.application.pipeline.market_data import (
    _STRATEGY_NAME,
    MarketDataProvider,
    NoValidKiteSession,
    _candles_to_dataframe,
    _dataframe_to_candles,
    _is_kite_token_error,
    _market_price,
    _notify_kite_expired_once_per_day,
    _select_provider,
)
from trading_scanner.application.pipeline.orchestrator import (
    _ENGINE_SETTINGS,
    _MAX_CONCURRENT_SYMBOLS,
    _process_symbol,
    run_signal_pipeline,
)

__all__ = [
    "MIN_SCORE",
    "MarketDataProvider",
    "NoValidKiteSession",
    "_BACKFILL_WINDOW_DAYS",
    "_ENGINE_SETTINGS",
    "_FULL_HISTORY",
    "_HEDGE_OTM_PCT",
    "_MAX_CONCURRENT_SYMBOLS",
    "_MINIMUM_CANDLES",
    "_RECENT_WINDOW_DAYS",
    "_STRATEGY_NAME",
    "_candles_to_dataframe",
    "_close_derivatives_shadow",
    "_close_futures_paper",
    "_close_paper_position",
    "_collect_and_open_ranked_positions",
    "_dataframe_to_candles",
    "_evaluate_from_stored_candles",
    "_evaluate_symbol",
    "_finalize_cash_entry",
    "_is_kite_token_error",
    "_market_price",
    "_notify_exit",
    "_notify_kite_expired_once_per_day",
    "_notify_missed_cash_entry",
    "_open_derivatives_shadow",
    "_open_futures_paper",
    "_open_paper_position",
    "_persist_entry_decision",
    "_process_symbol",
    "_rank_and_open_cash_positions",
    "_rank_and_open_futures_positions",
    "_rank_and_open_paper_positions",
    "_select_provider",
    "_win_rate_summary",
    "run_signal_pipeline",
]
