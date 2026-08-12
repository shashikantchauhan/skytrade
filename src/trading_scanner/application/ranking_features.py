"""Build the full Stage A feature table (see NOTES.md's ranking-model
roadmap) from the ``trades`` table plus stored candles -- the feature
engineering step shared by ``train_ranking_model.py`` (offline training) and,
eventually, the live ranking gate once a trained model replaces
``application/ranking.py``'s heuristic ``score_candidate``.

Eight features total. Six come straight off each ``Trade`` row (already
logged by ``application/backtest.py``'s replay) or its timestamp; the last
two need cross-symbol context computed here:

* ``sector``: looked up from ``analysis/sector_mapping.SECTOR_MAP`` (curated
  NSE sector-index mapping for this 220-symbol universe -- not derivable
  from the codebase itself, see that module's own docstring).
* ``correlation_to_open_positions``: the average Pearson correlation of
  trailing daily-bar returns between this candidate's symbol and every
  *other* symbol with an overlapping open BUY trade at this entry's
  timestamp (reconstructed from the trades table's own entry/exit
  intervals -- 0.0 when nothing else was open).
"""

import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading_scanner.domain.models import Candle, SignalSide, Trade
from trading_scanner.domain.ports import CandleRepository

# analysis/ is a sibling of src/, not part of the installed package -- see
# that directory's own scripts (sector_loss_analysis.py) for the same
# pattern of importing it standalone rather than folding it into the
# trading_scanner package (it is offline research tooling, not something
# the live pipeline imports).
_ANALYSIS_DIR = Path(__file__).resolve().parents[3] / "analysis"
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))
from sector_mapping import SECTOR_MAP  # noqa: E402

_CORRELATION_WINDOW_BARS = 50
_UNKNOWN_SECTOR = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """One BUY trade's Stage A feature snapshot plus its win/loss label."""

    symbol: str
    entry_timestamp: datetime
    prediction_at_entry: int
    adx: float
    regime_normalized: float
    volatility_margin: float
    sector: str
    day_of_week: int
    hour_of_day: int
    days_since_last_trade: float
    correlation_to_open_positions: float
    label: int  # 1 = win (pnl_percent > 0), 0 = loss


def _open_intervals(trades: list[Trade]) -> dict[str, list[tuple[datetime, datetime]]]:
    """Per symbol, every BUY trade's (entry, exit-or-now) interval.

    A still-open trade (``exit_timestamp is None``) is treated as open
    through the latest timestamp seen anywhere in the dataset -- the
    conservative choice: it may overstate how long it stayed open, never
    understate it, which only makes the correlation feature more
    conservative (more candidates count as "something was open"), not
    look-ahead biased.
    """
    latest = max(
        (trade.exit_timestamp or trade.entry_timestamp for trade in trades), default=None
    )
    intervals: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
    for trade in trades:
        end = trade.exit_timestamp or latest or trade.entry_timestamp
        intervals[trade.symbol].append((trade.entry_timestamp, end))
    return intervals


def _symbols_open_at(
    intervals: dict[str, list[tuple[datetime, datetime]]], symbol: str, at: datetime
) -> list[str]:
    open_symbols = []
    for other_symbol, spans in intervals.items():
        if other_symbol == symbol:
            continue
        if any(start <= at <= end for start, end in spans):
            open_symbols.append(other_symbol)
    return open_symbols


def _return_series(candles: list[Candle]) -> pd.Series:
    closes = pd.Series(
        [float(candle.close) for candle in candles],
        index=pd.DatetimeIndex([candle.timestamp for candle in candles]),
    )
    return closes.pct_change().dropna()


async def build_feature_table(
    trades: list[Trade], candle_repository: CandleRepository, interval: str
) -> list[FeatureRow]:
    """Build one FeatureRow per closed BUY trade with a known feature snapshot.

    Trades logged before this migration (no ADX/regime/volatility -- see
    ``application/backtest.py``) are skipped rather than imputed: a trained
    model should not learn from silently-fabricated feature values.
    """
    buy_trades = [
        trade
        for trade in trades
        if trade.side == SignalSide.BUY
        and trade.status == "closed"
        and trade.pnl_percent is not None
        and trade.adx_at_entry is not None
    ]
    intervals = _open_intervals([t for t in trades if t.side == SignalSide.BUY])

    # Cache each symbol's return series once -- correlation needs every
    # symbol's price history, not just the candidate's, and re-downloading
    # per trade would be 8-12k redundant repository calls.
    return_series_by_symbol: dict[str, pd.Series] = {}

    async def _returns_for(symbol: str) -> pd.Series:
        if symbol not in return_series_by_symbol:
            candles = await candle_repository.get_candles(symbol, interval, limit=None)
            return_series_by_symbol[symbol] = _return_series(list(candles))
        return return_series_by_symbol[symbol]

    last_entry_by_symbol: dict[str, datetime] = {}
    rows: list[FeatureRow] = []
    for trade in sorted(buy_trades, key=lambda t: t.entry_timestamp):
        sector = SECTOR_MAP.get(trade.symbol) or _UNKNOWN_SECTOR

        previous_entry = last_entry_by_symbol.get(trade.symbol)
        days_since_last_trade = (
            (trade.entry_timestamp - previous_entry).total_seconds() / 86400.0
            if previous_entry is not None
            else -1.0  # Sentinel: first-ever trade seen for this symbol.
        )
        last_entry_by_symbol[trade.symbol] = trade.entry_timestamp

        open_symbols = _symbols_open_at(intervals, trade.symbol, trade.entry_timestamp)
        correlation = 0.0
        if open_symbols:
            candidate_returns = await _returns_for(trade.symbol)
            candidate_window = candidate_returns[
                candidate_returns.index < trade.entry_timestamp
            ].tail(_CORRELATION_WINDOW_BARS)
            correlations = []
            for other_symbol in open_symbols:
                other_returns = await _returns_for(other_symbol)
                other_window = other_returns[other_returns.index < trade.entry_timestamp].tail(
                    _CORRELATION_WINDOW_BARS
                )
                aligned = pd.concat([candidate_window, other_window], axis=1).dropna()
                if len(aligned) >= 10:  # Too few overlapping bars -> not a meaningful correlation.
                    correlations.append(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            if correlations:
                correlation = float(np.nanmean(correlations))

        rows.append(
            FeatureRow(
                symbol=trade.symbol,
                entry_timestamp=trade.entry_timestamp,
                prediction_at_entry=trade.prediction_at_entry,
                adx=trade.adx_at_entry,
                regime_normalized=trade.regime_normalized_at_entry or 0.0,
                volatility_margin=trade.volatility_margin_at_entry or 0.0,
                sector=sector,
                day_of_week=trade.entry_timestamp.weekday(),
                hour_of_day=trade.entry_timestamp.hour,
                days_since_last_trade=days_since_last_trade,
                correlation_to_open_positions=correlation,
                label=1 if trade.pnl_percent > 0 else 0,
            )
        )
    return rows
