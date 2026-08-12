"""Test whether a trade's mapped sector index entering the same side at the
same time is a real, statistically meaningful edge -- not a guess from a
couple of hand-picked examples (e.g. PNB.NS BUY alongside ^NSEBANK BUY,
ZYDUSLIFE.NS BUY alongside ^CNXPHARMA BUY).

Two things already exist that sound similar but are not this:

* ``config.index_symbol`` (usually the broad NIFTY index) is evaluated and
  attached to every Telegram message purely for reading -- confirmed in
  ``application/signal_pipeline.py``'s own docstring, "never used to
  suppress a signal." Informational only, and it's the broad market index,
  not a stock's own sector index.
* Stage A's ``sector`` feature (``application/ranking_features.py``) is a
  *static* categorical label ("this stock is Banking") -- not "did the
  Banking index also fire a signal right now." A static label and a live,
  same-timestamp confirmation are very different kinds of feature.

This module builds the live confirmation signal directly from data the
system already has: since every sector index (``^NSEBANK``, ``^CNXPHARMA``,
etc.) is itself listed in ``config/symbols.txt`` and backtested like any
other symbol, its own BUY/SELL entries are already sitting in the
``trades`` table. No new AlphaEngine computation is needed -- just check
whether the mapped index has a trade of the same side at the same
``entry_timestamp`` as the stock's own trade.
"""

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from trading_scanner.domain.models import SignalSide, Trade

_ANALYSIS_DIR = Path(__file__).resolve().parents[3] / "analysis"
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))
from sector_mapping import SECTOR_MAP  # noqa: E402


@dataclass(frozen=True, slots=True)
class ConfirmationStats:
    n: int
    win_rate: Decimal | None
    avg_win: Decimal | None
    avg_loss: Decimal | None
    expectancy: Decimal | None


def annotate_sector_confirmation(trades: Sequence[Trade]) -> dict[int, bool | None]:
    """Map each trade (by ``id()``) to whether its sector index agreed.

    ``True``: the mapped sector index has its own trade of the same side
    starting at the exact same ``entry_timestamp``. ``False``: the symbol
    has a sector mapping but the index did not confirm. ``None``: the
    symbol has no sector mapping (see ``SECTOR_MAP``'s own docstring on
    which symbols are intentionally left unmapped) -- excluded from the
    comparison rather than counted as "disagreed", since there's nothing to
    check.
    """
    index_entries = {(trade.symbol, trade.side, trade.entry_timestamp) for trade in trades}
    confirmation: dict[int, bool | None] = {}
    for trade in trades:
        sector = SECTOR_MAP.get(trade.symbol)
        if sector is None:
            confirmation[id(trade)] = None
            continue
        confirmation[id(trade)] = (sector, trade.side, trade.entry_timestamp) in index_entries
    return confirmation


def _summarize(trades: list[Trade]) -> ConfirmationStats:
    closed = [t for t in trades if t.status == "closed" and t.pnl_percent is not None]
    if not closed:
        return ConfirmationStats(0, None, None, None, None)
    wins = [t.pnl_percent for t in closed if t.pnl_percent > 0]
    losses = [t.pnl_percent for t in closed if t.pnl_percent <= 0]
    return ConfirmationStats(
        n=len(closed),
        win_rate=Decimal(100 * len(wins)) / len(closed),
        avg_win=sum(wins) / len(wins) if wins else None,
        avg_loss=sum(losses) / len(losses) if losses else None,
        expectancy=sum(t.pnl_percent for t in closed) / len(closed),
    )


def compare_confirmed_vs_unconfirmed(
    trades: Sequence[Trade], side: SignalSide
) -> dict[str, ConfirmationStats]:
    """The actual test: does the confirmed subset beat the unconfirmed
    subset, across the full trade history -- not two examples."""
    side_trades = [trade for trade in trades if trade.side == side]
    confirmation = annotate_sector_confirmation(side_trades)
    confirmed = [t for t in side_trades if confirmation[id(t)] is True]
    unconfirmed = [t for t in side_trades if confirmation[id(t)] is False]
    unmapped = [t for t in side_trades if confirmation[id(t)] is None]
    return {
        "confirmed": _summarize(confirmed),
        "unconfirmed": _summarize(unconfirmed),
        "unmapped (no sector index)": _summarize(unmapped),
    }
