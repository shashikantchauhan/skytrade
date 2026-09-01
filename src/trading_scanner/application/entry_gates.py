"""Composable entry-gate wrappers around this codebase's existing, already
walk-forward-validated cash-entry filters -- Phase 1-2 of
`projectedPlann.md` (see docs/architecture/000-audit.md).

Zero logic changes: every gate here calls the exact same function, with
the exact same arguments and thresholds, that ``signal_pipeline.py``
already called inline in two separate places
(``_collect_and_open_ranked_positions`` and ``_process_symbol``'s unranked
cash fallback). Before this module, those two blocks independently
recomputed the same "track record + entry_quality_filter +
conviction_filter" chain with hand-written reason strings -- a real,
already-duplicated piece of business logic that could silently drift
between the two call sites (e.g. one updated, the other forgotten). This
replaces both with calls into one shared implementation; the underlying
``paper_trading.is_eligible`` / ``entry_quality_filter.
passes_indicator_filter`` / ``conviction_filter.passes_conviction_filter``
are untouched.

Ranking (``application/ranking.py``), and the capital/position-limit/
entry-cutoff checks inside ``live_cash_execution.execute_cash_entry``, are
deliberately NOT wrapped here yet: ranking needs the whole candidate batch
at once (not a per-candidate gate), and the capacity/cutoff checks must
stay live-re-checked at the moment of order placement -- including inside
``execute_cash_entry``'s own retry loop -- for TOCTOU safety against
concurrent symbols (see that function's own 2026-08-25 lock comment on
exactly this class of race). Pre-computing them into a cached
``EntryDecision`` would reintroduce that risk. Revisit once decision
persistence (Phase 3) needs a uniform shape for those too.
"""

from decimal import Decimal

from trading_scanner.application import conviction_filter, entry_quality_filter, paper_trading
from trading_scanner.domain.gates import EntryDecision, GateResult
from trading_scanner.domain.ports import TradeRepository

# Reason strings match exactly what both call sites already produced --
# preserving these keeps every existing Telegram note/log line byte-for-
# byte identical after migrating to this module.
_NOT_ELIGIBLE_REASON = "not eligible yet (win_rate<55% or insufficient trade history)"
_QUALITY_REASON = "entry_quality_filter"
_CONVICTION_REASON = "conviction filter -- weak entry candle"


async def evaluate_track_record_gate(
    symbol: str, interval: str, trade_repository: TradeRepository
) -> GateResult:
    """Wraps ``paper_trading.is_eligible`` -- the track-record gate shared
    by both the paper and cash books."""
    passed = await paper_trading.is_eligible(symbol, interval, trade_repository)
    return GateResult("track_record", passed, None if passed else _NOT_ELIGIBLE_REASON)


def evaluate_cash_quality_gates(
    volatility_margin: float,
    regime_normalized: float,
    high: Decimal,
    low: Decimal,
    close: Decimal,
) -> EntryDecision:
    """The two extra gates a real cash order requires on top of track
    record -- ``entry_quality_filter`` then ``conviction_filter``, in that
    order, matching the reason-string priority both existing call sites
    already used (quality's reason wins if both fail)."""
    quality_passed = entry_quality_filter.passes_indicator_filter(
        volatility_margin, regime_normalized
    )
    conviction_passed = conviction_filter.passes_conviction_filter(high, low, close)
    gates = (
        GateResult(
            "entry_quality_filter", quality_passed, None if quality_passed else _QUALITY_REASON
        ),
        GateResult(
            "conviction_filter",
            conviction_passed,
            None if conviction_passed else _CONVICTION_REASON,
        ),
    )
    return EntryDecision(gates=gates)
