"""Hard entry-quality gate for real cash orders: reject a BUY candidate
outright when its own entry-time ``volatility_margin``/``regime_normalized``
readings are below their historical median -- on top of (not instead of)
``paper_trading.is_eligible``'s symbol-track-record gate.

Why these two, and why a hard floor (unlike ``application/ranking.py``'s
combined heuristic score, whose own docstring explicitly warns against
deriving a rejection floor from it -- a walk-forward-tested logistic-
regression version of that combined score only hit AUC 0.527, barely above
random): ``volatility_margin`` and ``regime_normalized`` were the two
factors in that same investigation that showed a real, monotonic,
temporally-stable pattern on their own. This module re-validated that
finding independently on 2026-08-25 against 6,340 real closed BUY trades
(2025-06-11 through 2026-08-25) with a stricter bar than a single train/test
split: walk-forward across FIVE independent split dates (2026-02-01,
2026-04-01, 2026-05-10, 2026-06-01, 2026-07-15), thresholds fit only on data
before each split and applied only to data at/after it (see
``analysis/bad_trade_filters.py``, keep that script if this threshold is
ever re-derived). Requiring BOTH readings to clear their own train-period
median, every single time:

    split        kept   win-rate lift (filtered vs. unfiltered, same split)
    2026-02-01   33%    +3.5pp
    2026-04-01   32%    +5.0pp
    2026-05-10   30%    +6.9pp
    2026-06-01   30%    +7.0pp
    2026-07-15   29%    +4.5pp

Never negative, never close to zero, across five non-overlapping windows
spanning six months -- the bar this codebase already treats as trustworthy
(see ``train_ranking_model.py``'s own three-split validation of per-symbol
expectancy). Ruled out in the same pass, so deliberately NOT gates here:
ADX floor (no effect), sector-index same-bar confirmation (no effect),
tightening the win-rate eligibility bar past 55% (made results worse
out-of-sample -- don't raise ``paper_trading.MIN_WIN_RATE`` further from
this evidence).

``_VOLATILITY_MARGIN_FLOOR``/``_REGIME_NORMALIZED_FLOOR`` are each symbol's
own real historical median as of 2026-08-25 across the full 220-symbol
universe (not a train-only split -- the split above was only to prove the
effect is real; the deployed floor uses all available history for the best
current estimate). Revisit periodically the same way
``application/ranking.py``'s decile cuts are revisited -- these will drift
as more trade history accumulates.

Only gates real cash order placement (``application/live_cash_execution.py``
-- see its call site in ``signal_pipeline.py``), same as
``paper_trading.is_eligible``: a rejected candidate still notifies, still
gets its ``Trade`` row, still feeds the paper-benchmark and eligibility
history -- it just never reaches ``execute_cash_entry``. Squaring off an
already-open real position is never gated by this (exits are never blocked
on entry-quality grounds, same reasoning as the eligibility gate).
"""

_VOLATILITY_MARGIN_FLOOR = 4.888648705300606
_REGIME_NORMALIZED_FLOOR = 0.9554664608541863


def passes_indicator_filter(volatility_margin: float, regime_normalized: float) -> bool:
    """True only when both readings clear their historical median floor."""
    return (
        volatility_margin >= _VOLATILITY_MARGIN_FLOOR
        and regime_normalized >= _REGIME_NORMALIZED_FLOOR
    )
