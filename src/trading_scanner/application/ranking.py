"""Rank a scan cycle's eligible BUY candidates when there are more than free capital slots.

Today (no ranking anywhere in the repo), ``signal_pipeline.py`` opens paper
positions first-come-first-served: whichever symbol's ``_process_symbol``
happens to finish first among the concurrently-processed batch gets the
capital, with no regard for which candidate is actually the stronger signal.
This module adds a scoring/ranking step so, when a cycle produces more
eligible candidates than free slots, the strongest candidates win the
capital instead of whichever finished first.

Two-stage rollout (see NOTES.md's ranking-model roadmap):

* **Stage A (here now)**: a heuristic score using ``prediction_at_entry``
  (the kNN vote strength -- Pine's own confidence signal, and the strongest
  single feature available before a trained model exists), tie-broken by
  ADX (trend strength) and how far the volatility filter clears its
  threshold. This is deliberately swappable: ``score_candidate`` is the only
  function that needs to change once a trained model
  (``application/train_ranking_model.py``) is ready, everything else
  (``rank_candidates``/``select_top_n``) is scoring-method-agnostic.
* **Stage B (later)**: swap ``score_candidate`` for a trained
  LGBMClassifier/CatBoostClassifier P(win) score once
  ``train_ranking_model.py`` has a validated, calibrated model from the
  feature-logged backtest data (``application/backtest.py``). A
  learning-to-rank upgrade (LambdaMART family) can follow once that
  classifier stage is validated.

Not wired into ``signal_pipeline.py`` yet: the live scan loop currently
evaluates and opens positions per-symbol, concurrently, as soon as each
symbol's own BUY signal is found -- there is no point in that flow today
where "this cycle's other candidates" are known yet. Using this module
requires collecting one cycle's BUY candidates first, then ranking, then
opening positions for the top N -- a structural change to the scan loop,
not just an added function call. See NOTES.md for the integration plan.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_scanner.domain.models import SignalSide

# Absolute selectivity floor, on top of (not instead of) relative ranking:
# a candidate scoring below this gets rejected outright even when capital
# is sitting free, rather than funded just because nothing stronger showed
# up in that cycle. 0 (default) preserves today's pure-relative behavior.
#
# 2026-08-14: an earlier baseline run here (floors of 81, then 92) was
# built on the pre-cap formula below, whose score distribution was
# dominated by volatility_margin's uncapped long tail -- once fixed, that
# same analysis re-run against the corrected formula showed NO real
# discriminating power (quintiles roughly flat 55-60% win rate, top decile
# actually *worse* PF than baseline; prediction_at_entry alone shows the
# same flat pattern). Deployed floor was reverted to 0 (no floor) the same
# session pending real evidence the heuristic discriminates quality --
# don't reintroduce a nonzero floor without a fresh baseline run against
# THIS (capped) formula first.
MIN_SCORE = float(os.getenv("TRADING_SCANNER_RANKING_MIN_SCORE", "0"))


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """One symbol's eligible entry this scan cycle, plus the feature
    snapshot needed to score it (mirrors what ``application/backtest.py``
    now logs per trade -- see ``PineTrade``).

    ``direction`` defaults to BUY since that's the only side ever ranked
    today (SELL never competes for capital in the cash-only system -- see
    ``signal_pipeline.py``'s ``if result.signal != "BUY": continue``). It's
    here so ``score_candidate`` scores conviction *in the trade's own
    direction* correctly once SELL/short candidates start competing for
    capital too (futures/options), not just so it type-checks today.
    """

    symbol: str
    entry_timestamp: datetime
    entry_price: Decimal
    prediction_at_entry: int
    adx: float
    regime_normalized: float
    volatility_margin: float
    direction: SignalSide = SignalSide.BUY


# 2026-08-14: found by hand, checking real trade examples against this
# formula -- volatility_margin is NOT a small, bounded quantity the way ADX
# is (see this constant's own reasoning below). Its real distribution across
# 6,292 production trades: median ~4.9, p90 ~57, p99 ~438, max ~2160. Added
# uncapped, it silently stopped being a tie-breaker for a meaningful chunk
# of trades (44.5% of the top score quintile got there mainly *because of*
# a large volatility_margin, not strong prediction -- directly contradicting
# this function's own "prediction dominates" claim below, which was true in
# intent but false in practice until this cap). Capped at 5.0: prediction's
# own range is 10-80 (neighbors_count=8, so -8..+8 * 10), so a 5-point cap
# can never outweigh even a single prediction-vote's difference (10 points)
# -- restores tie-breaker-only behavior for volatility_margin. Chosen close
# to the real median (4.9) specifically so typical trades are barely
# affected; only the pathological long tail gets reined in.
_MAX_VOLATILITY_MARGIN_CONTRIBUTION = 5.0


def score_candidate(candidate: RankedCandidate) -> float:
    """Stage A heuristic score: higher is a stronger candidate, regardless
    of direction.

    ``prediction_at_entry`` is the kNN vote sum (Pine's own confidence
    signal -- roughly -neighbors_count..+neighbors_count, positive leaning
    BUY and negative leaning SELL), so it dominates the score. It's signed
    by ``direction`` first: a BUY wants a strongly positive vote, a SELL
    wants a strongly negative one, and without this flip a strongly
    bearish SELL candidate would score *lower* than a weak one -- exactly
    backwards. A no-op for today's BUY-only ranking (direction is always
    BUY live), but correct the moment SELL candidates are ranked too.

    ADX and the (capped) volatility margin break ties between similarly-
    confident signals: prefer a trending market (higher ADX) with
    volatility clearly above its filter threshold (positive margin) over a
    borderline one -- capped at ``_MAX_VOLATILITY_MARGIN_CONTRIBUTION`` so
    an extreme volatility reading can nudge a tie but never override a
    real prediction-strength difference (see that constant's own docstring
    for why this cap exists and how it was chosen).

    Replace this function's body with a trained model's ``predict_proba``
    call once ``train_ranking_model.py`` has one validated -- callers only
    depend on this signature (``RankedCandidate -> float``), not on how the
    score is computed.
    """
    direction_sign = 1.0 if candidate.direction == SignalSide.BUY else -1.0
    volatility_contribution = min(
        max(candidate.volatility_margin, 0.0), _MAX_VOLATILITY_MARGIN_CONTRIBUTION
    )
    return (
        float(candidate.prediction_at_entry) * direction_sign * 10.0
        + candidate.adx
        + volatility_contribution
    )


def rank_candidates(candidates: list[RankedCandidate]) -> list[RankedCandidate]:
    """Sort candidates strongest-first."""
    return sorted(candidates, key=score_candidate, reverse=True)


def select_top_n(candidates: list[RankedCandidate], free_slots: int) -> tuple[
    list[RankedCandidate], list[RankedCandidate]
]:
    """Split ranked candidates into (take, skip) at the free-slot boundary.

    ``free_slots`` is expected to be computed the same way
    ``paper_trading.try_open_position`` already sizes capacity (equity /
    TARGET_SLOTS, floored at MIN_POSITION_SIZE) -- this function only does
    the selection, not the capital math, so it stays correct if that sizing
    logic changes.
    """
    if free_slots <= 0:
        return [], list(candidates)
    ranked = rank_candidates(candidates)
    return ranked[:free_slots], ranked[free_slots:]
