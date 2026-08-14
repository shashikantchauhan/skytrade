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

import bisect
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


# 2026-08-14: which features actually deserve weight, decided by checking
# real outcomes against each of the 4 entry-time indicators independently,
# on all 6,292 closed BUY trades (not a trained model -- just win rate per
# decile bucket of each raw indicator, plus a temporal-stability check
# splitting the history in half to catch anything that was only true in
# aggregate). Results:
#
#   indicator            weak-bucket win% -> strong-bucket win%   stable across both halves?
#   volatility_margin      51.8%     ->     62.9%                  yes (holds in older AND newer half)
#   regime_normalized      54.9%     ->     61.9%                  partially (weak/not-significant in newer half)
#   adx                    56.4%     ->     60.2%  (non-monotonic) no -- noise
#   prediction_at_entry    54.6%     ->     60.4%  (non-monotonic) no -- noise
#
# So volatility_margin and regime_normalized get real, evidence-weighted
# priority; adx and prediction_at_entry get folded in at low weight since
# neither shows a trustworthy pattern on its own -- prediction_at_entry
# still matters for *direction* (see direction_sign below), just not as a
# quality signal.
#
# IMPORTANT CAVEAT (don't lose this): this ranks candidates *relative to
# each other* within a scan cycle -- it decides who gets capital first when
# several signals compete. It is NOT validated as an absolute predictor: a
# walk-forward test of a logistic-regression version of this same idea
# (train on past months, score the next unseen month, exactly like Stage
# B's CatBoost test) came back at mean AUC 0.527 -- barely above random,
# one month scored *below* random (0.431). That means don't trust this
# formula to reliably tell you "trade or don't trade" (no floor should be
# set from it), only to break ties when capital is scarcer than signals.
#
# Percentile buckets (deciles, 0/10/.../90) computed from the same 6,292
# trades' real distributions -- so a new candidate's raw value gets scored
# by where it falls in the *historical* distribution, not an arbitrary
# fixed scale.
_VOLATILITY_MARGIN_DECILE_CUTS = [0.3966, 0.9614, 1.8114, 3.0316, 4.8878, 8.0392, 14.1294, 25.6973, 57.273]
_REGIME_NORMALIZED_DECILE_CUTS = [0.0667, 0.2351, 0.4403, 0.6821, 0.9504, 1.3043, 1.7584, 2.4345, 3.7582]
_VOLATILITY_MARGIN_WEIGHT = 1.0  # strongest, stable evidence -- full weight
_REGIME_NORMALIZED_WEIGHT = 0.5  # real but less stable -- half weight
_PREDICTION_MAGNITUDE_WEIGHT = 2.0  # no quality signal found -- kept small, mostly for direction (see below)
_ADX_WEIGHT = 1.0  # no quality signal found -- raw value is already small (~0.07-0.65), stays a minor nudge


def _decile_score(value: float, cuts: list[float]) -> float:
    """0, 10, ..., 90 -- which decile of the real historical distribution
    ``value`` falls into, per the cut points above. Values beyond the
    observed historical range clamp to the nearest end bucket rather than
    extrapolating."""
    return float(bisect.bisect_right(cuts, value) * 10)


def score_candidate(candidate: RankedCandidate) -> float:
    """Stage A heuristic score: higher is a stronger candidate, regardless
    of direction.

    Weighted by real evidence (see the block comment above this function):
    ``volatility_margin`` and ``regime_normalized`` are the two indicators
    that actually showed a real, outcome-correlated pattern in production
    trade history, scored by percentile against that history and weighted
    accordingly. ``adx`` and ``prediction_at_entry``'s *magnitude* showed no
    such pattern, so they're weighted low -- ``prediction_at_entry`` is kept
    mainly for its *sign*: it's the kNN vote (Pine's own confidence signal,
    positive leaning BUY and negative leaning SELL), flipped by
    ``direction`` so a strongly bearish SELL candidate scores as strong, not
    weak -- a no-op for today's BUY-only ranking, but correct the moment
    SELL candidates are ranked too.

    This is a relative sort, not a validated absolute predictor -- see the
    walk-forward AUC caveat above. Don't derive a rejection floor from this
    score without new evidence.
    """
    direction_sign = 1.0 if candidate.direction == SignalSide.BUY else -1.0
    volatility_score = _decile_score(candidate.volatility_margin, _VOLATILITY_MARGIN_DECILE_CUTS)
    regime_score = _decile_score(candidate.regime_normalized, _REGIME_NORMALIZED_DECILE_CUTS)
    return (
        volatility_score * _VOLATILITY_MARGIN_WEIGHT
        + regime_score * _REGIME_NORMALIZED_WEIGHT
        + float(candidate.prediction_at_entry) * direction_sign * _PREDICTION_MAGNITUDE_WEIGHT
        + candidate.adx * _ADX_WEIGHT
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
