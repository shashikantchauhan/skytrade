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

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """One symbol's eligible BUY entry this scan cycle, plus the feature
    snapshot needed to score it (mirrors what ``application/backtest.py``
    now logs per trade -- see ``PineTrade``)."""

    symbol: str
    entry_timestamp: datetime
    entry_price: Decimal
    prediction_at_entry: int
    adx: float
    regime_normalized: float
    volatility_margin: float


def score_candidate(candidate: RankedCandidate) -> float:
    """Stage A heuristic score: higher is a stronger candidate.

    ``prediction_at_entry`` is the kNN vote sum (Pine's own confidence
    signal -- roughly -neighbors_count..+neighbors_count for a BUY-leaning
    vote), so it dominates the score. ADX and the volatility margin break
    ties between similarly-confident signals: prefer a trending market
    (higher ADX) with volatility clearly above its filter threshold
    (positive margin) over a borderline one.

    Replace this function's body with a trained model's ``predict_proba``
    call once ``train_ranking_model.py`` has one validated -- callers only
    depend on this signature (``RankedCandidate -> float``), not on how the
    score is computed.
    """
    return float(candidate.prediction_at_entry) * 10.0 + candidate.adx + max(
        candidate.volatility_margin, 0.0
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
