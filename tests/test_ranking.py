from datetime import UTC, datetime
from decimal import Decimal

from trading_scanner.application.ranking import (
    RankedCandidate,
    rank_candidates,
    score_candidate,
    select_top_n,
)
from trading_scanner.domain.models import SignalSide


def _candidate(
    symbol: str,
    prediction: int,
    adx: float = 0.2,
    vol_margin: float = 0.0,
    direction: SignalSide = SignalSide.BUY,
):
    return RankedCandidate(
        symbol=symbol,
        entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        entry_price=Decimal("100"),
        prediction_at_entry=prediction,
        adx=adx,
        regime_normalized=0.0,
        volatility_margin=vol_margin,
        direction=direction,
    )


def test_rank_candidates_sorts_strongest_prediction_first():
    weak = _candidate("WEAK", prediction=1)
    strong = _candidate("STRONG", prediction=6)
    medium = _candidate("MEDIUM", prediction=3)

    ranked = rank_candidates([weak, strong, medium])

    assert [candidate.symbol for candidate in ranked] == ["STRONG", "MEDIUM", "WEAK"]


def test_rank_candidates_breaks_ties_on_adx_and_volatility_margin():
    low_adx = _candidate("LOW_ADX", prediction=4, adx=0.1, vol_margin=0.0)
    high_adx = _candidate("HIGH_ADX", prediction=4, adx=0.5, vol_margin=0.0)

    ranked = rank_candidates([low_adx, high_adx])

    assert ranked[0].symbol == "HIGH_ADX"


def test_select_top_n_splits_at_free_slot_boundary():
    candidates = [_candidate(f"SYM{i}", prediction=i) for i in range(5)]

    take, skip = select_top_n(candidates, free_slots=2)

    assert [candidate.symbol for candidate in take] == ["SYM4", "SYM3"]
    assert {candidate.symbol for candidate in skip} == {"SYM0", "SYM1", "SYM2"}


def test_select_top_n_with_no_free_slots_skips_everyone():
    candidates = [_candidate("ONLY", prediction=5)]

    take, skip = select_top_n(candidates, free_slots=0)

    assert take == []
    assert skip == candidates


def test_score_candidate_is_direction_aware_for_sell():
    # Strongly bearish (prediction=-6) should score HIGHER than weakly
    # bearish (prediction=-1) for a SELL candidate -- without flipping the
    # sign by direction, the raw formula would rank these backwards, since
    # prediction_at_entry is only "more positive = better" for a BUY.
    strong_short = _candidate("STRONG_SHORT", prediction=-6, direction=SignalSide.SELL)
    weak_short = _candidate("WEAK_SHORT", prediction=-1, direction=SignalSide.SELL)

    assert score_candidate(strong_short) > score_candidate(weak_short)


def test_score_candidate_buy_and_sell_conviction_are_symmetric():
    # A BUY at prediction=+6 and a SELL at prediction=-6 represent equally
    # strong conviction in their own direction, so should score identically
    # (all else equal).
    strong_buy = _candidate("STRONG_BUY", prediction=6, direction=SignalSide.BUY)
    strong_short = _candidate("STRONG_SHORT", prediction=-6, direction=SignalSide.SELL)

    assert score_candidate(strong_buy) == score_candidate(strong_short)


def test_score_candidate_clamps_extreme_volatility_margin_to_top_decile():
    # A pathological outlier (real production data reaches into the
    # thousands) must not get an ever-growing score just for being more
    # extreme -- percentile bucketing clamps anything at/above the p90 cut
    # point to the same top bucket, same idea as the old hard cap but
    # derived from the real historical distribution instead of a fixed
    # number.
    at_p90_cut = _candidate("AT_P90", prediction=1, vol_margin=57.273)
    pathological_outlier = _candidate("OUTLIER", prediction=1, vol_margin=50000.0)

    assert score_candidate(at_p90_cut) == score_candidate(pathological_outlier)


def test_score_candidate_weighs_volatility_margin_above_prediction_magnitude():
    # 2026-08-14: intentional, evidence-based change -- checking real
    # production outcomes showed volatility_margin has a real win-rate
    # pattern (weak decile 51.8% win vs strong decile 62.9% win, stable
    # across both halves of trade history) while prediction_at_entry's
    # *magnitude* does not (54.6%-60.4%, non-monotonic, no real pattern).
    # So a candidate with strong historical volatility_margin standing now
    # legitimately outranks one with only a strong prediction vote and
    # unremarkable volatility_margin -- this replaces the old "prediction
    # always dominates" assumption, which real outcomes didn't support.
    # See ranking.py's block comment above score_candidate for the full
    # evidence table and the walk-forward caveat (this is a relative sort,
    # not a validated absolute predictor).
    strong_volatility_weak_prediction = _candidate("VOL", prediction=1, vol_margin=100.0)
    weak_volatility_strong_prediction = _candidate("PRED", prediction=8, vol_margin=0.1)

    assert score_candidate(strong_volatility_weak_prediction) > score_candidate(
        weak_volatility_strong_prediction
    )


def test_score_candidate_volatility_margin_still_breaks_ties_within_the_cap():
    same_pred_low_vol = _candidate("LOW_VOL", prediction=5, vol_margin=0.5)
    same_pred_high_vol = _candidate("HIGH_VOL", prediction=5, vol_margin=3.0)

    assert score_candidate(same_pred_high_vol) > score_candidate(same_pred_low_vol)
