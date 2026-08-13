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
