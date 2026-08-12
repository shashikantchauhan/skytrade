from datetime import UTC, datetime
from decimal import Decimal

from trading_scanner.application.ranking import RankedCandidate, rank_candidates, select_top_n


def _candidate(symbol: str, prediction: int, adx: float = 0.2, vol_margin: float = 0.0):
    return RankedCandidate(
        symbol=symbol,
        entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        entry_price=Decimal("100"),
        prediction_at_entry=prediction,
        adx=adx,
        regime_normalized=0.0,
        volatility_margin=vol_margin,
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
