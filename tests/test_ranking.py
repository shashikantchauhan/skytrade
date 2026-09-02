from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_scanner.application.ranking import (
    MIN_EXPECTANCY_TRADES,
    RankedCandidate,
    rank_candidates,
    score_candidate,
    select_top_n,
    symbol_expectancy,
)
from trading_scanner.domain.models import SignalSide, Trade


def _candidate(
    symbol: str,
    prediction: int,
    adx: float = 0.2,
    vol_margin: float = 0.0,
    direction: SignalSide = SignalSide.BUY,
    expectancy: float | None = None,
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
        expectancy=expectancy,
    )


class _FakeTradeRepository:
    """Minimal in-memory TradeRepository fake -- returns whatever closed
    trades it was given, ignoring the symbol/interval filter (matches the
    other fakes' style across this test suite)."""

    def __init__(self, trades: list[Trade]) -> None:
        self._trades = trades

    async def get_trades(self, symbol, interval):
        return self._trades

    async def open_trade(self, interval, trade):
        raise NotImplementedError

    async def close_open_trade(self, symbol, interval, side, exit_timestamp, exit_price):
        raise NotImplementedError

    async def abandon_open_trade(self, symbol, interval, side):
        raise NotImplementedError


def _closed_trade(side: SignalSide, pnl_percent: str) -> Trade:
    return Trade(
        symbol="RELIANCE.NS",
        side=side,
        entry_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        entry_price=Decimal("100"),
        prediction_at_entry=1,
        is_early_signal_flip=False,
        exit_timestamp=datetime(2026, 1, 2, tzinfo=UTC),
        exit_price=Decimal("101"),
        pnl_percent=Decimal(pnl_percent),
        status="closed",
    )


def test_retry_of_signal_timestamp_defaults_none_and_does_not_affect_scoring():
    # 2026-09-02 (delayed re-entry window, docs/decisions/
    # 011-delayed-reentry-window.md) -- purely a traceability field, must
    # not change score_candidate's output.
    fresh = _candidate("FRESH", prediction=4, adx=0.3, vol_margin=1.0)
    retry = RankedCandidate(
        symbol="RETRY", entry_timestamp=fresh.entry_timestamp, entry_price=fresh.entry_price,
        prediction_at_entry=4, adx=0.3, regime_normalized=0.0, volatility_margin=1.0,
        retry_of_signal_timestamp=datetime(2026, 8, 31, 10, 15, tzinfo=UTC),
    )

    assert fresh.retry_of_signal_timestamp is None
    assert retry.retry_of_signal_timestamp == datetime(2026, 8, 31, 10, 15, tzinfo=UTC)
    assert score_candidate(fresh) == score_candidate(retry)


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


def test_score_candidate_rewards_higher_expectancy():
    # 2026-08-14: expectancy is the one factor here with real walk-forward
    # support (see the block comment above _EXPECTANCY_WEIGHT) -- a
    # candidate whose symbol+side has a real track record of strong
    # expectancy should outrank an otherwise-identical one with weak
    # expectancy.
    strong_expectancy = _candidate("STRONG_EXP", prediction=1, expectancy=3.0)
    weak_expectancy = _candidate("WEAK_EXP", prediction=1, expectancy=-1.0)

    assert score_candidate(strong_expectancy) > score_candidate(weak_expectancy)


def test_score_candidate_treats_missing_expectancy_as_neutral_not_penalized():
    # A brand-new symbol+side with no track record yet (expectancy=None)
    # should score the same as one sitting exactly at the median of the
    # real historical expectancy distribution -- "unknown" isn't "bad."
    no_track_record = _candidate("NEW", prediction=1, expectancy=None)
    at_median = _candidate("MEDIAN", prediction=1, expectancy=0.9889)  # the median decile cut

    assert score_candidate(no_track_record) == score_candidate(at_median)

    # But a genuinely weak track record should still score below "unknown."
    weak_track_record = _candidate("WEAK", prediction=1, expectancy=-1.3)
    assert score_candidate(weak_track_record) < score_candidate(no_track_record)


@pytest.mark.asyncio
async def test_symbol_expectancy_returns_none_below_minimum_trade_count():
    trades = [_closed_trade(SignalSide.BUY, "5.0") for _ in range(MIN_EXPECTANCY_TRADES - 1)]
    repo = _FakeTradeRepository(trades)

    result = await symbol_expectancy("RELIANCE.NS", SignalSide.BUY, "1h", repo)

    assert result is None


@pytest.mark.asyncio
async def test_symbol_expectancy_computes_win_rate_weighted_average():
    # 8 wins at +2%, 2 losses at -1% -> win_rate=0.8
    # expectancy = 0.8*2 + 0.2*(-1) = 1.6 - 0.2 = 1.4
    trades = [_closed_trade(SignalSide.BUY, "2.0") for _ in range(8)] + [
        _closed_trade(SignalSide.BUY, "-1.0") for _ in range(2)
    ]
    repo = _FakeTradeRepository(trades)

    result = await symbol_expectancy("RELIANCE.NS", SignalSide.BUY, "1h", repo)

    assert result == pytest.approx(1.4)


@pytest.mark.asyncio
async def test_symbol_expectancy_only_counts_matching_side():
    # SELL trades shouldn't leak into a BUY-side expectancy calculation.
    trades = [_closed_trade(SignalSide.BUY, "2.0") for _ in range(10)] + [
        _closed_trade(SignalSide.SELL, "-5.0") for _ in range(10)
    ]
    repo = _FakeTradeRepository(trades)

    result = await symbol_expectancy("RELIANCE.NS", SignalSide.BUY, "1h", repo)

    assert result == pytest.approx(2.0)  # all 10 BUY trades won at +2%, SELL losses ignored
