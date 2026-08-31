"""Tests for the Breakout Probability port (see application/
breakout_probability.py's own docstring for what it measures and how it
differs from the original Pine indicator)."""

from datetime import UTC, datetime
from decimal import Decimal

from trading_scanner.application.breakout_probability import (
    BreakoutStats,
    LevelCounts,
    advance_breakout_stats,
    bullish_bias,
    compute_breakout_stats,
    project_next_bar_levels,
)
from trading_scanner.domain.models import Candle


def _candle(open_, high, low, close, day=1) -> Candle:
    return Candle(
        symbol="TEST.NS",
        timestamp=datetime(2026, 1, day, tzinfo=UTC),
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=1000,
    )


def test_green_candle_followed_by_a_new_high_counts_as_a_green_high_hit():
    candles = [
        _candle(100, 105, 99, 104, day=1),  # green (close > open)
        _candle(104, 108, 103, 107, day=2),  # makes a new high over day 1's 105
    ]

    stats = compute_breakout_stats(candles, step_percent=1.0, num_levels=1)

    assert stats.green_total == 1
    assert stats.red_total == 0
    assert stats.levels[0].green_high_hits == 1
    assert stats.levels[0].green_low_hits == 0
    probs = stats.level_probabilities()[0]
    assert probs.green_high == 100.0
    assert probs.red_high is None  # no red candles seen yet -- None, not 0%


def test_red_candle_followed_by_a_new_low_counts_as_a_red_low_hit():
    candles = [
        _candle(100, 101, 95, 96, day=1),  # red (close < open)
        _candle(96, 97, 90, 91, day=2),  # makes a new low under day 1's 95
    ]

    stats = compute_breakout_stats(candles, step_percent=1.0, num_levels=1)

    assert stats.red_total == 1
    assert stats.levels[0].red_low_hits == 1
    assert stats.level_probabilities()[0].red_low == 100.0


def test_a_doji_previous_candle_counts_toward_neither_color():
    candles = [
        _candle(100, 102, 98, 100, day=1),  # doji: close == open
        _candle(100, 105, 97, 103, day=2),
    ]

    stats = compute_breakout_stats(candles, step_percent=1.0, num_levels=1)

    assert stats.green_total == 0
    assert stats.red_total == 0
    assert stats.levels[0].green_high_hits == 0
    assert stats.levels[0].red_high_hits == 0


def test_higher_levels_require_a_proportionally_bigger_move():
    # Day 2's high (110) clears day 1's high (100) by exactly 10, which is
    # step_percent=1% of day 2's own close (1000) times threshold index 1
    # (step = 1000 * 0.01 = 10) -- level 1 hits, level 2 (needs +20) doesn't.
    candles = [
        _candle(90, 100, 89, 95, day=1),  # green
        _candle(95, 110, 94, 1000, day=2),
    ]

    stats = compute_breakout_stats(candles, step_percent=1.0, num_levels=3)

    assert stats.levels[0].green_high_hits == 1  # any new high at all
    assert stats.levels[1].green_high_hits == 1  # cleared +10
    assert stats.levels[2].green_high_hits == 0  # needed +20, only got +10


def test_directional_backtest_scores_a_win_when_the_picked_side_is_reached():
    # Day 1 green with no prior history -> level-0 green_high_hits (0) ==
    # green_low_hits (0), so target_is_high defaults true (>=) -> target is
    # day 1's high. Day 2 clears it -> a win.
    candles = [
        _candle(100, 105, 99, 104, day=1),
        _candle(104, 108, 103, 107, day=2),
    ]

    stats = compute_breakout_stats(candles, step_percent=1.0, num_levels=1)

    assert stats.directional_wins == 1
    assert stats.directional_losses == 0
    assert stats.directional_win_rate == 100.0


def test_directional_backtest_scores_a_loss_when_the_picked_side_is_missed():
    candles = [
        _candle(100, 105, 99, 104, day=1),  # green -> target is day 1's high (105)
        _candle(104, 104.5, 103, 104, day=2),  # never reaches 105
    ]

    stats = compute_breakout_stats(candles, step_percent=1.0, num_levels=1)

    assert stats.directional_wins == 0
    assert stats.directional_losses == 1


def test_project_next_bar_levels_uses_the_latest_candle_as_the_reference():
    candles = [
        _candle(100, 105, 99, 104, day=1),
        _candle(104, 110, 103, 109, day=2),  # green, becomes the projection base
    ]
    stats = compute_breakout_stats(candles, step_percent=1.0, num_levels=2)

    projected = project_next_bar_levels(candles[-1], stats, step_percent=1.0, num_levels=2)

    assert len(projected) == 2
    assert projected[0].high_target == Decimal("110")  # level 0: latest high + 0
    assert projected[0].low_target == Decimal("103")  # level 0: latest low - 0
    # green latest candle -> probabilities come from the green-conditioned side
    assert projected[0].high_probability == stats.level_probabilities()[0].green_high
    assert projected[1].high_target == candles[-1].high + (candles[-1].close * Decimal("0.01"))


def test_bullish_bias_is_none_with_no_prior_history_of_that_color():
    stats = BreakoutStats(levels=[LevelCounts()])
    prev = _candle(100, 105, 99, 104, day=1)  # green, but stats has zero green history

    assert bullish_bias(stats, prev) is None


def test_bullish_bias_is_true_when_new_highs_outpace_new_lows_for_that_color():
    stats = BreakoutStats(levels=[LevelCounts()])
    stats.green_total = 10
    stats.levels[0].green_high_hits = 7
    stats.levels[0].green_low_hits = 2
    prev = _candle(100, 105, 99, 104)  # green

    assert bullish_bias(stats, prev) is True


def test_bullish_bias_is_false_when_new_lows_outpace_new_highs_for_that_color():
    stats = BreakoutStats(levels=[LevelCounts()])
    stats.green_total = 10
    stats.levels[0].green_high_hits = 2
    stats.levels[0].green_low_hits = 7
    prev = _candle(100, 105, 99, 104)  # green

    assert bullish_bias(stats, prev) is False


def test_bullish_bias_reads_the_red_conditioned_side_for_a_red_prev_candle():
    stats = BreakoutStats(levels=[LevelCounts()])
    stats.green_total = 10
    stats.levels[0].green_high_hits = 9  # would say bullish if read from the green side
    stats.levels[0].green_low_hits = 1
    stats.red_total = 10
    stats.levels[0].red_high_hits = 1
    stats.levels[0].red_low_hits = 9
    prev = _candle(104, 105, 95, 100)  # red: close < open

    assert bullish_bias(stats, prev) is False


def test_advance_breakout_stats_snapshot_matches_a_full_recompute_of_the_prior_bars_only():
    # advance_breakout_stats is meant to be called incrementally, snapshotting
    # `stats` before folding in the newest bar -- confirm that snapshot always
    # equals compute_breakout_stats() run over just the bars seen so far.
    candles = [
        _candle(100, 105, 99, 104, day=1),
        _candle(104, 108, 103, 107, day=2),
        _candle(107, 106, 101, 102, day=3),
        _candle(102, 112, 101, 111, day=4),
    ]
    stats = BreakoutStats(levels=[LevelCounts()])
    for i in range(1, len(candles)):
        prev, curr = candles[i - 1], candles[i]
        expected = compute_breakout_stats(candles[:i], step_percent=1.0, num_levels=1)
        assert stats.level_probabilities() == expected.level_probabilities()
        assert stats.green_total == expected.green_total
        assert stats.red_total == expected.red_total
        advance_breakout_stats(stats, prev, curr, step_percent=1.0)
