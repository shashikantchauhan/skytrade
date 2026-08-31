"""Python port of Zeiierman's "Breakout Probability" Pine Script indicator
(TradingView, CC BY-NC-SA 4.0 -- NonCommercial -- personal use only, per the
user; this is a standalone reference tool, never wired into the live
pipeline or any real-money decision).

What it measures: conditioned on whether the *previous* candle closed green
or red, how often has price gone on to make a new high/low at least N%
("a level") beyond that previous candle's own high/low? Purely a historical
frequency table over one-candle-lag momentum -- no relation to
AlphaEngine's Lorentzian feature set (RSI/WT/CCI/ADX), and not a prediction
model in the sense the live gates use.

Differs from the original Pine script in one deliberate way: the original
only refreshes a level's displayed probability at the moment of a hit, so a
level's shown percentage can lag its true up-to-date value when hits are
rare (the denominator keeps growing between hits, but the ratio isn't
recomputed until the next one). This port always reports the true final
hits/total ratio for every level -- a correctness fix, not a different
measurement. Everything else (the per-level threshold math, the doji
handling, the level-0-only directional "backtest") mirrors the original
bar-by-bar.

See ``analysis/breakout_probability_report.py`` for the CLI that loads real
candles and prints this as a table -- this module is the pure, DB-free
logic underneath it (same split as ``sector_confirmation.py`` /
``analysis/sector_loss_analysis.py``).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from trading_scanner.domain.models import Candle


@dataclass(slots=True)
class LevelCounts:
    """Raw hit counts for one level (a fixed % distance beyond the previous
    candle's own high/low), split by the previous candle's color."""

    green_high_hits: int = 0
    green_low_hits: int = 0
    red_high_hits: int = 0
    red_low_hits: int = 0


@dataclass(slots=True)
class LevelProbabilities:
    """``None`` means "no prior candles of that color yet" -- not 0%."""

    green_high: float | None
    green_low: float | None
    red_high: float | None
    red_low: float | None


@dataclass(slots=True)
class BreakoutStats:
    levels: list[LevelCounts] = field(default_factory=list)
    green_total: int = 0
    red_total: int = 0
    directional_wins: int = 0
    directional_losses: int = 0

    def level_probabilities(self) -> list[LevelProbabilities]:
        return [
            LevelProbabilities(
                green_high=_pct(lvl.green_high_hits, self.green_total),
                green_low=_pct(lvl.green_low_hits, self.green_total),
                red_high=_pct(lvl.red_high_hits, self.red_total),
                red_low=_pct(lvl.red_low_hits, self.red_total),
            )
            for lvl in self.levels
        ]

    @property
    def directional_win_rate(self) -> float | None:
        decided = self.directional_wins + self.directional_losses
        return None if decided == 0 else 100 * self.directional_wins / decided


def _pct(hits: int, total: int) -> float | None:
    return None if total == 0 else round(100 * hits / total, 2)


def compute_breakout_stats(
    candles: Sequence[Candle], *, step_percent: float, num_levels: int
) -> BreakoutStats:
    """Walks ``candles`` chronologically exactly as the Pine indicator does
    bar by bar: for each pair of consecutive candles, classifies the
    earlier one's own color (green if it closed above its open, red if
    below -- a doji, close == open, counts toward neither side, same as the
    original), then checks whether the later candle's high/low cleared each
    of ``num_levels`` thresholds beyond the earlier candle's own high/low.

    ``step_percent``: 1.0 means each level sits 1% of the *later* candle's
    own close further out than the previous level -- level 0 is always the
    loosest threshold ("any new high/low at all", zero distance).

    Also accumulates the original's level-0-only directional "backtest":
    each bar, pick the higher-probability side so far (up if the previous
    candle was green and green-conditioned P(new high) >= P(new low), else
    down; the same comparison against red-conditioned probabilities
    whenever the previous candle was red *or* a doji -- matching the
    original's own green ? ... : ... branch exactly), then score whether
    this bar actually reached that target.
    """
    stats = BreakoutStats(levels=[LevelCounts() for _ in range(num_levels)])
    for prev, curr in zip(candles, candles[1:], strict=False):
        advance_breakout_stats(stats, prev, curr, step_percent)
    return stats


def advance_breakout_stats(
    stats: BreakoutStats, prev: Candle, curr: Candle, step_percent: float
) -> None:
    """Folds one more (``prev``, ``curr``) candle pair into ``stats`` in
    place -- the same per-bar update ``compute_breakout_stats`` applies bar
    by bar. Exposed separately (not just inlined in that loop) so a
    walk-forward backtest against real trade history can snapshot ``stats``
    *before* folding in the bar a trade entered on, giving the causal,
    no-look-ahead probability as it would have read at the moment of entry
    -- see ``analysis/breakout_probability_trade_backtest.py``."""
    green = prev.close > prev.open
    red = prev.close < prev.open
    if green:
        stats.green_total += 1
    elif red:
        stats.red_total += 1

    step = curr.close * (Decimal(str(step_percent)) / 100)
    for i, level in enumerate(stats.levels):
        threshold = step * i
        hit_high = curr.high >= prev.high + threshold
        hit_low = curr.low <= prev.low - threshold
        if green and hit_high:
            level.green_high_hits += 1
        if green and hit_low:
            level.green_low_hits += 1
        if red and hit_high:
            level.red_high_hits += 1
        if red and hit_low:
            level.red_low_hits += 1

    _score_directional_bias(stats, prev, curr, green)


def _score_directional_bias(stats: BreakoutStats, prev: Candle, curr: Candle, green: bool) -> None:
    level0 = stats.levels[0]
    if green:
        target_is_high = level0.green_high_hits >= level0.green_low_hits
    else:
        target_is_high = level0.red_high_hits >= level0.red_low_hits
    if target_is_high:
        hit = curr.high >= prev.high
    else:
        hit = curr.low <= prev.low
    if hit:
        stats.directional_wins += 1
    else:
        stats.directional_losses += 1


@dataclass(slots=True)
class ProjectedLevel:
    index: int
    high_target: Decimal
    low_target: Decimal
    high_probability: float | None
    low_probability: float | None


def project_next_bar_levels(
    latest: Candle, stats: BreakoutStats, *, step_percent: float, num_levels: int
) -> list[ProjectedLevel]:
    """The forward-looking table the original chart actually displays: price
    targets and their historical probabilities for the *next* bar, based on
    the most recently closed candle's own high/low and color.

    Uses ``latest``'s own close as the step basis, same proxy the live
    chart uses while the next bar is still forming (its real close isn't
    known yet)."""
    green = latest.close > latest.open
    probabilities = stats.level_probabilities()
    step = latest.close * (Decimal(str(step_percent)) / 100)
    projected = []
    for i in range(num_levels):
        threshold = step * i
        probs = probabilities[i]
        projected.append(
            ProjectedLevel(
                index=i,
                high_target=latest.high + threshold,
                low_target=latest.low - threshold,
                high_probability=probs.green_high if green else probs.red_high,
                low_probability=probs.green_low if green else probs.red_low,
            )
        )
    return projected


def bullish_bias(stats: BreakoutStats, prev: Candle) -> bool | None:
    """The level-0 directional call as of right after ``prev`` closes --
    the same "which side has the higher hit rate" comparison the original
    indicator's own backtest panel uses (see ``_score_directional_bias``),
    exposed standalone so a caller can snapshot it against real trade
    history without needing the rest of ``advance_breakout_stats``'s
    bookkeeping.

    Returns ``True`` (bullish -- P(new high) >= P(new low)) or ``False``
    (bearish), conditioned on ``prev``'s own color. ``None`` if there
    isn't at least one prior candle of that color yet -- not enough
    history to make a call either way, not a bearish call."""
    probs = stats.level_probabilities()[0]
    green = prev.close > prev.open
    high, low = (probs.green_high, probs.green_low) if green else (probs.red_high, probs.red_low)
    if high is None or low is None:
        return None
    return high >= low
