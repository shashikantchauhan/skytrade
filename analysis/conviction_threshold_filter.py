"""Test conviction (close-location-value at the entry candle) as a
STANDALONE selectivity filter on top of the real-gated baseline BUY
population -- not baked into the Lorentzian model (a 6-feature
AlphaEngineConviction variant that fed close-location-value into the KNN
search itself was tried and then removed -- full-universe result was only
marginal, win rate up ~1.3pp, expectancy flat), just a cheap post-hoc
threshold on signals we already have. The question here is different: can
conviction pick out a SMALLER, BETTER subset of the trades the system
already generates, rather than changing which trades get generated in the
first place? This module found the answer is yes -- see
``application/conviction_filter.py``, now wired into real cash entries.

Cheap by design: no Lorentzian KNN replay at all. Pulls the real ``trades``
table, causally reconstructs the real gate stack (55%/5-trade eligibility +
entry_quality_filter) exactly like every other backtest this session, then
joins each surviving BUY's entry candle (and previous candle) from the
``candles`` table to compute close-location-value = (close-low)/(high-low)
-- 1.0 means it closed at the bar's high, 0.0 at the low, 0.5 for a
zero-range bar.

Sweeps thresholds on: entry-candle CLV alone, previous-candle CLV alone,
and both combined -- reporting n / win rate / expectancy / sum P&L for
each, so a shrinking-but-improving population is visible directly (unlike
a hard gate wired into a live capital-simulation, this is just "what would
happen to the population" -- capacity-dilution effects come after, only if
this stage shows a real signal worth testing further).

Run:
    PYTHONPATH=src python analysis/conviction_threshold_filter.py <db-path>
"""

import sqlite3
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from trading_scanner.application import entry_quality_filter, paper_trading  # noqa: E402


@dataclass
class Row:
    symbol: str
    interval: str
    entry_timestamp: str
    pnl_percent: float
    volatility_margin: float | None
    regime_normalized: float | None


def _clv(high: float, low: float, close: float) -> float:
    rng = high - low
    return 0.5 if rng == 0 else (close - low) / rng


def load_gated_rows(db_path: str) -> list[tuple[Row, float, float | None]]:
    """Returns (row, entry_candle_clv, prev_candle_clv) for every real-gated
    closed BUY trade -- eligibility/quality-filter reconstructed causally
    per symbol, exactly like every other backtest this session."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT symbol, interval, entry_timestamp, pnl_percent,
               volatility_margin_at_entry, regime_normalized_at_entry
        FROM trades
        WHERE side = 'buy' AND status = 'closed' AND pnl_percent IS NOT NULL
        ORDER BY symbol, entry_timestamp
        """
    )
    by_symbol: dict[str, list[Row]] = {}
    for r in cur.fetchall():
        row = Row(
            symbol=r["symbol"],
            interval=r["interval"],
            entry_timestamp=r["entry_timestamp"],
            pnl_percent=r["pnl_percent"],
            volatility_margin=r["volatility_margin_at_entry"],
            regime_normalized=r["regime_normalized_at_entry"],
        )
        by_symbol.setdefault(row.symbol, []).append(row)

    gated: list[Row] = []
    for rows in by_symbol.values():
        prior_pnls: list[float] = []
        for row in rows:
            eligible = len(prior_pnls) >= paper_trading.MIN_CLOSED_TRADES and (
                100 * len([p for p in prior_pnls if p > 0]) / len(prior_pnls)
                >= float(paper_trading.MIN_WIN_RATE)
            )
            if (
                eligible
                and row.volatility_margin is not None
                and row.regime_normalized is not None
                and entry_quality_filter.passes_indicator_filter(
                    row.volatility_margin, row.regime_normalized
                )
            ):
                gated.append(row)
            prior_pnls.append(row.pnl_percent)

    # Join candle OHLC for the entry bar and the bar immediately before it.
    result: list[tuple[Row, float, float | None]] = []
    for row in gated:
        cur.execute(
            """
            SELECT timestamp, open, high, low, close FROM candles
            WHERE symbol = ? AND interval = ? AND timestamp <= ?
            ORDER BY timestamp DESC LIMIT 2
            """,
            (row.symbol, row.interval, row.entry_timestamp),
        )
        bars = cur.fetchall()
        if not bars or bars[0]["timestamp"] != row.entry_timestamp:
            continue  # entry candle itself not found, skip
        entry_clv = _clv(bars[0]["high"], bars[0]["low"], bars[0]["close"])
        prev_clv = (
            _clv(bars[1]["high"], bars[1]["low"], bars[1]["close"]) if len(bars) > 1 else None
        )
        result.append((row, entry_clv, prev_clv))
    conn.close()
    return result


def summarize(pnls: list[float]) -> str:
    if not pnls:
        return "n=0"
    wins = [p for p in pnls if p > 0]
    return (
        f"n={len(pnls):>5}  win_rate={100 * len(wins) / len(pnls):5.1f}%  "
        f"expectancy={statistics.mean(pnls):+.3f}%  sum_pnl={sum(pnls):+8.1f}%"
    )


def main() -> None:
    db_path = sys.argv[1] if len(sys.argv) > 1 else "local.db"
    data = load_gated_rows(db_path)
    print(f"Loaded {len(data)} real-gated BUY trades with entry-candle data.\n")

    print("=== Baseline: no conviction filter ===")
    print(summarize([r.pnl_percent for r, _, _ in data]))

    print("\n=== Filtered by ENTRY-candle CLV >= threshold ===")
    for threshold in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        pnls = [r.pnl_percent for r, entry_clv, _ in data if entry_clv >= threshold]
        print(f"CLV >= {threshold:.1f}:  {summarize(pnls)}")

    print("\n=== Filtered by PREVIOUS-candle CLV >= threshold ===")
    for threshold in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        pnls = [
            r.pnl_percent
            for r, _, prev_clv in data
            if prev_clv is not None and prev_clv >= threshold
        ]
        print(f"CLV >= {threshold:.1f}:  {summarize(pnls)}")

    print("\n=== Filtered by BOTH entry AND previous candle CLV >= threshold ===")
    for threshold in (0.0, 0.5, 0.6, 0.7, 0.8):
        pnls = [
            r.pnl_percent
            for r, entry_clv, prev_clv in data
            if prev_clv is not None and entry_clv >= threshold and prev_clv >= threshold
        ]
        print(f"CLV >= {threshold:.1f}:  {summarize(pnls)}")

    print("\n=== Filtered by AVERAGE(entry, previous) CLV >= threshold ===")
    for threshold in (0.0, 0.5, 0.6, 0.7, 0.8):
        pnls = [
            r.pnl_percent
            for r, entry_clv, prev_clv in data
            if prev_clv is not None and (entry_clv + prev_clv) / 2 >= threshold
        ]
        print(f"CLV >= {threshold:.1f}:  {summarize(pnls)}")


if __name__ == "__main__":
    main()
