import pandas as pd

from analysis.hourly_swing_comparison import aggregate_complete_hourly


def test_hourly_aggregation_uses_nse_aligned_pairs_and_drops_stub() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-09-10 03:45:00+00:00",
            "2026-09-10 04:15:00+00:00",
            "2026-09-10 04:45:00+00:00",
            "2026-09-10 05:15:00+00:00",
            "2026-09-10 09:45:00+00:00",
        ]
    )
    bars = pd.DataFrame(
        {
            "symbol": "TEST.NS",
            "timestamp": timestamps,
            "date": timestamps.date,
            "open": [100, 101, 103, 104, 106],
            "high": [102, 104, 105, 107, 108],
            "low": [99, 100, 102, 103, 105],
            "close": [101, 103, 104, 106, 107],
            "volume": [10, 20, 30, 40, 50],
        }
    )

    result = aggregate_complete_hourly(bars)

    assert len(result) == 2
    assert result.open.tolist() == [100, 103]
    assert result.close.tolist() == [103, 106]
    assert result.high.tolist() == [104, 107]
    assert result.low.tolist() == [99, 102]
    assert result.volume.tolist() == [30, 70]
    assert result.timestamp.tolist() == timestamps[[1, 3]].tolist()
