import pandas as pd
import pytest

from analysis.nifty500_swing_backtest import _cusum


def test_cusum_accumulates_only_after_drift() -> None:
    values = pd.Series([0.1, 0.4, 0.5, -0.9])

    result = _cusum(values, drift=0.2)

    assert result.tolist() == pytest.approx([0.0, 0.2, 0.5, -0.7])
