import time

import pytest

from trading_scanner.alpha_engine import AlphaEngine
from trading_scanner.application.fast_predict import bootstrap_queue_state, evaluate_latest_bar
from trading_scanner.infrastructure.yahoo import YahooFinanceProvider
from trading_scanner.validation.runner import _changed, _signal_state

# Real production config -- see signal_pipeline._ENGINE_SETTINGS.
_ENGINE_SETTINGS = {"include_full_history": True, "use_dynamic_exits": True}


@pytest.fixture(scope="module")
def real_history():
    """Real AARTIIND.NS 1h history, downloaded once and reused by every test."""
    return YahooFinanceProvider().get_recent_history("AARTIIND.NS", "1h", 729).sort_index()


def _ground_truth(engine: AlphaEngine, window):
    """One full (expensive) AlphaEngine pass, exposing what analyze() hides:
    the raw signal state array and dynamic-exit booleans, so tests can check
    exactly what a real hourly run should produce at the final bar.
    """
    open_ = window["Open"].to_numpy(dtype=float)
    high = window["High"].to_numpy(dtype=float)
    low = window["Low"].to_numpy(dtype=float)
    close = window["Close"].to_numpy(dtype=float)
    source = close.copy()

    features = engine._calculate_features(close, high, low)
    filter_all = engine._calculate_filters(open_, source, high, low, close)
    kernel = engine._kernel_regression(source)
    events = engine._predict(features, source, filter_all, kernel, close)
    signal_array = _signal_state(events["prediction"], filter_all)
    changed_array = _changed(signal_array)

    ground_truth_signal = (
        "BUY"
        if events["start_long"][-1]
        else "SELL" if events["start_short"][-1] else "NEUTRAL"
    )
    is_early_signal_flip = bool(changed_array[-1]) and any(
        bool(value) for value in changed_array[-4:-1]
    )
    return {
        "prediction": int(events["prediction"][-1]),
        "signal": ground_truth_signal,
        "end_long": bool(events["end_long"][-1]),
        "end_short": bool(events["end_short"][-1]),
        "is_early_signal_flip": is_early_signal_flip,
        # The raw signal state this window's own last bar ended up in --
        # what `signal_previous` should equal after bootstrapping/evaluating
        # up through that bar.
        "signal_at_last_bar": int(signal_array[-1]),
    }


@pytest.mark.parametrize("length", [1500, 5046])
def test_bootstrap_and_evaluate_matches_ground_truth(real_history, length) -> None:
    """The actual production first-time-seeing-a-symbol path --
    bootstrap_queue_state(window[:-1]) then evaluate_latest_bar(window) --
    must agree with a full AlphaEngine pass at the latest bar, for both
    entries (signal/prediction) and exits (end_long/end_short/early flip).
    This exercises bootstrap_queue_state's own output directly (not a
    separately-derived one), which is what production actually does; an
    earlier version of this test fed in a manually-derived `signal_previous`
    instead and did not catch that bootstrap_queue_state was returning 0
    rather than the correctly-replayed value.
    """
    window = real_history.tail(length)
    engine = AlphaEngine(**_ENGINE_SETTINGS)

    bootstrap = bootstrap_queue_state(engine, window.iloc[:-1])
    fast_result = evaluate_latest_bar(
        engine, window, bootstrap.signal_previous, bootstrap.queue_state, bootstrap.exit_state
    )

    truth_full = _ground_truth(engine, window)
    assert fast_result.prediction == truth_full["prediction"]
    assert fast_result.signal == truth_full["signal"]
    assert fast_result.end_long == truth_full["end_long"]
    assert fast_result.end_short == truth_full["end_short"]
    assert fast_result.is_early_signal_flip == truth_full["is_early_signal_flip"]


def test_bootstrap_signal_previous_matches_ground_truth(real_history) -> None:
    """bootstrap_queue_state's signal_previous must equal the raw signal
    state a full AlphaEngine pass would show at the bar just before the
    newest one -- not 0. Below-`filter_all` bars carry forward whatever the
    state was arbitrarily far in the past, so 0 is only correct by luck."""
    window = real_history.tail(1500)
    engine = AlphaEngine(**_ENGINE_SETTINGS)

    bootstrap = bootstrap_queue_state(engine, window.iloc[:-1])
    truth = _ground_truth(engine, window.iloc[:-1])

    assert bootstrap.signal_previous == truth["signal_at_last_bar"]


def test_consecutive_hourly_runs_match_a_single_batch_pass(real_history) -> None:
    """Stepping evaluate_latest_bar forward a few bars at a time (as real
    hourly runs would, carrying signal_previous/queue_state/exit_state
    between calls, starting from the real bootstrap output) must converge to
    the same result a single full batch pass over the same final window
    gives -- this is the actual shape production runs in, for both entries
    and exits."""
    window = real_history.tail(1500)
    engine = AlphaEngine(**_ENGINE_SETTINGS)
    steps = 5  # simulate 5 consecutive hourly runs

    bootstrap = bootstrap_queue_state(engine, window.iloc[:-steps])
    signal_previous = bootstrap.signal_previous
    queue_state = bootstrap.queue_state
    exit_state = bootstrap.exit_state

    result = None
    for end in range(len(window) - steps + 1, len(window) + 1):
        result = evaluate_latest_bar(
            engine, window.iloc[:end], signal_previous, queue_state, exit_state
        )
        signal_previous = result.signal_previous
        queue_state = result.queue_state
        exit_state = result.exit_state

    truth_full = _ground_truth(engine, window)
    assert result.prediction == truth_full["prediction"]
    assert result.signal == truth_full["signal"]
    assert result.end_long == truth_full["end_long"]
    assert result.end_short == truth_full["end_short"]
    assert result.is_early_signal_flip == truth_full["is_early_signal_flip"]


def test_evaluate_latest_bar_is_fast(real_history) -> None:
    engine = AlphaEngine(**_ENGINE_SETTINGS)
    bootstrap = bootstrap_queue_state(engine, real_history.iloc[:-1])

    start = time.time()
    evaluate_latest_bar(
        engine, real_history, bootstrap.signal_previous, bootstrap.queue_state, bootstrap.exit_state
    )
    elapsed = time.time() - start

    assert elapsed < 5.0, f"evaluate_latest_bar took {elapsed:.2f}s, expected well under 5s"
