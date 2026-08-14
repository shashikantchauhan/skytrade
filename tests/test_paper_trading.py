from decimal import Decimal

from trading_scanner.application import paper_trading
from trading_scanner.application.paper_trading import stop_loss_price, trailing_stop_price


def test_stop_loss_price_is_below_entry_by_the_configured_percent():
    assert stop_loss_price(Decimal("100")) == Decimal("97")  # 3% default


def test_trailing_stop_not_activated_below_the_activation_threshold():
    # Default activation is 15% -- a peak only 10% above entry hasn't
    # triggered the trail yet, so there is no trailing-stop price at all
    # (the position is only ever protected by the hard stop-loss here).
    entry = Decimal("100")
    peak = Decimal("110")
    assert trailing_stop_price(entry, peak) is None


def test_trailing_stop_activates_once_peak_clears_the_threshold():
    entry = Decimal("100")
    peak = Decimal("115")  # exactly the 15% activation threshold
    assert trailing_stop_price(entry, peak) == Decimal("115") * (1 - Decimal("3") / 100)


def test_trailing_stop_follows_new_peaks_not_entry():
    # Once activated, the trail is anchored to the highest price seen, not
    # a fixed distance from entry -- a stronger run tightens the floor
    # upward as it goes, exactly like a real trailing stop order would.
    entry = Decimal("100")
    low_peak_stop = trailing_stop_price(entry, Decimal("120"))
    high_peak_stop = trailing_stop_price(entry, Decimal("150"))
    assert high_peak_stop > low_peak_stop


def test_trailing_stop_respects_env_overridden_thresholds(monkeypatch):
    monkeypatch.setattr(paper_trading, "TRAILING_STOP_ACTIVATION_PCT", Decimal("5"))
    monkeypatch.setattr(paper_trading, "TRAILING_STOP_TRAIL_PCT", Decimal("1"))

    entry = Decimal("100")
    # 4% up -- below the overridden 5% activation.
    assert paper_trading.trailing_stop_price(entry, Decimal("104")) is None
    # 5% up -- activates, trails 1% below the peak.
    assert paper_trading.trailing_stop_price(entry, Decimal("105")) == Decimal("105") * (
        1 - Decimal("1") / 100
    )
