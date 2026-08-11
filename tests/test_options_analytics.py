from datetime import UTC, datetime
from decimal import Decimal

from trading_scanner.application.options_analytics import compute_greeks, enrich_trade


def test_compute_greeks_put_returns_sane_values():
    result = compute_greeks(
        "PE",
        underlying_price=Decimal("1400"),
        strike=Decimal("1400"),
        premium=Decimal("25"),
        expiry="2026-08-26",
        as_of=datetime(2026, 8, 11, tzinfo=UTC),
    )

    assert result is not None
    assert 0 < result["implied_volatility"] < 100
    assert -1 < result["delta"] < 0  # a put's delta is always negative
    assert result["theta"] < 0  # time decay always hurts a long option
    assert result["gamma"] > 0
    assert result["vega"] > 0


def test_compute_greeks_call_delta_is_positive():
    result = compute_greeks(
        "CE",
        underlying_price=Decimal("1400"),
        strike=Decimal("1400"),
        premium=Decimal("25"),
        expiry="2026-08-26",
        as_of=datetime(2026, 8, 11, tzinfo=UTC),
    )

    assert result is not None
    assert 0 < result["delta"] < 1


def test_compute_greeks_none_when_expiry_already_passed():
    result = compute_greeks(
        "CE",
        underlying_price=Decimal("1400"),
        strike=Decimal("1400"),
        premium=Decimal("25"),
        expiry="2026-08-01",
        as_of=datetime(2026, 8, 11, tzinfo=UTC),
    )

    assert result is None


def test_compute_greeks_none_when_premium_below_intrinsic_value():
    # A CE at strike 1000 with spot 1400 has intrinsic value >= 400 --
    # a quoted premium of 1 is not a valid Black-Scholes input.
    result = compute_greeks(
        "CE",
        underlying_price=Decimal("1400"),
        strike=Decimal("1000"),
        premium=Decimal("1"),
        expiry="2026-08-26",
        as_of=datetime(2026, 8, 11, tzinfo=UTC),
    )

    assert result is None


def test_enrich_trade_open_position_has_no_exit_greeks():
    result = enrich_trade(
        "PE",
        strike=Decimal("1400"),
        expiry="2026-08-26",
        entry_timestamp=datetime(2026, 8, 11, tzinfo=UTC),
        underlying_price_at_entry=Decimal("1400"),
        entry_premium=Decimal("25"),
    )

    assert result["entry"] is not None
    assert result["exit"] is None


def test_enrich_trade_closed_position_has_both_entry_and_exit_greeks():
    result = enrich_trade(
        "PE",
        strike=Decimal("1400"),
        expiry="2026-08-26",
        entry_timestamp=datetime(2026, 8, 11, tzinfo=UTC),
        underlying_price_at_entry=Decimal("1400"),
        entry_premium=Decimal("25"),
        exit_timestamp=datetime(2026, 8, 15, tzinfo=UTC),
        underlying_price_at_exit=Decimal("1380"),
        exit_premium=Decimal("35"),
    )

    assert result["entry"] is not None
    assert result["exit"] is not None
