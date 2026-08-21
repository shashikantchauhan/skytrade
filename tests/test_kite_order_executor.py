"""Tests for KiteOrderExecutor's cash order placement -- specifically the
2026-08-21 fix: Kite rejects a plain market order placed via the API
("Market orders without market protection are not allowed via API"), so
these must go out as a protected LIMIT order instead. Uses a real
KiteConnect instance (so the real order-type/variety/product constants are
exercised, not redefined by hand) with place_order swapped for a capturing
stub -- no real network call.
"""

from decimal import Decimal

from kiteconnect import KiteConnect

from trading_scanner.infrastructure.kite import KiteOrderExecutor


class _CapturingKite(KiteConnect):
    def __init__(self) -> None:
        super().__init__(api_key="test")
        self.calls: list[dict] = []

    def place_order(self, **kwargs):
        self.calls.append(kwargs)
        return "order-1"


def test_cash_buy_order_is_a_protected_limit_order_above_reference_price():
    kite = _CapturingKite()
    executor = KiteOrderExecutor(kite)

    executor.place_cash_market_order("RELIANCE", "BUY", 5, Decimal("1000"))

    call = kite.calls[0]
    assert call["order_type"] == kite.ORDER_TYPE_LIMIT
    assert call["variety"] == kite.VARIETY_REGULAR
    assert call["exchange"] == "NSE"
    assert call["product"] == kite.PRODUCT_CNC
    assert call["transaction_type"] == "BUY"
    assert call["quantity"] == 5
    # 0.5% protection above the reference price for a BUY -- still fills
    # immediately against real liquidity, just satisfies Kite's requirement
    # for an explicit price instead of a bare market order.
    assert call["price"] == 1005.0


def test_cash_sell_order_is_a_protected_limit_order_below_reference_price():
    kite = _CapturingKite()
    executor = KiteOrderExecutor(kite)

    executor.place_cash_market_order("RELIANCE", "SELL", 5, Decimal("1000"))

    call = kite.calls[0]
    assert call["order_type"] == kite.ORDER_TYPE_LIMIT
    assert call["transaction_type"] == "SELL"
    # 0.5% protection *below* the reference price for a SELL.
    assert call["price"] == 995.0


def test_cash_order_price_rounds_to_two_decimals():
    kite = _CapturingKite()
    executor = KiteOrderExecutor(kite)

    executor.place_cash_market_order("RELIANCE", "BUY", 5, Decimal("1309.65"))

    # 1309.65 * 1.005 = 1316.199825 -> rounds to 1316.20
    assert kite.calls[0]["price"] == 1316.20
