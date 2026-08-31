"""Tests for KiteOrderExecutor's cash order placement.

2026-08-21: Kite rejects a plain market order placed via the API ("Market
orders without market protection are not allowed via API"). 2026-08-31:
place_order() has always accepted a market_protection parameter directly
-- place_cash_market_order now sends a real ORDER_TYPE_MARKET order with
market_protection=-1 instead of a synthetic protected LIMIT order priced
off a (possibly stale) reference price. Uses a real KiteConnect instance
(so the real order-type/variety/product constants are exercised, not
redefined by hand) with place_order/instruments swapped for capturing/
faking stubs -- no real network call.

tick_size/round_to_tick are no longer exercised by cash order placement
itself (a real MARKET order has no price to round) -- see
test_gtt_bracket.py for where tick rounding still matters (GTT bracket
LIMIT legs).
"""

from decimal import Decimal

from kiteconnect import KiteConnect

from trading_scanner.infrastructure.kite import KiteOrderExecutor


class _CapturingKite(KiteConnect):
    def __init__(
        self,
        tick_size: str = "0.05",
        net_positions: list[dict] | None = None,
        holdings: list[dict] | None = None,
    ) -> None:
        super().__init__(api_key="test")
        self.calls: list[dict] = []
        self._tick_size = tick_size
        self._net_positions = net_positions or []
        self._holdings = holdings or []

    def place_order(self, **kwargs):
        self.calls.append(kwargs)
        return "order-1"

    def instruments(self, exchange=None):
        return [
            {
                "exchange": "NSE",
                "tradingsymbol": "RELIANCE",
                "instrument_token": 1,
                "tick_size": self._tick_size,
            }
        ]

    def positions(self):
        return {"net": self._net_positions, "day": []}

    def holdings(self):
        return self._holdings


def test_cash_buy_order_is_a_real_market_order_with_protection():
    kite = _CapturingKite()
    executor = KiteOrderExecutor(kite)

    executor.place_cash_market_order("RELIANCE", "BUY", 5, Decimal("1000"))

    call = kite.calls[0]
    assert call["order_type"] == kite.ORDER_TYPE_MARKET
    assert call["variety"] == kite.VARIETY_REGULAR
    assert call["exchange"] == "NSE"
    assert call["product"] == kite.PRODUCT_CNC
    assert call["transaction_type"] == "BUY"
    assert call["quantity"] == 5
    # -1 = Kite's own automatic protection band, applied against the
    # *current* exchange price at execution -- not a price we compute here.
    assert call["market_protection"] == -1
    assert "price" not in call


def test_cash_sell_order_is_a_real_market_order_with_protection():
    kite = _CapturingKite()
    executor = KiteOrderExecutor(kite)

    executor.place_cash_market_order("RELIANCE", "SELL", 5, Decimal("1000"))

    call = kite.calls[0]
    assert call["order_type"] == kite.ORDER_TYPE_MARKET
    assert call["transaction_type"] == "SELL"
    assert call["market_protection"] == -1
    assert "price" not in call


def test_holding_quantity_adds_a_same_day_buy_to_a_prior_day_holding():
    kite = _CapturingKite(
        net_positions=[{"product": "CNC", "tradingsymbol": "RELIANCE", "quantity": 5}],
        holdings=[{"product": "CNC", "tradingsymbol": "RELIANCE", "quantity": 2, "t1_quantity": 0}],
    )
    executor = KiteOrderExecutor(kite)

    assert executor.holding_quantity("RELIANCE") == 7


def test_holding_quantity_ignores_a_negative_same_day_sell_quantity():
    # 2026-08-28 regression: positions()['net'] shows a *negative* quantity
    # for a same-day SELL of shares that came from yesterday's holdings --
    # confirmed live against UNIONBANK.NS right after this app sold it:
    # holdings() had already dropped to 0, but summing the -26 straight in
    # produced a nonsensical negative "holding".
    kite = _CapturingKite(
        net_positions=[{"product": "CNC", "tradingsymbol": "RELIANCE", "quantity": -26}],
        holdings=[{"product": "CNC", "tradingsymbol": "RELIANCE", "quantity": 0, "t1_quantity": 0}],
    )
    executor = KiteOrderExecutor(kite)

    assert executor.holding_quantity("RELIANCE") == 0


def test_holding_quantity_counts_a_t1_unsettled_lot():
    kite = _CapturingKite(
        holdings=[{"product": "CNC", "tradingsymbol": "RELIANCE", "quantity": 0, "t1_quantity": 4}],
    )
    executor = KiteOrderExecutor(kite)

    assert executor.holding_quantity("RELIANCE") == 4


def test_holding_quantity_ignores_other_symbols_and_products():
    kite = _CapturingKite(
        net_positions=[{"product": "CNC", "tradingsymbol": "TCS", "quantity": 5}],
        holdings=[
            {"product": "CNC", "tradingsymbol": "RELIANCE", "quantity": 1, "t1_quantity": 0},
            {"product": "NRML", "tradingsymbol": "RELIANCE", "quantity": 99, "t1_quantity": 0},
        ],
    )
    executor = KiteOrderExecutor(kite)

    assert executor.holding_quantity("RELIANCE") == 1
