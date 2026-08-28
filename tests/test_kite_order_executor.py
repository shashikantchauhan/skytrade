"""Tests for KiteOrderExecutor's cash order placement -- specifically the
2026-08-21 fix: Kite rejects a plain market order placed via the API
("Market orders without market protection are not allowed via API"), so
these must go out as a protected LIMIT order instead. Uses a real
KiteConnect instance (so the real order-type/variety/product constants are
exercised, not redefined by hand) with place_order/instruments swapped for
capturing/faking stubs -- no real network call.

Also covers the 2026-08-25 fix: the protected price must round to this
instrument's own real tick size (from Kite's instrument dump), not a flat
2 decimals -- a real production bug (InputException: "Tick size for this
script is 0.10 ...") that rejected 10 of 14 real orders in one session.
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


def test_cash_order_price_rounds_to_the_instruments_tick_size():
    kite = _CapturingKite(tick_size="0.05")
    executor = KiteOrderExecutor(kite)

    executor.place_cash_market_order("RELIANCE", "BUY", 5, Decimal("1309.65"))

    # 1309.65 * 1.005 = 1316.199825 -> nearest 0.05 multiple -> 1316.20
    assert kite.calls[0]["price"] == 1316.20


def test_cash_order_price_rounds_to_a_010_tick_size():
    # 2026-08-25 regression: RELIANCE-like stocks tick at 0.05, but plenty
    # of real NSE stocks tick at 0.10 -- a flat round(price, 2) guess
    # produces a price Kite rejects outright for those.
    kite = _CapturingKite(tick_size="0.10")
    executor = KiteOrderExecutor(kite)

    executor.place_cash_market_order("RELIANCE", "BUY", 3, Decimal("1380.5"))

    # 1380.5 * 1.005 = 1387.4025 -> nearest 0.10 multiple -> 1387.40
    assert kite.calls[0]["price"] == 1387.40


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
