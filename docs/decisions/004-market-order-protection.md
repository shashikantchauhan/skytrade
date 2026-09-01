# 004: Real MARKET orders with explicit market_protection, not a synthetic limit

## Status
Accepted, 2026-08-31 (documented retroactively as part of this refactor).

## Context
The very first real order this system ever placed (2026-08-21) was
rejected: "Market orders without market protection are not allowed via
API." The initial fix priced a synthetic protected LIMIT order off the
signal's reference price instead of a true market order. That reference
price can already be a few seconds stale by the time the order reaches
the exchange; if price had moved past the fixed limit in the meantime,
the order could sit unfilled rather than execute -- working around the
symptom, not the actual constraint.

Kite's `place_order` has always accepted a `market_protection` parameter
directly; the LIMIT workaround was never necessary.

## Decision
`KiteOrderExecutor.place_cash_market_order` places a true
`ORDER_TYPE_MARKET` order with an explicit `market_protection` percentage
(2%, not Kite's `-1` "automatic" setting -- a Kite Connect developer forum
thread surfaced real reports of `-1` picking a band too narrow during a
fast move, with Zerodha's own team recommending an explicit percentage
instead). A real MARKET order prices its protection band off the
*current* exchange price at execution, not a stale reference price.

## Consequences
- No more synthetic LIMIT-order slippage risk from a stale reference
  price.
- Execution past NSE's own exchange-side price-protection band still
  can't be guaranteed -- 2% reduces, does not eliminate, the chance of an
  order sitting unfilled during an extreme move. Acceptable given this
  system's position sizing (~Rs50,000/trade).

See `infrastructure/kite.py`'s `place_cash_market_order` docstring for the
full reasoning and the exact incident dates.
