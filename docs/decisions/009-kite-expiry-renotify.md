# 009: Re-alert on an expired Kite session every 15 minutes, not once a day

## Status
Accepted, 2026-09-02.

## Context
Kite's access token expires on a fixed daily reset with no documented exact
time, requiring a manual re-login (password + OTP -- Kite Connect has no
refresh-token flow) every trading morning. The pipeline already detected
this (a cheap `kite.profile()` call before ever touching the ticker) and
sent one "please log in again" Telegram alert per calendar day
(`kite_session.expiry_notified_date`), then backed off and retried the
check every ~60s without saying anything further.

In production, that single daily alert was missed or seen late on three
consecutive trading days -- the session sat expired at market open for
30-104 minutes each morning (2026-08-31: ~71 min, 2026-09-01: ~30 min,
2026-09-02: ~104 min) before anyone logged back in. The pipeline is
completely idle during that window: no candle evaluation, no order
placement, real capital sitting unused for no reason visible without
reading server logs by hand.

## Decision
`kite_session` gains a new `expiry_notified_at` column (an ISO timestamp,
alongside the now-unused `expiry_notified_date`). The renamed
`_notify_kite_expired_periodically` (was `_notify_kite_expired_once_per_
day`) re-sends the same alert every 15 minutes (`_EXPIRY_RENOTIFY_
INTERVAL`) for as long as the session stays broken, instead of once per
calendar day. Every call site (`live_pipeline.py`'s always-on ticker path,
the older `run_signal_pipeline` download path, and the ticker watchdog)
already re-runs this check on its own ~60s/hourly cadence -- the only
change is how long a "no need to resend yet" window this function itself
enforces.

15 minutes is not a full fix -- it still requires a human to see the alert
and act. It bounds how long one missed ping can silently cost, rather than
solving "make sure someone always sees Telegram."

## Consequences
- A missed morning login now costs at most ~15 minutes of unawareness
  instead of potentially the rest of the day.
- `expiry_notified_date`/`get_expiry_notified_date`/`set_expiry_notified_
  date` are kept (real, already-persisted columns/methods) but nothing
  writes to them anymore -- `expiry_notified_at` is the live path.
- New tests: `tests/test_kite_expiry_notification.py` (first-call send,
  no-resend-within-window, resend-after-window, no-notifier no-op,
  round-trip persistence).
- Does not address the underlying limitation that Kite Connect has no way
  to keep a session alive without a human logging in each day -- that's a
  real constraint of the API, not something this fixes.

See `application/pipeline/market_data.py`'s `_notify_kite_expired_
periodically` and `infrastructure/db/kite_session.py`'s `expiry_notified_
at` for the full reasoning.
