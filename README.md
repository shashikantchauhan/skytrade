# SkyTrade

This project downloads and validates market candles so that a strategy can be
added in a later milestone.

## ⚠️ Read this first — Project Plan & Status

This section exists so work can resume from a completely different
machine/account with zero prior context. Read this before touching code.

### What this project actually is

A Python port of a TradingView "Lorentzian Classification" Pine Script
strategy (`strategy/lorentzian.pine`, `strategy/MLExtensions_v2.pine`), run
hourly against ~220 NSE F&O stocks + sector indices, that:
1. Generates BUY/SELL signals matching the TradingView chart bar-for-bar.
2. Tracks every entry/exit as a `Trade` for real win-rate statistics.
3. Runs a simulated (paper) ₹8,00,000 trading account on top of that,
   long-only, gated by each symbol's own track record and available capital,
   with dynamically-sized position slots that scale as the account compounds.
4. Notifies everything via Telegram.

The end goal (not yet reached) is: validate the paper account is actually
profitable for about a week, then connect it to a real Zerodha account for
live trading with real accounting.

- **Long-only, always.** NSE cash market does not allow short selling for
  multi-day (delivery) holds — only intraday MIS, squared off same day. This
  strategy's average holding period is ~3.5 days, so **SELL signals can
  never become real trades** in the cash market. They still notify
  (informational only) but must never touch the paper account or (later)
  place a real order. This is already implemented correctly — do not "fix"
  it into placing short orders.
- **`alpha_engine.py` (the Pine Script translation) is not to be modified.**
  All fixes/extensions go in the `application/` layer around it, reusing its
  private methods, so the core translation stays a faithful 1:1 port that
  can always be checked against TradingView.
- Personal coding standards from `~/.claude/CLAUDE.md` apply throughout: no
  mutation, pure functions, flat/early-return logic, fail loudly (no silent
  `None`/swallowed exceptions), comment only the *why*.

### Key technical decisions already made (don't re-litigate)

- **Pine-faithful trade scoring**: every trade (live and backtest) prices
  entry/exit at `(high + low + open + open) / 4` — Pine's own `ml.backtest`
  convention, not the close. See `application/backtest.py`'s module
  docstring for the full derivation and why this (plus single-active-
  position tracking) was the actual root cause of an earlier win-rate
  mismatch against TradingView's own on-chart stats table.
- **Single active position per symbol**: a new opposite-side entry silently
  *abandons* (never scores as win/loss) whatever position was still open —
  mirrors Pine's own overwritable `start_long_trade`/`start_short_trade`
  variables exactly. See `TradeRepository.abandon_open_trade`.
- **`AlphaEngine(include_full_history=True, use_dynamic_exits=True)`** is the
  production config everywhere (live pipeline and backtest) — matches the
  real TradingView chart's input overrides.
- **Residual TradingView gap accepted as unfixable**: even with the above
  fixes, win rate doesn't match TradingView 100% exactly, attributed to
  Yahoo Finance's shorter historical data depth vs. TradingView's own feed
  (`include_full_history=True`'s neighbor search sees whatever history is
  available — deeper on TradingView). Not fixable without a paid longer-
  history data source. Accepted as good enough; do not keep chasing this.
- **Stock-split data corruption**: Yahoo's incremental candle upsert never
  retroactively re-adjusts old rows after a real split, permanently
  corrupting the price scale at the split date (e.g. BAJFINANCE.NS showed a
  fake +80% "profit" from an actual 1:5 split). Fixed once via a full
  wipe-and-redownload of all candle history. If this happens again (watch
  for implausible >40% single-bar jumps), the fix is the same: wipe
  `candles`/`engine_state`/`trades` for the affected symbol(s) and
  re-download fresh in one call (self-consistent split adjustment).
- **Paper-trading sizing rationale (updated after the full 220-symbol
  backtest)**: real concurrent-position demand, via Little's Law (concurrent
  positions ≈ entries/day × average holding period) computed only over
  symbols that actually clear the 55% eligibility bar (ineligible symbols
  never reach `try_open_position`), works out to **~32 concurrent slots** —
  not the ~6-7 an earlier, much smaller partial-data estimate suggested.
  `INITIAL_CAPITAL` is now ₹8,00,000 and `TARGET_SLOTS` is 32, giving a
  starting slot size of ~₹25,000. **Slot size is dynamic**, not fixed:
  every entry recomputes `total_equity / TARGET_SLOTS` (floored at
  `MIN_POSITION_SIZE`, ₹25,000), so slots grow automatically as the account
  compounds profit — no manual re-tuning needed. The ₹25,000 floor keeps the
  flat per-trade DP charge (~₹18, sell-side only) under ~5% of an average
  winning trade's profit; below that, flat fees start eating a
  disproportionate share of returns.

### Current status (last touched)

- Paper-trading account code is **built, wired in, and fully tested**
  (33/33 tests passing) — see the "Paper trading account" section below for
  what it does. Nothing further needed here unless requirements change.
- A full 220-symbol historical backtest re-run (`python -m
  trading_scanner.backtest`, rebuilding the `trades` table with the
  Pine-faithful scoring above) was intentionally **stopped partway through**
  (~85/220 symbols, around `GRASIM.NS`) so the machine could be freed up to
  move to a different laptop/account. **This needs to be re-run to
  completion** before the paper-trading eligibility filter has a full,
  reliable picture across all 220 symbols (it degrades gracefully with
  partial data — symbols never backtested just show "not eligible yet" —
  but the plan calls for full coverage). Before re-running: confirm no
  other backtest process is already running (`ps aux | grep
  trading_scanner.backtest`) to avoid the double-DELETE race that happened
  once earlier.
- `config/symbols.txt` is fully populated: 220 symbols (208 real NSE F&O
  stocks + `AARTIIND.NS` + 11 sector indices). Nothing to redo here.
- GitHub Actions schedule (`.github/workflows/hourly-signals.yml`) is
  already set to NSE market hours only. Nothing to redo here unless the
  schedule needs adjusting.

### Pending / next steps, in order

1. Re-run the full 220-symbol backtest to completion (see above).
2. Share the completed backtest results **one symbol at a time** (total
   wins, total losses, win rate %, WL ratio only — explicitly NOT a CSV
   export, NOT trade-by-trade detail — the user corrected this format
   explicitly once already).
3. Run the live hourly pipeline for real (locally or via the GitHub Action)
   for about a week and let the paper account accumulate real signals/trades.
4. Review paper-account performance after that week. Only if genuinely
   profitable, proceed to Phase 2.
5. **Phase 2 (future, not yet started)**: buy a Zerodha Kite Connect
   market-data subscription (~₹2k, user has already agreed to this spend
   conditionally on Phase 1 being profitable) — needed for real-time
   execution regardless. Then explore buying PE (put) options as a
   synthetic short mechanism for SELL signals (impossible to backtest
   properly today since Yahoo has no Indian stock options data; Kite
   Connect does).
6. Eventually: real Zerodha live-trading integration with real accounting,
   once paper trading has proven the strategy out.

### Secrets that must travel with the code (not in git, not in this repo folder by default)

`.env` holds real production credentials — copy it separately/securely, not
alongside a public code share:
- `TRADING_SCANNER_TURSO_URL`, `TRADING_SCANNER_TURSO_AUTH_TOKEN` (hosted DB —
  already has real accumulated candle/trade/paper-account history in it)
- `TRADING_SCANNER_TELEGRAM_BOT_TOKEN`, `TRADING_SCANNER_TELEGRAM_CHAT_ID`

## Project Structure

```text
src/trading_scanner/
├── alpha_engine.py      # TradingView Lorentzian Classification translation
├── application/
│   ├── scanner.py       # Per-symbol download, validation, and summary output
│   ├── symbols.py       # SymbolLoader for the symbols text file
│   └── validation.py    # OHLCV candle validation
├── config/
│   └── settings.py      # Centralized application configuration
├── infrastructure/
│   └── yahoo.py         # YahooProvider market-data adapter
├── validation/
│   └── runner.py         # Per-bar AlphaEngine CSV validation
├── main.py              # Application orchestration
└── validate.py          # Validation command-line entry point
config/
└── symbols.txt          # One market symbol per line
```

## Running Project

Python 3.12 is required.

```bash
poetry install
poetry run trading-scanner
```

Alternatively, install the project dependencies and run:

```bash
PYTHONPATH=src python -m trading_scanner.main
```

## Configuration

Configuration is centralized in `trading_scanner.config.settings`. Defaults can
be overridden with environment variables or a `.env` file:

| Setting | Environment variable | Default |
| --- | --- | --- |
| Scan interval | `TRADING_SCANNER_SCAN_INTERVAL_HOURS` | `1` hour |
| Candle interval | `TRADING_SCANNER_CANDLE_INTERVAL` | `1h` |
| Candle history | `TRADING_SCANNER_CANDLE_HISTORY` | `300` candles |
| Symbols file | `TRADING_SCANNER_SYMBOLS_FILE` | `config/symbols.txt` |
| Logging level | `TRADING_SCANNER_LOGGING_LEVEL` | `INFO` |

The scan interval is configuration only at this stage; the application runs one
scan and does not schedule recurring work.

## Symbols File

Place one Yahoo Finance symbol per line in `config/symbols.txt`. Blank lines,
whitespace, and duplicate symbols are ignored. A missing file produces a clear
error during startup.

## Yahoo Provider

`YahooProvider` downloads unadjusted historical OHLCV data with `yfinance`.
Results are ordered by datetime, incomplete rows are removed, and the returned
data is limited to the configured candle history. Before display, the scanner
requires at least 200 rows and the `Open`, `High`, `Low`, `Close`, and `Volume`
columns.

## Example Output

```text
2026-08-06 10:00:00 INFO: Application Started
2026-08-06 10:00:00 INFO: Loaded 5 symbols
2026-08-06 10:00:00 INFO: Downloading RELIANCE.NS
--------------------------------------------------

RELIANCE.NS

Rows:
300

Latest Candle

Time: 2026-08-06 09:15:00+05:30
Open: 1420.5
High: 1428.0
Low: 1418.1
Close: 1425.7
Volume: 1250000
--------------------------------------------------
2026-08-06 10:00:01 INFO: Completed RELIANCE.NS
2026-08-06 10:00:05 INFO: Application Finished
```

## Alpha Engine Validation

Export every candle's Alpha Engine result to compare it bar-by-bar with the
TradingView indicator:

```bash
PYTHONPATH=src python -m trading_scanner.validate \
  --symbol AARTIIND.NS \
  --interval 1h \
  --days 10
```

The command downloads all available hourly candles from the requested calendar
day window, evaluates AlphaEngine once for every historical candle, and writes
the result to `validation/validation_AARTIIND_1h.csv`. The exchange suffix is
removed from the filename; for example, `AARTIIND.NS` becomes `AARTIIND`.

The CSV includes each bar's prediction, filter states, entry/exit conditions,
and final BUY/SELL event. Use it to compare every `start_long_trade` and
`start_short_trade` value against TradingView.

### Debug CSV

Every run also writes `validation/validation_debug_AARTIIND_1h.csv`. It
contains the same OHLC and signal columns as the main CSV plus every
intermediate value that feeds the final signal, so a mismatch against
TradingView can be traced bar-by-bar without re-deriving anything:

`timestamp`, `open`, `high`, `low`, `close`, `prediction`, `is_bullish`,
`is_bearish`, `is_ema_uptrend`, `is_ema_downtrend`, `is_sma_uptrend`,
`is_sma_downtrend`, `filter_all`, `bars_held`, `is_new_buy_signal`,
`is_new_sell_signal`, `start_long_trade`, `start_short_trade`,
`end_long_trade`, `end_short_trade`, `kernel_estimate`, `kernel_bullish`,
`kernel_bearish`, `signal`.

When one signal differs from TradingView, open this file at that timestamp
and compare each intermediate column to find exactly which filter or state
diverged.

### Verbose mode

Add `--verbose` to print every BUY/SELL bar to the console while the CSVs are
generated:

```bash
PYTHONPATH=src python -m trading_scanner.validate \
  --symbol AARTIIND.NS \
  --interval 1h \
  --days 10 \
  --verbose
```

```text
2026-08-02 11:00:00+05:30
Prediction: 8
BUY
--------------------------------
```

Bars with a `NEUTRAL` signal are not printed.

## Hourly Signal Pipeline

`trading_scanner.signals` runs the scanner across every symbol in
`config/symbols.txt` on a schedule, accumulating candles in a database so
AlphaEngine has real warm-up history instead of re-downloading a short window
every run. It stores candles in [Turso](https://turso.tech) (hosted
libSQL/SQLite), tracks which signals have already been notified, and sends
BUY/SELL signals through `TelegramNotifier` (or logs them if Telegram isn't
configured).

For each symbol, every run:

1. Downloads a small recent window from Yahoo Finance (just enough to cover
   any gap since the last run) and upserts it into the `candles` table.
2. Loads the **full** accumulated history for that symbol — never truncated,
   since matching TradingView requires the neighbor search to always see the
   same starting point (see `application/fast_predict.py`'s module docstring).
3. Skips the symbol with a "warming up" log line until at least 200 candles
   are stored — AlphaEngine's regime filter needs that much history to be
   meaningful.
4. Evaluates only the newest bar (`application/fast_predict.evaluate_latest_bar`)
   instead of AlphaEngine's full-history `analyze()` — the ANN neighbor queue
   and exit-tracking state are persisted between runs (`engine_state` table)
   so each run costs ~1-3s instead of the ~43s a full re-derivation would
   take. A new symbol pays a one-time "bootstrapping" cost to build that
   state from its accumulated history.
5. Notifies once per new BUY/SELL signal (de-duplicated by `Signal.fingerprint`,
   so re-running the same hour never double-notifies), and records the entry
   in the `trades` table. When AlphaEngine's own dynamic exit
   (`end_long`/`end_short`) fires, the matching open trade is closed and its
   `pnl_percent` computed — long profits on price rising, short on price
   falling — for later win-rate/backtest analysis. Trade bookkeeping mirrors
   Pine's own `ml.backtest` scoring exactly (single active position per
   symbol, `(high+low+open+open)/4` pricing) — see
   `application/backtest.py`'s module docstring.
6. Appends that symbol's own historical win rate (e.g. `win_rate=66.0%
   (33W/17L)`, computed from its closed trades so far) to every BUY/SELL
   notification's rationale — so you never see a fresh signal without
   knowing how this symbol's past trades have actually performed. Omitted
   entirely for a symbol with no closed trades yet.

Only entries (BUY/SELL) trigger a Telegram notification. When AlphaEngine's
own dynamic exit fires, a distinct "CLOSE" notification is sent too, showing
the realized `pnl_percent`.

### Paper trading account

`application/paper_trading.py` simulates a real-money account (₹8,00,000
starting capital) alongside the strategy's own signals — long-only, since
NSE cash market doesn't allow short selling for multi-day (delivery) holds
and this strategy's average holding period is ~3.5 days. Every BUY entry is
gated on two checks before it becomes a real paper position:

1. **Eligibility**: the symbol needs at least 5 closed BUY-only trades and a
   BUY-only win rate of at least 55% (computed from the same `trades` table
   the win-rate notification summary uses — both now filter to BUY-only
   consistently, so the number shown in a notification always matches the
   number eligibility actually used). A symbol with no track record yet, or
   a weak one, still notifies — just tagged `paper: not eligible yet`.
2. **Capacity**: positions are sized **dynamically** — every entry
   recomputes `total_equity / TARGET_SLOTS` (32 slots, floored at
   `MIN_POSITION_SIZE` ₹25,000), where total_equity is cash plus all open
   positions' allocated capital. Starting slot size is ~₹25,000; slots grow
   automatically as the account compounds profit, no manual re-tuning
   needed. `TARGET_SLOTS` (32) matches real signal demand computed via
   Little's Law over eligible symbols only, after the full 220-symbol
   backtest completed. Once the account's cash balance can't cover one more
   slot, the entry is skipped and tagged `paper: SKIPPED (no capital
   available)` rather than silently dropped.

SELL signals never touch the paper account — they still notify, tagged
`informational only — not tradeable in NSE cash market`, since they can't be
executed as real cash-market positions. When a paper position's matching
`end_long` exit fires, it closes with a distinct `lorentzian-paper-exit`
notification showing the realized rupee P&L, and its capital (plus/minus
P&L) returns to the account's cash balance for the next eligible entry.

This is a deliberate first phase before any real capital or broker
connection: once a week of paper results confirms the account is actually
profitable, the plan is to buy a Zerodha Kite Connect market-data
subscription (needed for real-time execution anyway) and go live with real
money — and, in a later phase, use that same real option-chain data to
explore buying PE (put) options as a synthetic short for SELL signals, since
direct short selling isn't available in the cash market.

### Market index context

Set `TRADING_SCANNER_INDEX_SYMBOL` (defaults to `^NSEI`, NIFTY 50) to have a
broad market index evaluated the same way every run. Its current
signal/prediction/early-flip state is appended to every stock notification's
rationale — purely informational, to help judge whether a signal lines up
with the broader market or looks like noise against it. It never suppresses
a stock signal, even when the two disagree. Set it to an empty string to
disable index tracking entirely.

### Setting up Turso

1. Create a free database: `turso db create trading-scanner` (or sign up at
   [turso.tech](https://turso.tech)).
2. Get the connection details: `turso db show trading-scanner --url` and
   `turso db tokens create trading-scanner`.
3. Add them as GitHub Actions repository secrets: `TURSO_DATABASE_URL` and
   `TURSO_AUTH_TOKEN`.
4. Optionally add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` secrets to
   receive signals on Telegram instead of just the logs.

### Running locally with no account

`libsql-client` also speaks directly to a local SQLite file, so the whole
pipeline can be developed and tested without a Turso account:

```bash
TRADING_SCANNER_TURSO_URL="file:local.db" \
PYTHONPATH=src python -m trading_scanner.signals
```

Candles accumulate in `local.db` across repeated runs exactly as they would
against a hosted Turso database — only the URL changes between local
development and production.

### Scheduling

[`.github/workflows/hourly-signals.yml`](.github/workflows/hourly-signals.yml)
runs only during NSE market hours (9:15 AM-3:30 PM IST), weekdays only, via
`cron: "50 4-9 * * 1-5"` -- each trigger fires 5 minutes after an hourly
candle closes (10:15, 11:15, 12:15, 13:15, 14:15, 15:15 IST), giving Yahoo
Finance time to actually have that candle ready. Also triggerable manually
via `workflow_dispatch`. This doesn't account for NSE holidays; on a holiday
the pipeline just finds no new candle for every symbol and skips gracefully.
Since this repository is public, GitHub Actions minutes are free regardless
of run frequency or symbol count.

## Web Dashboard

`trading_scanner.webapp` is a small FastAPI app that reads live from the same
Turso database the hourly pipeline writes to. It shows the paper account's
cash balance, total equity, open positions, recent closed trades and
buy-only win rate, and the pipeline's last-run status — and lets you trigger
a manual pipeline run or edit the paper-trading capital/slots/min-position
sizing, all from a browser instead of asking for a status update.

Run it locally:

```bash
TRADING_SCANNER_DASHBOARD_PASSWORD=<pick-a-password> \
PYTHONPATH=src python -m trading_scanner.webapp
```

Then open `http://localhost:8000` and log in with that password. Sessions
are cookie-based (30 days), kept in memory — restarting the dashboard logs
everyone out, which is fine for a single-user tool.

On the VPS, `vps_setup.sh` installs it as a systemd service
(`p-trade-dashboard`, auto-restarts, survives reboot) listening on port
8000. Open that port in your VPS provider's firewall to reach it, and set a
real `TRADING_SCANNER_DASHBOARD_PASSWORD` in `.env` before exposing it —
there's no other access control.

Editing capital/slots/min-position from the dashboard's Config panel
rewrites `.env` directly; it takes effect on the *next* pipeline run (the
already-running dashboard process keeps its own already-imported values
until it's restarted too).

## Current Limitations

The exported validation CSV is intended for manual bar-by-bar comparison
against TradingView; it is separate from the accumulated candle history used
by the hourly signal pipeline.
