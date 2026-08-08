# Trading Scanner

This project downloads and validates market candles so that a strategy can be
added in a later milestone.

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

`application/paper_trading.py` simulates a real-money account (₹5,00,000
starting capital) alongside the strategy's own signals — long-only, since
NSE cash market doesn't allow short selling for multi-day (delivery) holds
and this strategy's average holding period is ~3.5 days. Every BUY entry is
gated on two checks before it becomes a real paper position:

1. **Eligibility**: the symbol needs at least 5 closed BUY-only trades and a
   BUY-only win rate of at least 55% (computed from the same `trades` table
   the win-rate notification summary uses). A symbol with no track record
   yet, or a weak one, still notifies — just tagged
   `paper: not eligible yet`.
2. **Capacity**: positions are sized at a fixed ₹75,000 (chosen so the
   BUY-only average winning trade, ~3.34%, nets ~₹2,500 — the target
   average win). ₹5,00,000 / ₹75,000 gives ~6-7 concurrent slots. Once the
   account's cash balance can't cover one more slot, the entry is skipped
   and tagged `paper: SKIPPED (no capital available)` rather than silently
   dropped.

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

## Current Limitations

The exported validation CSV is intended for manual bar-by-bar comparison
against TradingView; it is separate from the accumulated candle history used
by the hourly signal pipeline.
