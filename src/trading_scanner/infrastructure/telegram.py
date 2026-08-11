import logging
import re

import httpx

from trading_scanner.domain.models import Signal, SignalSide

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Sends formatted, HTML-parsed Telegram messages.

    Every message is deliberately shaped so the *category* is obvious at a
    glance (a distinct emoji/header) before you read a single number --
    that was the actual complaint: everything looked like the same wall of
    text regardless of whether it was a fresh signal, a strategy-only exit
    (informational), a real paper P&L close, or a system alert. See
    ``Signal.category`` for what drives the header choice.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat_id = chat_id

    async def send_signal(self, signal: Signal) -> None:
        message = _format_signal(signal)
        await self.send_text(message)

    async def send_text(self, message: str) -> None:
        """Free-form notification, e.g. the Kite-session-expired alert (see
        ``application/signal_pipeline.py``'s ``_select_provider``) -- not
        every notification is a trade signal. HTML parse mode so callers can
        bold/italic without hand-escaping every message; plain text still
        renders fine under HTML mode as long as it has no bare ``<``/``&``.
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                self._url,
                json={"chat_id": self._chat_id, "text": message, "parse_mode": "HTML"},
            )
            response.raise_for_status()


def _format_signal(signal: Signal) -> str:
    parts = [segment.strip() for segment in signal.rationale.split(";") if segment.strip()]
    if signal.category == "paper_exit":
        return _format_paper_exit(signal, parts)
    if signal.category == "exit":
        return _format_exit(signal, parts)
    return _format_entry(signal, parts)


def _format_entry(signal: Signal, parts: list[str]) -> str:
    is_buy = signal.side == SignalSide.BUY
    is_actionable = is_buy and any(p.startswith("paper: opened") for p in parts)
    if is_buy and is_actionable:
        header = "🟢 <b>BUY SIGNAL</b> -- paper trade opened"
    elif is_buy:
        header = "🟡 <b>BUY SIGNAL</b> <i>(watch only, no paper trade)</i>"
    else:
        header = "🔵 <b>SELL SIGNAL</b> <i>(info only -- not tradeable in cash market)</i>"
    lines = [
        header,
        f"<b>{_escape(signal.symbol)}</b> @ ₹{signal.price}",
    ]
    for part in parts:
        if part.startswith("prediction="):
            continue  # internal engine detail, not useful in a Telegram message
        lines.append(f"• {_escape(part)}")
    return "\n".join(lines)


def _format_exit(signal: Signal, parts: list[str]) -> str:
    detail = next((p for p in parts if "pnl=" in p), "")
    pnl_str, is_loss = _extract_pnl_percent(detail)
    header = "⚪ <b>STRATEGY EXIT</b> <i>(informational -- no paper capital involved)</i>"
    lines = [
        header,
        f"<b>{_escape(signal.symbol)}</b> ({signal.side.upper()}) @ ₹{signal.price}",
    ]
    if detail:
        lines.append(f"• {_escape(detail)}")
    if pnl_str:
        emoji = "📉" if is_loss else "📈"
        lines.append(f"{emoji} {pnl_str}")
    return "\n".join(lines)


def _format_paper_exit(signal: Signal, parts: list[str]) -> str:
    detail = "; ".join(parts)
    pnl_str, is_loss = _extract_pnl_percent(detail)
    header = ("🔴" if is_loss else "🟢") + " <b>PAPER TRADE CLOSED</b>"
    lines = [
        header,
        f"<b>{_escape(signal.symbol)}</b> @ ₹{signal.price}",
        f"• {_escape(detail)}",
    ]
    if pnl_str:
        emoji = "❌" if is_loss else "✅"
        lines.append(f"{emoji} {pnl_str}")
    return "\n".join(lines)


def _extract_pnl_percent(text: str) -> tuple[str, bool]:
    """Pulls the P&L percentage out of a rationale fragment for a standalone
    highlighted line -- handles both "pnl=4.20%" (raw exit) and
    "pnl=₹500 (4.20%)" (paper exit) shapes. Returns ("", False) if none is
    present."""
    matches = re.findall(r"([-+]?\d+\.\d+)%", text)
    if not matches:
        return "", False
    value = float(matches[-1])
    return f"P&L: {value:+.2f}%", value < 0


def _escape(text: str) -> str:
    """Telegram's HTML parse mode requires literal &, <, > to be escaped."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class LoggingNotifier:
    async def send_signal(self, signal: Signal) -> None:
        logger.info("Signal: %s", signal)

    async def send_text(self, message: str) -> None:
        logger.info("Notification: %s", message)
