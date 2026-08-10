import logging

import httpx

from trading_scanner.domain.models import Signal


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat_id = chat_id

    async def send_signal(self, signal: Signal) -> None:
        # An exit uses a distinct strategy tag (see signal_pipeline._notify_exit)
        # so it isn't mistaken for a fresh entry in the same side/symbol.
        is_exit = signal.strategy.endswith("-exit")
        action = f"CLOSE {signal.side.upper()}" if is_exit else signal.side.upper()
        message = (
            f"{action} {signal.symbol} at {signal.price}\n"
            f"Strategy: {signal.strategy}\nReason: {signal.rationale}"
        )
        await self.send_text(message)

    async def send_text(self, message: str) -> None:
        """Free-form notification, e.g. the Kite-session-expired alert (see
        ``application/signal_pipeline.py``'s ``_select_provider``) -- not
        every notification is a trade signal."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                self._url, json={"chat_id": self._chat_id, "text": message}
            )
            response.raise_for_status()


class LoggingNotifier:
    async def send_signal(self, signal: Signal) -> None:
        logging.getLogger(__name__).info("Signal: %s", signal)

    async def send_text(self, message: str) -> None:
        logging.getLogger(__name__).info("Notification: %s", message)
