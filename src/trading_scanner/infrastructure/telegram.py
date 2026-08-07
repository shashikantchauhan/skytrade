import logging

import httpx

from trading_scanner.domain.models import Signal


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat_id = chat_id

    async def send_signal(self, signal: Signal) -> None:
        message = (
            f"{signal.side.upper()} {signal.symbol} at {signal.price}\n"
            f"Strategy: {signal.strategy}\nReason: {signal.rationale}"
        )
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                self._url, json={"chat_id": self._chat_id, "text": message}
            )
            response.raise_for_status()


class LoggingNotifier:
    async def send_signal(self, signal: Signal) -> None:
        logging.getLogger(__name__).info("Signal: %s", signal)
