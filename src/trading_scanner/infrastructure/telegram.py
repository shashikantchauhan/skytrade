import logging

import httpx

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Sends free-form, HTML-parsed Telegram messages -- every call site
    (application/live_cash_execution.py, live_pipeline.py,
    application/live_execution.py, application/signal_pipeline.py,
    application/gtt_bracket.py) builds its own message text and calls
    ``send_text`` directly; nothing here formats a ``Signal`` itself.

    2026-08-31: removed the ``Signal``-formatting path (``send_signal`` and
    its ``_format_*`` helpers) -- dead code, superseded by direct
    ``send_text`` calls at every real call site and unreferenced anywhere
    in the codebase.
    """

    def __init__(self, bot_token: str, chat_id: str, label: str = "") -> None:
        """``chat_id``: one Telegram chat ID, or several comma-separated
        (e.g. "1152740946,8834658819") to fan the same message out to
        multiple people -- each person only needs to message the bot once,
        ever, to get a chat ID; after that they're just another entry here.

        ``label``: shown in every message header (e.g. "Nifty50",
        "Smallcap") -- needed because more than one deployment can share
        the same bot/chat ID (see skytrade-smallcap's .env), and without
        it a message gives no clue which system it came from."""
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat_ids = [c.strip() for c in chat_id.split(",") if c.strip()]
        self._label = label

    async def send_text(self, message: str, label: str | None = None) -> None:
        """Free-form notification, e.g. the Kite-session-expired alert (see
        ``application/signal_pipeline.py``'s ``_select_provider``). HTML
        parse mode so callers can bold/italic without hand-escaping every
        message; plain text still renders fine under HTML mode as long as
        it has no bare ``<``/``&``.

        A label (e.g. "Cash", "Smallcap") is prepended as its own first
        line whenever one is set -- ``label`` if given, else this
        instance's own deployment-wide default (``self._label``).
        Centralized here, not in each caller, so every message type is
        tagged consistently with minimal per-call-site changes. Matters
        because more than one deployment can share the same bot/chat ID
        (see skytrade-smallcap's .env) and, without this, a message gives
        no clue which system sent it.

        Sent to every configured chat ID independently -- one recipient's
        failure (e.g. they blocked the bot) is logged but never blocks
        delivery to the others.
        """
        label = label if label is not None else self._label
        if label:
            message = f"<i>[{_escape(label)}]</i>\n{message}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            for chat_id in self._chat_ids:
                try:
                    response = await client.post(
                        self._url,
                        json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
                    )
                    response.raise_for_status()
                except Exception:
                    logger.warning("Telegram send failed for chat_id=%s", chat_id, exc_info=True)


def _escape(text: str) -> str:
    """Telegram's HTML parse mode requires literal &, <, > to be escaped."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class LoggingNotifier:
    async def send_text(self, message: str) -> None:
        logger.info("Notification: %s", message)
