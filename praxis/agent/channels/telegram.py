"""
TelegramChannel — long-polling Telegram Bot API via stdlib urllib.

No new dependencies. Uses urllib.request for all HTTP calls.

Usage
-----
    channel = TelegramChannel(
        token="your-bot-token",
        allowed_chat_ids={"123456789"},   # optional whitelist
        poll_timeout=30,                  # seconds per long-poll
    )
    for msg in channel.poll():
        reply = agent.chat(msg.text, ctx)
        channel.send(msg.chat_id, reply)

Bot setup (one-time)
--------------------
1. Message @BotFather on Telegram → /newbot → follow prompts → copy token
2. Message your bot once to open a chat
3. Get your chat id:  https://api.telegram.org/bot<TOKEN>/getUpdates
4. Set allowed_chat_ids to your chat id for security (recommended)

Markdown
--------
send() uses parse_mode="Markdown" so the agent can return formatted replies.
Telegram supports a limited Markdown subset — bold, italic, code, code blocks.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterator

from praxis.agent.channels.base import Channel, InboundMessage

_BASE = "https://api.telegram.org/bot{token}/{method}"
_POLL_TIMEOUT = 30       # seconds for Telegram long-poll (server-side wait)
_RETRY_DELAY = 5         # seconds to wait after a network error
_MAX_MESSAGE_LEN = 4096  # Telegram hard limit


class TelegramChannel(Channel):
    """
    Telegram Bot API channel using getUpdates long-polling.

    Parameters
    ----------
    token:
        Bot token from @BotFather.
    allowed_chat_ids:
        If non-empty, only messages from these chat ids are processed.
        Strongly recommended in production to prevent strangers from
        using your agent.
    poll_timeout:
        Long-poll timeout in seconds (sent to Telegram). Default: 30.
    trigger_word:
        Optional trigger word — messages that don't start with this are
        ignored. Useful in group chats. Default: None (all messages accepted).
    """

    def __init__(
        self,
        token: str,
        allowed_chat_ids: set[str] | None = None,
        poll_timeout: int = _POLL_TIMEOUT,
        trigger_word: str | None = None,
    ) -> None:
        self._token = token
        self._allowed = allowed_chat_ids or set()
        self._poll_timeout = poll_timeout
        self._trigger = trigger_word.lower() if trigger_word else None
        self._offset = 0
        self._running = True

    # ── Channel interface ──────────────────────────────────────────────────

    def poll(self) -> Iterator[InboundMessage]:
        """
        Long-poll Telegram indefinitely, yielding each text message.
        Skips non-text updates, commands, and non-whitelisted chats.
        """
        while self._running:
            try:
                updates = self._get_updates()
            except urllib.error.URLError as exc:
                # Network hiccup — wait and retry
                print(f"[TelegramChannel] network error: {exc}. Retrying in {_RETRY_DELAY}s")
                time.sleep(_RETRY_DELAY)
                continue

            for update in updates:
                self._offset = update["update_id"] + 1
                msg = self._extract_message(update)
                if msg is not None:
                    yield msg

    def send(self, chat_id: str, text: str) -> None:
        """Send a text reply, splitting at 4096 chars if needed."""
        for chunk in _split_message(text):
            self._api_call("sendMessage", {
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "Markdown",
            })

    def send_typing(self, chat_id: str) -> None:
        """Send a typing indicator (no-op on error — non-critical)."""
        try:
            self._api_call("sendChatAction", {
                "chat_id": chat_id,
                "action": "typing",
            })
        except Exception:
            pass

    def stop(self) -> None:
        """Signal the poll loop to exit after the current long-poll returns."""
        self._running = False

    # ── Internal helpers ───────────────────────────────────────────────────

    def _get_updates(self) -> list[dict]:
        params = {
            "offset": self._offset,
            "timeout": self._poll_timeout,
            "allowed_updates": json.dumps(["message"]),
        }
        data = self._api_call("getUpdates", params)
        return data.get("result", [])

    def _extract_message(self, update: dict) -> InboundMessage | None:
        """Return an InboundMessage or None if the update should be skipped."""
        raw_msg = update.get("message") or update.get("edited_message")
        if not raw_msg:
            return None

        text = raw_msg.get("text", "").strip()
        if not text:
            return None  # photo, sticker, etc.

        chat_id = str(raw_msg["chat"]["id"])
        user_id = str(raw_msg.get("from", {}).get("id", chat_id))

        # Whitelist check
        if self._allowed and chat_id not in self._allowed:
            return None

        # Trigger word check
        if self._trigger:
            lower = text.lower()
            if not lower.startswith(self._trigger):
                return None
            # Strip trigger word from message
            text = text[len(self._trigger):].strip()
            if not text:
                return None

        return InboundMessage(
            chat_id=chat_id,
            user_id=user_id,
            text=text,
            raw=raw_msg,
        )

    def _api_call(self, method: str, params: dict) -> dict:
        """Make a Telegram Bot API call and return parsed JSON."""
        url = _BASE.format(token=self._token, method=method)
        data = urllib.parse.urlencode(params).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")

        with urllib.request.urlopen(req, timeout=self._poll_timeout + 5) as resp:
            body = resp.read()

        result = json.loads(body)
        if not result.get("ok"):
            raise RuntimeError(f"Telegram API error: {result.get('description', result)}")
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split_message(text: str, limit: int = _MAX_MESSAGE_LEN) -> list[str]:
    """Split text into chunks that fit Telegram's 4096-char limit."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks
