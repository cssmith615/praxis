"""
Channel ABC — pluggable message transport.

Any Channel implementation must:
  - yield InboundMessage objects via poll()
  - send text back via send(chat_id, text)

Adding a new channel (Discord, WhatsApp, stdin) requires only a subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass
class InboundMessage:
    """A normalised inbound message from any channel."""
    chat_id: str          # opaque id for the conversation thread
    user_id: str          # sender id
    text: str             # message body
    raw: dict             # original payload for channel-specific handling


class Channel(ABC):
    """Abstract base for agent message transports."""

    @abstractmethod
    def poll(self) -> Iterator[InboundMessage]:
        """
        Yield inbound messages, blocking as needed.

        Implementations should be interruptible (check a stop flag or use
        a short timeout so AgentRunner can shut down cleanly).
        """

    @abstractmethod
    def send(self, chat_id: str, text: str) -> None:
        """Send a text reply to the given chat_id."""

    def send_typing(self, chat_id: str) -> None:
        """Optional: send a typing indicator. No-op by default."""
