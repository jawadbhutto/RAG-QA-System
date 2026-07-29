"""
memory.py
---------
Single responsibility: hold conversation history so follow-up questions
("what about its risks?") can be understood in context, and expose it
in the shapes other modules need (plain text for prompts, LangChain
message objects for the contextualizing LLM call).

This is intentionally simple, in-memory, per-session state — no
database, no persistence across app restarts. For a Streamlit app, one
`ConversationMemory` instance lives in `st.session_state` per user.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.config import settings


@dataclass
class Turn:
    """One question/answer exchange."""

    question: str
    answer: str


@dataclass
class ConversationMemory:
    """Rolling window of the last N conversation turns."""

    turns: list[Turn] = field(default_factory=list)
    max_turns: int = settings.max_history_turns

    def add_turn(self, question: str, answer: str) -> None:
        """Record a question/answer pair, trimming to `max_turns`."""
        self.turns.append(Turn(question=question, answer=answer))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def is_empty(self) -> bool:
        return len(self.turns) == 0

    def clear(self) -> None:
        self.turns = []

    def as_messages(self) -> list[BaseMessage]:
        """Return history as alternating Human/AI LangChain messages."""
        messages: list[BaseMessage] = []
        for turn in self.turns:
            messages.append(HumanMessage(content=turn.question))
            messages.append(AIMessage(content=turn.answer))
        return messages

    def as_text(self) -> str:
        """Return history as a plain-text transcript, for debugging/display."""
        lines = []
        for turn in self.turns:
            lines.append(f"Q: {turn.question}")
            lines.append(f"A: {turn.answer}")
        return "\n".join(lines)
