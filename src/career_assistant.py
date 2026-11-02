"""
TechPulse Phase 8 — AI Career Assistant orchestrator.

Ties together: intent classification → grounded response generation →
optional LLM enhancement → session memory. All factual content comes from
existing TechPulse modules; the LLM (optional) only rephrases grounded facts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src import llm_provider
from src.intent_classifier import classify_intent
from src.response_generator import (ASSISTANT_DISCLAIMER, SUGGESTED_QUESTIONS,
                                    CareerAssistant)


@dataclass
class ChatTurn:
    role: str          # "user" | "assistant"
    text: str
    intent: str = ""
    confidence: float = 0.0


@dataclass
class ChatSession:
    """Session-scoped memory. Lives in st.session_state — never persisted to
    disk, never shared between users, cleared on demand."""

    history: list[ChatTurn] = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    max_turns: int = 40            # cap to bound session memory

    def add(self, role: str, text: str, intent: str = "", confidence: float = 0.0):
        self.history.append(ChatTurn(role=role, text=text, intent=intent,
                                     confidence=confidence))
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def clear(self):
        self.history.clear()

    def last_user_message(self) -> str:
        for t in reversed(self.history):
            if t.role == "user":
                return t.text
        return ""


class CareerAssistantApp:
    """Stateless controller: feed it (df, message, session) → assistant reply."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def handle(self, session: ChatSession, message: str,
               use_llm: bool = False, resume_consent: bool = False,
               resume_text: str = "") -> ChatTurn:
        """Classify → route → respond. Empty messages get a gentle prompt."""
        message = (message or "").strip()
        if not message:
            reply = ("It looks like your message is empty — ask me something like: "
                     + SUGGESTED_QUESTIONS[0].lower())
            session.add("assistant", reply)
            return ChatTurn("assistant", reply)

        intent_res = classify_intent(message)
        assistant = CareerAssistant(self.df, profile=session.profile)
        try:
            body = assistant.respond(message, intent_res.intent,
                                     use_llm=use_llm,
                                     resume_consent=resume_consent,
                                     resume_text=resume_text)
        except Exception:
            body = ("Something went wrong while generating that answer, but the "
                    "rest of TechPulse is unaffected. Try rephrasing your question.")
        reply = f"{body}\n\n---\n*{ASSISTANT_DISCLAIMER}*\n*Routing: {intent_res.intent} (confidence {intent_res.confidence:.0%})*"
        session.add("user", message, intent_res.intent, intent_res.confidence)
        session.add("assistant", reply)
        return ChatTurn("assistant", reply, intent_res.intent, intent_res.confidence)


__all__ = ["CareerAssistantApp", "ChatSession", "ChatTurn",
           "SUGGESTED_QUESTIONS", "classify_intent", "llm_provider"]
