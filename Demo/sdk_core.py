"""Core AI agent logic for the Multi-Agent SDK Sandbox."""

from __future__ import annotations

import time
import re

from google import genai
from google.genai import types


MODEL_NAME = "gemini-3.8-flash"
REQUEST_TIMEOUT_MS = 15_000


class NPCAgent:
    """A stateful NPC that remembers its conversation with Gemini."""

    def __init__(self, name: str, persona: str, api_key: str) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(
            api_key=api_key.strip(),
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )
        self.name = name
        self.persona = persona
        self.chat = self.client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=96,
                response_modalities=["TEXT"],
                system_instruction=(
                    f"You are the NPC named {name} in a fictional game world. "
                    f"Your personality and role: {persona}. "
                    "Respond in English with one complete sentence of 20 to 30 words. "
                    "State your immediate action and show one in-character emotion. "
                    "Stay strictly in character and never identify yourself as AI."
                )
            ),
        )

    def react(self, event_context: str) -> str:
        """Send an event to the NPC and return its text response."""
        if not event_context.strip():
            raise ValueError("event_context cannot be empty.")

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self.chat.send_message(event_context.strip())
                text = self._extract_text(response)
                if self._is_complete_response(text):
                    return text

                if attempt < 1:
                    time.sleep(0.75 * (attempt + 1))
                    continue

                details = self._response_details(response)
                last_error = RuntimeError(
                    f"{self.name} returned an incomplete response ({details})."
                )
                break
            except Exception as error:
                last_error = error
                error_text = str(error).upper()
                transient = any(
                    marker in error_text
                    for marker in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")
                )
                if not transient or attempt == 1:
                    raise
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"Gemini request failed: {last_error}") from last_error

        raise RuntimeError(f"{self.name} request failed: {last_error}") from last_error

    @staticmethod
    def _is_complete_response(text: str) -> bool:
        """Require at least one meaningful, complete sentence from the NPC."""
        normalized = " ".join(text.split())
        if len(normalized.split()) < 3:
            return False
        return bool(re.search(r"[.!?]['\"]?$", normalized))

    @staticmethod
    def _extract_text(response: object) -> str:
        """Read text safely from the SDK response, including candidate parts."""
        chunks: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    chunks.append(str(part_text))
        if chunks:
            return "".join(chunks).strip()

        try:
            text = getattr(response, "text", None)
        except (AttributeError, ValueError):
            text = None
        return str(text).strip() if text else ""

    @staticmethod
    def _response_details(response: object) -> str:
        """Return safe diagnostic fields when Gemini returns no text."""
        details: list[str] = []
        status = getattr(response, "model_status", None)
        feedback = getattr(response, "prompt_feedback", None)
        if status:
            details.append(f"model_status={status}")
        if feedback:
            details.append(f"prompt_feedback={feedback}")
        for candidate in getattr(response, "candidates", None) or []:
            reason = getattr(candidate, "finish_reason", None)
            if reason:
                details.append(f"finish_reason={reason}")
        return ", ".join(details) or "no text parts"
