from __future__ import annotations

from openai import OpenAI


class LLMServiceError(Exception):
    pass


SYSTEM_PROMPT = """You are a helpful assistant.
Keep responses concise and useful."""


class LLMService:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_answer(
        self,
        question: str,
        conversation_history: list[dict],
    ) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(conversation_history[-10:])
        messages.append({"role": "user", "content": question})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
            )
        except Exception as exc:  # pragma: no cover - SDK errors vary
            raise LLMServiceError(f"LLM API failure: {exc}") from exc

        answer = response.choices[0].message.content or ""
        answer = answer.strip()
        if not answer:
            raise LLMServiceError("LLM API failure: empty response from model.")
        return answer
