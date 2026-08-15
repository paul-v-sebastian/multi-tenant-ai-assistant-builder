from __future__ import annotations

from openai import OpenAI


class LLMServiceError(Exception):
    pass


SYSTEM_PROMPT = """You are a helpful assistant.
Keep responses concise and useful."""

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document context.
Use only the uploaded PDF context below to answer the user's question.
If the uploaded PDF does not contain the answer, explicitly say that you have no information on that topic from the uploaded PDF.
Do not answer from general knowledge."""

NO_CONTEXT_SYSTEM_PROMPT = """You are a helpful assistant.
The uploaded PDF does not contain relevant information for the user's question.
Explicitly state that you have no information on that topic from the uploaded PDF.
Do not answer from general knowledge."""


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

    def generate_answer_with_context(
        self,
        question: str,
        context: str,
        conversation_history: list[dict],
    ) -> str:
        user_content = f"Context:\n{context}\n\nQuestion: {question}"
        messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
        messages.extend(conversation_history[-10:])
        messages.append({"role": "user", "content": user_content})
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

    def generate_no_context_answer(
        self,
        question: str,
        conversation_history: list[dict],
    ) -> str:
        messages = [{"role": "system", "content": NO_CONTEXT_SYSTEM_PROMPT}]
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
