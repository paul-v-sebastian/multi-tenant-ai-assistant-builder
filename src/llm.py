from __future__ import annotations

from openai import OpenAI

from src.retrieval import RetrievedChunk, format_citation


class LLMServiceError(Exception):
    pass


SYSTEM_PROMPT = """You are a helpful assistant.
Keep responses concise and useful."""
RAG_PROMPT = """Use retrieved context when it is relevant to the question.
If the context is incomplete, rely on it carefully and say when you are unsure."""


class LLMService:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_answer(
        self,
        question: str,
        conversation_history: list[dict],
        retrieved_chunks: list[RetrievedChunk] | None = None,
    ) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if retrieved_chunks:
            citations = "\n\n".join(
                f"{format_citation(chunk)}\n{chunk.text}"
                for chunk in retrieved_chunks
            )
            messages.append({"role": "system", "content": RAG_PROMPT})
            messages.append({"role": "system", "content": f"Retrieved context:\n{citations}"})
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
