from __future__ import annotations

from openai import OpenAI

from src.retrieval import RetrievedChunk, format_citation


class LLMServiceError(Exception):
    pass


SYSTEM_PROMPT = """You are a PDF RAG assistant.
Answer only from the retrieved document context.
Do not invent facts.
Do not use outside knowledge.
If the context does not contain the answer, say that the answer was not found in the uploaded document.
Keep answers concise and useful.
Include citations in the format [Source: filename.pdf, page X, chunk Y]."""


class LLMService:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_answer(
        self,
        question: str,
        retrieval_results: list[RetrievedChunk],
        conversation_history: list[dict],
    ) -> str:
        context_blocks = []
        for chunk in retrieval_results:
            context_blocks.append(
                f"{format_citation(chunk)}\n{chunk.text}"
            )
        history_blocks = [
            f"{message['role'].capitalize()}: {message['content']}"
            for message in conversation_history[-6:]
        ]
        user_prompt = (
            "Conversation history:\n"
            + ("\n".join(history_blocks) if history_blocks else "None")
            + "\n\n"
            + "Question:\n"
            + question
            + "\n\nRetrieved document context:\n"
            + "\n\n".join(context_blocks)
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
        except Exception as exc:  # pragma: no cover - SDK errors vary
            raise LLMServiceError(f"LLM API failure: {exc}") from exc

        answer = response.choices[0].message.content or ""
        answer = answer.strip()
        if not answer:
            raise LLMServiceError("LLM API failure: empty response from model.")

        if retrieval_results and "[Source:" not in answer:
            citations = "\n".join(format_citation(chunk) for chunk in retrieval_results)
            answer = f"{answer}\n\n{citations}"
        return answer

