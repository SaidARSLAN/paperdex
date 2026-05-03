from llama_index.core import Settings as LlamaSettings

from paperdex.exceptions import GenerationError, PaperdexError
from paperdex.models import QueryResponse
from paperdex.retrieval import Retriever


class Generator:
    def __init__(self) -> None:
        self._retriever = Retriever()

    def answer(self, question: str, top_k: int | None = None) -> QueryResponse:
        try:
            chunks = self._retriever.retrieve(question=question, top_k=top_k)
            if len(chunks) == 0:
                raise GenerationError("No context found, cannot answer question.")

            context = "\n\n".join(f"[Source {i}] {s.text}" for i, s in enumerate(chunks, start=1))

            prompt = f"""Answer the question based on the context below.
If the context does not contain the answer, say 'I don't know'.
Reply in the same language as the question.

Context:
{context}

Question: {question}
Answer:"""

            response = LlamaSettings.llm.complete(prompt)
            answer_text = str(response).strip()
            return QueryResponse(answer=answer_text, sources=chunks)

        except PaperdexError:
            raise
        except Exception as e:
            raise GenerationError(f"Generation error: {e}") from e
