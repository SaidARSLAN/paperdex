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
                raise GenerationError("Bağlam bulunamadı, sorulara cevap verilemiyor.")

            context = "\n\n".join(
                f"[Kaynak {i}] {s.text}" for i, s in enumerate(chunks, start=1)
            )

            prompt = f"""Aşağıdaki bağlama dayanarak soruyu cevapla. Bağlam dışında bilgi varsa 'Bilmiyorum' de. Türkçe cevap ver.

Bağlam:
{context}

Soru: {question}
Cevap:"""

            response = LlamaSettings.llm.complete(prompt)
            answer_text = str(response).strip()
            return QueryResponse(answer=answer_text, sources=chunks)

        except PaperdexError:
            raise
        except Exception as e:
            raise GenerationError(f"Generation hatası: {e}") from e
