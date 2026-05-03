import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from paperdex.exceptions import PaperdexError, RetrievalError
from paperdex.models import Source
from paperdex.settings import settings


class Retriever:
    def __init__(self) -> None:
        self._chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
        self._chroma_collection = self._chroma_client.get_or_create_collection(
            settings.chroma_collection
        )
        self._vector_store = ChromaVectorStore(
            chroma_collection=self._chroma_collection
        )

        self._index = VectorStoreIndex.from_vector_store(self._vector_store)

    def retrieve(self, question: str, top_k: int | None = None) -> list[Source]:
        try:
            if top_k is None:
                top_k = settings.top_k
            retriever = self._index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(question)
            return [
                Source(
                    text=n.node.get_content(),
                    score=n.score or 0.0,
                    metadata=n.node.metadata or {},
                )
                for n in nodes
            ]
        except PaperdexError:
            raise
        except Exception as e:
            raise RetrievalError(f"Retrieve Hatası : {e}") from e
