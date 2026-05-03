import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore

from paperdex.exceptions import EmptyDocumentError, IngestError, PaperdexError
from paperdex.settings import settings


class Ingestor:
    def __init__(self) -> None:
        self._chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
        self._chroma_collection = self._chroma_client.get_or_create_collection(
            settings.chroma_collection
        )
        self._vector_store = ChromaVectorStore(
            chroma_collection=self._chroma_collection
        )
        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )
        Settings.embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
        Settings.llm = Groq(model=settings.groq_model, api_key=settings.groq_api_key)
        Settings.chunk_size = settings.chunk_size
        Settings.chunk_overlap = settings.chunk_overlap

    def ingest_directory(self, data_dir: str) -> int:
        try:
            documents = SimpleDirectoryReader(
                input_dir=data_dir, exclude_hidden=False
            ).load_data()
            if len(documents) == 0:
                raise EmptyDocumentError("Klasörde okunabilir döküman yok")
            index = VectorStoreIndex.from_documents(
                documents, storage_context=self._storage_context
            )

            return self._chroma_collection.count()
        except PaperdexError:
            raise
        except Exception as e:
            raise IngestError(f"Ingest hatası: {e}") from e
