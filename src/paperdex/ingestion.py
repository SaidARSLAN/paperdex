import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from paperdex.exceptions import EmptyDocumentError, IngestError, PaperdexError
from paperdex.settings import settings


class Ingestor:
    def __init__(self) -> None:
        self._chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
        self._chroma_collection = self._chroma_client.get_or_create_collection(
            settings.chroma_collection
        )
        self._vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)

    def ingest_directory(self, data_dir: str) -> int:
        try:
            documents = SimpleDirectoryReader(input_dir=data_dir, exclude_hidden=False).load_data()
            if len(documents) == 0:
                raise EmptyDocumentError("No readable documents found in directory")
            VectorStoreIndex.from_documents(documents, storage_context=self._storage_context)

            return self._chroma_collection.count()
        except PaperdexError:
            raise
        except Exception as e:
            raise IngestError(f"Ingest error: {e}") from e
