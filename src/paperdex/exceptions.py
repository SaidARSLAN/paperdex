class PaperdexError(Exception):
    """Paperdex projesinin tüm hatalarının ortak atası."""


class IngestError(PaperdexError):
    """Doküman ingest sırasında hata (PDF parse, chunking, embedding)."""


class EmptyDocumentError(IngestError):
    """Doküman açıldı ama metin içermiyor (boş veya yalnızca resim)."""


class RetrievalError(PaperdexError):
    """Vector store'da arama sırasında hata (ChromaDB erişimi vs)."""


class GenerationError(PaperdexError):
    """LLM çağrısı sırasında hata (Groq API down, rate limit, timeout)."""
