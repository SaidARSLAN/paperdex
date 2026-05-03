class PaperdexError(Exception):
    """Paperdex projesinin tüm hatalarının ortak atası."""


class IngestError(PaperdexError):
    """PDF parse, chunking veya embedding sırasında hata."""


class EmptyDocumentError(IngestError):
    """Doküman açıldı ama metin içermiyor (boş veya yalnızca resim)."""


class RetrievalError(PaperdexError):
    """Vector store'da arama sırasında hata."""


class GenerationError(PaperdexError):
    """LLM çağrısı sırasında hata (Groq down, rate limit, timeout)."""
