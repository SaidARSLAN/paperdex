class PaperdexError(Exception):
    """Base class for all Paperdex-specific errors."""


class IngestError(PaperdexError):
    """Error during PDF parse, chunking, or embedding."""


class EmptyDocumentError(IngestError):
    """Document opened but contains no extractable text (empty or image-only)."""


class RetrievalError(PaperdexError):
    """Error while searching the vector store."""


class GenerationError(PaperdexError):
    """Error during LLM call (Groq down, rate limit, timeout)."""
