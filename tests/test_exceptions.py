"""Test exception hierarchy.

Verifies that all custom exceptions inherit correctly from PaperdexError,
and that the EmptyDocumentError -> IngestError chain works.
"""

from paperdex.exceptions import (
    EmptyDocumentError,
    GenerationError,
    IngestError,
    PaperdexError,
    RetrievalError,
)


def test_all_inherit_from_paperdex_error() -> None:
    """All custom exceptions must descend from PaperdexError."""
    assert issubclass(IngestError, PaperdexError)
    assert issubclass(RetrievalError, PaperdexError)
    assert issubclass(GenerationError, PaperdexError)


def test_empty_document_is_ingest_error() -> None:
    """EmptyDocumentError is a specialized IngestError."""
    err = EmptyDocumentError("empty pdf")

    # Both checks must pass — inheritance chain
    assert isinstance(err, IngestError)
    assert isinstance(err, PaperdexError)


def test_message_preserved() -> None:
    """Exception message survives str() conversion."""
    err = IngestError("custom message")
    assert str(err) == "custom message"
