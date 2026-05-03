"""Test Pydantic model validation.

Validates that:
- Valid input creates instances cleanly.
- Invalid input raises ValidationError (Pydantic's standard error).
- Default values work as expected.
"""

import pytest
from pydantic import ValidationError

from paperdex.models import IngestResponse, QueryRequest, Source


def test_valid_query_request() -> None:
    """A normal question with min_length passes."""
    req = QueryRequest(question="What is RAG?")

    assert req.question == "What is RAG?"
    assert req.top_k is None  # default


def test_query_request_rejects_short_question() -> None:
    """Pydantic enforces min_length=3."""
    with pytest.raises(ValidationError):
        QueryRequest(question="ab")  # 2 chars, too short


def test_query_request_rejects_too_long_question() -> None:
    """Pydantic enforces max_length=500."""
    with pytest.raises(ValidationError):
        QueryRequest(question="x" * 501)


def test_source_score_in_range() -> None:
    """Score must be between 0.0 and 1.0."""
    src = Source(text="example", score=0.85, metadata={"page": 3})
    assert src.score == 0.85
    assert src.metadata == {"page": 3}


def test_source_rejects_invalid_score() -> None:
    """Score > 1.0 is rejected."""
    with pytest.raises(ValidationError):
        Source(text="x", score=1.5)


def test_source_default_metadata_is_empty_dict() -> None:
    """metadata defaults to empty dict (mutable default safety via factory)."""
    src = Source(text="x", score=0.5)
    assert src.metadata == {}


def test_ingest_response_chunks_must_be_non_negative() -> None:
    """chunks >= 0 is enforced."""
    with pytest.raises(ValidationError):
        IngestResponse(doc_id="abc", filename="x.pdf", chunks=-1)
