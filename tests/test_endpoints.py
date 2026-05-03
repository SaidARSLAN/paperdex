"""Test FastAPI endpoints via TestClient.

NOTE: importing paperdex.main triggers Generator() and Ingestor() at module
level, which loads the embedding model and connects to ChromaDB. The first
test run is slow (~10-15s warm-up). Subsequent runs in the same session are
fast.

For Phase C: mock LLM/embedding/ChromaDB so tests run in milliseconds without
network calls or model downloads.
"""

from fastapi.testclient import TestClient

from paperdex.main import app

client = TestClient(app)


def test_root_returns_health_status() -> None:
    """GET / returns service health payload."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "paperdex"}


def test_query_rejects_short_question() -> None:
    """POST /query with question < 3 chars returns 422 (Pydantic validation)."""
    response = client.post("/query", json={"question": "ab"})

    assert response.status_code == 422
    # Pydantic's error structure includes 'loc' pointing to the bad field
    body = response.json()
    assert "detail" in body
    assert any("question" in str(err.get("loc", [])) for err in body["detail"])


def test_query_missing_question_field() -> None:
    """POST /query without 'question' field returns 422."""
    response = client.post("/query", json={})

    assert response.status_code == 422
