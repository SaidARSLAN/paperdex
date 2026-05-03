import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse

from paperdex.exceptions import (
    EmptyDocumentError,
    GenerationError,
    IngestError,
    PaperdexError,
    RetrievalError,
)
from paperdex.generation import Generator
from paperdex.ingestion import Ingestor
from paperdex.models import IngestResponse, QueryRequest, QueryResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("paperdex")

app = FastAPI(
    title="Paperdex",
    description="Multilingual document Q&A with open-weight LLM",
    version="0.1.0",
)

generator = Generator()
ingestor = Ingestor()


@app.exception_handler(EmptyDocumentError)
async def empty_doc_handler(request: Request, exc: EmptyDocumentError) -> JSONResponse:
    logger.warning(f"EmptyDocumentError: {exc}")
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(IngestError)
async def ingest_handler(request: Request, exc: IngestError) -> JSONResponse:
    logger.error(f"IngestError: {exc}", exc_info=True)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(RetrievalError)
async def retrieval_handler(request: Request, exc: RetrievalError) -> JSONResponse:
    logger.error(f"RetrievalError: {exc}", exc_info=True)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(GenerationError)
async def generation_handler(request: Request, exc: GenerationError) -> JSONResponse:
    logger.error(f"GenerationError: {exc}", exc_info=True)
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(PaperdexError)
async def paperdex_handler(request: Request, exc: PaperdexError) -> JSONResponse:
    logger.error(f"PaperdexError: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Unexpected error"})


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "paperdex"}


@app.post("/query")
def query_endpoint(request: QueryRequest) -> QueryResponse:
    logger.info(f"query received: {request.question[:50]!r} top_k={request.top_k}")
    response = generator.answer(request.question, top_k=request.top_k)
    logger.info(f"query answered: {len(response.sources)} sources")
    return response


@app.post("/ingest")
async def ingest_endpoint(file: UploadFile) -> IngestResponse:
    filename = file.filename or "upload.pdf"
    logger.info(f"ingest received: {filename}")
    content = await file.read()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / filename
        tmp_path.write_bytes(content)
        chunks = ingestor.ingest_directory(tmpdir)

    doc_id = uuid.uuid4().hex
    logger.info(f"ingest complete: {filename} → {chunks} chunks")
    return IngestResponse(doc_id=doc_id, filename=filename, chunks=chunks)
