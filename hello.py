"""Phase A — Hello World RAG.

Single-file minimal pipeline:
PDF -> text -> chunk -> embedding -> ChromaDB -> query -> Groq LLM -> answer
"""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore

# --- Load .env (GROQ_API_KEY) ---
load_dotenv(Path(__file__).parent / ".env")

# --- LlamaIndex global config ---
Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
Settings.llm = Groq(
    model="openai/gpt-oss-120b",  # verify model name in Groq console
    api_key=os.environ["GROQ_API_KEY"],
)

# --- Load documents (all PDFs in data/) ---
data_dir = Path(__file__).parent / "data"
documents = SimpleDirectoryReader(
    input_dir=str(data_dir),
    exclude_hidden=False,
).load_data()
print(f"Loaded documents: {len(documents)}")

# --- ChromaDB (in-memory client, not persistent — switched in Phase B) ---
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("paperdex")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- Build index (chunk + embed + store) ---
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

# --- Query engine ---
query_engine = index.as_query_engine()

# --- Ask a question ---
question = "Which safety procedures are described?"
print(f"\nQuestion: {question}")
response = query_engine.query(question)
print(f"\nAnswer: {response}")
