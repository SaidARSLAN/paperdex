"""Paperdex package — global LlamaIndex setup.

When the module is imported, the global Settings.embed_model and Settings.llm
are configured once. All modules (ingestion, retrieval, generation) share
the same configuration.
"""

from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from paperdex.settings import settings

LlamaSettings.embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
LlamaSettings.llm = Groq(
    model=settings.groq_model,
    api_key=settings.groq_api_key,
)
LlamaSettings.chunk_size = settings.chunk_size
LlamaSettings.chunk_overlap = settings.chunk_overlap
