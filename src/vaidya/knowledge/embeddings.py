"""Embedding configuration for the knowledge store.

Phase 1
-------
Uses ChromaDB's default embedding function (all-MiniLM-L6-v2 via
sentence-transformers).  This is sufficient for English-language scheme
descriptions and keyword-heavy queries during the prototype stage.

Phase 2
-------
Switch to a multilingual embedding model that handles Hindi, Tamil,
Bengali, and English equally well.  Candidates:

- **multilingual-e5-large** (intfloat) -- strong cross-lingual retrieval,
  ~560 M params.  Available via HuggingFace or self-hosted.
- **Sarvam embeddings** -- if/when Sarvam releases a dedicated embedding
  endpoint, prefer it for consistency with the rest of the stack.

To swap in a custom embedding function, instantiate
:class:`chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction`
(or a custom ``EmbeddingFunction`` subclass) and pass it to the
``KnowledgeStore`` constructor at collection-creation time.

Example (Phase 2)::

    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction,
    )

    ef = SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large",
        device="cpu",  # or "cuda"
    )
    collection = client.get_or_create_collection(
        "schemes",
        embedding_function=ef,
    )
"""
