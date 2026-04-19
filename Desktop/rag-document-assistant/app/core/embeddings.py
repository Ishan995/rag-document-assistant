"""HuggingFace sentence-transformers embeddings via LangChain."""

from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings

import config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    try:
        return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    except ImportError as e:
        raise ImportError(
            "sentence-transformers (and typically torch) must be installed for embeddings. "
            "Run: pip install sentence-transformers torch"
        ) from e
