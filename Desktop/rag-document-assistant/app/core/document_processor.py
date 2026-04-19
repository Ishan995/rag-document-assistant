"""PDF loading and text chunking."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_pdf_documents(file_paths: list[Path]) -> list:
    """Load one or more PDF files and return LangChain Documents."""
    all_docs = []
    for path in file_paths:
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        for doc in docs:
            doc.metadata.setdefault("source", str(path))
            doc.metadata["filename"] = path.name
        all_docs.extend(docs)
    return all_docs


def chunk_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)
