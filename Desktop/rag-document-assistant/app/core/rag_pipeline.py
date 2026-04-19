"""RAG retrieval, query rewriting, and generation with citations."""

from pathlib import Path
from typing import Any, Literal

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

import config
from app.core.document_processor import chunk_documents, load_pdf_documents
from app.core.embeddings import get_embeddings

_faiss_store: FAISS | None = None


def _ollama_llm(*, temperature: float | None = None, num_predict: int | None = None) -> ChatOllama:
    kwargs: dict[str, Any] = {
        "model": config.OLLAMA_LLM_MODEL,
        "base_url": config.OLLAMA_BASE_URL,
        "num_ctx": config.OLLAMA_NUM_CTX,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if num_predict is not None:
        kwargs["num_predict"] = num_predict
    return ChatOllama(**kwargs)


def _truncate_chunk_text(text: str) -> str:
    text = (text or "").strip()
    max_c = config.CONTEXT_CHARS_PER_CHUNK
    if len(text) <= max_c:
        return text
    return text[: max_c - 1].rstrip() + "…"


def _vectorstore_path() -> Path:
    return config.VECTORSTORE_DIR


def _faiss_folder() -> Path:
    return _vectorstore_path()


def vectorstore_exists() -> bool:
    folder = _faiss_folder()
    index_file = folder / f"{config.FAISS_INDEX_NAME}.faiss"
    pkl_file = folder / f"{config.FAISS_INDEX_NAME}.pkl"
    return index_file.is_file() and pkl_file.is_file()


def load_vectorstore() -> FAISS | None:
    global _faiss_store
    if _faiss_store is not None:
        return _faiss_store
    if not vectorstore_exists():
        return None
    embeddings = get_embeddings()
    _faiss_store = FAISS.load_local(
        str(_faiss_folder()),
        embeddings,
        index_name=config.FAISS_INDEX_NAME,
        allow_dangerous_deserialization=True,
    )
    return _faiss_store


def save_vectorstore(store: FAISS) -> None:
    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(_faiss_folder()), index_name=config.FAISS_INDEX_NAME)


def ingest_pdf_paths(paths: list[Path]) -> dict[str, Any]:
    """Load PDFs, chunk, add to FAISS, persist."""
    global _faiss_store
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = [Path(p).resolve() for p in paths]
    documents = load_pdf_documents(paths)
    if not documents:
        raise ValueError(
            "Could not read any pages from the PDF file(s). "
            "The file may be corrupt, encrypted, or not a valid PDF."
        )
    chunks = chunk_documents(documents)
    if not chunks:
        raise ValueError(
            "No text could be extracted from the PDF(s). "
            "They may be image-only scans—add OCR text—or contain only blank pages."
        )
    embeddings = get_embeddings()
    existing = load_vectorstore()
    if existing is None:
        store = FAISS.from_documents(chunks, embeddings)
    else:
        existing.add_documents(chunks)
        store = existing
    save_vectorstore(store)
    _faiss_store = store
    return {
        "files": [p.name for p in paths],
        "pages_loaded": len(documents),
        "chunks_added": len(chunks),
    }


def reset_vectorstore() -> None:
    global _faiss_store
    _faiss_store = None
    folder = _faiss_folder()
    for pattern in (f"{config.FAISS_INDEX_NAME}.faiss", f"{config.FAISS_INDEX_NAME}.pkl"):
        p = folder / pattern
        if p.is_file():
            p.unlink()


def _rewrite_query(user_question: str) -> str:
    llm = _ollama_llm(
        temperature=config.QUERY_REWRITE_TEMPERATURE,
        num_predict=config.OLLAMA_NUM_PREDICT_REWRITE,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You rewrite user questions for better document search. "
                "Output ONLY the improved search query, no quotes or explanation.",
            ),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm
    out = chain.invoke({"question": user_question})
    text = out.content if hasattr(out, "content") else str(out)
    return text.strip() or user_question


def _format_context_with_citations(docs: list[Document]) -> tuple[str, list[dict[str, Any]]]:
    lines = []
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        filename = meta.get("filename") or Path(meta.get("source", "unknown")).name
        page = meta.get("page")
        if page is None:
            page = 0
        key = (filename, int(page))
        if key not in seen:
            seen.add(key)
            sources.append({"filename": filename, "page": int(page) + 1})
        body = _truncate_chunk_text(doc.page_content)
        lines.append(f"[Excerpt {i} — {filename}, page {int(page) + 1}]\n{body}")
    return "\n\n".join(lines), sources


def _build_messages_for_answer(
    question: str,
    context: str,
    sources_summary: str,
    mode: Literal["simple", "detailed"],
    chat_history: list[dict[str, str]],
):
    if mode == "simple":
        system = (
            "You are a helpful assistant. Answer using ONLY the provided context excerpts. "
            "Be concise (2–4 sentences). "
            "At the end, add a 'Sources:' line listing each cited PDF filename and page number "
            "you relied on, matching the excerpt labels."
        )
    else:
        system = (
            "You are a helpful assistant. Answer using ONLY the provided context excerpts. "
            "Give a structured, detailed answer with clear sections where appropriate. "
            "At the end, add a 'Sources:' line listing each cited PDF filename and page number "
            "you relied on, matching the excerpt labels."
        )
    messages = [("system", system)]
    for turn in chat_history[-config.CHAT_HISTORY_MAX_TURNS :]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            messages.append(("human", content))
        elif role == "assistant":
            messages.append(("assistant", content))
    messages.extend(
        [
            (
                "human",
                "Context excerpts:\n{context}\n\n"
                "Available source list (filename + page):\n{sources_summary}\n\n"
                "Question: {question}",
            ),
        ]
    )
    return ChatPromptTemplate.from_messages(messages)


def query_rag(
    question: str,
    mode: Literal["simple", "detailed"] = "simple",
    chat_history: list[dict[str, str]] | None = None,
    rewrite_query: bool | None = None,
) -> dict[str, Any]:
    store = load_vectorstore()
    if store is None:
        return {
            "answer": "No documents have been indexed yet. Upload PDFs and ingest first.",
            "rewritten_query": question,
            "sources": [],
            "error": "empty_index",
        }
    chat_history = chat_history or []
    use_rewrite = (
        config.QUERY_REWRITE_ENABLED if rewrite_query is None else rewrite_query
    )
    rewritten = _rewrite_query(question) if use_rewrite else question.strip()
    k = config.RETRIEVER_K
    docs = store.similarity_search(rewritten, k=k)
    if not docs:
        return {
            "answer": "No relevant passages were found in the indexed documents.",
            "rewritten_query": rewritten,
            "sources": [],
        }
    context, sources = _format_context_with_citations(docs)
    sources_summary = "\n".join(
        f"- {s['filename']} (page {s['page']})" for s in sources
    )
    temp = (
        config.GENERATION_TEMPERATURE_SIMPLE
        if mode == "simple"
        else config.GENERATION_TEMPERATURE_DETAILED
    )
    num_predict = (
        config.OLLAMA_NUM_PREDICT_SIMPLE
        if mode == "simple"
        else config.OLLAMA_NUM_PREDICT_DETAILED
    )
    llm = _ollama_llm(temperature=temp, num_predict=num_predict)
    prompt = _build_messages_for_answer(question, context, sources_summary, mode, chat_history)
    chain = prompt | llm
    out = chain.invoke(
        {
            "context": context,
            "sources_summary": sources_summary,
            "question": question,
        }
    )
    answer_text = out.content if hasattr(out, "content") else str(out)
    return {
        "answer": answer_text.strip(),
        "rewritten_query": rewritten,
        "sources": sources,
    }


def check_ollama_reachable() -> bool:
    try:
        llm = _ollama_llm(temperature=0, num_predict=8)
        llm.invoke([HumanMessage(content="ping")])
        return True
    except Exception:
        return False
