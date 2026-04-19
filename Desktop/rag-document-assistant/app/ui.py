"""Streamlit chat UI for RAG Document Assistant."""

import os

import httpx
import streamlit as st

import config

DEFAULT_API = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000")


def _raise_for_status(r: httpx.Response) -> None:
    if r.status_code < 400:
        return
    try:
        body = r.json()
        detail = body.get("detail", r.text)
        if isinstance(detail, list):
            detail = "; ".join(
                str(item.get("msg", item)) if isinstance(item, dict) else str(item)
                for item in detail
            )
    except Exception:
        detail = r.text or r.reason_phrase
    raise RuntimeError(detail if detail else f"HTTP {r.status_code}")


def _api_base() -> str:
    return st.session_state.get("api_url", DEFAULT_API).rstrip("/")


def _get_status():
    try:
        r = httpx.get(f"{_api_base()}/status", timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _ingest_files(files: list) -> dict:
    multipart = []
    for f in files:
        multipart.append(("files", (f.name, f.getvalue(), "application/pdf")))
    r = httpx.post(f"{_api_base()}/ingest", files=multipart, timeout=600.0)
    _raise_for_status(r)
    return r.json()


def _query_api(
    question: str,
    mode: str,
    chat_history: list[dict],
    rewrite_query: bool,
) -> dict:
    payload = {
        "question": question,
        "mode": mode,
        "rewrite_query": rewrite_query,
        "chat_history": chat_history,
    }
    r = httpx.post(f"{_api_base()}/query", json=payload, timeout=300.0)
    _raise_for_status(r)
    return r.json()


def _reset_api() -> dict:
    r = httpx.post(f"{_api_base()}/reset", timeout=60.0)
    _raise_for_status(r)
    return r.json()


def main():
    st.set_page_config(
        page_title="RAG Document Assistant",
        page_icon="📄",
        layout="wide",
    )
    st.title("RAG Document Assistant")
    st.caption("Local PDF Q&A — powered by Ollama, FAISS, and sentence-transformers")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "answer_mode" not in st.session_state:
        st.session_state.answer_mode = "simple"
    if "rewrite_query" not in st.session_state:
        st.session_state.rewrite_query = config.QUERY_REWRITE_ENABLED

    with st.sidebar:
        st.header("Settings")
        st.session_state.api_url = st.text_input(
            "API base URL",
            value=st.session_state.get("api_url", DEFAULT_API),
            help="FastAPI server URL (default: http://127.0.0.1:8000)",
        )
        mode = st.radio(
            "Answer mode",
            options=["simple", "detailed"],
            index=0 if st.session_state.answer_mode == "simple" else 1,
            horizontal=True,
        )
        st.session_state.answer_mode = mode
        st.session_state.rewrite_query = st.checkbox(
            "Rewrite query before search",
            value=st.session_state.rewrite_query,
            help="Runs an extra Ollama call to refine your question for retrieval. Slower, sometimes better recall.",
        )
        st.caption(
            "**Faster replies:** keep **Simple** mode and rewrite **off**. "
            "Most delay is the local LLM—try a smaller Ollama model in `config.py` "
            "(e.g. `llama3.2:1b` after `ollama pull llama3.2:1b`)."
        )

        st.divider()
        st.subheader("Upload PDFs")
        uploads = st.file_uploader(
            "Choose one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("Ingest PDFs", type="primary", disabled=not uploads):
            with st.spinner("Ingesting documents (embedding may take a while)..."):
                try:
                    result = _ingest_files(uploads)
                    st.success(
                        f"Ingested {result.get('chunks_added', 0)} chunks from "
                        f"{len(result.get('files', []))} file(s)."
                    )
                except Exception as e:
                    st.error(f"Ingest failed: {e}")

        st.divider()
        if st.button("Reset index & uploads", type="secondary"):
            try:
                _reset_api()
                st.session_state.messages = []
                st.success("Reset complete.")
                st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {e}")

        st.divider()
        st.subheader("Backend status")
        status = _get_status()
        if "error" in status and "ollama_reachable" not in status:
            st.warning(f"Cannot reach API: {status.get('error', status)}")
        else:
            st.write("Ollama:", "OK" if status.get("ollama_reachable") else "unreachable")
            st.write("Vector index:", "ready" if status.get("vectorstore_ready") else "empty")
            st.caption(f"Model: {status.get('ollama_model', '—')}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                src = msg["sources"]
                lines = [f"- **{s.get('filename', '?')}** — page {s.get('page', '?')}" for s in src]
                st.caption("Sources:\n" + "\n".join(lines))

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history = []
        for m in st.session_state.messages[:-1]:
            history.append({"role": m["role"], "content": m["content"]})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    data = _query_api(
                        prompt,
                        st.session_state.answer_mode,
                        history,
                        st.session_state.rewrite_query,
                    )
                    answer = data.get("answer", "")
                    sources = data.get("sources") or []
                    rw = data.get("rewritten_query")
                    st.markdown(answer)
                    if sources:
                        lines = [
                            f"- **{s.get('filename', '?')}** — page {s.get('page', '?')}"
                            for s in sources
                        ]
                        st.caption("Sources:\n" + "\n".join(lines))
                    if rw and rw != prompt:
                        st.caption(f"Search query used: _{rw}_")
                except Exception as e:
                    answer = f"Error: {e}"
                    sources = []
                    st.error(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )
        st.rerun()


if __name__ == "__main__":
    main()
