"""FastAPI backend for RAG Document Assistant."""

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

import config
from app.core import rag_pipeline

app = FastAPI(
    title="RAG Document Assistant API",
    description="Local RAG API using Ollama, FAISS, and sentence-transformers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Home page when you open the API URL in a browser (GET / is not an error)."""
    return {
        "service": "RAG Document Assistant API",
        "docs": "/docs",
        "endpoints": {
            "GET /status": "Health and index status",
            "POST /ingest": "Upload PDFs (multipart form, field name: files)",
            "POST /query": "Ask a question (JSON body)",
            "POST /reset": "Clear vector store and uploaded PDFs",
        },
        "ui": "Run Streamlit separately: python -m streamlit run app/ui.py",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: Literal["simple", "detailed"] = "simple"
    rewrite_query: bool = Field(
        default=False,
        description="If true, runs an extra LLM step to rewrite the question before search (slower).",
    )
    chat_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Prior turns only: [{'role': 'user'|'assistant', 'content': '...'}]",
    )


@app.get("/status")
def status():
    ollama_ok = rag_pipeline.check_ollama_reachable()
    indexed = rag_pipeline.vectorstore_exists()
    return {
        "ollama_reachable": ollama_ok,
        "ollama_url": config.OLLAMA_BASE_URL,
        "ollama_model": config.OLLAMA_LLM_MODEL,
        "vectorstore_ready": indexed,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
    }


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    pdf_files = [f for f in files if f.filename and f.filename.lower().endswith(".pdf")]
    if not pdf_files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF file.")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    try:
        for upload in pdf_files:
            safe_name = Path(upload.filename).name
            dest = config.DATA_DIR / safe_name
            content = await upload.read()
            dest.write_bytes(content)
            saved_paths.append(dest)
        result = rag_pipeline.ingest_pdf_paths(saved_paths)
        return {"status": "ok", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/query")
def query(body: QueryRequest):
    try:
        out = rag_pipeline.query_rag(
            question=body.question.strip(),
            mode=body.mode,
            chat_history=body.chat_history,
            rewrite_query=body.rewrite_query,
        )
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/reset")
def reset():
    try:
        rag_pipeline.reset_vectorstore()
        if config.DATA_DIR.is_dir():
            for p in config.DATA_DIR.iterdir():
                if p.is_file() and p.suffix.lower() == ".pdf":
                    p.unlink()
        return {"status": "ok", "message": "Vector store and uploaded PDFs cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
