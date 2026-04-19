<div align="center">

# 📄 RAG Document Assistant

**Ask questions about your PDFs using local AI — no paid APIs, runs fully on your machine.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_AI-000000?style=for-the-badge)](https://ollama.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-121212?style=for-the-badge)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

<sub>🔒 Privacy-first · 💰 Zero paid API keys · 🖥️ 100% on-device inference</sub>

</div>

---

## ✨ Features

| | |
|---:|---|
| 📚 | **Multi-PDF upload** — ingest several documents at once |
| 🔍 | **Semantic search** — meaning-based retrieval over your corpus |
| ✍️ | **Query rewriting** — optional LLM step to refine questions before search |
| 📝 | **Simple & detailed answer modes** — tune depth vs. speed |
| 📑 | **Source citations with page numbers** — trace every answer back to the PDF |
| 💾 | **Persistent vector store** — FAISS index on disk; survives restarts |
| 🔌 | **REST API** — FastAPI backend for programmatic access |
| 💬 | **Chat UI** — Streamlit frontend with conversational history |

---

## 🛠️ Tech Stack

| Layer | Technology |
|------|------------|
| **LLM** | Ollama (`llama3.2`) |
| **Embeddings** | Hugging Face `sentence-transformers` |
| **Vector DB** | FAISS |
| **Orchestration** | LangChain |
| **Backend** | FastAPI |
| **Frontend** | Streamlit |

---

## ⚙️ Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/rag-document-assistant.git
   cd rag-document-assistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**

   **Windows (PowerShell)**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS / Linux**
   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Pull the Ollama model**
   ```bash
   ollama pull llama3.2
   ```

   Ensure the [Ollama](https://ollama.com/) app is running (default: `http://localhost:11434`).

---

## 🚀 Running

Open **two** terminals from the project root (with the venv activated).

| | Command |
|---:|---|
| 🖥️ **Backend** | `uvicorn app.api:app --reload` |
| 🎨 **Frontend** | `streamlit run app/ui.py` |

Then open the Streamlit URL in your browser (usually `http://localhost:8501`). The API serves interactive docs at `http://127.0.0.1:8000/docs` by default.

> **Tip:** If port `8000` is busy, bind another port, e.g. `uvicorn app.api:app --reload --port 8001`, and set the API URL in the Streamlit sidebar accordingly.

---

## 📁 Project Structure

```
rag-document-assistant/
├── app/
│   ├── __init__.py
│   ├── api.py                 # FastAPI REST API
│   ├── ui.py                  # Streamlit chat UI
│   └── core/
│       ├── __init__.py
│       ├── document_processor.py   # PDF load & chunking
│       ├── embeddings.py           # HuggingFace embeddings
│       └── rag_pipeline.py       # RAG retrieval & generation
├── data/                      # Uploaded PDFs (runtime)
├── vectorstore/               # Persisted FAISS index (runtime)
├── config.py                  # App configuration
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

```text
┌─────────┐    chunk & embed     ┌──────────────┐    persist    ┌───────┐
│   PDF   │ ──────────────────► │    FAISS     │ ────────────► │ disk  │
└─────────┘                    └──────────────┘               └───────┘
                                     ▲
                                     │ semantic search
┌──────────────┐    rewrite*    ┌────┴─────┐    retrieve     ┌─────────┐
│ User question│ ────────────► │ Pipeline │ ◄────────────── │ chunks  │
└──────────────┘               └────┬─────┘                 └─────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               │
             ┌─────────────┐                        │
             │ Ollama LLM  │ ◄── reads matched chunks + question
             └─────────────┘
                    │
                    ▼
             Answer + citations
```

1. **PDF → chunks** — Documents are split into overlapping text chunks.  
2. **Chunks → embeddings** — Each chunk is embedded with Hugging Face `sentence-transformers`.  
3. **Store in FAISS** — Vectors are indexed and **saved to disk** for persistence.  
4. **User asks a question** — Your question enters the pipeline.  
5. **Question → rewritten** — Optionally, the LLM **rewrites** the query for better retrieval *(toggle in UI)*.  
6. **FAISS retrieval** — The index returns the **most relevant chunks** to the (possibly rewritten) question.  
7. **Ollama answers** — The **llama3.2** model reads those chunks and produces an answer **with source citations** (PDF name + page).

---

<div align="center">

**⭐ If this project helped you, consider starring the repo!**

Made with ❤️ using local AI

</div>
