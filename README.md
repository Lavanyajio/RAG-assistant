# 🔎 Project 1 — Hybrid RAG Assistant

Chat with your own PDFs / TXT / Markdown files. Answers are grounded in the documents and cite every fact.

**What is new compared with a basic RAG demo**

| Technique | Why it matters |
|---|---|
| **Hybrid retrieval** — HF embeddings + FAISS (meaning) **and** BM25 (exact keywords) | Finds both "how do I fix a battery problem" *and* exact codes like `E07`, model numbers, clause IDs |
| **Reciprocal Rank Fusion (RRF)** | Merges the two rankings without fragile score weighting |
| **LLM query rewriting + multi-query** | Follow-ups like "and how long does *that* take?" become standalone searches; 2 extra phrasings improve recall |
| **Citations + retrieval trace** | Each answer shows `[n]` citations and an expandable panel with semantic rank, keyword rank and fused score per passage |
| **Persistent index + de-duplication** | Index survives restarts (`storage/index`); re-uploading the same file is skipped (SHA-256) |

Sample data: two fictional e-scooter documents (`data/sample_docs`) — click **Load sample docs**.

## Flow

```
upload → load (pypdf) → paragraph-aware chunks → HF embeddings ─┐
                                                                ├─ HybridIndex (FAISS + BM25) → disk
question + history → LLM rewrite (standalone + 2 variants)      │
   → dense top-20 + BM25 top-20 per variant → RRF fusion → top-k passages
   → LLM (Hugging Face) answers ONLY from passages with [n] citations
```

## Project layout

```
app.py                       Gradio UI (entry point)
rag_assistant/config.py      All settings (env / .env)
rag_assistant/llm.py         Hugging Face chat client with retries
rag_assistant/embeddings.py  HF Inference API embeddings (or local sentence-transformers)
rag_assistant/documents.py   Loading + chunking
rag_assistant/hybrid_index.py FAISS + BM25 + RRF, save/load
rag_assistant/pipeline.py    Ingest, rewrite, retrieve, answer
tests/                       Offline tests (fake LLM + fake embedder, no token needed)
Dockerfile · Jenkinsfile · requirements*.txt · .env.example
```

## Run in PyCharm

1. **File → Open** this folder.
2. **Settings → Project → Python Interpreter → Add Interpreter → Virtualenv** (Python 3.10–3.12).
3. Open the terminal in PyCharm: `pip install -r requirements-dev.txt`
4. Copy `.env.example` to `.env` and set `HF_TOKEN`.
5. Right-click `app.py` → **Run 'app'** → open http://localhost:7860
6. Tests: right-click the `tests` folder → **Run 'pytest in tests'** (works without a token).

## Docker

```bash
docker build -t hybrid-rag-assistant .
docker run -p 7861:7860 --env-file .env -v rag-storage:/app/storage hybrid-rag-assistant
```

## Jenkins

Create a **Pipeline** job pointing at this folder's repo (Script Path: `Jenkinsfile`). Stages: Setup → flake8 →
pytest (JUnit + coverage) → Docker build → container smoke test → optional push (`DOCKER_REGISTRY` param,
credential `docker-registry-creds`) → optional deploy (`DEPLOY` param, credential `hf-token`, port `DEPLOY_PORT`).

## Settings worth trying

| Variable | Default | Effect |
|---|---|---|
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Any chat model on HF Inference Providers |
| `EMBEDDING_BACKEND` | `api` | `local` runs embeddings on your machine (`pip install sentence-transformers`) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 150 | Smaller chunks = more precise citations |
| `TOP_K`, `CANDIDATE_POOL`, `RRF_K` | 5, 20, 60 | Retrieval breadth and fusion smoothing |

After changing the embedding model, click **Clear index** and re-index (vector sizes differ).
