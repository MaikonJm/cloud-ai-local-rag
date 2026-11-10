# Cloud AI: Local RAG Assistant

> A self-hosted AI assistant that ingests technical documents (including scanned PDFs), classifies and indexes them, and answers questions with **Retrieval-Augmented Generation**. It runs 100% locally with Docker Compose, with no external AI API.

![Docker](https://img.shields.io/badge/Docker_Compose-2496ED?logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?logo=ollama&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-DC244C)
![CI](https://github.com/MaikonJm/cloud-ai-local-rag/actions/workflows/ci.yml/badge.svg)

![Architecture](docs/architecture.png)

## TL;DR

| | |
|---|---|
| **What** | Knowledge API (FastAPI) that receives documents, runs OCR when needed, auto-classifies them, generates embeddings and answers questions with sources |
| **Stack** | Docker Compose · Python/FastAPI · Ollama (`qwen2.5:3b`, `nomic-embed-text`) · Qdrant · SQLite · Tesseract OCR · Open WebUI |
| **Result** | 4 containerized services; end-to-end RAG query answered with sources in **~36 s on CPU only** (8 vCPU / 16 GB VM, no GPU) |
| **Write-up** | [Full step-by-step article on Medium (PT-BR)](https://medium.com/@maikonjm) |

## Architecture

| Service | Role |
|---|---|
| **knowledge-api** | FastAPI app: auth, ingestion, OCR, classification, indexing and RAG queries. Exposes an OpenAI-compatible API |
| **ollama** | Runs the local LLM and the embedding model |
| **qdrant** | Vector database for semantic search |
| **open-webui** | Chat interface; the assistant shows up as the `cloud-ai` model |

**Ingestion:** upload → stored and registered in SQLite → text extraction (OCR fallback) → auto-classification → approve or send to review → embeddings → Qdrant.

**Query:** question → embedding → Qdrant similarity search → retrieved context → LLM → answer + sources.

### Key design decisions

- **Separate embedding and generation models.** `nomic-embed-text` is small and fast for vectors; `qwen2.5:3b` gave the best latency/quality trade-off on CPU. `qwen3:4b` was tested and dropped for being too slow without a GPU.
- **OCR fallback per page.** A scanned AWS study PDF returned no text with normal extraction. Pages without a text layer are now rendered with PyMuPDF and read by Tesseract.
- **Asynchronous ingestion.** OCR + classification + embeddings can exceed an HTTP timeout, so `/upload` returns `202` with an ID and processing runs in the background (`pending → processing → approved | review | error`).
- **OpenAI-compatible endpoints** (`/v1/models`, `/v1/chat/completions`). Open WebUI talks to the RAG pipeline with no custom plugin.
- **Human review for low-confidence documents.** The LLM's confidence score is not statistically calibrated, so it only routes documents to review; it never deletes or blocks them.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `POST /upload` | Upload a document (async, returns an ID) |
| `GET /documents/{id}` | Processing status |
| `GET /review` · `POST /review/{id}/approve` | Review queue |
| `POST /retry/{id}` | Reprocess a document |
| `POST /ask` | RAG question with sources |
| `GET /v1/models` · `POST /v1/chat/completions` | OpenAI-compatible API for Open WebUI |

All endpoints except `/health` require `Authorization: Bearer <API_TOKEN>`.

## How to run

**Requirements:** Linux with Docker Engine and Docker Compose. The original environment was an Ubuntu Server VM (VMware) with 8 vCPU, 16 GB RAM and 120 GB disk, using CPU-only inference.

```bash
cp .env.example .env
sed -i "s/REPLACE_WITH_RANDOM_SECRET/$(openssl rand -hex 32)/;s/REPLACE_WITH_RANDOM_TOKEN/$(openssl rand -hex 32)/" .env
chmod 600 .env

docker compose up -d --build

docker compose exec ollama ollama pull qwen2.5:3b
docker compose exec ollama ollama pull nomic-embed-text
```

### Test the pipeline

```bash
curl http://127.0.0.1:8000/health
TOKEN=$(sed -n 's/^API_TOKEN=//p' .env | tr -d '\r"')

# upload a sample document
curl -X POST http://127.0.0.1:8000/upload \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "file=@examples/aws-sqs-test.md"

# check processing status
curl http://127.0.0.1:8000/documents/DOCUMENT_ID \
  -H "Authorization: Bearer ${TOKEN}"
```

Then open Open WebUI at `http://127.0.0.1:3000` and select the **cloud-ai** model.

## Security

- Secrets live in `.env` (git-ignored, `chmod 600`); the repo only ships `.env.example`, filled with random values from `openssl rand`.
- Ports are published on **loopback only** (`127.0.0.1`). Ollama and Qdrant are not published at all and are reachable only on the internal Docker network.
- Token comparison uses `secrets.compare_digest` (constant-time).
- The API's interactive docs (`/docs`, `/redoc`) are disabled.
- Private documents, vector storage and metadata DBs are excluded from Git.

**Known gaps (lab scope):** a single shared token (no per-user access control), no TLS, and Open WebUI signup is enabled.

## Troubleshooting & lessons learned

- **Scanned PDF returned empty text** (27.5 MB, 5 pages of images) → added the OCR fallback with PyMuPDF + Tesseract.
- **Embedding timeouts under concurrency** → limited Ollama to 1 loaded model and 1 parallel request (`OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_NUM_PARALLEL`) to fit CPU/RAM.
- **Uploading a file in the Open WebUI chat does not feed the Qdrant collection.** It uses Open WebUI's own RAG, so ingestion stays centralized in `POST /upload`.

## Roadmap

- [ ] Persistent job queue (tasks survive restarts)
- [ ] Page-level source references for PDFs
- [ ] Deduplicate repeated sources in answers
- [ ] Observability: metrics and dashboards
- [ ] Automated answer-quality tests
- [ ] Specialized agents (AWS, Azure, GCP, OCI, DevOps, SRE): definitions already in `agents/`, not yet wired in
- [ ] Cloud deployment version

---

**Maikon Marquez** · Cloud & DevOps Engineer · [LinkedIn](https://www.linkedin.com/in/maikon-marquez) · [Medium](https://medium.com/@maikonjm)
