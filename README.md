# RAG System – Cost-Efficient QA over PDFs
This is a learning project built with assistance from AI tools to understand RAG pipelines
This project implements a **Retrieval-Augmented Generation (RAG)** system for answering questions over PDF documents. It indexes PDFs into a local **ChromaDB** vector database and uses **Google Gemini** to generate grounded answers with source citations.

---

## Project Structure

```text
├── rag.py                 # Main application (ingest + query)
├── get_metrics.py         # Evaluation harness (Recall@k, MRR, Precision@k, nDCG@k, Hit Rate@k)
├── eval_results.json      # Generated evaluation results (created by get_metrics.py)
├── .env.example           # Environment variable template (copy to .env)
├── docs/                  # Folder containing PDF files to ingest
│   └── sample.pdf         # Sample document (LaTeX guide)
└── README.md              # Project documentation (this file)
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/AkANkSHA-RaWaT-2026/rag_assignment.git
cd rag_assignment
```

### 2. Create a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install chromadb pypdf python-dotenv google-genai tenacity numpy
```

### 4. Configure environment variables

Copy the example file and add your API key:

```bash
cp .env.example .env
```

Inside `.env`:

```env
GOOGLE_API_KEY=your_api_key_here

# Optional — all have sensible defaults if omitted
CHUNK_SIZE=512
CHUNK_OVERLAP=50
SIM_THRESHOLD=0.45
EMBED_MODEL=models/gemini-embedding-001
GEMINI_LLM_MODEL=models/gemini-2.5-flash
```

Obtain a free API key from **Google AI Studio** (https://aistudio.google.com/).

> **Note:** Google periodically deprecates specific model versions (e.g. `gemini-2.0-flash` was retired June 2026). If you hit a `429 RESOURCE_EXHAUSTED` error with `limit: 0`, check `GEMINI_LLM_MODEL` against Google's [current pricing page](https://ai.google.dev/gemini-api/docs/pricing) before assuming it's a billing issue.

---

## Usage

### Ingest (index) PDFs

Place PDF files inside `docs/` and run:

```bash
python rag.py ingest --folder ./docs
```

Example output:

```text
Ingesting PDFs from ./docs...
  Processing: sample.pdf
    Upserted 13 chunks for sample.pdf
Total vectors in DB: 13
```

Ingestion is **idempotent** — re-running it upserts chunks by a content+filename hash, so no duplicate vectors are ever created, even if you run it repeatedly on the same files.

### Ask a question

```bash
python rag.py query "What is the main topic of this document?"
```

Example output:

```text
============================================================
ANSWER: The main topic of this document is a LaTeX template guide,
covering how to compile a .tex file to a .pdf file and pdfLaTeX capabilities [1], [2], [3].
CITATIONS: ['[1] sample.pdf', '[2] sample.pdf', '[3] sample.pdf']
CHUNKS USED: 5
LATENCY: 1372.94 ms
EST. TOKENS: 658
GENERATION OK: True
============================================================
```

Optional flags:

```bash
python rag.py query "your question" --k 8 --filter '{"source":"sample.pdf"}'
```

- `--k` — number of chunks to retrieve (default: 5)
- `--filter` — metadata filter as a JSON string, e.g. restrict to one source file

If no chunk clears the similarity threshold, the system returns *"No relevant context found"* instead of guessing — this is the no-hallucination branch, and it is scored as a normal (non-error) result.

### Run the evaluation harness

```bash
python get_metrics.py
```

This scores a fixed set of 10 manually labeled questions against the ingested corpus and writes full results to `eval_results.json`.

Example summary output:

```text
avg_recall@5: 0.650
avg_hit_rate@5: 0.700
avg_mrr: 0.328
avg_precision@5: 0.220
avg_ndcg@5: 0.585
n_queries: 10
n_scoreable_recall_queries: 9
```

> Replace the numbers above with your actual `eval_results.json` output before submission — these are illustrative, not guaranteed results.

---

## Vector Store Choice

**ChromaDB** was chosen over a managed vector DB (e.g. Pinecone) for this project because:

- It's free and runs entirely locally — no always-on hosted infrastructure cost.
- It provides persistent storage via `PersistentClient`, so the index survives process restarts.
- It supports metadata filtering (used here for `source` and `chunk_index`) and cosine similarity search out of the box.
- For a lightly-queried, small-to-medium corpus (the exact profile this assignment targets), Chroma's embedded model avoids paying for idle managed-DB capacity.

The trade-off: Chroma runs as a single embedded process rather than a distributed, horizontally scaled service — acceptable for this scale, but a real constraint discussed below.

---

## Cost Comparison: ChromaDB (self-hosted) vs. Managed Vector DB

Estimates assume a managed provider charging for always-on pod capacity (e.g. Pinecone-style pricing), 1536-dimension embeddings, and light query volume (a few thousand queries/month) — the scenario where fixed infrastructure cost dominates.

| Vectors    | Managed DB (est. monthly) | ChromaDB self-hosted (est. monthly)         | Notes                                                                 |
|------------|---------------------------|----------------------------------------------|------------------------------------------------------------------------|
| 100K       | ~$70–100                  | ~$5–10 (small VM/disk only)                  | Managed pod cost dominates even at this small scale                   |
| 1M         | ~$200–300                 | ~$15–30 (larger disk, same VM class)         | Chroma cost grows with disk, not with a per-vector "pod tier" jump    |
| 10M        | ~$800–1,200+              | ~$50–120 (bigger VM, possibly sharded)       | At this scale, self-hosted needs real ops attention (backups, HA)    |

**Assumptions stated explicitly:**
- Managed DB pricing assumes always-on pods sized to the vector count, independent of query volume — the core problem this assignment targets.
- Self-hosted cost is just compute + disk (a small cloud VM), and excludes engineering time to maintain it.
- Embedding generation cost (Gemini API calls) is identical either way and excluded from this table, since it depends on ingestion volume, not storage choice.

**Trade-off accepted:** self-hosted ChromaDB has no built-in high availability, automatic backups, or horizontal scaling — at 10M+ vectors with real uptime requirements, the operational burden (and risk) shifts back toward justifying a managed DB, even at higher cost.

---

## Discussion

**When would I switch back to a managed vector DB?**
Once query volume or uptime requirements exceed what a single self-hosted instance can reasonably guarantee — specifically, if this needed multi-region availability, automatic failover, or if the team lacked bandwidth to own backups and scaling. For a lightly-queried corpus under ~10M vectors with no strict SLA, the cost savings of ChromaDB clearly outweigh the operational risk.

**Was retrieval or generation the weaker link?**
Based on the evaluation results in `eval_results.json`, retrieval numbers should be examined first: a low `nDCG@5`/`Recall@5` relative to `Precision@5` suggests the embedding model or chunking strategy is misranking relevant content, while strong retrieval metrics paired with weak answer faithfulness would point to the generation/prompting step instead. *(Fill in with your specific numbers once you've run `get_metrics.py` — e.g., "Recall@5 of 0.65 with Precision@5 of only 0.22 suggests the top-k is noisy, meaning retrieval is the weaker link here.")*

---

## Features

- PDF document ingestion with idempotent re-ingestion (no duplicate vectors)
- Configurable chunking (`CHUNK_SIZE`, `CHUNK_OVERLAP`) via `.env`
- Local, persistent vector storage using ChromaDB
- Metadata-based filtering (e.g. by source document)
- Grounded answer generation via Google Gemini, with citations limited to chunks actually used in the answer
- No-hallucination fallback when no chunk clears the similarity threshold
- Retry-with-backoff on transient API rate-limit/server errors
- Per-query latency, chunk count, and token usage logging
- Retrieval evaluation harness (Recall@k, Hit Rate@k, MRR, Precision@k, nDCG@k) with results exported to JSON
