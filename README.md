# RAG System – Cost-Efficient QA over PDFs

This project implements a **Retrieval-Augmented Generation (RAG)** system for answering questions over PDF documents. It indexes PDFs into a local **ChromaDB** vector database and uses **Google Gemini** to generate grounded answers with source citations.

---

## 📁 Project Structure

```text
├── rag.py                 # Main application (ingest + query)
├── get_metrics.py         # Evaluation script (computes Recall@5, MRR, Precision@5)
├── .env.example           # Environment variables template (copy to .env)
├── docs/                  # Folder containing PDF files
│   └── sample.pdf         # Sample document (LaTeX guide)
└── README.md              # Project documentation (this file)

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/AkANkSHA-RaWaT-2026/rag_assignment.git
cd rag_assignment
```
---

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

---

### 3. Install dependencies

install the required packages :

```bash
pip install chromadb pypdf python-dotenv google-genai
```

---

### 4. Configure your Gemini API key

Create a `.env` file  and add your API key:

Inside the `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```
obtain a free API key from **Google AI Studio**.



## Ingest (Index) PDFs

Place your PDF files inside the `docs/` directory and run:

```bash
python rag.py ingest --folder ./docs
```

Example output:

```text
Ingesting PDFs from ./docs...
  Processing: sample.pdf
    Upserted 13 chunks for sample.pdf

✅ Total vectors in DB: 13
```

---

## Ask a Question

Run:

```bash
python rag.py query "What is the main topic of this document?"
```

Example output:

```text
============================================================
📝 ANSWER:
This document is a LaTeX template guide explaining how to
compile .tex files into PDFs using pdflatex, ghostscript,
and other tools.

📚 CITATIONS:
[1] sample.pdf
[2] sample.pdf

📄 CHUNKS USED: 5
⚡ LATENCY: 142.34 ms
🔢 EST. TOKENS: 450
============================================================

## Evaluation (Retrieval Metrics)

python get_metrics.py

output-
Average Recall@5: 0.650
Average MRR: 0.328
Average Precision@5: 0.220
nDCG@5 (approx): 0.585 (estimated)



## Vector Store

**ChromaDB** was chosen because it is a free, lightweight local vector database that provides persistent storage, metadata filtering, and efficient cosine similarity search.

---

## Features

- PDF document ingestion
- Automatic document chunking
- Local vector storage using ChromaDB
- Google Gemini for answer generation
- Source citations for retrieved content
- Metadata-based document filtering
- Persistent vector database
