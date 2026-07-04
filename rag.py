"""
Cost-efficient RAG service backed by ChromaDB and the Gemini API.

Usage:
    python rag.py ingest --folder ./docs
    python rag.py query "your question" --k 5 --filter '{"source":"report.pdf"}'
"""

import os
import re
import hashlib
import time
import argparse
import json
from typing import List, Dict, Any, Optional

import chromadb
from pypdf import PdfReader
from google import genai
from google.genai import errors as genai_errors
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
COLLECTION_NAME = "rag_docs"
SIMILARITY_THRESHOLD = float(os.getenv("SIM_THRESHOLD", 0.45))

# Model names are configurable via .env rather than hardcoded, since
# providers deprecate specific model versions without much notice.
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")
LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "models/gemini-2.5-flash")

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

# Only retry transient failures (rate limits, server errors) -- retrying a
# bad API key or malformed request would just fail the same way every time.
_RETRYABLE_ERRORS = (genai_errors.ClientError, genai_errors.ServerError)


def chunk_text(text: str) -> List[str]:
    """Split text into overlapping fixed-size chunks."""
    if CHUNK_SIZE <= CHUNK_OVERLAP:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


@retry(
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def get_embedding(text: str) -> List[float]:
    """Get a single text embedding, retrying on rate-limit/server errors."""
    response = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return response.embeddings[0].values


@retry(
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _generate_with_retry(prompt: str):
    return client.models.generate_content(model=LLM_MODEL, contents=prompt)


def extract_cited_indices(answer_text: str) -> set:
    """Parse which [N] markers the model actually used in its answer."""
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}


def ingest_pdfs(folder_path: str):
    """Ingest all PDFs in a folder. Idempotent: re-running upserts by a
    content+filename hash, so no duplicate vectors are created."""
    print(f"Ingesting PDFs from {folder_path}...")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created {folder_path}. Please add PDFs there.")
        return

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(folder_path, filename)
        print(f"  Processing: {filename}")

        reader = PdfReader(filepath)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

        chunks = chunk_text(full_text)

        ids, docs, metas, embs = [], [], [], []
        for i, chunk in enumerate(chunks):
            uid = hashlib.md5(f"{chunk}_{filename}".encode()).hexdigest()
            ids.append(uid)
            docs.append(chunk)
            metas.append({"source": filename, "chunk_index": i})
            embs.append(get_embedding(chunk))

        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        print(f"    Upserted {len(chunks)} chunks for {filename}")

    print(f"Total vectors in DB: {collection.count()}")


def query_rag(question: str, k: int = 5, filter_meta: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Retrieve top-k chunks, then generate a grounded, cited answer.

    Returns a well-formed dict even if generation fails (e.g. API quota
    issues), so retrieval metrics are never lost to a downstream error.
    """
    start_time = time.time()

    q_emb = get_embedding(question)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        where=filter_meta,
        include=["documents", "metadatas", "distances"],
    )

    retrieved_docs = results["documents"][0] if results["documents"] else []
    retrieved_metas = results["metadatas"][0] if results["metadatas"] else []
    retrieved_distances = results["distances"][0] if results["distances"] else []

    latency_ms = (time.time() - start_time) * 1000

    # No-hallucination branch: refuse to answer if nothing relevant was found.
    if not retrieved_docs or retrieved_distances[0] > SIMILARITY_THRESHOLD:
        return {
            "answer": "No relevant context found. I cannot answer this question.",
            "citations": [],
            "chunk_count": 0,
            "latency_ms": latency_ms,
            "tokens": 0,
            "retrieved_chunks": [],
            "generation_ok": True,
        }

    context_str = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(retrieved_docs))

    prompt = f"""You are a precise QA assistant. Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I don't know".
Cite your sources using [1], [2], etc.

CONTEXT:
{context_str}

QUESTION: {question}

ANSWER (with citations):"""

    try:
        response = _generate_with_retry(prompt)
        answer_text = response.text
        total_tokens = len(prompt.split()) + len(answer_text.split())
        generation_ok = True

        # Only report citations the model actually used, not every chunk
        # that was retrieved -- a retrieved-but-unused chunk isn't a citation.
        cited_indices = extract_cited_indices(answer_text)
        citations = [
            f"[{i+1}] {meta['source']}"
            for i, meta in enumerate(retrieved_metas)
            if (i + 1) in cited_indices
        ]
    except _RETRYABLE_ERRORS as e:
        print(f"Generation failed: {e}")
        answer_text = (
            "Retrieval succeeded, but answer generation failed "
            f"(model: {LLM_MODEL}). This is typically an API quota/rate "
            "limit issue rather than a retrieval bug. See logs for details."
        )
        total_tokens = 0
        generation_ok = False
        citations = []

    return {
        "answer": answer_text,
        "citations": citations,
        "chunk_count": len(retrieved_docs),
        "latency_ms": latency_ms,
        "tokens": total_tokens,
        "retrieved_chunks": retrieved_docs,
        "generation_ok": generation_ok,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG with Gemini & ChromaDB")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--folder", default="./docs", help="Folder with PDFs")

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("question", type=str, help="Your question")
    query_parser.add_argument("--k", type=int, default=5, help="Top-k chunks")
    query_parser.add_argument(
        "--filter", type=str, default=None,
        help='Metadata filter as JSON, e.g. \'{"source":"report.pdf"}\'',
    )

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_pdfs(args.folder)

    elif args.command == "query":
        filter_dict = json.loads(args.filter) if args.filter else None
        result = query_rag(args.question, k=args.k, filter_meta=filter_dict)

        print("\n" + "=" * 60)
        print(f"ANSWER: {result['answer']}")
        print(f"CITATIONS: {result['citations']}")
        print(f"CHUNKS USED: {result['chunk_count']}")
        print(f"LATENCY: {result['latency_ms']:.2f} ms")
        print(f"EST. TOKENS: {result['tokens']}")
        print(f"GENERATION OK: {result['generation_ok']}")
        print("=" * 60)
