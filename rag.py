import os
import hashlib
import time
import argparse
import json
from typing import List, Dict, Optional

import chromadb
from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
COLLECTION_NAME = "rag_docs"
SIMILARITY_THRESHOLD = float(os.getenv("SIM_THRESHOLD", 0.45))

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
EMBED_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "models/gemini-2.0-flash"

db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

def chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def get_embedding(text: str) -> List[float]:
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text
    )
    return resp.embeddings[0].values

def ingest_pdfs(folder_path: str):
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
    print(f"✅ Total vectors in DB: {collection.count()}")

def query_rag(question: str, k: int = 5, filter_meta: Optional[Dict[str, str]] = None):
    start_time = time.time()
    q_emb = get_embedding(question)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        where=filter_meta
    )
    retrieved_docs = results["documents"][0] if results["documents"] else []
    retrieved_metas = results["metadatas"][0] if results["metadatas"] else []
    retrieved_distances = results["distances"][0] if results["distances"] else []
    latency_ms = (time.time() - start_time) * 1000

    if not retrieved_docs or retrieved_distances[0] > SIMILARITY_THRESHOLD:
        return {
            "answer": "No relevant context found. I cannot answer this question.",
            "citations": [],
            "chunk_count": 0,
            "latency_ms": latency_ms,
            "tokens": 0,
            "retrieved_chunks": []
        }

    context_parts = [f"[{i+1}] {doc}" for i, doc in enumerate(retrieved_docs)]
    context_str = "\n\n".join(context_parts)
    prompt = f"""You are a precise QA assistant. Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I don't know".
Cite your sources using [1], [2], etc.

CONTEXT:
{context_str}

QUESTION: {question}

ANSWER (with citations):"""
    
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )
    total_tokens = len(prompt.split()) + len(response.text.split())
    return {
        "answer": response.text,
        "citations": [f"[{i+1}] {meta['source']}" for i, meta in enumerate(retrieved_metas)],
        "chunk_count": len(retrieved_docs),
        "latency_ms": latency_ms,
        "tokens": total_tokens,
        "retrieved_chunks": retrieved_docs
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--folder", default="./docs")
    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("question", type=str)
    query_parser.add_argument("--k", type=int, default=5)
    query_parser.add_argument("--filter", type=str, default=None)
    args = parser.parse_args()
    if args.command == "ingest":
        ingest_pdfs(args.folder)
    elif args.command == "query":
        filter_dict = json.loads(args.filter) if args.filter else None
        result = query_rag(args.question, k=args.k, filter_meta=filter_dict)
        print("\n" + "="*60)
        print(f"📝 ANSWER: {result['answer']}")
        print(f"📚 CITATIONS: {result['citations']}")
        print(f"📄 CHUNKS USED: {result['chunk_count']}")
        print(f"⚡ LATENCY: {result['latency_ms']:.2f} ms")
        print(f"🔢 EST. TOKENS: {result['tokens']}")
        print("="*60)
