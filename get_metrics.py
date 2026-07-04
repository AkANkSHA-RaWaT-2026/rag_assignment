"""
Retrieval evaluation for the RAG pipeline.

Computes Recall@k, Hit Rate@k, MRR, Precision@k, and nDCG@k against a
manually labeled set of relevant chunks, then writes results to
eval_results.json for submission alongside the code.

Usage:
    python get_metrics.py
"""

import json
import numpy as np
from rag import get_embedding, collection

K = 5  # top-k used throughout; change this one constant to evaluate a different k


def detect_source_filename() -> str:
    """Auto-detect the ingested PDF's filename from the collection, so it
    doesn't need to be hardcoded and kept in sync by hand."""
    all_metas = collection.get(include=["metadatas"])["metadatas"]
    sources = {m["source"] for m in all_metas}
    if len(sources) == 1:
        return next(iter(sources))
    raise ValueError(
        f"Expected exactly one ingested source file, found {sources}. "
        "Set SOURCE manually below if you're intentionally evaluating "
        "against multiple documents."
    )


SOURCE = detect_source_filename()

# Manually labeled relevant chunks, keyed by (source_filename, chunk_index).
# Chunk indices come from the order chunks were created during ingestion --
# check chroma_db contents or ingestion logs to verify these against your PDF.
relevant_chunks = {
    "What tools are needed to compile a .tex file?": {(SOURCE, 0), (SOURCE, 1), (SOURCE, 2)},
    "How to use pdflatex?": {(SOURCE, 1), (SOURCE, 2)},
    "What is the purpose of spell-checking?": {(SOURCE, 3)},
    "What are the three kinds of horizontal dashes?": {(SOURCE, 4)},
    "What is the role of ghostscript?": {(SOURCE, 1)},
    "How to view a .pdf file?": {(SOURCE, 2), (SOURCE, 5)},
    "What is the difference between LaTeX and pdflatex?": {(SOURCE, 6)},
    "What are reserved characters in LaTeX?": {(SOURCE, 5)},
    "What is the 'no relevant context' branch?": set(),  # expects no retrieval / low similarity
    "How to send a fax using .ps file?": {(SOURCE, 1), (SOURCE, 2)},
}

questions = list(relevant_chunks.keys())


def dcg(relevances):
    """Binary-relevance DCG: sum(rel_i / log2(i + 2))."""
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(retrieved_ids, relevant_set, k):
    """nDCG@k against a binary-relevance gold set."""
    relevances = [1 if rid in relevant_set else 0 for rid in retrieved_ids[:k]]
    actual = dcg(relevances)

    n_relevant_possible = min(len(relevant_set), k)
    ideal_relevances = [1] * n_relevant_possible + [0] * (k - n_relevant_possible)
    ideal = dcg(ideal_relevances)

    return actual / ideal if ideal > 0 else 0.0


def evaluate_query(question: str) -> dict:
    """Run one query through retrieval and score it against the gold set."""
    q_emb = get_embedding(question)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=K,
        include=["metadatas", "distances"],
    )
    metas = results["metadatas"][0]
    dist = results["distances"][0]

    # (source, chunk_index) uniquely identifies a chunk, unlike its position
    # in the results list, which tells you nothing about which chunk it is.
    retrieved_ids = [(m["source"], m["chunk_index"]) for m in metas]

    relevant = relevant_chunks.get(question, set())
    retrieved_relevant = [rid for rid in retrieved_ids if rid in relevant]

    # np.nan for queries with no gold-relevant chunks: they're excluded from
    # the Recall/Hit Rate averages rather than counted as failures, since
    # there's nothing to "recall" in the first place.
    recall = len(retrieved_relevant) / len(relevant) if relevant else np.nan
    precision = len(retrieved_relevant) / K
    hit = 1.0 if retrieved_relevant else (np.nan if not relevant else 0.0)

    mrr = 0.0
    for rank, rid in enumerate(retrieved_ids):
        if rid in relevant:
            mrr = 1.0 / (rank + 1)
            break

    ndcg = ndcg_at_k(retrieved_ids, relevant, K)

    return {
        "question": question,
        "retrieved_ids": [list(rid) for rid in retrieved_ids],  # tuples aren't JSON-serializable
        "top_distance": dist[0] if dist else None,
        "recall": recall,
        "hit": hit,
        "mrr": mrr,
        "precision": precision,
        "ndcg": ndcg,
    }


def main():
    per_query_results = [evaluate_query(q) for q in questions]

    recalls = [r["recall"] for r in per_query_results if not np.isnan(r["recall"])]
    hits = [r["hit"] for r in per_query_results if not np.isnan(r["hit"])]
    mrrs = [r["mrr"] for r in per_query_results]
    precisions = [r["precision"] for r in per_query_results]
    ndcgs = [r["ndcg"] for r in per_query_results]

    for r in per_query_results:
        print(f"Q: {r['question'][:50]}")
        print(f"  Retrieved: {r['retrieved_ids']}")
        recall_display = f"{r['recall']:.2f}" if not np.isnan(r["recall"]) else "N/A (no gold relevant)"
        print(f"  Recall@{K}: {recall_display}, MRR: {r['mrr']:.2f}, "
              f"Precision@{K}: {r['precision']:.2f}, nDCG@{K}: {r['ndcg']:.2f}")
        print()

    summary = {
        f"avg_recall@{K}": float(np.mean(recalls)),
        f"avg_hit_rate@{K}": float(np.mean(hits)),
        "avg_mrr": float(np.mean(mrrs)),
        f"avg_precision@{K}": float(np.mean(precisions)),
        f"avg_ndcg@{K}": float(np.mean(ndcgs)),
        "n_queries": len(questions),
        "n_scoreable_recall_queries": len(recalls),
    }

    print("=" * 50)
    for key, value in summary.items():
        print(f"{key}: {value}")

    with open("eval_results.json", "w") as f:
        json.dump({"summary": summary, "per_query": per_query_results}, f, indent=2, default=str)
    print("\nSaved detailed results to eval_results.json")


if __name__ == "__main__":
    main()
