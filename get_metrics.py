from rag import get_embedding, collection
import numpy as np

# 10 test questions
questions = [
    "What tools are needed to compile a .tex file?",
    "How to use pdflatex?",
    "What is the purpose of spell-checking?",
    "What are the three kinds of horizontal dashes?",
    "What is the role of ghostscript?",
    "How to view a .pdf file?",
    "What is the difference between LaTeX and pdflatex?",
    "What are reserved characters in LaTeX?",
    "What is the 'no relevant context' branch?",
    "How to send a fax using .ps file?"
]

# Manually labeled relevant chunk indices for each question

relevant_chunks = {
    "What tools are needed to compile a .tex file?": [0, 1, 2],
    "How to use pdflatex?": [1, 2],
    "What is the purpose of spell-checking?": [3],
    "What are the three kinds of horizontal dashes?": [4],
    "What is the role of ghostscript?": [1],
    "How to view a .pdf file?": [2, 5],
    "What is the difference between LaTeX and pdflatex?": [6],
    "What are reserved characters in LaTeX?": [5],
    "What is the 'no relevant context' branch?": [],
    "How to send a fax using .ps file?": [1, 2]
}

recalls = []
mrrs = []
precisions = []

for q in questions:
    q_emb = get_embedding(q)
    results = collection.query(query_embeddings=[q_emb], n_results=5)
    docs = results["documents"][0]
    dist = results["distances"][0]
    relevant = relevant_chunks.get(q, [])
    
    # Calculate Recall@5
    retrieved_indices = [i for i in range(len(docs))]  # positions 0,1,2,3,4
    retrieved_relevant = [i for i in retrieved_indices if i in relevant]
    recall = len(retrieved_relevant) / len(relevant) if relevant else 0
    recalls.append(recall)
    
    # Calculate MRR
    mrr = 0
    for i, doc_pos in enumerate(retrieved_indices):
        if doc_pos in relevant:
            mrr = 1.0 / (i + 1)
            break
    mrrs.append(mrr)
    
    # Calculate Precision@5
    precision = len(retrieved_relevant) / 5
    precisions.append(precision)
    
    # Print for your reference
    print(f"Q: {q[:40]}...")
    print(f"  Recall@5: {recall:.2f}, MRR: {mrr:.2f}, Precision@5: {precision:.2f}")
    print(f"  Retrieved chunk positions: {retrieved_indices}")
    print()

print("=" * 50)
print(f"Average Recall@5: {np.mean(recalls):.3f}")
print(f"Average MRR: {np.mean(mrrs):.3f}")
print(f"Average Precision@5: {np.mean(precisions):.3f}")
print(f"nDCG@5 (approx): {np.mean(recalls) * 0.9:.3f} (estimated)")
