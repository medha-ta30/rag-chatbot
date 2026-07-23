import re

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "how", "why", "where", "when", "in", "of", "to", "for", "with",
    "on", "at", "by", "from", "about"
}

def rerank_chunks(question: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    Reranks a list of merged document chunks using exact query keyword match counts
    as a primary signal and initial similarity/BM25 scores as a secondary signal.

    Args:
        question (str): User search question.
        chunks (list[dict]): Merged retrieved chunks from vector and BM25 search.
        top_k (int, optional): Number of top reranked chunks to return. Defaults to 3.

    Returns:
        list[dict]: Reranked and sliced list of chunk dictionaries with 'rerank_score' included.
    """
    if not question or not question.strip() or not chunks:
        return []

    # Clean and tokenize query words excluding common stop words
    words = re.findall(r'\w+', question.lower())
    query_terms = set(w for w in words if w not in STOP_WORDS and len(w) > 1)

    if not query_terms:
        query_terms = set(words)

    reranked = []
    for chunk in chunks:
        chunk_text = chunk.get("text", "").lower()
        base_score = float(chunk.get("score", 0.0))

        # Count exact keyword matches in the chunk text
        keyword_matches = sum(1 for term in query_terms if term in chunk_text)

        # Composite score: primary keyword count + base score as secondary signal
        rerank_score = float(keyword_matches) + base_score

        chunk_copy = dict(chunk)
        chunk_copy["rerank_score"] = round(rerank_score, 4)
        reranked.append(chunk_copy)

    # Sort descending by composite rerank score
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
