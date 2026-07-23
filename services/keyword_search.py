from rank_bm25 import BM25Okapi

def perform_keyword_search(
    question: str, 
    chunks: list[dict], 
    top_k: int = 3,
    filename_filter: str = None
) -> list[dict]:
    """
    Performs BM25 keyword search over a list of document chunks.

    Args:
        question (str): The search query / user question.
        chunks (list[dict]): A list of chunk dictionaries containing 'text' and metadata.
        top_k (int, optional): The maximum number of top matching chunks to return. Defaults to 3.
        filename_filter (str, optional): Optional filename to filter chunks before ranking. Defaults to None.

    Returns:
        list[dict]: A list of top matching chunk dictionaries with their BM25 score included.
    """
    if not question or not question.strip() or not chunks:
        return []

    # Filter chunks by filename if specified
    candidate_chunks = chunks
    if filename_filter:
        candidate_chunks = [c for c in chunks if c.get("filename") == filename_filter]

    if not candidate_chunks:
        return []

    # Tokenize corpus and query for BM25
    tokenized_corpus = [chunk.get("text", "").lower().split() for chunk in candidate_chunks]
    tokenized_query = question.lower().split()

    if not tokenized_query:
        return []

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    scored_chunks = []
    for idx, score in enumerate(scores):
        if score > 0:  # Only include chunks with positive keyword relevance
            chunk_copy = dict(candidate_chunks[idx])
            chunk_copy["score"] = float(score)
            scored_chunks.append(chunk_copy)

    # Sort in descending order of BM25 score
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]
