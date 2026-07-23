def check_retrieval_confidence(
    retrieved_chunks: list[dict],
    vector_results: list[dict] = None,
    min_vector_threshold: float = 0.35
) -> tuple[bool, str]:
    """
    Evaluates whether the retrieved results meet the confidence criteria for LLM response generation.

    Args:
        retrieved_chunks (list[dict]): Merged/reranked chunk dictionaries.
        vector_results (list[dict], optional): Original vector search results containing similarity scores.
        min_vector_threshold (float, optional): Minimum vector similarity threshold. Defaults to 0.35.

    Returns:
        tuple[bool, str]:
            - confidence_passed (bool): True if confidence criteria are met, False otherwise.
            - confidence_reason (str): Detailed reason for logging.
    """
    if not retrieved_chunks:
        return False, "No document chunks retrieved."

    # Extract vector similarity scores from vector search results or retrieved chunks
    candidate_source = vector_results if vector_results is not None else retrieved_chunks
    vector_scores = [float(c.get("score", 0.0)) for c in candidate_source if "score" in c]

    if not vector_scores:
        return True, "Confidence passed (chunk matches found)."

    top_vector_score = max(vector_scores)

    if top_vector_score < min_vector_threshold:
        return (
            False,
            f"Top vector similarity score ({top_vector_score:.4f}) is below threshold ({min_vector_threshold:.2f})."
        )

    return True, f"Confidence passed with top vector score {top_vector_score:.4f}."
