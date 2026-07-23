import config
from google import genai
from qdrant_client.models import Filter, FieldCondition, MatchValue
from services.vector_db import get_qdrant_client
from services.logger import logger
from services.memory import get_recent_conversation_context
from services.keyword_search import perform_keyword_search


def retrieve_relevant_chunks(
    question: str, 
    top_k: int = 3, 
    filename_filter: str = None,
    chat_history: list[dict] = None,
    document_chunks: list[dict] = None
) -> list[dict]:
    """
    Performs hybrid retrieval using vector search (Qdrant) and BM25 keyword search,
    merging and deduplicating the results by chunk_id.

    Args:
        question (str): The search query / question from the user.
        top_k (int, optional): The number of top relevant chunks to retrieve for each strategy. Defaults to 3.
        filename_filter (str, optional): Filename to filter the search. Defaults to None.
        chat_history (list[dict], optional): List of previous message dictionaries. Defaults to None.
        document_chunks (list[dict], optional): List of available in-memory document chunks for BM25. Defaults to None.

    Returns:
        list[dict]: A list of merged, deduplicated chunk dictionaries.

    Raises:
        ValueError: If the input question is empty, GEMINI_API_KEY is missing, or the Qdrant collection does not exist.
        Exception: For Qdrant connection issues or general failures.
    """
    if not question or not question.strip():
        raise ValueError("Search question cannot be empty.")

    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured in the environment variables.")

    # 1. Format conversation context from chat history
    context_str, turns_used = get_recent_conversation_context(chat_history or [], max_turns=3)

    if turns_used > 0:
        logger.info("Conversation history used=True turns=%d", turns_used)
        effective_query = f"Previous Conversation:\n{context_str}\n\nCurrent Question: {question}"
    else:
        logger.info("Conversation history used=False turns=0")
        effective_query = question

    # 2. Vector Search (Qdrant)
    try:
        genai_client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = genai_client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=effective_query
        )

        query_embedding = response.embeddings[0].values
    except Exception as e:
        raise RuntimeError(f"Search query execution failed in Qdrant: {e}") from e

    try:
        qdrant_client = get_qdrant_client()
    except Exception as e:
        raise Exception(f"Failed to connect to Qdrant database: {str(e)}")

    try:
        collection_exists = qdrant_client.collection_exists(collection_name=config.QDRANT_COLLECTION_NAME)
    except Exception as e:
        raise Exception(f"Failed to check Qdrant collection status: {str(e)}")

    if not collection_exists:
        raise ValueError(
            f"Collection '{config.QDRANT_COLLECTION_NAME}' does not exist. "
            "Please upload and process some PDF documents first to initialize the database."
        )

    query_filter = None
    if filename_filter:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="filename",
                    match=MatchValue(value=filename_filter)
                )
            ]
        )

    try:
        results = qdrant_client.query_points(
            collection_name=config.QDRANT_COLLECTION_NAME,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            timeout=30,
            with_payload=True,
        ).points
    except Exception as e:
        raise Exception(f"Search query execution failed in Qdrant: {str(e)}")

    vector_results = []
    for hit in results:
        payload = hit.payload or {}
        vector_results.append({
            "chunk_id": str(hit.id),
            "filename": payload.get("filename", "unknown"),
            "page_number": payload.get("page_number", 0),
            "chunk_index": payload.get("chunk_index", 0),
            "text": payload.get("text", ""),
            "score": hit.score
        })

    # 3. BM25 Keyword Search
    keyword_results = perform_keyword_search(
        question=question,
        chunks=document_chunks or [],
        top_k=top_k,
        filename_filter=filename_filter
    )

    # 4. Merge Results & Deduplicate by chunk_id
    seen_ids = set()
    merged_chunks = []

    for chunk in vector_results:
        chunk_id = chunk.get("chunk_id")
        if chunk_id and chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            merged_chunks.append(chunk)

    for chunk in keyword_results:
        chunk_id = chunk.get("chunk_id")
        if chunk_id and chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            merged_chunks.append(chunk)

    # 5. Re-ranking Stage
    from services.reranker import rerank_chunks
    final_chunks = rerank_chunks(question, merged_chunks, top_k=top_k)

    # 6. Confidence Check Stage
    from services.confidence import check_retrieval_confidence
    confidence_passed, confidence_reason = check_retrieval_confidence(
        final_chunks, 
        vector_results=vector_results, 
        min_vector_threshold=0.35
    )

    top_vector_score = max([float(c.get("score", 0.0)) for c in vector_results]) if vector_results else 0.0

    # 7. Logging
    logger.info(
        "Hybrid Retrieval vector_results=%d keyword_results=%d merged_results=%d",
        len(vector_results), len(keyword_results), len(merged_chunks)
    )
    logger.info(
        "Re-ranking merged_chunks=%d final_chunks=%d",
        len(merged_chunks), len(final_chunks)
    )
    logger.info(
        "Confidence Check passed=%s reason='%s' chunks_count=%d top_vector_score=%.4f",
        confidence_passed, confidence_reason, len(final_chunks), top_vector_score
    )

    if not confidence_passed:
        return [], False, confidence_reason

    return final_chunks, True, confidence_reason
