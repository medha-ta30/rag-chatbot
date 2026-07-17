import config
from google import genai
from qdrant_client.models import Filter, FieldCondition, MatchValue
from services.vector_db import get_qdrant_client

def retrieve_relevant_chunks(question: str, top_k: int = 3, filename_filter: str = None) -> list[dict]:
    """
    Generates an embedding for the user's question and searches the Qdrant
    vector database for the top_k most similar document chunks, optionally filtered by filename.

    Args:
        question (str): The search query / question from the user.
        top_k (int, optional): The number of top relevant chunks to retrieve. Defaults to 3.
        filename_filter (str, optional): Filename to filter the search. Defaults to None.

    Returns:
        list[dict]: A list of dictionaries representing the matching chunks and metadata:
            [
                {
                    "chunk_id": str,
                    "filename": str,
                    "page_number": int,
                    "chunk_index": int,
                    "text": str,
                    "score": float
                }
            ]

    Raises:
        ValueError: If the input question is empty, GEMINI_API_KEY is missing, or the Qdrant collection does not exist.
        Exception: For Qdrant connection issues or general failures.
    """
    # 1. Input validation
    if not question or not question.strip():
        raise ValueError("Search question cannot be empty.")

    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured in the environment variables.")

    # 2. Generate embedding for the question
    try:
        genai_client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = genai_client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=question
        )

        print(response)

        query_embedding = response.embeddings[0].values
    except Exception as e:
        raise RuntimeError(f"Search query execution failed in Qdrant: {e}") from e

    # 3. Connect to Qdrant and search
    try:
        qdrant_client = get_qdrant_client()
    except Exception as e:
        raise Exception(f"Failed to connect to Qdrant database: {str(e)}")

    # Check if collection exists
    try:
        collection_exists = qdrant_client.collection_exists(collection_name=config.QDRANT_COLLECTION_NAME)
    except Exception as e:
        raise Exception(f"Failed to check Qdrant collection status: {str(e)}")

    if not collection_exists:
        raise ValueError(
            f"Collection '{config.QDRANT_COLLECTION_NAME}' does not exist. "
            "Please upload and process some PDF documents first to initialize the database."
        )

    # Build filter if filename_filter is specified
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

    # 4. Format and return results
    retrieved_chunks = []
    for hit in results:
        payload = hit.payload or {}
        retrieved_chunks.append({
            "chunk_id": str(hit.id),
            "filename": payload.get("filename", "unknown"),
            "page_number": payload.get("page_number", 0),
            "chunk_index": payload.get("chunk_index", 0),
            "text": payload.get("text", ""),
            "score": hit.score
        })

    return retrieved_chunks
