
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
import config
import streamlit as st

@st.cache_resource
def get_qdrant_client() -> QdrantClient:
    """
    Initializes and returns a QdrantClient instance using the URL and API key from configuration.

    Returns:
        QdrantClient: An active client instance configured for the Qdrant database.

    Raises:
        Exception: If the connection configuration is missing or invalid.
    """
    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY if config.QDRANT_API_KEY else None,
        timeout=60
    )

def create_collection_if_not_exists(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """
    Checks if a collection exists in Qdrant; if it does not, creates it with Cosine distance.

    Args:
        client (QdrantClient): An active QdrantClient instance.
        collection_name (str): The name of the collection to check/create.
        vector_size (int): The dimensionality of the vectors to configure (e.g. 768).

    Returns:
        None

    Raises:
        Exception: If collection status query or collection creation fails.
    """
    exists = client.collection_exists(collection_name=collection_name)
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

def get_available_filenames(client: QdrantClient, collection_name: str) -> list[str]:
    """
    Returns the distinct source filenames currently stored in the collection,
    for populating a document-scope selector in the UI.

    Args:
        client (QdrantClient): An active QdrantClient instance.
        collection_name (str): The collection to inspect.

    Returns:
        list[str]: Sorted, deduplicated list of filenames found in stored payloads.
    """
    if not client.collection_exists(collection_name=collection_name):
        return []

    records, _ = client.scroll(
        collection_name=collection_name,
        limit=1000,
        with_payload=["filename"],
    )
    return sorted({r.payload["filename"] for r in records if r.payload.get("filename")})

def create_collection_if_not_exists(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    exists = client.collection_exists(collection_name=collection_name)
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        # Required for filtering by filename in retrieve_relevant_chunks
        client.create_payload_index(
            collection_name=collection_name,
            field_name="filename",
            field_schema=PayloadSchemaType.KEYWORD,
        )

def upsert_embeddings(client: QdrantClient, collection_name: str, embedded_chunks: list[dict]) -> None:
    """
    Upserts a list of embedded chunks into the specified Qdrant collection.
    Generates deterministic UUIDs for each point from their chunk IDs.

    Args:
        client (QdrantClient): An active QdrantClient instance.
        collection_name (str): The name of the collection to upsert points into.
        embedded_chunks (list[dict]): A list of dictionaries representing chunks with embeddings:
            [{"chunk_id": "...", "filename": "...", "page_number": 1, "chunk_index": 0, "text": "...", "embedding": [...]}]

    Returns:
        None

    Raises:
        Exception: If vector formatting or upsert communication fails.
    """
    points = []
    
    for chunk in embedded_chunks:
        chunk_id = chunk["chunk_id"]
        
        # Generate a deterministic UUID based on the chunk_id string
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
        
        # Build payload containing metadata fields
        payload = {
            "filename": chunk["filename"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"]
        }
        
        point = PointStruct(
            id=point_id,
            vector=chunk["embedding"],
            payload=payload
        )
        
        points.append(point)

    BATCH_SIZE = 100

    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]

        result = client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True,
            timeout=60
        )

        # Optional status check (depends on qdrant-client version)
        try:
            if hasattr(result, "status") and str(result.status).lower() != "completed":
                raise RuntimeError(f"Batch {i // BATCH_SIZE + 1} upload failed: {result.status}")
            print(f"Uploaded batch {i // BATCH_SIZE + 1}: {len(batch)} vectors")
        except Exception:
            pass
