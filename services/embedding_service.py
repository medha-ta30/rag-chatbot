import os
from google import genai
import config

def generate_embeddings(chunks: list[dict]) -> list[dict]:
    """
    Generates embedding vectors for a list of text chunks using Google's text-embedding-004 model.

    Args:
        chunks (list[dict]): A list of chunk dictionaries:
            [{"chunk_id": "...", "filename": "...", "page_number": 1, "chunk_index": 0, "text": "..."}]

    Returns:
        list[dict]: A new list of chunk dictionaries with the 'embedding' vector field appended:
            [{
                "chunk_id": "...",
                "filename": "...",
                "page_number": 1,
                "chunk_index": 0,
                "text": "...",
                "embedding": [0.184, -0.392, ...]
            }]

    Raises:
        ValueError: If the GEMINI_API_KEY is not configured or the chunks list is empty.
    """
    # 1. Validation
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured in the environment variables.")
    
    if not chunks:
        raise ValueError("The list of chunks provided is empty.")

    # 2. Initialize Google GenAI client
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    embedded_chunks = []

    # 3. Process each chunk
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        chunk_id = chunk.get("chunk_id", "unknown")
        
        # Skip empty chunks
        if not text:
            print(f"Warning: Skipped empty chunk '{chunk_id}'.")
            continue

        try:
            # Call Google GenAI Embeddings API
            response = client.models.embed_content(
                model=config.EMBEDDING_MODEL,
                contents=text
            )
            
            # Extract embedding values (usually a list of floats)
            embedding_vector = response.embedding.values
            
            # Append the 'embedding' key at the end of the new dictionary
            new_chunk = {
                "chunk_id": chunk["chunk_id"],
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "embedding": embedding_vector
            }
            
            embedded_chunks.append(new_chunk)
            
        except Exception as e:
            # Print/log the warning and continue processing other chunks
            print(f"Warning: Failed to generate embedding for chunk '{chunk_id}': {str(e)}")

    return embedded_chunks
