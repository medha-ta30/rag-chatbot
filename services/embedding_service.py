import time
import config
from google import genai

def generate_embeddings(chunks: list[dict], batch_size: int = 20, max_retries: int = 5) -> list[dict]:
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured in the environment variables.")
    if not chunks:
        raise ValueError("The list of chunks provided is empty.")

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    embedded_chunks = []

    # Filter out empty chunks up front, same as before
    valid_chunks = []
    for chunk in chunks:
        if chunk.get("text", "").strip():
            valid_chunks.append(chunk)
        else:
            print(f"Warning: Skipped empty chunk '{chunk.get('chunk_id', 'unknown')}'.")

    for i in range(0, len(valid_chunks), batch_size):
        batch = valid_chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]

        attempt = 0
        while attempt <= max_retries:
            try:
                response = client.models.embed_content(
                    model=config.EMBEDDING_MODEL,
                    contents=texts,   # list in, list of embeddings out — one request for the whole batch
                )
                for chunk, emb in zip(batch, response.embeddings):
                    embedded_chunks.append({**chunk, "embedding": emb.values})
                break  # batch succeeded, move to next batch

            except Exception as e:
                if "429" in str(e) and attempt < max_retries:
                    wait_seconds = 2 ** attempt * 5   # 5s, 10s, 20s, 40s, 80s
                    print(f"Rate limited on batch {i // batch_size}; retrying in {wait_seconds}s...")
                    time.sleep(wait_seconds)
                    attempt += 1
                else:
                    print(f"Failed to embed batch starting at chunk {i}: {e}")
                    break  # give up on this batch after exhausting retries, don't loop forever

    return embedded_chunks