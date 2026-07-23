import time
import config
from google import genai
from services.logger import logger


def generate_embeddings(chunks: list[dict], batch_size: int = 20, max_retries: int = 5) -> list[dict]:
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured in the environment variables.")
    if not chunks:
        raise ValueError("The list of chunks provided is empty.")

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    embedded_chunks = []

    valid_chunks = []
    for chunk in chunks:
        if chunk.get("text", "").strip():
            valid_chunks.append(chunk)
        else:
            logger.warning("Skipped empty chunk chunk_id=%s", chunk.get("chunk_id", "unknown"))

    skipped_count = len(chunks) - len(valid_chunks)
    if skipped_count > 0:
        logger.info("Embedding Generation skipped_chunks=%d", skipped_count)

    for i in range(0, len(valid_chunks), batch_size):
        batch = valid_chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]

        attempt = 0
        while attempt <= max_retries:
            try:
                response = client.models.embed_content(
                    model=config.EMBEDDING_MODEL,
                    contents=texts,
                )
                for chunk, emb in zip(batch, response.embeddings):
                    embedded_chunks.append({**chunk, "embedding": emb.values})
                break

            except Exception as e:
                if "429" in str(e) and attempt < max_retries:
                    wait_seconds = 2 ** attempt * 5
                    logger.warning(
                        "Rate limited on batch batch=%d retrying_in=%ds",
                        i // batch_size, wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    attempt += 1
                else:
                    logger.error(
                        "Failed to embed batch batch=%d error=%s",
                        i // batch_size, e, exc_info=True,
                    )
                    break

    if embedded_chunks:
        vector_dim = len(embedded_chunks[0]["embedding"]) if embedded_chunks else 0
        logger.info(
            "Embedding Generation embedding_model=%s vectors_generated=%d vector_dimension=%d",
            config.EMBEDDING_MODEL, len(embedded_chunks), vector_dim,
        )

    return embedded_chunks
