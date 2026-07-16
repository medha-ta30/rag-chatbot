import os
import config

def chunk_document_pages(
    pages: list[dict], 
    chunk_size: int = config.DEFAULT_CHUNK_SIZE, 
    chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP
) -> list[dict]:
    """
    Chunks extracted pages into smaller, overlapping segments without splitting words.

    Args:
        pages (list[dict]): A list of dictionaries representing pages:
            [{"filename": "...", "page_number": 1, "text": "..."}]
        chunk_size (int): The target maximum size of each chunk in characters.
        chunk_overlap (int): The target overlap size between adjacent chunks in characters.

    Returns:
        list[dict]: A list of dictionaries representing chunks:
            [{
                "chunk_id": "filename_page_X_chunk_Y",
                "filename": "...",
                "page_number": X,
                "chunk_index": Y,
                "text": "..."
            }]
    """
    # Validate parameters
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and strictly less than chunk_size.")

    all_chunks = []

    for page in pages:
        filename = page.get("filename", "unknown")
        page_number = page.get("page_number", 0)
        text = page.get("text", "").strip()

        # Skip pages with no extractable text
        if not text:
            continue

        words = text.split()
        if not words:
            continue

        words_count = len(words)
        start_idx = 0
        chunk_index = 0

        while start_idx < words_count:
            current_words = []
            curr_char_len = 0

            # Step 1: Pack words into the current chunk
            i = start_idx
            while i < words_count:
                word = words[i]
                space_padding = 1 if len(current_words) > 0 else 0
                next_len = curr_char_len + space_padding + len(word)

                # Stop if adding the next word exceeds chunk_size
                if next_len > chunk_size and len(current_words) > 0:
                    break

                current_words.append(word)
                curr_char_len = next_len
                i += 1

            chunk_text = " ".join(current_words)
            chunk_id = f"{filename}_page_{page_number}_chunk_{chunk_index}"

            all_chunks.append({
                "chunk_id": chunk_id,
                "filename": filename,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text
            })

            chunk_index += 1

            # If we reached the end of words list, we are done with this page
            if i >= words_count:
                break

            # Step 2: Determine next start_idx incorporating character overlap
            backtrack_chars = 0
            backtrack_words_count = 0

            for word in reversed(current_words):
                space_padding = 1 if backtrack_words_count > 0 else 0
                backtrack_chars += len(word) + space_padding
                backtrack_words_count += 1
                if backtrack_chars >= chunk_overlap:
                    break

            # Safety fallback: always progress by at least one word to prevent infinite loops
            if backtrack_words_count >= len(current_words):
                backtrack_words_count = len(current_words) - 1

            if backtrack_words_count < 0:
                backtrack_words_count = 0

            start_idx = i - backtrack_words_count

    return all_chunks
