from services.logger import logger

RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant.

Answer the user's question using ONLY the information provided in the context below.

Rules:
1. Do not use outside knowledge or make up information.
2. If the answer cannot be found in the context at all, state clearly that the information is not available in the uploaded document(s).
3. If asked to summarize, synthesize and paraphrase key points from the context.
4. Keep the answer clear, concise, and factual.
5. Whenever relevant, reference the source document and page number (e.g., "(Source: filename.pdf, Page: X)") using the metadata provided in each context block.

Context:
{context}

Question:
{question}

Answer:"""

def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Constructs a structured prompt for the LLM by combining system instructions,
    retrieved document chunks (with filename and page metadata), and the user query.

    Args:
        question (str): The user's question.
        retrieved_chunks (list[dict]): A list of dictionaries representing retrieved chunks.

    Returns:
        str: The compiled prompt string ready to be sent to the LLM.

    Raises:
        ValueError: If the question is empty, no retrieved chunks are provided, or if chunk texts are empty.
    """
    if not question or not question.strip():
        raise ValueError("The search question cannot be empty.")

    if not retrieved_chunks:
        raise ValueError("At least one retrieved document chunk must be provided.")

    formatted_chunks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk.get("text", "").strip()
        if not text:
            raise ValueError(f"Retrieved chunk {idx} contains empty or missing text.")

        metadata_lines = []
        filename = chunk.get("filename")
        if filename and filename != "unknown":
            metadata_lines.append(f"Source: {filename}")

        page = chunk.get("page_number")
        if page and page > 0:
            metadata_lines.append(f"Page: {page}")

        header = "\n".join(metadata_lines)
        if header:
            header += "\n\n"

        chunk_str = f"----------------------------------------\n{header}Content:\n{text}\n----------------------------------------"
        formatted_chunks.append(chunk_str)

    context_str = "\n\n".join(formatted_chunks)

    full_prompt = RAG_PROMPT_TEMPLATE.format(
        context=context_str,
        question=question.strip()
    )

    logger.info("Prompt Construction completed chunks_count=%d", len(retrieved_chunks))

    return full_prompt
