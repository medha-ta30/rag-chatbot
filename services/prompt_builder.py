RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant.

Answer the user's question using ONLY the information provided in the context below.

If asked to summarize, synthesize and paraphrase the key points from the context into
your own summary — the context does not need to already contain a pre-written summary.

If the context partially answers the question, answer with what's available and note
what's missing. If the answer cannot be found in the context at all, respond exactly with:
"I don't have enough information to answer this question."

Do not make up information not present in the context.

Context:
{context}

Question:
{question}

Answer:"""

def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Constructs a structured prompt for the LLM by combining system instructions,
    retrieved document chunks (with metadata and similarity scores), and the user query.

    Args:
        question (str): The user's question.
        retrieved_chunks (list[dict]): A list of dictionaries representing retrieved chunks:
            [{"chunk_id": "...", "filename": "...", "page_number": 1, "chunk_index": 0, "text": "...", "score": 0.95}]

    Returns:
        str: The compiled prompt string ready to be sent to the LLM.

    Raises:
        ValueError: If the question is empty, no retrieved chunks are provided, or if chunk texts are empty.
    """
    # 1. Validation
    if not question or not question.strip():
        raise ValueError("The search question cannot be empty.")

    if not retrieved_chunks:
        raise ValueError("At least one retrieved document chunk must be provided.")

    # 2. Build the context section from the chunks
    formatted_chunks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        filename = chunk.get("filename", "unknown")
        page = chunk.get("page_number", 0)
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "").strip()

        if not text:
            raise ValueError(f"Retrieved chunk {idx} contains empty or missing text.")

        chunk_str = (
            f"[Chunk {idx}]\n"
            f"Filename: {filename}\n"
            f"Page: {page}\n"
            f"Text:\n{text}"
        )
        formatted_chunks.append(chunk_str)

    # Join the chunks with a separator
    context_str = "\n\n------------------------\n\n".join(formatted_chunks)

    # 3. Format into final prompt template
    full_prompt = RAG_PROMPT_TEMPLATE.format(
        context=context_str,
        question=question.strip()
    )

    return full_prompt
