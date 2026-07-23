import os
import time
import streamlit as st
import config
from services.logger import logger
from services.pdf_loader import extract_text_from_pdf
from services.chunker import chunk_document_pages
from services.embedding_service import generate_embeddings
from services.vector_db import get_qdrant_client, create_collection_if_not_exists, upsert_embeddings, get_available_filenames

logger.info("Application Startup")

def initialize_directories():
    for directory in [config.UPLOAD_DIR, config.LOG_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

initialize_directories()

st.set_page_config(
    page_title="RAG Chatbot",
    layout="centered",
)

st.title("RAG Chatbot 🤖")

# 1. Sidebar Configuration & Clear Chat
st.sidebar.title("Configuration")

model_option = st.sidebar.selectbox(
    "Select LLM Model",
    options=config.LLM_MODELS,
)

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []

# 2. Document Ingestion Section
st.subheader("Document Processing")
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

process_button = st.button("Process PDFs")

if process_button:
    if not uploaded_files:
        st.warning("Please upload at least one PDF file before processing.")
    else:
        total_pages_processed = 0
        all_extracted_pages = []
        successful_files_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            file_size = uploaded_file.size
            status_text.text(f"Processing '{filename}'...")
            logger.info("PDF Upload filename=%s size=%d bytes", filename, file_size)
            
            save_path = os.path.join(config.UPLOAD_DIR, filename)
            try:
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            except Exception as e:
                logger.error("Failed to save PDF filename=%s error=%s", filename, e, exc_info=True)
                st.error(f"Failed to save {filename}: {str(e)}")
                continue
                
            try:
                extracted_pages = extract_text_from_pdf(save_path)
                total_pages_processed += len(extracted_pages)
                all_extracted_pages.extend(extracted_pages)
                successful_files_count += 1
                logger.info("PDF Processing filename=%s pages_extracted=%d", filename, len(extracted_pages))
            except Exception as e:
                logger.error("Failed to process PDF filename=%s error=%s", filename, e, exc_info=True)
                st.error(f"Failed to process '{filename}': {str(e)}")
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        status_text.empty()
        progress_bar.empty()
        
        if all_extracted_pages:
            chunks = []
            try:
                chunks = chunk_document_pages(all_extracted_pages)
                logger.info(
                    "Chunking total_chunks=%d chunk_size=%d overlap=%d",
                    len(chunks), config.DEFAULT_CHUNK_SIZE, config.DEFAULT_CHUNK_OVERLAP,
                )
            except Exception as e:
                logger.error("Failed during chunking error=%s", e, exc_info=True)
                st.error(f"Failed during chunking: {str(e)}")
            
            if chunks:
                st.session_state.processed_chunks = chunks
                embedded_chunks = []
                status_text.text("Generating embeddings...")
                try:
                    embedded_chunks = generate_embeddings(chunks)
                    status_text.empty()
                except Exception as e:
                    status_text.empty()
                    logger.error("Failed during embedding generation error=%s", e, exc_info=True)
                    st.error(f"Failed during embedding generation: {str(e)}")
                
                if embedded_chunks:
                    status_text.text("Storing vectors...")
                    try:
                        client = get_qdrant_client()
                        vector_size = len(embedded_chunks[0]["embedding"])
                        collection_name = config.QDRANT_COLLECTION_NAME
                        
                        create_collection_if_not_exists(client, collection_name, vector_size)
                        logger.info(
                            "Qdrant collection_name=%s vectors_stored=%d vector_dimension=%d",
                            collection_name, len(embedded_chunks), vector_size,
                        )
                        upsert_embeddings(client, collection_name, embedded_chunks)
                        
                        status_text.empty()
                        st.success(f"Successfully processed and indexed {successful_files_count} PDF document(s).")
                    except Exception as e:
                        status_text.empty()
                        logger.error("Qdrant upsert failed error=%s", e, exc_info=True)
                        st.error(
                            f"Could not connect or write to vector database. "
                            f"Please verify that Qdrant is running. Error details: {str(e)}"
                        )

# 3. Document Selector
st.markdown("---")
try:
    client_for_scope = get_qdrant_client()
    available_files = get_available_filenames(client_for_scope, config.QDRANT_COLLECTION_NAME)
except Exception:
    available_files = []

scope_option = st.selectbox(
    "Search within:",
    options=["All documents"] + available_files,
)
filename_filter = None if scope_option == "All documents" else scope_option

# 4. Chat Interface Section
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_chunks" not in st.session_state:
    st.session_state.processed_chunks = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Collect user chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message and store in history
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Validate model API keys
    if model_option == config.MODEL_GEMINI and not config.GEMINI_API_KEY:
        error_msg = "GEMINI_API_KEY is missing. Please check your environment configuration."
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    elif model_option == config.MODEL_QWEN and not config.HF_API_KEY:
        error_msg = "Hugging Face API Key (HF_API_KEY) is missing. Please check your environment configuration."
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        with st.chat_message("assistant"):
            status_text = st.empty()
            status_text.text("Retrieving context and generating answer...")
            try:
                from services.retrieval import retrieve_relevant_chunks
                from services.prompt_builder import build_prompt
                from services.llm_service import generate_response
                
                # Step 1: Conversation-Aware Hybrid Retrieval & Confidence Check
                past_history = st.session_state.messages[:-1] if len(st.session_state.messages) > 1 else []
                results, confidence_passed, confidence_reason = retrieve_relevant_chunks(
                    prompt, 
                    top_k=config.DEFAULT_TOP_K, 
                    filename_filter=filename_filter,
                    chat_history=past_history,
                    document_chunks=st.session_state.get("processed_chunks", [])
                )
                
                if not confidence_passed:
                    status_text.empty()
                    refusal_msg = "I couldn't find enough relevant information in the uploaded document(s) to answer that question."
                    st.info(refusal_msg)
                    st.session_state.messages.append({"role": "assistant", "content": refusal_msg})
                else:
                    logger.info(
                        "Retrieval query=%s filter=%s chunks_retrieved=%d similarity_scores=%s",
                        prompt, filename_filter, len(results),
                        [round(r.get("score", 0.0), 4) for r in results],
                    )

                    # Step 2: Prompt Construction
                    full_prompt = build_prompt(prompt, results)
                    logger.info("Prompt Construction prompt_created=True")

                    # Step 3: LLM Generation
                    llm_start = time.time()
                    response_text = generate_response(full_prompt, model_option)
                    generation_time = round(time.time() - llm_start, 2)
                    logger.info("LLM Generation model=%s generation_time=%.2fs", model_option, generation_time)

                    status_text.empty()
                    logger.info("Response success=True")
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                status_text.empty()
                logger.error("Response success=False error=%s", e, exc_info=True)
                error_msg = f"Error generating answer: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
