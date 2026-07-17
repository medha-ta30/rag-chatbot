import os
import streamlit as st
import config
from services.pdf_loader import extract_text_from_pdf
from services.chunker import chunk_document_pages
from services.embedding_service import generate_embeddings
from services.vector_db import get_qdrant_client, create_collection_if_not_exists, upsert_embeddings, get_available_filenames
import re

# 1. Initialize directory structure
def initialize_directories():
    for directory in [config.UPLOAD_DIR ]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

initialize_directories()

# 2. Configure Streamlit Page Layout
st.set_page_config(
    page_title="RAG Chatbot",
    layout="centered",
)

st.title("RAG Chatbot 🤖")

# 4. Sidebar Configuration
st.sidebar.title("Configuration")

# Model Selection
model_option = st.sidebar.selectbox(
    "Select LLM Model",
    options=config.LLM_MODELS,
)


# 5. Main Page Content (PDF Upload & Ingestion)
uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
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
        
        # Create progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            status_text.text(f"Processing '{filename}'...")
            
            # Save file to uploads directory for processing and debugging
            save_path = os.path.join(config.UPLOAD_DIR, filename)
            try:
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            except Exception as e:
                st.error(f"Failed to save {filename}: {str(e)}")
                continue
                
            # Perform text extraction
            try:
                extracted_pages = extract_text_from_pdf(save_path)
                total_pages_processed += len(extracted_pages)
                all_extracted_pages.extend(extracted_pages)
                successful_files_count += 1
            except Exception as e:
                st.error(f"Failed to process '{filename}': {str(e)}")
            
            # Update progress
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        status_text.empty()
        progress_bar.empty()
        
        if all_extracted_pages:
            # Step 2: Custom Chunking
            chunks = []
            try:
                chunks = chunk_document_pages(all_extracted_pages)
            except Exception as e:
                st.error(f"Failed during chunking: {str(e)}")
            
            # Step 3: Generate Embeddings
            if chunks:
                embedded_chunks = []
                status_text.text("Generating embeddings...")
                try:
                    embedded_chunks = generate_embeddings(chunks)
                    status_text.empty()
                except Exception as e:
                    status_text.empty()
                    st.error(f"Failed during embedding generation: {str(e)}")
                
                # Step 4: Store in Qdrant Vector Database
                if embedded_chunks:
                    if len(embedded_chunks) < len(chunks):
                        skipped_count = len(chunks) - len(embedded_chunks)
                        st.warning(f"Warning: {skipped_count} chunk(s) failed to generate embeddings and were skipped.")
                        
                    db_stored_count = 0
                    qdrant_connected = False
                    status_text.text("Connecting and storing vectors in Qdrant...")
                    
                    try:
                        # Connect to database
                        client = get_qdrant_client()
                        vector_size = len(embedded_chunks[0]["embedding"])
                        collection_name = config.QDRANT_COLLECTION_NAME
                        
                        # Initialize collection if not exists
                        create_collection_if_not_exists(client, collection_name, vector_size)
                        
                        # Upsert points
                        upsert_embeddings(client, collection_name, embedded_chunks)
                        
                        db_stored_count = len(embedded_chunks)
                        qdrant_connected = True
                        status_text.empty()
                    except Exception as e:
                        status_text.empty()
                        st.error(
                            f"Could not connect or write to Qdrant vector database. "
                            f"Please verify that Qdrant is running and check your configuration settings. "
                            f"Error details: {str(e)}"
                        )
                    
                    # Display processing summary
                    st.success(
                        f"Successfully processed {successful_files_count} PDF(s) "
                        f"with a total of {total_pages_processed} page(s), "
                        f"generating {len(chunks)} chunk(s), "
                        f"and storing {db_stored_count} vector(s) in Qdrant."
                    )
                    
                    # Dashboard Grid
                    st.markdown("### Process Dashboard")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Embeddings", len(embedded_chunks))
                        st.write(f"**Model:** `{config.EMBEDDING_MODEL}`")
                    with col2:
                        st.metric("Vector Dimension", len(embedded_chunks[0]["embedding"]))
                    with col3:
                        conn_status = "🟢 Connected" if qdrant_connected else "🔴 Disconnected"
                        st.metric("Qdrant Status", conn_status)
                        st.write(f"**Collection:** `{config.QDRANT_COLLECTION_NAME}`")
                        
                    # Preview first 3 chunks with their embedding metadata
                    st.markdown("### Chunking & Database Preview")
                    preview_count = min(3, len(embedded_chunks))
                    
                    with st.expander(f"🔍 Preview Chunks & Vectors (First {preview_count})"):
                        for i in range(preview_count):
                            chunk = embedded_chunks[i]
                            chunk_preview = [round(val, 4) for val in chunk["embedding"][:5]]
                            st.markdown(f"**Chunk ID**: `{chunk['chunk_id']}`")
                            st.write(f"- **Source File**: {chunk['filename']}")
                            st.write(f"- **Page Number**: {chunk['page_number']}")
                            st.write(f"- **Chunk Index**: {chunk['chunk_index']}")
                            st.write(f"- **Embedding (first 5 dimensions)**: `{chunk_preview}...`")
                            st.code(chunk['text'], language=None)
                            st.markdown("---")

# 6. Semantic Retrieval & Response Generation Section
st.markdown("---")
st.markdown("### 🤖 Query & Response Generation")
st.write("Submit a question to retrieve relevant context and generate an answer.")

# Document scope selector — lets HR staff narrow a question to one specific policy doc
client_for_scope = get_qdrant_client()
available_files = get_available_filenames(client_for_scope, config.QDRANT_COLLECTION_NAME)
scope_option = st.selectbox(
    "Search within:",
    options=["All documents"] + available_files,
)
filename_filter = None if scope_option == "All documents" else scope_option

search_query = st.text_input("Enter your question:")
search_button = st.button("Generate Answer")

if search_button:
    if not search_query.strip():
        st.warning("Please enter a question.")
    elif model_option == config.MODEL_GEMINI and not config.GEMINI_API_KEY:
        st.error("GEMINI_API_KEY is missing. Please check your environment configuration.")
    elif model_option == config.MODEL_QWEN and not config.HF_API_KEY:
        st.error("Hugging Face API Key (HF_API_KEY) is missing. Please check your environment configuration.")
    else:
        status_text = st.empty()
        status_text.text("Retrieving relevant chunks...")
        try:
            import time
            from services.retrieval import retrieve_relevant_chunks
            from services.prompt_builder import build_prompt
            from services.llm_service import generate_response
            
            # Step 1: Retrieval
            results = retrieve_relevant_chunks(search_query, top_k=config.DEFAULT_TOP_K, filename_filter=filename_filter)
            
            if not results:
                status_text.empty()
                st.info("No matching chunks found in the database. Cannot generate response.")
            else:
                # Step 2: Prompt Construction
                full_prompt = build_prompt(search_query, results)
                
                # Step 3: LLM Generation (with timing)
                status_text.text(f"Generating response from {model_option}...")
                start_time = time.time()
                
                response_text = generate_response(full_prompt, model_option)
                
                generation_time = time.time() - start_time
                status_text.empty()
                
                # Success & Info Dashboard
                st.success("Response generated successfully!")
                
                st.markdown("#### Execution Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Selected Model:** `{model_option}`")
                    st.write(f"**Question:** '{search_query}'")
                with col2:
                    st.write(f"**Context Chunks:** {len(results)}")
                    st.write(f"**Qdrant Collection:** `{config.QDRANT_COLLECTION_NAME}`")
                with col3:
                    st.write(f"**Generation Time:** `{generation_time:.2f}s`")
                    st.write("**Response Status:** 🟢 Successful")
                
                # Render Final Answer
                st.markdown("### 🤖 Answer")
                st.info(response_text)
                
                # Collapsible Expander for the prompt
                with st.expander("📝 View Constructed Prompt"):
                    st.text_area("Final Prompt String", value=full_prompt, height=300, disabled=True)
                
                # Detailed retrieved chunks inside expanders
                st.markdown("#### Top Retrieved Chunks")
                for idx, match in enumerate(results):
                    with st.expander(f"📄 Match #{idx+1} | Score: {match['score']:.4f} | {match['filename']} (Page {match['page_number']})"):
                        st.markdown(f"**Chunk ID:** `{match['chunk_id']}`")
                        st.markdown(f"**Source File:** {match['filename']}")
                        st.markdown(f"**Page:** {match['page_number']} | **Chunk Index:** {match['chunk_index']}")
                        st.markdown(f"**Similarity Score:** `{match['score']:.4f}`")
                        st.markdown("**Chunk Preview (first 250 characters):**")
                        preview_text = match['text'][:250] + ("..." if len(match['text']) > 250 else "")
                        st.code(preview_text, language=None)
        except Exception as e:
            status_text.empty()
            st.error(f"Error during execution: {str(e)}")
st.markdown("---")
