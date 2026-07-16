import os
import streamlit as st
import config
from services.pdf_loader import extract_text_from_pdf
from services.chunker import chunk_document_pages

# 1. Initialize directory structure
def initialize_directories():
    for directory in [config.UPLOAD_DIR, config.LOG_DIR, config.DOCS_DIR, config.TESTS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

initialize_directories()

# 2. Configure Streamlit Page Layout
st.set_page_config(
    page_title="Week 9 RAG Chatbot",
    layout="centered",
)

# 3. Main Page Header
st.title("Week 9 RAG Chatbot 🤖")
st.subheader("Current Phase: Phase 3 - Chunking")

# 4. Sidebar Configuration
st.sidebar.title("Configuration")

# Model Selection
model_option = st.sidebar.selectbox(
    "Select LLM Model",
    options=["Gemini 2.5 Flash", "Mistral-7B-Instruct"],
)

# Environment Status
st.sidebar.subheader("Environment Status")

gemini_key_status = "🟢 Configured" if config.GEMINI_API_KEY else "🔴 Missing"
st.sidebar.write(f"Gemini API Key: {gemini_key_status}")

hf_key_status = "🟢 Configured" if config.HF_API_KEY else "🔴 Missing"
st.sidebar.write(f"Hugging Face API Key: {hf_key_status}")

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
            try:
                chunks = chunk_document_pages(all_extracted_pages)
                
                # Display processing summary
                st.success(
                    f"Successfully processed {successful_files_count} PDF(s) "
                    f"with a total of {total_pages_processed} page(s), "
                    f"generating {len(chunks)} chunk(s)."
                )
                
                # Preview first 3 chunks
                st.markdown("### Chunking Verification Preview")
                preview_count = min(3, len(chunks))
                
                with st.expander(f"🔍 Preview Chunks (First {preview_count} chunks)"):
                    for i in range(preview_count):
                        chunk = chunks[i]
                        st.markdown(f"**Chunk ID**: `{chunk['chunk_id']}`")
                        st.write(f"- **Source File**: {chunk['filename']}")
                        st.write(f"- **Page Number**: {chunk['page_number']}")
                        st.write(f"- **Chunk Index**: {chunk['chunk_index']}")
                        st.code(chunk['text'], language=None)
                        st.markdown("---")
            except Exception as e:
                st.error(f"Failed during chunking: {str(e)}")


