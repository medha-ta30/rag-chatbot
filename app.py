import os
import streamlit as st
import config

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
st.subheader("Current Phase: Phase 1 - Project Setup")

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

# 5. Main Page Content
st.success("Project setup completed. Waiting for Phase 2.")
