# Product Requirements Document (PRD)

## RAG Chatbot

| Field              | Value                    |
| ------------------ | ------------------------ |
| **Version**        | 1.0                      |
| **Date**           | July 2026                |
| **Status**         | Active                   |
| **Application**    | Streamlit Web Interface  |

---

## 1. Project Overview

The RAG Chatbot is a web-based question-answering application that allows users to upload PDF documents and query their contents using a Retrieval-Augmented Generation (RAG) pipeline. The system extracts text from PDFs, chunks and embeds the content into a vector database, and retrieves semantically relevant passages at query time to ground an LLM's response. Users can toggle between two LLM backends — Google Gemini Flash and Qwen 2.5 — to generate answers.

---

## 2. Problem Statement

Organizations and individuals accumulate large volumes of PDF documents — policy manuals, handbooks, reports, and technical papers. Manually searching through these documents to find specific answers is time-consuming and error-prone. Keyword-based search fails to capture semantic intent, and traditional chatbots lack access to private, document-specific knowledge.

There is a need for a system that:

- Ingests multiple PDF documents and makes their content machine-searchable.
- Retrieves semantically relevant passages in response to natural language questions.
- Grounds LLM-generated answers in actual document content, reducing hallucination.
- Supports multiple LLM backends for flexibility in response generation.

---

## 3. Objectives

1. **Document Ingestion** — Enable users to upload one or more PDF files and have them automatically processed, chunked, embedded, and stored in a vector database.
2. **Semantic Retrieval** — Retrieve the most relevant document chunks for a given natural language query using cosine similarity search over embedding vectors.
3. **Grounded Generation** — Generate answers using an LLM with retrieved document context as the sole source of information, preventing fabrication.
4. **Multi-Model Support** — Allow users to choose between two LLM providers (Gemini Flash and Qwen 2.5) for response generation.
5. **Document-Scoped Search** — Support filtering queries to a specific uploaded document for targeted retrieval.

---

## 4. Scope

### In Scope

- PDF document upload and text extraction.
- Configurable word-level chunking with overlap.
- Embedding generation using Google Gemini.
- Vector storage and retrieval using Qdrant.
- Dual-LLM response generation (Gemini Flash and Qwen 2.5).
- Streamlit-based single-page web interface.
- Document scope filtering at query time.
- Processing dashboard with metrics and chunk previews.
- API key validation for selected LLM.

### Out of Scope

- Support for non-PDF file formats (DOCX, TXT, HTML, images).
- OCR or image-based text extraction.
- Conversation history or multi-turn chat.
- User authentication and authorization.
- Cross-page chunking or document-level chunking strategies.
- Hybrid search (keyword/BM25 combined with vector search).
- Re-ranking of retrieved results.
- Real-time streaming of LLM responses.
- Document deletion or management UI.
- Deployment infrastructure (Docker, Kubernetes, CI/CD).
- Rate limiting, quotas, or usage tracking.

---

## 5. Functional Requirements

### 5.1 Document Ingestion

| ID    | Requirement                                                                                       |
| ----- | ------------------------------------------------------------------------------------------------- |
| FR-01 | The system shall accept one or more PDF files via a file uploader widget.                         |
| FR-02 | The system shall extract text from each PDF page using `pypdf`.                                   |
| FR-03 | The system shall chunk extracted text into segments of up to 1000 characters with 200-character overlap using word-boundary splitting. |
| FR-04 | The system shall generate embedding vectors for each chunk using the `gemini-embedding-001` model. |
| FR-05 | The system shall process embedding requests in batches of 20 with exponential backoff on rate limit errors (HTTP 429). |
| FR-06 | The system shall store chunks and their embeddings in a Qdrant collection named `rag_documents`.   |
| FR-07 | The system shall assign deterministic UUIDs (UUID5) to each chunk based on its chunk ID, ensuring idempotent upserts. |
| FR-08 | The system shall save uploaded PDF files to the `uploads/` directory on disk.                       |
| FR-09 | The system shall display a processing dashboard with total pages, total chunks, and processing time after ingestion. |
| FR-10 | The system shall display chunk previews with filename, page number, and text excerpt after ingestion. |

### 5.2 Query and Retrieval

| ID    | Requirement                                                                                       |
| ----- | ------------------------------------------------------------------------------------------------- |
| FR-11 | The system shall accept a natural language question via a text input field.                        |
| FR-12 | The system shall embed the query using the same `gemini-embedding-001` model used for document chunks. |
| FR-13 | The system shall retrieve the top 3 most similar chunks from Qdrant using cosine similarity.       |
| FR-14 | The system shall support optional filtering of search results to a specific document by filename.   |
| FR-15 | The system shall construct a structured prompt containing retrieved chunk metadata (filename, page) and chunk text alongside the user's question. |
| FR-16 | The system shall send the constructed prompt to the selected LLM for answer generation.            |
| FR-17 | The system shall display the generated answer, the full constructed prompt (expandable), and each retrieved chunk with metadata (expandable). |
| FR-18 | The system shall display retrieval and generation timing in an execution summary.                  |

### 5.3 Multi-Model Support

| ID    | Requirement                                                                                       |
| ----- | ------------------------------------------------------------------------------------------------- |
| FR-19 | The system shall provide a sidebar dropdown to select between Gemini Flash and Qwen 2.5.          |
| FR-20 | The system shall route generation requests to the appropriate LLM backend based on the user's selection. |
| FR-21 | The system shall validate that the required API key is present for the selected model before generation. |
| FR-22 | Gemini Flash shall use the `google-genai` SDK with model ID `gemini-flash-latest`.                |
| FR-23 | Qwen 2.5 shall use the Hugging Face Inference API with model ID `Qwen/Qwen2.5-7B-Instruct` and a maximum of 500 tokens. |

### 5.4 Document Scope Filtering

| ID    | Requirement                                                                                       |
| ----- | ------------------------------------------------------------------------------------------------- |
| FR-24 | The system shall populate a "Search within:" dropdown with the list of unique filenames stored in the Qdrant collection. |
| FR-25 | The system shall allow users to select "All documents" or a specific filename to scope their query. |
| FR-26 | The system shall apply the filename filter as a Qdrant `FieldCondition` keyword match during retrieval. |

---

## 6. Non-Functional Requirements

### 6.1 Performance

| ID    | Requirement                                                                                       |
| ----- | ------------------------------------------------------------------------------------------------- |
| NFR-01 | The system shall process document ingestion end-to-end within reasonable time for PDFs up to 50 pages. |
| NFR-02 | Embedding generation shall use batching (batch size 20) to minimize API round trips.              |
| NFR-03 | Vector upserts shall use batching (batch size 100) with synchronous waits to ensure data consistency. |
| NFR-04 | The Qdrant client shall be cached across Streamlit reruns using `@st.cache_resource`.             |

### 6.2 Reliability

| ID    | Requirement                                                                                       |
| ----- | ------------------------------------------------------------------------------------------------- |
| NFR-05 | The system shall handle rate limit errors (HTTP 429) from the Gemini embedding API with exponential backoff (5s, 10s, 20s, 40s, 80s, up to 5 retries). |
| NFR-06 | The system shall skip empty chunks during embedding and continue processing remaining chunks.      |
| NFR-07 | The system shall display user-facing error messages via `st.error()` for failures at each pipeline stage. |
| NFR-08 | The system shall handle missing API keys gracefully by validating before making API calls.         |

### 6.3 Usability

| ID    | Requirement                                                                                       |
| ----- | ------------------------------------------------------------------------------------------------- |
| NFR-09 | The application shall be accessible via a web browser through Streamlit.                           |
| NFR-10 | The UI shall provide visual feedback during document processing via progress bars and status text. |
| NFR-11 | The UI shall display results in a structured layout with expandable sections for prompts and chunks. |
| NFR-12 | The application shall operate as a single-page interface with ingestion at the top and querying at the bottom. |

### 6.4 Configuration

| ID    | Requirement                                                                                       |
| ----- | ------------------------------------------------------------------------------------------------- |
| NFR-13 | API keys and connection settings shall be loaded from a `.env` file via `python-dotenv`.          |
| NFR-14 | All pipeline parameters (chunk size, overlap, top_k, model IDs) shall be defined as constants in `config.py`. |
| NFR-15 | The Qdrant collection name shall default to `rag_documents`.                                      |

---

## 7. User Flow

### 7.1 Ingestion Flow

```
1. User opens the Streamlit application.
2. User uploads one or more PDF files via the file uploader.
3. User clicks "Process PDFs".
4. System saves files to uploads/ directory.
5. System extracts text from each PDF page.
6. System chunks text into overlapping segments.
7. System generates embedding vectors for all chunks.
8. System creates the Qdrant collection (if not exists) with a payload index on filename.
9. System upserts chunks and vectors into Qdrant.
10. System displays a processing dashboard with metrics and chunk previews.
```

### 7.2 Query Flow

```
1.  User optionally selects a document scope from the "Search within:" dropdown.
2.  User enters a question in the text input field.
3.  User clicks "Generate Answer".
4.  System validates the API key for the selected LLM model.
5.  System embeds the query using gemini-embedding-001.
6.  System searches Qdrant for the top 3 similar chunks (optionally filtered by filename).
7.  System constructs a RAG prompt with retrieved chunk context and metadata.
8.  System sends the prompt to the selected LLM (Gemini Flash or Qwen 2.5).
9.  System displays the answer, execution summary, constructed prompt, and retrieved chunks.
```

---

## 8. Features

### 8.1 Core Features

| Feature                    | Description                                                                                   |
| -------------------------- | --------------------------------------------------------------------------------------------- |
| PDF Upload                 | Multi-file PDF upload with automatic text extraction using pypdf.                             |
| Semantic Chunking          | Custom word-boundary chunking with configurable chunk size (1000 chars) and overlap (200 chars). |
| Embedding Generation       | Gemini embedding-001 model with batch processing (batch size 20) and rate-limit retry logic.  |
| Vector Storage             | Qdrant cloud integration with cosine similarity, deterministic UUIDs, and payload indexing.    |
| Semantic Search            | Query embedding and top-k cosine similarity search with optional document-scoped filtering.   |
| RAG Response Generation    | LLM-generated answers grounded in retrieved document context with hallucination prevention.    |
| Processing Dashboard       | Post-ingestion metrics display (pages, chunks, time) and chunk preview with metadata.         |
| Execution Summary          | Per-query timing breakdown for retrieval and generation steps.                                |
| Prompt Transparency        | Expandable display of the full constructed prompt sent to the LLM.                             |
| Chunk Inspection           | Expandable display of each retrieved chunk with filename, page number, and text preview.      |

### 8.2 Multi-Model Toggle

| Feature                    | Description                                                                                   |
| -------------------------- | --------------------------------------------------------------------------------------------- |
| Gemini Flash               | Google's Gemini Flash model via the `google-genai` SDK.                                       |
| Qwen 2.5                   | Hugging Face's Qwen 2.5 7B Instruct model via the Inference API with 500-token limit.        |
| Model Selector             | Sidebar dropdown for switching between models at query time.                                  |
| API Key Validation         | Pre-flight check that the required API key exists before calling the selected LLM.            |

### 8.3 Document Scope Filtering

| Feature                    | Description                                                                                   |
| -------------------------- | --------------------------------------------------------------------------------------------- |
| Scope Selector             | Dropdown populated with unique filenames from the Qdrant collection.                          |
| Filename-Based Filtering   | Qdrant keyword match filter applied during retrieval to narrow results to a specific document. |

---

## 9. Tech Stack

| Layer                  | Technology                        | Version   | Purpose                                    |
| ---------------------- | --------------------------------- | --------- | ------------------------------------------ |
| **Frontend**           | Streamlit                         | 1.59.2    | Web UI framework                           |
| **Embeddings**         | Google Gemini Embedding API       | —         | Vector embedding generation                |
| **Embedding Model**    | gemini-embedding-001              | —         | Text-to-vector model                       |
| **Vector Database**    | Qdrant (Cloud)                    | —         | Vector storage, indexing, and retrieval    |
| **Vector DB Client**   | qdrant-client                     | 1.18.0    | Python client for Qdrant                   |
| **LLM (Primary)**      | Gemini Flash                      | —         | Response generation via Google GenAI       |
| **LLM (Secondary)**    | Qwen 2.5 7B Instruct             | —         | Response generation via Hugging Face API   |
| **LLM SDK (Gemini)**   | google-genai                      | 2.12.0    | Google Gemini API client                   |
| **LLM SDK (Qwen)**     | huggingface_hub                   | 1.23.0    | Hugging Face Inference Client              |
| **PDF Processing**     | pypdf                             | 6.14.2    | PDF text extraction                        |
| **Numerical**          | numpy                             | 2.5.1     | Embedding vector operations                |
| **Configuration**      | python-dotenv                     | 1.2.2     | Environment variable loading               |
| **Validation**         | pydantic                          | 2.13.4    | Data validation (used by Qdrant/GenAI)     |
| **Runtime**            | Python                            | 3.14      | Application runtime                        |

---

## 10. Success Criteria

| ID    | Criterion                                                                                      | Measurement                                    |
| ----- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| SC-01 | Users can upload one or more PDFs and see them fully ingested into the vector database.        | All chunks appear in Qdrant after processing.  |
| SC-02 | Users can ask natural language questions and receive relevant, context-grounded answers.        | Retrieved chunks are semantically related to the query. |
| SC-03 | The system prevents hallucination by restricting answers to provided context.                   | LLM responds with the designated fallback when context is insufficient. |
| SC-04 | Users can switch between Gemini Flash and Qwen 2.5 without code changes.                       | Both models generate responses for the same query. |
| SC-05 | Users can narrow queries to a specific document and receive document-specific answers.          | Retrieved chunks belong to the selected document. |
| SC-06 | The ingestion pipeline handles rate limits gracefully without data loss.                        | Chunks skipped due to rate limits are reported to the user. |
| SC-07 | Re-uploading the same document does not create duplicate entries in the vector database.         | Chunk count remains stable across re-uploads.  |
| SC-08 | API key validation prevents cryptic errors when a required key is missing.                      | A clear error message is shown before API calls. |

---

## 11. Future Scope

| Area                         | Description                                                                                   |
| ---------------------------- | --------------------------------------------------------------------------------------------- |
| **Multi-Format Support**     | Extend document ingestion to support DOCX, TXT, HTML, and Markdown files.                     |
| **OCR Integration**          | Add optical character recognition for scanned PDFs and image-based documents.                 |
| **Conversation History**     | Implement multi-turn chat with session-persisted message history and context carryover.        |
| **Cross-Page Chunking**      | Implement document-level chunking that respects semantic boundaries across page breaks.        |
| **Hybrid Search**            | Combine vector similarity with keyword-based BM25 search for improved retrieval accuracy.      |
| **Re-Ranking**               | Add a cross-encoder re-ranking step after initial retrieval to improve result relevance.       |
| **Streaming Responses**      | Stream LLM responses token-by-token for improved perceived latency.                           |
| **Configurable Parameters**  | Expose chunk size, overlap, top_k, and other parameters through the UI.                       |
| **Document Management**      | Add UI for listing, viewing, and deleting previously uploaded documents.                      |
| **User Authentication**      | Add login system with per-user document collections and query history.                        |
- **Rate Limiting & Quotas**    | Implement per-user usage tracking and rate limiting.                                         |
| **Testing Suite**             | Add unit and integration tests for all pipeline components.                                  |
| **Logging & Monitoring**      | Implement structured logging and performance monitoring with alerting.                        |
| **Docker Deployment**         | Containerize the application for consistent deployment across environments.                   |
| **Additional LLM Backends**   | Integrate OpenAI, Anthropic, or self-hosted models as additional generation options.          |
| **Conversation Memory**       | Maintain chat context across multiple queries within a session for follow-up questions.        |
| **Prompt Templates**          | Allow users to select or customize prompt templates for different use cases.                  |
| **Multi-Collection Support** | Organize documents into separate collections for different projects or teams.                 |

---

*End of Document*
