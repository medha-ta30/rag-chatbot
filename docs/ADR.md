# Architecture Decision Records (ADR)

## RAG Chatbot

| Field           | Value         |
| --------------- | ------------- |
| **Version**     | 1.0           |
| **Date**        | July 2026     |
| **Status**      | Active        |

---

## ADR-01: Streamlit as the UI Framework

### Decision

Use Streamlit as the sole frontend and application framework for the RAG chatbot.

### Context

The application requires a web-based interface for PDF upload, query input, and result display. The target users are internal stakeholders who need a functional interface without the overhead of a full frontend stack. Development velocity and simplicity are priorities over customization and performance at scale.

### Alternatives Considered

| Alternative          | Consideration                                                                 |
| -------------------- | ----------------------------------------------------------------------------- |
| **Flask + React**    | Full separation of concerns; requires separate frontend build, API design, CORS handling, and deployment of two services. Significantly higher complexity for a prototype-stage application. |
| **FastAPI + React**  | Similar to Flask + React but with async support and automatic API docs. Still introduces two-service deployment overhead and frontend build tooling. |
| **Gradio**           | Simpler than Streamlit for single-function demos; however, limited layout control makes it unsuitable for a multi-section page with sidebars, expanders, progress indicators, and dashboard metrics. |
| **Streamlit**        | Single-file Python application with no frontend build step, built-in widgets (file uploader, selectbox, text input, progress bars, expanders), and rapid iteration. Deploys with a single command. |

### Rationale

Streamlit eliminates the need for separate frontend development, API layer design, and multi-service deployment. The application's requirements — file upload, form inputs, text display, expandable sections, and sidebar controls — map directly to Streamlit's built-in widget set. The entire application runs as a single Python process, which is appropriate for a prototype-to-MVP-stage tool. The trade-off in UI customization and scalability is acceptable given the current scope.

### Consequences

**Positive:**
- Single-file application with no build step.
- Rapid prototyping and iteration.
- Built-in widgets cover all required UI patterns.
- Single-process deployment via `streamlit run`.

**Negative:**
- Limited control over UI layout and styling compared to custom React.
- Every user interaction triggers a full page rerun, which can be inefficient.
- Not designed for high-concurrency multi-user deployment.
- Session state management is implicit and can be unintuitive.

---

## ADR-02: Gemini Embedding Model

### Decision

Use Google's `gemini-embedding-001` model via the `google-genai` SDK for all embedding generation — both document chunks and user queries.

### Context

The RAG pipeline requires converting text into dense vector representations for semantic similarity search. The embedding model must produce consistent vectors for both ingestion and query time, support batch processing, and be accessible via a managed API to avoid self-hosting overhead.

### Alternatives Considered

| Alternative                    | Consideration                                                                 |
| ------------------------------ | ----------------------------------------------------------------------------- |
| **OpenAI text-embedding-3-small** | High-quality embeddings with broad adoption; introduces a dependency on OpenAI's ecosystem and separate API key management. |
| **Hugging Face sentence-transformers** | Free and self-hosted; requires model download, GPU allocation, and inference server management. Adds infrastructure complexity. |
| **Cohere embed-v3**            | Strong multilingual support; adds a third API provider and API key, increasing configuration surface. |
| **Gemini embedding-001**       | Already required for the Gemini Flash LLM integration (shared API key), batch-capable, and accessed via the same `google-genai` SDK already in use. |

### Rationale

Since the project already integrates Gemini Flash as an LLM backend and the `google-genai` SDK is a dependency, using Gemini for embeddings avoids introducing an additional API provider. Both the embedding model and the LLM share the same `GEMINI_API_KEY`, reducing configuration complexity. The model supports batch embedding (batch size 20 in implementation), which minimizes API round trips during ingestion. The same model is used for query embedding, ensuring vector space consistency.

### Consequences

**Positive:**
- Single SDK (`google-genai`) handles both embeddings and LLM generation.
- Shared API key with Gemini Flash LLM reduces configuration overhead.
- Batch processing support minimizes API calls.
- Consistent vector space between ingestion and retrieval.

**Negative:**
- Gemini API key is always required, even when Qwen is selected as the LLM.
- No option to use a different embedding model without architectural changes.
- Rate limiting on the Gemini API can bottleneck ingestion (mitigated by exponential backoff).
- Embedding model quality is not independently tunable from the LLM provider choice.

---

## ADR-03: Dual-LLM Support (Gemini Flash + Qwen 2.5)

### Decision

Support two LLM backends — Google Gemini Flash and Qwen 2.5 7B Instruct — with runtime selection via a sidebar dropdown. Embeddings remain tied to Gemini regardless of the selected LLM.

### Context

Different LLMs have different strengths, latency profiles, and cost structures. Users may have preferences or organizational requirements around specific model providers. Supporting multiple backends provides flexibility without requiring separate application instances.

### Alternatives Considered

| Alternative                  | Consideration                                                                 |
| ---------------------------- | ----------------------------------------------------------------------------- |
| **Single LLM (Gemini only)** | Simpler architecture; removes routing logic and dual API key management. Limits user choice and creates a single-vendor dependency. |
| **Single LLM (Qwen only)**  | Free and open-source; quality may be lower than Gemini Flash for complex reasoning. Requires Hugging Face API key but not Gemini. |
| **Plugin-based LLM system** | Highly extensible; adds significant complexity for dynamically loading models, managing connection pools, and handling diverse API formats. Over-engineered for two models. |
| **Dual LLM with runtime toggle** | Supports two specific models with a simple routing function; balances flexibility with implementation simplicity. |

### Rationale

Two models were selected to provide a concrete choice without over-engineering the routing layer. Gemini Flash is a high-quality, low-latency proprietary model. Qwen 2.5 is a capable open-source alternative accessible via Hugging Face's Inference API. The implementation uses a simple conditional branch in `llm_service.py` to route requests, avoiding the complexity of a plugin system. The API key validation in `app.py` ensures the correct key exists before making calls.

### Consequences

**Positive:**
- Users can compare responses from different model providers.
- Organizational flexibility — supports both Google and Hugging Face ecosystems.
- Simple routing logic (single conditional branch).
- Pre-flight API key validation prevents runtime errors.

**Negative:**
- Two API keys must be managed (`GEMINI_API_KEY` and `HF_API_KEY`).
- Qwen responses are capped at 500 tokens (hardcoded), creating an asymmetric capability.
- Embeddings remain Gemini-only — Qwen cannot be used as a standalone backend.
- Model-specific error handling must be maintained for both providers.

---

## ADR-04: Qdrant Cloud as Vector Database

### Decision

Use Qdrant Cloud as the vector database for storing, indexing, and querying embedding vectors. The collection uses cosine similarity with a KEYWORD payload index on the `filename` field.

### Context

The application needs a vector database that supports high-dimensional vector storage, cosine similarity search, metadata filtering, and payload indexing. Managed infrastructure is preferred to avoid self-hosting and operational overhead.

### Alternatives Considered

| Alternative              | Consideration                                                                 |
| ------------------------ | ----------------------------------------------------------------------------- |
| **Pinecone**             | Fully managed, low operational overhead; introduces vendor lock-in and pricing complexity. API-only access limits local development flexibility. |
| **Weaviate**             | Feature-rich with hybrid search and GraphQL API; heavier operational footprint for a simple use case. Requires more configuration for minimal benefit. |
| **Chroma**               | Lightweight and easy to embed; lacks production-grade persistence, scaling, and cloud hosting options. Suitable for local development but not for deployed applications. |
| **FAISS (local)**        | Extremely fast in-memory search; no persistence, no metadata filtering, no server architecture. Requires manual index management and serialization. |
| **Qdrant Cloud**         | REST and gRPC APIs, native payload filtering with KEYWORD indexes, cosine distance metric, managed cloud hosting with free tier, and simple Python client. |

### Rationale

Qdrant provides the right balance of features and simplicity for this application. The `qdrant-client` Python library offers a clean API for collection creation, upsert, and query operations. Native support for payload indexing (KEYWORD type on `filename`) enables efficient metadata filtering without secondary databases. Cosine similarity is directly supported as a distance metric, aligning with the embedding model's output. The cloud deployment option removes the need for self-hosting while maintaining a local development fallback via `localhost:6333`.

### Consequences

**Positive:**
- Native cosine similarity search without manual normalization.
- KEYWORD payload index on `filename` enables efficient document-scoped queries.
- Managed cloud hosting with a local development fallback.
- Simple Python client with synchronous and async support.
- Deterministic UUID upserts prevent duplicate entries.

**Negative:**
- Cloud dependency introduces a network hop for all vector operations.
- Free tier limits may require monitoring as document volume grows.
- `get_available_filenames` uses a scroll with `limit=1000`, which may miss filenames in very large collections.
- No built-in re-ranking or hybrid search capabilities.

---

## ADR-05: Custom Word-Boundary Chunking

### Decision

Implement a custom word-level chunking algorithm with configurable chunk size (1000 characters) and overlap (200 characters), operating per-page without cross-page spanning.

### Context

The ingestion pipeline must split extracted PDF text into overlapping segments suitable for embedding and retrieval. Chunk quality directly impacts retrieval accuracy and LLM response quality. Chunks must be small enough to fit within embedding model context windows but large enough to retain semantic coherence.

### Alternatives Considered

| Alternative                       | Consideration                                                                 |
| --------------------------------- | ----------------------------------------------------------------------------- |
| **LangChain RecursiveCharacterTextSplitter** | Well-tested, widely used; adds a heavy dependency (LangChain ecosystem) for a single function. Recursive splitting adds complexity without proportional benefit for PDF content. |
| **Fixed-size character splitting** | Simplest implementation; splits mid-word, breaking semantic coherence and producing unnatural text boundaries. |
| **Sentence-level splitting (spaCy/NLTK)** | Produces semantically clean chunks; requires additional NLP dependencies (model downloads, tokenization overhead) and adds latency to ingestion. |
| **Custom word-boundary chunking** | Avoids mid-word splits, retains natural text flow, requires no external NLP dependencies, and provides configurable overlap for context continuity. |

### Rationale

A custom word-boundary chunker was chosen to avoid the dependency overhead of LangChain or NLP libraries while still producing clean, semantically meaningful chunks. The algorithm splits text at whitespace boundaries, ensuring no word is broken across chunks. The 200-character overlap provides context continuity between adjacent chunks, improving retrieval of spans that sit near chunk boundaries. The per-page approach aligns with the PDF extraction model (page-by-page via pypdf) and produces deterministic chunk IDs that encode the source page.

### Consequences

**Positive:**
- No external NLP or chunking library dependencies.
- Word-boundary splitting preserves natural text flow.
- Overlap mechanism improves context continuity across chunks.
- Deterministic chunk IDs (`{filename}_page_{N}_chunk_{M}`) enable idempotent upserts.
- Simple, readable, and maintainable implementation.

**Negative:**
- Per-page chunking breaks semantic coherence across page boundaries — a sentence spanning two pages will be split into separate chunk sequences.
- No semantic-aware splitting (e.g., paragraph or section boundaries are not respected).
- Character-based target size with word-based execution means actual chunk sizes may vary.
- Overlap is approximate (character target, word execution), not precise.

---

## ADR-06: Metadata Filtering via Qdrant Payload Index

### Decision

Store document metadata (`filename`, `page_number`, `chunk_index`, `text`) as Qdrant payload on each vector point. Create a KEYWORD payload index on the `filename` field to enable efficient document-scoped filtering at query time.

### Context

Users need the ability to narrow queries to specific documents when the collection contains multiple uploaded PDFs. Metadata must be stored alongside vectors to support filtering without external lookups. The filtering mechanism must be efficient enough to avoid full-collection scans.

### Alternatives Considered

| Alternative                  | Consideration                                                                 |
| ---------------------------- | ----------------------------------------------------------------------------- |
| **Separate metadata store** | Store metadata in a relational database (SQLite, PostgreSQL) and join with Qdrant results by ID. Adds complexity and a second data store for a simple use case. |
| **In-memory filtering** | Retrieve top-k results from Qdrant, then filter by filename in Python. Wastes retrieval budget — if most results are from the wrong document, the effective top-k is reduced. |
| **Qdrant payload filtering** | Filter at the database level before scoring, ensuring the full top-k budget is spent on relevant chunks from the target document. |
| **Qdrant payload filtering + KEYWORD index** | Same as above but with an index on the `filename` field to avoid linear scan during filter evaluation. |

### Rationale

Qdrant's native payload filtering applies the filter during vector search, not after, ensuring the top-k results are always from the target document when a scope is selected. The KEYWORD index on `filename` accelerates filter evaluation by avoiding a full payload scan. Storing metadata as payload (rather than a separate store) keeps the architecture simple and avoids cross-system consistency issues. The metadata stored (`filename`, `page_number`, `chunk_index`, `text`) is sufficient for prompt construction and chunk attribution in the UI.

### Consequences

**Positive:**
- Filter applied at search time, not post-retrieval — full top-k budget is used efficiently.
- KEYWORD index on `filename` ensures fast filter evaluation.
- Metadata travels with the vector — no external joins or lookups required.
- Sufficient metadata for prompt construction and UI display.

**Negative:**
- KEYWORD index requires exact string matching — no fuzzy or partial filename matching.
- `get_available_filenames` uses a scroll with `limit=1000`, which may not capture all filenames in large collections.
- No support for multi-field compound filters (e.g., filename + page range).
- Payload is immutable after upsert — metadata corrections require re-ingestion.

---

## ADR-07: Cosine Similarity for Vector Search

### Decision

Use cosine similarity as the distance metric for all vector search operations in the Qdrant collection.

### Context

The retrieval pipeline must rank document chunks by semantic similarity to the user's query. The distance metric determines how similarity is calculated between the query embedding and stored chunk embeddings.

### Alternatives Considered

| Alternative                | Consideration                                                                 |
| -------------------------- | ----------------------------------------------------------------------------- |
| **Euclidean (L2) distance** | Measures absolute vector distance; sensitive to magnitude differences. Works well for normalized vectors but less intuitive for semantic similarity. |
| **Dot product**            | Measures orientation and magnitude combined; if vectors are not normalized, magnitude bias can distort similarity rankings. |
| **Cosine similarity**      | Measures angle between vectors, ignoring magnitude. Naturally suited for text embeddings where direction encodes meaning and magnitude varies by text length. |

### Rationale

Cosine similarity is the standard metric for text embedding comparison. The `gemini-embedding-001` model produces normalized or near-normalized vectors where angular distance best captures semantic similarity. Cosine is invariant to vector magnitude, making it robust to variations in chunk length. Qdrant supports cosine distance natively as `Distance.COSINE`, requiring no custom distance computation.

### Consequences

**Positive:**
- Semantically meaningful — angular distance correlates with meaning similarity.
- Magnitude-invariant — robust to chunk length variations.
- Native Qdrant support — no custom scoring logic required.
- Industry standard for text embedding retrieval.

**Negative:**
- Does not account for lexical overlap — semantically similar but terminologically different passages score equally with unrelated but vectorially close passages.
- No hybrid scoring — purely vector-based, without BM25 or keyword relevance signals.
- Sensitive to embedding model quality — poor embeddings produce poor similarity rankings regardless of the metric.

---

## ADR-08: Single-Page Stateless Application Design

### Decision

Design the application as a single-page Streamlit interface with two sequential sections (ingestion and query) and no persistent session state, conversation history, or chat loop.

### Context

The application serves as a functional RAG prototype. The primary use case is document upload followed by independent question-answering. There is no requirement for multi-turn conversation, user accounts, or session-persisted query results.

### Alternatives Considered

| Alternative                    | Consideration                                                                 |
| ------------------------------ | ----------------------------------------------------------------------------- |
| **Multi-page Streamlit app**   | Separate ingestion and query pages with navigation; adds routing complexity and requires cross-page state management for minimal UX benefit. |
| **Chat-based UI**              | `st.chat_message` / `st.chat_input` with conversation history; requires `st.session_state` management, token budget tracking, and context windowing. Adds significant complexity for a feature not yet required. |
| **Stateless single-page design** | Linear workflow on one page, each query independent, no session persistence. Simplest implementation, easiest to debug, no state-related bugs. |

### Rationale

A stateless single-page design matches the current requirements and minimizes implementation complexity. Each query is processed independently, which simplifies debugging and eliminates entire classes of state management bugs (stale state, race conditions, session expiry). The only cross-rerun state is the Qdrant client, cached via `@st.cache_resource` for connection reuse. Processing results are stored in Qdrant, not in session state, so they persist across reruns without explicit management.

### Consequences

**Positive:**
- No `st.session_state` usage eliminates state-related bugs.
- Each query is independently reproducible.
- Simple debugging — no hidden state influences behavior.
- Processing results persist in Qdrant, not in volatile session memory.

**Negative:**
- No conversation context — each query is independent, no follow-up questions.
- Processing dashboard disappears after any UI interaction (no session persistence).
- Users must re-select document scope for every query.
- Not suitable for multi-turn conversational use cases.

---

## ADR-09: Deterministic UUIDs for Idempotent Upserts

### Decision

Generate deterministic UUIDs (UUID5 with DNS namespace) from chunk ID strings for Qdrant point identifiers, ensuring that re-processing the same document overwrites existing vectors rather than creating duplicates.

### Context

Users may re-upload the same document after corrections or updates. The ingestion pipeline must handle this gracefully without creating duplicate vectors that would pollute search results and waste storage.

### Alternatives Considered

| Alternative                  | Consideration                                                                 |
| ---------------------------- | ----------------------------------------------------------------------------- |
| **Random UUIDs (UUID4)**     | Simple; re-uploading creates duplicates, requiring explicit deduplication logic or collection truncation before re-ingestion. |
| **Filename-based point IDs** | One point per filename; loses per-chunk granularity and prevents storing multiple chunks from the same document. |
| **Chunk ID as string key**   | Use the chunk ID string directly as the point ID; Qdrant supports string IDs but UUIDs are more standard for vector databases. |
| **Deterministic UUIDs (UUID5)** | Derived from chunk ID; same document always produces the same point IDs, enabling idempotent upserts without deduplication logic. |

### Rationale

UUID5 generation from chunk IDs (`{filename}_page_{N}_chunk_{M}`) ensures that the same document always produces the same set of point identifiers. Qdrant's upsert operation overwrites existing points with matching IDs, so re-ingestion naturally replaces old vectors with new ones. This eliminates the need for a separate deduplication step or collection-level truncation. The chunk ID format is human-readable and deterministic, making it easy to inspect and debug.

### Consequences

**Positive:**
- Re-uploading the same document is idempotent — no duplicates created.
- No separate deduplication logic required.
- Human-readable chunk IDs aid debugging.
- Deterministic behavior simplifies testing.

**Negative:**
- No user confirmation or diff display when overwriting existing chunks.
- Changing the chunking parameters (size, overlap) produces different chunk IDs, creating new points without removing old ones.
- Deleting a single document's chunks requires iterating by filename filter, not by point ID pattern.

---

## ADR-10: Centralized Configuration Management

### Decision

Centralize all configuration — environment variables, constants, model identifiers, and directory paths — in a single `config.py` module loaded at application startup via `python-dotenv`.

### Context

The application requires multiple configuration values: API keys, database connection settings, chunking parameters, model identifiers, and directory paths. These must be accessible across all service modules without circular imports or scattered `os.getenv()` calls.

### Alternatives Considered

| Alternative                    | Consideration                                                                 |
| ------------------------------ | ----------------------------------------------------------------------------- |
| **Scattered `os.getenv()` calls** | Each module reads its own config; no central source of truth, duplicated default values, difficult to audit all settings. |
| **YAML/TOML config file**     | Human-readable, supports nested configuration; adds a parsing dependency and file I/O, and does not natively support environment variable interpolation. |
| **Pydantic Settings**         | Type-validated, auto-documented configuration; adds a dependency and complexity for a flat configuration surface that does not require validation. |
| **Centralized `config.py`**   | Single import gives access to all settings; flat module with no class instantiation; `python-dotenv` handles `.env` loading with fallback defaults. |

### Rationale

A flat `config.py` module is the simplest approach that meets all requirements. Every service module imports `config` and accesses settings as module-level attributes. The `python-dotenv` integration ensures `.env` values are loaded before any `os.getenv()` calls. Hardcoded constants (chunk size, overlap, model IDs) live alongside environment-derived values, providing a single source of truth for all configuration. No class instantiation or validation framework is needed for the current configuration surface.

### Consequences

**Positive:**
- Single import for all configuration values.
- `.env` integration with fallback defaults.
- Flat, readable module with no framework overhead.
- Easy to audit all settings in one file.

**Negative:**
- No runtime validation — invalid values (e.g., non-integer chunk size) fail at point of use, not at startup.
- Unused constants (`LOG_DIR`, `DOCS_DIR`, `TESTS_DIR`) remain in the module without enforcement.
- All modules see all configuration — no principle of least privilege for secrets.
- No environment-based switching (dev/staging/prod) without modifying the `.env` file.

---

*End of Document*
