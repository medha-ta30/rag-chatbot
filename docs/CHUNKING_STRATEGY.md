# Chunking Strategy

## RAG Chatbot

| Field           | Value              |
| --------------- | ------------------ |
| **Module**      | `services/chunker.py` |
| **Parameters**  | `config.py` lines 24-25 |
| **Algorithm**   | Word-boundary sliding window with character-based overlap |

---

## 1. Overview

The chunking strategy converts raw PDF page text into fixed-size, overlapping text segments suitable for embedding and retrieval. The implementation uses a custom word-level sliding window algorithm that respects word boundaries and provides configurable overlap between adjacent chunks.

---

## 2. Configuration

| Parameter                | Value   | Source                      |
| ------------------------ | ------- | --------------------------- |
| `DEFAULT_CHUNK_SIZE`     | 1000    | `config.py` line 24         |
| `DEFAULT_CHUNK_OVERLAP`  | 200     | `config.py` line 25         |

These values are passed as default arguments to `chunk_document_pages()` and can be overridden programmatically, though the Streamlit UI does not expose them.

### Validation Constraints

```python
chunk_size > 0
0 <= chunk_overlap < chunk_size
```

The function raises `ValueError` if these constraints are violated (`chunker.py` lines 29-32).

---

## 3. Algorithm

### 3.1 Entry Point

```python
chunk_document_pages(pages, chunk_size=1000, chunk_overlap=200)
```

**Input:** A list of page dictionaries from `pdf_loader.py`:

```python
{"filename": "policy.pdf", "page_number": 3, "text": "Full page text..."}
```

**Output:** A list of chunk dictionaries:

```python
{"chunk_id": "policy.pdf_page_3_chunk_0", "filename": "policy.pdf", "page_number": 3, "chunk_index": 0, "text": "Chunk text..."}
```

### 3.2 Per-Page Processing

The algorithm processes each page independently (`chunker.py` line 36). Pages with no extractable text are skipped entirely (`chunker.py` lines 42-43). Text is split into words by whitespace using Python's `str.split()` (`chunker.py` line 45), which handles multiple spaces, tabs, and newlines uniformly.

### 3.3 Word Packing (Step 1)

Starting from `start_idx`, the algorithm iterates through words and accumulates them into `current_words` until adding the next word would exceed `chunk_size` characters (`chunker.py` lines 58-70).

The character length calculation accounts for space padding between words:

```python
space_padding = 1 if len(current_words) > 0 else 0
next_len = curr_char_len + space_padding + len(word)
```

A chunk is only created when at least one word has been accumulated (`chunker.py` line 65: `len(current_words) > 0`), preventing empty chunks.

The resulting chunk text is reconstructed by joining the accumulated words with single spaces (`chunker.py` line 72):

```python
chunk_text = " ".join(current_words)
```

This means the output text may differ from the original whitespace formatting (e.g., multiple spaces are collapsed to single spaces).

### 3.4 Overlap Calculation (Step 2)

After creating a chunk, the algorithm determines how many words from the end of the current chunk should be included in the next chunk to satisfy the overlap requirement (`chunker.py` lines 89-98).

The overlap is computed by iterating backward through `current_words` and accumulating character lengths (including space padding) until the accumulated length meets or exceeds `chunk_overlap`:

```python
backtrack_chars = 0
backtrack_words_count = 0

for word in reversed(current_words):
    space_padding = 1 if backtrack_words_count > 0 else 0
    backtrack_chars += len(word) + space_padding
    backtrack_words_count += 1
    if backtrack_chars >= chunk_overlap:
        break
```

The next chunk's `start_idx` is then set to:

```python
start_idx = i - backtrack_words_count
```

This means the next chunk begins `backtrack_words_count` words before the end of the current chunk, creating the overlap region.

### 3.5 Infinite Loop Prevention

A safety check ensures the algorithm always progresses forward, even if the overlap is larger than the chunk itself (`chunker.py` lines 100-105):

```python
if backtrack_words_count >= len(current_words):
    backtrack_words_count = len(current_words) - 1

if backtrack_words_count < 0:
    backtrack_words_count = 0
```

This guarantees that at least one word is consumed per iteration, preventing the algorithm from re-processing the same words indefinitely.

### 3.6 Chunk ID Generation

Each chunk receives a deterministic, human-readable ID (`chunker.py` line 73):

```python
chunk_id = f"{filename}_page_{page_number}_chunk_{chunk_index}"
```

Example: `Access Control Policy.pdf_page_3_chunk_0`

The `chunk_index` is per-page, resetting to 0 for each new page.

---

## 4. Worked Example

Consider a page with the following text (120 characters):

> "The company provides health insurance benefits to all full-time employees who have completed their probationary period. Benefits include medical, dental, and vision coverage."

**Parameters:** `chunk_size=100`, `chunk_overlap=30`

**Step 1 — First chunk:**

Words are packed until character length exceeds 100:

```
"The company provides health insurance benefits to all full-time employees who" = 83 chars
Adding "have" would be 88 chars — still under 100.
Adding "completed" would be 98 chars — still under 100.
Adding "their" would be 104 chars — exceeds 100. Stop.
```

**Chunk 0:** `"The company provides health insurance benefits to all full-time employees who have completed their"` (98 chars)

**Step 2 — Overlap calculation:**

Backtracking from the end to find 30 characters of overlap:

```
"their" = 5 chars (backtrack: 1 word, 5 chars)
"completed" = 10 chars (backtrack: 2 words, 16 chars including space)
"have" = 5 chars (backtrack: 3 words, 22 chars including space)
"who" = 4 chars (backtrack: 4 words, 27 chars including space)
"employees" = 10 chars (backtrack: 5 words, 38 chars including space) — exceeds 30. Stop.
```

Backtrack count: 4 words (`who have completed their` = 27 chars)

**Next chunk starts at:** `start_idx = i - 4`

**Step 3 — Second chunk:**

```
"who have completed their probationary period. Benefits include medical, dental," = 80 chars
Adding "and" would be 84 chars.
Adding "vision" would be 91 chars.
Adding "coverage." would be 101 chars — exceeds 100. Stop.
```

**Chunk 1:** `"who have completed their probationary period. Benefits include medical, dental, and vision"` (91 chars)

**Overlap region:** `"who have completed their"` appears at the end of Chunk 0 and the beginning of Chunk 1.

---

## 5. Chunk Size and Overlap Rationale

### 5.1 Why 1000 Characters

| Factor                     | Rationale                                                                          |
| -------------------------- | ---------------------------------------------------------------------------------- |
| **Embedding model limit**  | Gemini `gemini-embedding-001` supports up to 2048 tokens per input. 1000 characters (~150-250 tokens) fits comfortably within this limit with room for metadata. |
| **Retrieval granularity**  | 1000 characters is roughly 150-200 words, which is long enough to capture a complete paragraph or section, but short enough to be a focused retrieval unit. |
| **Prompt context budget**  | With `DEFAULT_TOP_K=3`, up to 3 chunks are included in the prompt. 3 × 1000 = 3000 characters of context, which fits within the context windows of both Gemini Flash and Qwen 2.5. |
| **LLM response quality**  | Chunks that are too short lose context; chunks that are too long dilute the signal. 1000 characters strikes a balance for question-answering tasks. |

### 5.2 Why 200 Characters Overlap

| Factor                     | Rationale                                                                          |
| -------------------------- | ---------------------------------------------------------------------------------- |
| **Boundary continuity**    | Sentences or clauses that span chunk boundaries are captured in both adjacent chunks, ensuring no information is lost at boundaries. |
| **Overlap ratio**          | 200 / 1000 = 20% overlap. This is a standard ratio in RAG chunking strategies that balances context continuity against redundant storage and retrieval. |
| **Embedding consistency**  | Overlapping text produces partially similar embeddings for adjacent chunks, which can improve retrieval when a query matches content near a chunk boundary. |
| **Token budget impact**    | 200 characters (~30-50 tokens) of redundancy per chunk is negligible within the overall prompt budget. |

---

## 6. Benefits

### 6.1 Word Boundary Integrity

The algorithm never splits a word across chunks. Words are the atomic unit of splitting, which ensures that every chunk contains complete, readable text. This avoids the artifact of truncated words like `"employ"` / `"ees"` that character-level splitters produce.

### 6.2 Deterministic Output

The same input always produces the same chunks. There is no randomness, no dependency on external state, and no model-dependent tokenization. This makes the chunking step reproducible and debuggable.

### 6.3 Idempotent Chunk IDs

The chunk ID format (`{filename}_page_{N}_chunk_{M}`) is deterministic and human-readable. When combined with UUID5 generation in `vector_db.py`, re-processing the same document overwrites existing vectors rather than creating duplicates.

### 6.4 No External Dependencies

The chunker uses only Python built-ins (`str.split()`, `len()`, list operations). There are no dependencies on LangChain, NLTK, spaCy, or any external chunking library. This keeps the dependency footprint minimal and the code fully auditable.

### 6.5 Per-Page Isolation

Each page is chunked independently, which aligns with the PDF extraction model (`pypdf` extracts page-by-page). This makes it straightforward to trace any chunk back to its exact source page.

---

## 7. Limitations

### 7.1 No Cross-Page Chunking

Sentences or paragraphs that span page boundaries are split across separate page-based chunk sequences. A sentence that begins on page 3 and continues on page 4 will be chunked as two separate fragments, potentially losing semantic coherence.

**Impact:** Queries targeting content near page boundaries may retrieve incomplete fragments.

**Mitigation:** None in the current implementation. Cross-page chunking would require a post-extraction pass that merges text across page boundaries before chunking.

### 7.2 No Semantic Boundary Awareness

The algorithm does not respect paragraph breaks, section headings, or other semantic boundaries. A chunk may end mid-sentence or mid-paragraph, and a new chunk may begin in the middle of a different semantic unit.

**Impact:** Chunks may contain logically disconnected content, which can reduce retrieval quality for queries targeting a specific section.

**Mitigation:** The 200-character overlap partially addresses this by ensuring adjacent chunks share context, but does not guarantee clean semantic boundaries.

### 7.3 Whitespace Normalization

The output text is reconstructed via `" ".join(current_words)`, which collapses all original whitespace (multiple spaces, tabs, newlines, indentation) into single spaces. This can affect readability for code-heavy or formatted content.

**Impact:** Minimal for natural language PDFs. Potentially significant for technical documents with structured formatting.

### 7.4 Variable Actual Chunk Sizes

The algorithm targets `chunk_size` as a maximum, but actual chunk sizes vary based on word lengths. A chunk containing long words may be significantly shorter than 1000 characters, while a chunk that just barely fits may be close to 1000. The actual distribution depends on the text.

**Impact:** Inconsistent chunk sizes can lead to uneven embedding density and variable retrieval quality across chunks.

### 7.5 No Duplicate Detection

Re-uploading the same document with different content (e.g., an updated version with the same filename) will overwrite existing chunks due to deterministic chunk IDs. However, if the new version has a different number of pages or different page content, the old chunks from pages that no longer exist will remain in the database as orphaned vectors.

**Impact:** Stale chunks from previous versions may appear in search results.

**Mitigation:** Requires manual collection management or a dedicated document deletion feature.

### 7.6 Page-Level Reset of Chunk Index

The `chunk_index` resets to 0 for each new page. This means chunk IDs like `report.pdf_page_1_chunk_0` and `report.pdf_page_2_chunk_0` both end with `_chunk_0`, which can be confusing when scanning chunk IDs without the page number context.

**Impact:** Cosmetic issue only. The full chunk ID (including page number) is unique.

---

## 8. Trade-offs Summary

| Decision                        | Benefit                                      | Cost                                            |
| ------------------------------- | -------------------------------------------- | ----------------------------------------------- |
| Word-boundary splitting         | No broken words, readable output             | Variable chunk sizes, no character-level precision |
| 1000-character chunk size       | Fits embedding model limits, focused chunks   | May split semantic units, no semantic awareness |
| 200-character overlap           | Context continuity at boundaries              | 20% storage and embedding redundancy            |
| Per-page chunking               | Deterministic, traceable to source page       | Breaks cross-page semantic coherence            |
| No external dependencies        | Minimal footprint, fully auditable            | No access to advanced chunking algorithms       |
| Deterministic chunk IDs         | Idempotent upserts, reproducible output       | Orphaned chunks on document version changes     |
| Character-based overlap target  | Simple implementation, predictable behavior   | Actual overlap is approximate (word-granular)   |

---

## 9. Configuration Defaults

```python
# config.py lines 24-25

DEFAULT_CHUNK_SIZE = 1000      # Maximum characters per chunk
DEFAULT_CHUNK_OVERLAP = 200    # Target overlap between adjacent chunks (characters)
```

These values are used as the default arguments in `chunk_document_pages()` and are not currently exposed in the Streamlit UI. Changing them requires modifying `config.py` and restarting the application.

---

*End of Document*
