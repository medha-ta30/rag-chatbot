# Custom RAG Chatbot - Test Cases
 
## 1. Relevant question (clearly answerable from docs)
**Query:** "What embedding model is used for this project?"
**Expected Behavior:** The retriever successfully finds the chunk mentioning the architecture, and the generator states the exact model name.
**Observed Output:** "The embedding model used for this project is sentence-transformers 'all-MiniLM-L6-v2'."
 
## 2. Irrelevant question (not covered by docs at all)
**Query:** "How do I bake a chocolate cake?"
**Expected Behavior:** The retriever brings back low-score or unrelated chunks, and the generator strictly adheres to the prompt, stating it cannot find the answer in the provided documents.
**Observed Output:** "I cannot find the answer in the provided documents."
 
## 3. Ambiguous question (could match multiple docs/topics)
**Query:** "What is the best way to split text?"
**Expected Behavior:** The retriever finds chunks relating to standard LangChain splitters (if available) as well as the custom chunking logic. The generator synthesizes the answer highlighting the custom approach implemented in this codebase.
**Observed Output:** "Based on the provided documents, the best way to split text for this project is using a custom hand-written function that chunks by a fixed size with overlap, attempting to split on paragraph boundaries."
 
## 4. Empty query
**Query:** "" (empty string)
**Expected Behavior:** The retriever pipeline handles the empty string gracefully by either returning no chunks or an error message, and the generator returns an empty response or a prompt to enter a question.
**Observed Output:** (The UI prevents empty queries from being submitted, or the backend immediately returns "I couldn't find any relevant documents to answer your question.")
 
## 5. Multi-doc question (answer requires info from 2+ source documents)
**Query:** "How does the custom chunker prepare data for the vector store?"
**Expected Behavior:** The retriever finds a chunk from the chunking documentation and a chunk from the vector store documentation. The generator merges this information into a cohesive answer.
**Observed Output:** "The custom chunker splits the text into fixed-size segments with an overlap of 50 characters, and then these chunks, along with their metadata, are embedded and upserted into the local Qdrant vector store."
