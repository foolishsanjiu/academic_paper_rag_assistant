# Project Architecture

## Core Modules

### app.py
Streamlit Web interface and session-state management.

### config.py
Loads model API configuration from environment variables.

### logging_config.py
Configures application logging.

### document_loader.py
Loads PDF papers with PyMuPDF and converts each page into a
LangChain Document while preserving filename and page metadata.

### text_splitter.py
Splits page-level Documents into chunk-level Documents and adds
chunk metadata.

### vector_store.py
Loads the BGE-M3 embedding model and manages the persistent
Chroma vector store.

### retriever.py
Performs Top-k semantic retrieval and returns relevant chunks
with metadata and similarity information.

### llm_client.py
Provides an OpenAI-compatible LLM client for DeepSeek.

### rag_chain.py
Connects query rewriting, retrieval, context construction,
grounded prompting, answer generation, and source citation.

## Current Pipeline

PDF
→ Page Document
→ Chunk Document
→ BGE-M3 Embedding
→ Chroma
→ Top-k Retrieval
→ Context Construction
→ DeepSeek
→ Answer + Citation

## Relationship to LLM-Universe

This project was developed while referring to the
Datawhale LLM-Universe tutorials for RAG concepts and
implementation ideas.

The current Academic Paper RAG Assistant is maintained
as an independent project.

Project-specific implementation includes:

- PDF page-level parsing and parsing-quality inspection;
- SAR-paper metadata design;
- chunk-level metadata and chunk IDs;
- BGE-M3 embedding integration;
- persistent Chroma vector database;
- semantic retrieval with metadata and similarity scores;
- multi-turn retrieval query rewriting;
- grounded RAG prompt construction;
- paper/page/chunk citation mapping;
- knowledge-base refusal behavior;
- Streamlit RAG interface.