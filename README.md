# Academic Paper RAG Assistant

Academic Paper RAG Assistant is a retrieval-augmented
generation system for academic-paper question answering.

The current knowledge base focuses on synthetic aperture
radar (SAR) research papers. The system parses PDF papers,
splits them into retrieval chunks, generates dense embeddings
with BGE-M3, stores them in Chroma, retrieves relevant
contexts for user queries, and uses DeepSeek to generate
grounded answers with paper and page citations.

PDF Papers
    ↓
PyMuPDF
    ↓
Page Documents
    ↓
RecursiveCharacterTextSplitter
    ↓
Chunk Documents
    ↓
BGE-M3
    ↓
Chroma
    ↓
Top-k Retrieval
    ↓
Context + RAG Prompt
    ↓
DeepSeek
    ↓
Answer + Sources

## Current Features

- PDF paper upload and validation
- Automatic knowledge-base rebuilding
- PDF parsing with page metadata
- Configurable text chunking
- BGE-M3 dense embeddings
- Persistent Chroma vector database
- Semantic Top-k retrieval
- Multi-turn query rewriting
- DeepSeek-based grounded RAG answering
- Paper and PDF-page citations
- Knowledge-base refusal behavior
- Configurable Top-k
- Configurable Temperature
- Configurable Chunk Size
- Configurable Chunk Overlap
- Knowledge-base paper and chunk statistics