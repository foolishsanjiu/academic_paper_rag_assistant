# Evaluation Dataset

`questions.json` is the reproducible evaluation set for the Academic Paper
RAG Assistant. It contains 25 English questions grounded in the 15-paper SAR
image-generation corpus.

## Question types

- `fact`: 10 single-paper factual questions.
- `comparison`: 5 questions comparing two papers.
- `cross_document`: 5 questions requiring evidence from multiple papers.
- `no_answer`: 5 questions whose answers are not present in the corpus.

`source_files` and `source_pages` identify the expected evidence. Page numbers
are one-based PDF page numbers, matching the `page_number` metadata stored in
Chroma. For `no_answer` questions both fields are empty, and the expected
behavior is the standard knowledge-base refusal.

The initial set deliberately stays in English because all source papers are in
English. This keeps the baseline focused on retrieval and grounded generation
instead of mixing in cross-language retrieval as an additional variable.
