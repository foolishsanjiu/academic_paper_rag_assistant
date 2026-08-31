# Evaluation Dataset

`questions.json` is the reproducible evaluation set for the Academic Paper
RAG Assistant. It contains 100 questions grounded in the 15-paper SAR
image-generation corpus. The 25-question legacy set is retained unchanged,
with 25 development questions and 50 held-out test questions added in M2.

## Question types

- `fact`: 40 single-paper factual questions.
- `comparison`: 20 questions comparing two papers.
- `cross_document`: 20 questions requiring evidence from multiple papers.
- `no_answer`: 20 questions whose answers are not present in the corpus.

`source_files` and `source_pages` identify the expected evidence. Page numbers
are one-based PDF page numbers, matching the `page_number` metadata stored in
Chroma. For `no_answer` questions both fields are empty, and the expected
behavior is the standard knowledge-base refusal.

The expanded set includes 16 Chinese queries over English evidence to measure
cross-language retrieval. New questions also record language, difficulty,
retrieval challenge tags, expected key points, and a fixed data split.

## Expanded schema compatibility

The loader accepts the legacy fields above and adds these defaults for the
original 25 questions:

- `split`: `legacy`
- `language`: `en`
- `difficulty`: `unspecified`
- `tags`: `[]`
- `expected_key_points`: `[]`

New questions must set these fields explicitly. Supported splits are
`legacy`, `dev`, and `test`; test questions must not be used for parameter
selection.

Validate the current legacy dataset without loading an embedding model:

```bash
python evaluation/validate_dataset.py
```

## Chunk-level qrels

Rank-aware metrics use a separate `qrels.json` object keyed by question ID.
Each judgment records `chunk_id`, `file_name`, `page_number`, a relevance grade
from 0 to 3, and a short annotation reason. Relevance grades mean:

- `3`: directly and substantially supports the answer.
- `2`: supports a required key point.
- `1`: topically relevant but insufficient by itself.
- `0`: irrelevant or misleading hard negative.

Validate the complete qrels file against both schema and current Chroma
metadata with:

```bash
python evaluation/validate_dataset.py \
  --qrels evaluation/qrels.json \
  --require-qrels \
  --check-index
```

Passing `--qrels evaluation/qrels.json` to `evaluate.py` or
`evaluate_retrieval.py` adds Recall@k, MRR@k, and nDCG@k to question-level and
summary results. Runs without qrels keep the historical output compatible.

Run the M3 BM25-only retrieval baseline without loading the embedding model or
calling an LLM:

```bash
python evaluation/evaluate_retrieval.py \
  --method bm25 \
  --top-k 5 8 10 \
  --qrels evaluation/qrels.json
```

The runtime index uses `rank-bm25==0.2.2`, sorts the corpus by persistent
`chunk_id`, and records its tokenizer and BM25 parameters in the result file.
BM25 scores are stored as `bm25_score`, never mislabeled as cosine similarity.

Run the M4 Hybrid/RRF development grid without calling an LLM:

```bash
python evaluation/evaluate_retrieval.py \
  --method hybrid \
  --split dev \
  --top-k 10 \
  --candidate-k 20 32 50 \
  --rrf-k 30 60 \
  --fusion-k 40 \
  --qrels evaluation/qrels.json \
  --output evaluation/results/m4_hybrid_dev_grid.json
```

Hybrid retrieves equal-sized Dense and BM25 candidate sets, deduplicates by
`chunk_id`, and applies deterministic Reciprocal Rank Fusion. The selected M4
development parameters are candidate k 20, RRF k 60, and fusion k 40. Do not
use the `test` split for further parameter selection. See
`m4_hybrid_rrf_dev_report.md` for the ablation results and limitations.

Build a deduplicated, deterministic annotation pool from one or more retrieval
result files and every Chunk on the expected source pages:

```bash
python evaluation/build_annotation_pool.py \
  --results evaluation/results/m0_dense_baseline.json \
  --output evaluation/results/annotation_pool.json
```

Candidate ordering is a deterministic hash order rather than retrieval rank.
Retrieval-system provenance is stored separately from candidate rows so an
annotation interface can hide it while preserving auditability. Generated
pools remain under the Git-ignored `evaluation/results/` directory because
they contain paper excerpts.

For later corpus revisions, `propose_qrels.py` can create an offline semantic
draft plus a review report. The draft must be reviewed before changing the
explicit mapping in `finalize_qrels.py`; running the latter materializes the
tracked `qrels.json`. The finalized M2 set contains 213 relevance judgments:
direct evidence is graded 3 and complementary evidence is graded 2.
