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

## Expanded schema compatibility

The loader accepts the legacy fields above and adds these defaults until the
M2 dataset expansion is complete:

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

After M2 creates the complete qrels file, validate both schema and current
Chroma metadata with:

```bash
python evaluation/validate_dataset.py \
  --qrels evaluation/qrels.json \
  --require-qrels \
  --check-index
```

Passing `--qrels evaluation/qrels.json` to `evaluate.py` or
`evaluate_retrieval.py` adds Recall@k, MRR@k, and nDCG@k to question-level and
summary results. Runs without qrels keep the historical output compatible.

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
