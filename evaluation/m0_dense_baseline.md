# M0 Dense Retrieval Baseline

Baseline frozen on 2026-08-31 before M1 rank-aware metric implementation.

## Environment

- Branch: `codex/hybrid-rerank-evaluation`
- Baseline code commit: `de0a352`
- Python: `D:\CondaEnvs\llm-universe\python.exe` (Python 3.10)
- Embedding: `BAAI/bge-m3`, local cache with offline mode enabled
- Index: 15 PDFs, 218 pages, 2142 Chunks, 600/100 split
- Questions: 25 total, 20 with expected retrieval sources

Raw retrieval output is stored locally at
`evaluation/results/m0_dense_baseline.json`. The results directory remains
Git-ignored because raw outputs contain paper excerpts.

## Verification

The pre-change regression suite passed:

```text
Ran 47 tests in 0.084s
OK
```

Dense retrieval results:

| Metric | Top-k 5 | Top-k 8 |
|---|---:|---:|
| Any expected file hit | 100% | 100% |
| All expected files hit | 65% | 75% |
| Any expected page hit | 85% | 95% |
| All expected pages hit | 50% | 50% |
| Average unique files | 2.16 | 2.88 |
| Largest single-file share | 76.0% | 68.5% |
| Average warm retrieval latency | 0.0886 s | 0.0837 s |

The latency values are local observations rather than portable performance
claims. Recall@k, MRR@k, and nDCG@k are intentionally absent because the M0
dataset has no Chunk-level qrels yet.
