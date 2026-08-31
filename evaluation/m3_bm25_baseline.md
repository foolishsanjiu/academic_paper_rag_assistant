# M3 BM25 Sparse Retrieval Baseline

## Scope

- Configuration B only: BM25 enabled; Dense, RRF, and Reranker disabled.
- Dataset: frozen 100-question `evaluation/questions.json`.
- Ground truth: 213 reviewed Chunk judgments in `evaluation/qrels.json`.
- Corpus: 15 PDFs, 218 pages, 2142 Chunks.
- No embedding model or paid LLM API was loaded.

## Runtime identity

| Item | Value |
|---|---|
| Implementation | `rank-bm25==0.2.2` / Okapi BM25 |
| Tokenizer | `unicode_nfkc_ascii_v1` |
| Tokenization | NFKC, casefold, English/numeric/hyphenated terms |
| BM25 parameters | k1=1.5, b=0.75, epsilon=0.25 |
| Corpus order | ascending persistent `chunk_id` |
| Python / OS | 3.10.20 / Windows 10 build 26200 |
| Processor identity | Intel64 Family 6 Model 191 |
| Chroma open / BM25 build | 0.0917 s / 0.2235 s |

Raw output is stored at the Git-ignored path
`evaluation/results/m3_bm25_baseline.json` because it contains paper excerpts.

## Results

Metrics are macro averages over the 80 answerable questions for rank-aware
metrics and expected-source coverage. Latency includes one BM25 query over all
2142 in-memory Chunks, but excludes Chroma open and index construction.

| Top-k | Any file | All files | Any page | All pages | Recall | MRR | nDCG | P50 | P95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 96.25% | 65.00% | 58.75% | 35.00% | 0.2231 | 0.2477 | 0.2008 | 5.20 ms | 7.10 ms |
| 8 | 97.50% | 73.75% | 68.75% | 42.50% | 0.3015 | 0.2664 | 0.2327 | 4.85 ms | 6.51 ms |
| 10 | 97.50% | 76.25% | 75.00% | 48.75% | 0.3202 | 0.2717 | 0.2410 | 5.00 ms | 6.60 ms |

## Verification

- Repeating Top-10 retrieval for all 100 questions produced identical ordered
  `chunk_id` lists.
- Unit tests cover normalization, exact IDs, empty queries, unsupported tokens,
  duplicate IDs, unknown terms, stable score ties, and score serialization.
- The full project suite passes 79 tests.
- Dataset/qrels validation passes against the active Chroma index.

## Known limitation and stopping point

The tokenizer deliberately does not add Chinese segmentation. Three Chinese
queries containing no English or numeric term therefore return an empty BM25
ranking. This is an explicit configuration-B limitation; it must be addressed
by the Dense branch in M4 Hybrid/RRF rather than by changing the frozen test
set or silently adding a new tokenizer.

M3 stops here. No RRF, Hybrid Retrieval, Reranker, or default RAG-path switch is
included in this milestone.
