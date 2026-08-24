# Corpus and Index Audit

Audit date: 2026-08-24

## Decision

The current 15-paper corpus is sufficient for the project baseline. No Chinese
papers or papers from unrelated SAR tasks are required before evaluation.

The corpus is deliberately domain-focused and English-only. This keeps the
baseline centered on retrieval, evidence coverage, grounded answering, and
refusal rather than introducing cross-language retrieval as another variable.

## Coverage

- Language: 15 English journal or conference papers.
- GAN-family methods: 8 papers, including angle control, statistical guidance,
  few-shot generation, background conversion, adversarial autoencoding, and
  land-cover-conditioned scene synthesis.
- Diffusion-family methods: 7 papers, including frequency-aware generation,
  oil-spill synthesis, ship inpainting, noise-tolerant novel views, statistical
  imaging, and federated angle-aware generation.
- Application coverage: SAR targets, ships, oil spills, large scenes, target
  recognition data augmentation, detection data generation, and SAR imaging.

This coverage supports single-paper facts, same-family comparisons, GAN versus
diffusion comparisons, cross-document synthesis, and no-answer cases.

## Optional future expansion

Expansion should be a separate experiment rather than part of this baseline:

- Add 3-5 Chinese SAR-generation papers to measure cross-language retrieval.
- Add 4-6 SAR detection or recognition papers to measure cross-task routing and
  retrieval.

Neither expansion is needed for the current Academic Paper RAG Assistant scope.

## Rebuild verification

The active `academic_papers` Chroma collection was rebuilt from every PDF in
`data/papers` with the following verified state:

- Papers: 15
- PDF pages: 218
- Chunks produced: 2142
- Chroma records: 2142
- Chunk size: 600 characters
- Chunk overlap: 100 characters
- Embedding model: `BAAI/bge-m3`
- Manifest build time: `2026-08-24T23:18:51+08:00`

The rebuild was accepted only after the generated Chunk count, Chroma record
count, and `index_manifest.json` count all matched.
