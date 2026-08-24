# Baseline Manual Review

Run configuration:

- Questions: 25
- Top-k: 5
- Temperature: 0.2
- Chunk size: 600
- Chunk overlap: 100
- Embedding model: `BAAI/bge-m3`
- Answer score: `0 = incorrect`, `1 = partially correct`, `2 = mostly or fully correct`

## Automated results

| Metric | Result |
|---|---:|
| Completed without runtime error | 25/25 |
| Retrieval any-file hit | 100% |
| Retrieval all-files hit | 65% |
| Page any-hit | 85% |
| Page all-hit | 50% |
| Refusal behavior correct | 84% |
| Average latency | 4.12 seconds |

Retrieval rates exclude the five `no_answer` questions because those questions
have no expected source.

## Manual answer scores

| ID | Score | Review note |
|---|---:|---|
| fact_001 | 2 | Complete adaptive-interval and G0 explanation. |
| fact_002 | 2 | Correctly lists all three CG-DM innovations. |
| fact_003 | 2 | Correct generator and discriminator roles. |
| fact_004 | 2 | Correct attributes and episode-training strategy. |
| fact_005 | 2 | Correct Bysloss motivation and preserved features. |
| fact_006 | 1 | Gets the high-frequency hypothesis but says the FFT implementation is unavailable. |
| fact_007 | 2 | Correct Wasserstein and label-loss changes. |
| fact_008 | 1 | Reports the 5.77% gain but misses rotated cropping. |
| fact_009 | 2 | Correct co-domain augmentations. |
| fact_010 | 2 | Correct MAP conditional-guidance mechanism. |
| comparison_001 | 1 | Explains ATGAN but does not retrieve the second method. |
| comparison_002 | 2 | Correct statistical-domain versus frequency-domain comparison. |
| comparison_003 | 1 | Explains AGGAN fully but only gives a high-level AAE description. |
| comparison_004 | 1 | Explains Ship-Go but does not retrieve CG-DM. |
| comparison_005 | 1 | Identifies FAM but omits clutter pretraining and efficiency differences. |
| cross_001 | 1 | Covers CG-DM motivation only. |
| cross_002 | 1 | Retrieves all papers but lacks enough mechanism-level evidence. |
| cross_003 | 1 | Covers Ship-Go only. |
| cross_004 | 1 | Covers Ship-Go only. |
| cross_005 | 1 | Covers FAGD only. |
| no_answer_001 | 2 | Correct refusal. |
| no_answer_002 | 2 | Correct refusal. |
| no_answer_003 | 2 | Correct refusal. |
| no_answer_004 | 2 | Correct refusal. |
| no_answer_005 | 2 | Correct refusal with an appropriate domain explanation. |

Manual total: **39/50**  
Manual mean: **1.56/2.00**  
Score distribution: **14 score-2, 11 score-1, 0 score-0**

## Findings

1. Single-paper factual retrieval is strong: all ten fact questions retrieve
   the expected paper and expected page.
2. Top-5 is often filled by several chunks from one paper. This produces a
   100% any-file hit rate but only a 65% all-files hit rate and limits comparison
   and cross-document answers.
3. All five deliberately unanswerable questions are refused correctly.
4. The overall 84% refusal-behavior score is lower because four answerable
   multi-paper questions contain the standard refusal message after incomplete
   evidence retrieval.
5. The next parameter experiment should test larger Top-k values and, more
   importantly, source-diversified retrieval rather than assuming that Top-k
   alone guarantees multi-document coverage.
