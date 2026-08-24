# Optimized RAG Evaluation

## Configuration

- Final Top-k: 8
- Similarity candidate pool: 32
- Maximum chunks per PDF: 3
- Temperature: 0.2
- Chunk size: 600
- Chunk overlap: 100

## Baseline comparison

| Metric | Plain Top-k 5 | Diversified Top-k 8 | Change |
|---|---:|---:|---:|
| Retrieval any-file hit | 100% | 100% | 0 pp |
| Retrieval all-files hit | 65% | 90% | +25 pp |
| Page any-hit | 85% | 90% | +5 pp |
| Page all-hit | 50% | 55% | +5 pp |
| Refusal behavior correct | 84% | 96% | +12 pp |
| Average latency | 4.12 s | 6.78 s | +2.66 s |
| Manual answer score | 39/50 | 43/50 | +4 |

The five no-answer questions are excluded from retrieval hit-rate denominators.

## Manual answer scores

| ID | Score | Review note |
|---|---:|---|
| fact_001 | 2 | Complete adaptive-interval and G0 explanation. |
| fact_002 | 2 | Correctly identifies all three CG-DM innovations. |
| fact_003 | 2 | Correct generator and discriminator roles. |
| fact_004 | 2 | Correct category/aspect-angle conditioning and episode training. |
| fact_005 | 2 | Correct Bysloss motivation and preserved features. |
| fact_006 | 0 | Refuses despite relevant DiffuSAR evidence; clear regression. |
| fact_007 | 2 | Correct Wasserstein and label-loss changes. |
| fact_008 | 1 | Correct 5.77% gain but still misses rotated cropping. |
| fact_009 | 2 | Correct speckle and monotonic augmentations. |
| fact_010 | 2 | Correct received-signal conditional inference and MAP context. |
| comparison_001 | 2 | Compares both angle-control formulations. |
| comparison_002 | 2 | Correct statistical versus frequency-domain comparison. |
| comparison_003 | 1 | AGGAN is complete; AAE mechanism remains incomplete. |
| comparison_004 | 2 | Correctly distinguishes both conditioning/output designs. |
| comparison_005 | 2 | Correct DDPM baseline and DiffuSAR FAM increment. |
| cross_001 | 2 | Covers both papers' GAN limitations. |
| cross_002 | 1 | GAN mechanisms are useful; FAGD mechanism remains incomplete. |
| cross_003 | 1 | Covers BCNet and Ship-Go but misses AGGAN. |
| cross_004 | 1 | Covers CG-DM and Ship-Go; land-cover control is incomplete. |
| cross_005 | 2 | Correctly compares noise robustness with federated robustness. |
| no_answer_001 | 2 | Correct refusal. |
| no_answer_002 | 2 | Correct refusal. |
| no_answer_003 | 2 | Correct refusal. |
| no_answer_004 | 2 | Correct refusal. |
| no_answer_005 | 2 | Correct refusal with domain explanation. |

Manual total: **43/50**  
Manual mean: **1.72/2.00**  
Score distribution: **19 score-2, 5 score-1, 1 score-0**

## Decision

Source diversification is useful for comparisons and cross-document questions,
but it should not replace plain similarity retrieval for every query. The
three-chunk cap still causes one factual regression, and latency increases by
about 65%.

The project therefore keeps diversification optional. The later query-routing
layer should select diversified retrieval for explicit comparison or
multi-paper synthesis and plain retrieval for focused single-paper questions.
This decision is based on the measured tradeoff rather than a global parameter
change.
