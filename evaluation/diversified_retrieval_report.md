# Source-Diversified Retrieval Experiment

The diversified Retriever takes a larger similarity candidate pool and then
limits how many final chunks one PDF may contribute.

## One chunk per paper

| Top-k | All files | Any page | All pages | Avg. unique papers |
|---:|---:|---:|---:|---:|
| 5 | 90% | 45% | 25% | 3.80 |
| 8 | 100% | 45% | 30% | 5.56 |
| 10 | 100% | 45% | 30% | 6.80 |

This setting maximizes paper diversity but is too aggressive for factual
questions because only one local passage from the relevant paper survives.

## Two chunks per paper

| Top-k | All files | Any page | All pages | Avg. unique papers |
|---:|---:|---:|---:|---:|
| 5 | 85% | 65% | 40% | 3.04 |
| 8 | 90% | 65% | 40% | 4.56 |
| 10 | 95% | 65% | 45% | 5.36 |

## Selected configuration

The full-generation evaluation will use:

- Final Top-k: 8
- Similarity candidate pool: 32
- Maximum chunks per PDF: 2

This raises all-file retrieval from 75% for plain Top-k 8 to 90%, while keeping
two passages available for each selected paper. Top-k 10 adds context for only a
five-point all-file improvement, and the one-chunk cap risks under-supporting
single-paper questions.

The page metric is interpreted cautiously. Evaluation answers are anchored to
abstract pages for auditability, but another page from the same expected paper
may still contain valid and more detailed evidence. The full-generation review
therefore checks answer correctness rather than optimizing only for page 1.
