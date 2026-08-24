# Top-k Retrieval Experiment

The experiment evaluates retrieval only. It reuses the 25-question dataset and
does not call the LLM, so the comparison isolates the Retriever.

## Results

| Top-k | Any file | All files | Any page | All pages | Avg. unique papers | Largest source share |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 100% | 60% | 80% | 50% | 1.60 | 80.0% |
| 5 | 100% | 65% | 85% | 50% | 2.16 | 76.0% |
| 8 | 100% | 75% | 95% | 50% | 2.88 | 68.5% |
| 10 | 100% | 75% | 95% | 50% | 3.36 | 63.2% |

Rates are calculated over the 20 answerable questions. The five no-answer
questions have no expected source and are excluded from hit-rate denominators.

## Conclusions

1. Increasing Top-k from 3 to 8 improves multi-paper coverage.
2. Top-k 8 and Top-k 10 produce the same all-file and all-page hit rates.
3. Exact all-page coverage remains at 50% for every tested value, so increasing
   Top-k alone does not solve evidence completeness.
4. Larger Top-k values reduce source concentration, but Top-k 10 also adds more
   context without improving expected-source coverage.
5. The next experiment should retrieve a larger candidate pool and cap the
   number of final chunks contributed by one paper.

Top-k 8 is the best plain-similarity candidate for a full-generation comparison,
but source-diversified retrieval should be tested before changing the project
default.
