# Regression tests for the browser path

```bash
# 1. Build the fixture (needs the exported model; run on the build machine).
.venv-export/bin/python script/web_export/export_golden.py

# 2. Run the comparisons.
cd tests/js && npm install
npm test                    # both files
node preprocess.test.mjs    # the preprocessing and inference chain
node table.test.mjs         # the count-table reader
```

## What is being defended

The site runs the classifier in the browser, which means the preprocessing the
paper's pipeline performs has been re-implemented in JavaScript. If that port is
wrong, the page still produces a number, still draws a distribution, and is
still wrong — there is no error to notice. This test is the only thing standing
between a typo and a quietly incorrect percentile.

Two links in the chain are covered by other tests and are not repeated here:

- The PyTorch pipeline reproduces the training run's `pred_test.csv`, asserted
  by `export_dysbiosis.py` for all 10,276 reference samples before it writes
  anything.
- The exported ONNX reproduces PyTorch to 9.5e-07 on logits and 6.0e-08 on
  attention, asserted by the same script.

This test closes the last link: Python's preprocessing and inference against
JavaScript's, on the same samples, reading the same float16 embedding blob.

## Why the fixture holds arrays, not just a score

A single logit difference tells you the two paths disagree. It does not tell
you whether the rank normalization, the top-600 selection, the padding or the
vocabulary lookup is responsible. The fixture therefore records the raw counts
that go in and the token, abundance and mask arrays that come out, so a failure
says which step drifted.

## What is asserted, and how

For a sample that fits within `numSteps`, every check is exact or near-exact,
because both implementations keep every non-zero taxon and the only freedom is
floating-point ordering:

| Quantity | Tolerance | Observed |
|---|---|---|
| Token sequence | exact | exact on all 29 samples that fit |
| Mask | exact | exact |
| Abundance | 1e-6 | 5e-8 |
| Logit | 1e-4 | well under it |

For a sample deeper than `numSteps`, comparing position by position is
meaningless, and the test does not pretend otherwise. The model has no
positional encoding, so permuting equally abundant taxa is the same input;
what matters is *which* taxa were kept, and at what abundance. So the test
asserts:

- the abundances selected are the same multiset (the ranking agrees);
- no taxon was kept at a lower abundance than one left behind (the selection
  really is the top `numSteps`);
- the logit stays inside a looser bound.

### The one real divergence, measured

Python selects with `np.argsort(row)[::-1][:numSteps]`. Among equal values,
which of them survive is decided by introsort's internal ordering — an
artefact, not a rule, and not stable across numpy versions either. JavaScript
sorts by abundance descending, index ascending, which is deterministic.
Neither is more correct; the site's is at least reproducible.

The fixture contains exactly one such sample, `PD/ERR2730328`: 661 taxa, of
which 41 are tied at the cutoff abundance of 0.1649017. The two
implementations keep different 41-element subsets, 41 of the 600 positions
carry a different taxon, and the logits differ by **1.3e-3**. Twenty-nine of
the thirty samples take the exact path.

The reference cohort as a whole contains one sample over 600 taxa, so this is
the whole of the divergence rather than an example of it. A visitor's sample
can be deeper and can have larger tie blocks, which is why the tolerance is
5e-3 rather than the observed 1.3e-3 — and why the assertions above are about
the selection rather than about the logit.

## A note on the attention output

`topContributors` ranks sequence positions by attention. Attention weights are
compared to 6.0e-08 by the export, so the ranking is stable across runtimes
except where two positions differ by less than that — in which case their
relative order is not meaningful anyway.

## The count-table reader

`table.test.mjs` covers `web/assets/js/table.js`, which is where a visitor's
file first meets the page. The cases are the shapes that arrive in practice:
the banner line and `#OTU ID` header that `biom convert --to-tsv` and `qiime
tools export` write, the same table transposed by hand, a row with more cells
than the header, a comma-separated file, an all-zero table (empty, not an
error), and text with no delimiter at all (a sentence, not a TypeError).

PapaParse is a page global rather than an import, so the test installs a stub
for it that covers delimiter detection, skipped empty lines and the
`errors[0].type === 'Delimiter'` signal.

## One runtime, three places

`onnxruntime-node` is pinned here to the version `script/fetch_vendor.sh`
vendors for the page and `export_dysbiosis.py` checks against PyTorch. Moving
one of the three without the others means the graph visitors load is a graph
nothing verified.
