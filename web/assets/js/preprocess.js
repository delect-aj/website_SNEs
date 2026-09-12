/**
 * The browser half of the table-to-tensor contract.
 *
 * Mirrors `script/web_export/preprocessing.py`, which in turn mirrors
 * `read_imdb` and `truncate_pad` in `otu_attention.py`. Every step here is
 * load-bearing: rank the non-zero counts only, divide by the sample's largest
 * rank, keep the `numSteps` most abundant OTUs, and mask both padding and
 * out-of-vocabulary positions. Getting one of them wrong produces a number
 * that looks fine and is wrong, so `tests/js/` pins this against Python.
 *
 * This module is plain ES and has no DOM dependency, so the same code runs in
 * the browser and under node in the test suite.
 */

export const PAD_INDEX = 0;
export const UNK_INDEX = 1;

/**
 * Rank-normalize one sample's non-zero counts into (0, 1].
 *
 * Ties share their average rank, and the result is divided by the largest
 * rank, so the most abundant OTU always sits at 1. Zeros stay zero. The ranks
 * are returned over the dense index space because the caller needs to pick the
 * top `numSteps` across all of it.
 *
 * @param {number} nFeatures - Size of the dense index space.
 * @param {ArrayLike<number>} indices - Feature indices with a non-zero count.
 * @param {ArrayLike<number>} counts - Counts aligned with `indices`.
 * @returns {Float64Array} Rank-normalized counts, length `nFeatures`.
 */
export function rankNormalizeSample(nFeatures, indices, counts) {
  const dense = new Float64Array(nFeatures);
  for (let k = 0; k < indices.length; k += 1) {
    const value = counts[k];
    if (value > 0) dense[indices[k]] = value;
  }

  const nonzero = [];
  for (let i = 0; i < nFeatures; i += 1) {
    if (dense[i] !== 0) nonzero.push(i);
  }
  if (nonzero.length === 0) return dense;

  // Ascending by count; Array.prototype.sort is stable, so equal counts stay
  // in ascending index order, which is what np.argsort(kind='mergesort') does.
  // `order` holds feature indices; the ranks live in a parallel array so the
  // two never get confused. Writing a rank back over an index here would put
  // every rank on the wrong taxon and leave the embedding unchanged in shape
  // while changing what it means.
  const order = nonzero.slice().sort((a, b) => dense[a] - dense[b]);
  const ranks = new Float64Array(order.length);

  let start = 0;
  while (start < order.length) {
    let stop = start + 1;
    while (stop < order.length && dense[order[stop]] === dense[order[start]]) {
      stop += 1;
    }
    const rank = (start + stop - 1) / 2 + 1;
    for (let k = start; k < stop; k += 1) ranks[k] = rank;
    start = stop;
  }

  const maxRank = ranks[ranks.length - 1];
  for (let k = 0; k < order.length; k += 1) {
    dense[order[k]] = ranks[k] / maxRank;
  }
  return dense;
}

/**
 * Select, pad and encode one rank-normalized sample.
 *
 * The selection rule when a sample has at least `numSteps` non-zero OTUs is by
 * abundance descending with ties broken by ascending index. Python uses
 * `np.argsort(row)[::-1]`, whose tie order is an artefact of introsort; the
 * two can only differ when the value at the cutoff is tied, and the reference
 * cohort contains one such sample out of 10,276. `tests/js/` measures the
 * resulting difference rather than assuming it away.
 *
 * @param {Float64Array} dense - Output of {@link rankNormalizeSample}.
 * @param {number} numSteps - Sequence length.
 * @param {Map<string, number>} vocabIndex - OTU id to vocabulary index.
 * @param {ArrayLike<string>} featureIds - OTU id of each dense position.
 * @returns {{features: Int32Array, abundance: Float32Array, mask: Int32Array,
 *            nOtus: number}}
 *   `features` holds vocabulary indices, `abundance` the rank-normalized
 *   weights, `mask` is 0 at padding and out-of-vocabulary positions.
 */
export function encodeSample(dense, numSteps, vocabIndex, featureIds) {
  const nonzero = [];
  for (let i = 0; i < dense.length; i += 1) {
    if (dense[i] !== 0) nonzero.push(i);
  }

  let take;
  if (nonzero.length >= numSteps) {
    take = nonzero.slice().sort((a, b) => (dense[b] - dense[a]) || (a - b));
    take.length = numSteps;
  } else {
    take = nonzero;
  }

  const features = new Int32Array(numSteps);   // zero-filled: PAD_INDEX
  const abundance = new Float32Array(numSteps);
  const mask = new Int32Array(numSteps);

  for (let k = 0; k < take.length; k += 1) {
    const index = vocabIndex.get(featureIds[take[k]]);
    features[k] = index === undefined ? UNK_INDEX : index;
    abundance[k] = dense[take[k]];
  }
  for (let k = 0; k < numSteps; k += 1) {
    mask[k] = features[k] !== PAD_INDEX && features[k] !== UNK_INDEX ? 1 : 0;
  }

  return { features, abundance, mask, nOtus: nonzero.length };
}

/**
 * Run the whole contract over a sparse sample in one call.
 *
 * @param {{nFeatures: number, featureIds: ArrayLike<string>,
 *          vocabIndex: Map<string, number>, numSteps: number}} spec
 * @param {ArrayLike<number>} indices - Feature indices with a non-zero count.
 * @param {ArrayLike<number>} counts - Counts aligned with `indices`.
 * @returns {ReturnType<typeof encodeSample>}
 */
export function preprocessSample(spec, indices, counts) {
  const dense = rankNormalizeSample(spec.nFeatures, indices, counts);
  return encodeSample(dense, spec.numSteps, spec.vocabIndex, spec.featureIds);
}
