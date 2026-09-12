/**
 * Browser-side inference for the dysbiosis score.
 *
 * The model does not run on the server. The site ships the 13-fold ensemble as
 * a ~4.7 MB ONNX graph plus the frozen embedding table it shares, and the
 * visitor's own CPU does the forward pass. Two reasons: a 1-2 core server
 * should not spend itself on work the browser can do for free, and traffic
 * costs nothing to scale that way.
 *
 * The embedding table is deliberately outside the graph. All 416 checkpoints
 * share one frozen table, so baking it into each fold would ship it thirteen
 * times; gathering rows in JavaScript and feeding the encoder vectors keeps
 * the download at one copy.
 */

/**
 * Load the embedding table as float32.
 *
 * @param {Float32Array} table - Result of `halfToFloat` over the float16 blob,
 *   laid out row-major as `(nTokens, dModel)`.
 * @param {number} dModel
 */
export function makeGather(table, dModel) {
  return function gather(features) {
    const rows = features.length;
    const out = new Float32Array(rows * dModel);
    for (let i = 0; i < rows; i += 1) {
      const source = features[i] * dModel;
      out.set(table.subarray(source, source + dModel), i * dModel);
    }
    return out;
  };
}

/**
 * Build the three input tensors for one sample.
 *
 * @param {object} ort - The `ort` global from onnxruntime-web.
 * @param {{features: Int32Array, abundance: Float32Array, mask: Int32Array}} sample
 * @param {number} numSteps
 * @param {number} dModel
 * @param {Float32Array} gathered - Output of `gather(sample.features)`.
 */
export function makeFeeds(ort, sample, numSteps, dModel, gathered) {
  // onnxruntime-web has no int32 tensor for this input; the graph declares
  // int64 because that is what the PyTorch mask is.
  const mask = new BigInt64Array(numSteps);
  for (let i = 0; i < numSteps; i += 1) {
    mask[i] = BigInt(sample.mask[i]);
  }
  return {
    inputs: new ort.Tensor('float32', gathered, [1, numSteps, dModel]),
    weight: new ort.Tensor('float32', sample.abundance, [1, numSteps]),
    mask: new ort.Tensor('int64', mask, [1, numSteps]),
  };
}

/**
 * Run the ensemble over one prepared sample.
 *
 * @returns {{logit: number, attention: Float32Array}} `attention` is the
 *   fold-, head- and query-averaged weight per sequence position, which is
 *   what the contributing-taxa list ranks.
 */
export async function scoreSample(session, ort, sample, numSteps, dModel,
                                  gather) {
  const feeds = makeFeeds(ort, sample, numSteps, dModel,
                          gather(sample.features));
  const outputs = await session.run(feeds);
  const logit = outputs.logit.data[0];
  let attention = outputs.attention.data;
  if (!(attention instanceof Float32Array)) {
    attention = Float32Array.from(attention);
  }
  return { logit, attention: attention.slice(0, numSteps) };
}

/**
 * Share of the reference cohort scoring below `value`, as a percentage.
 *
 * The reference arrays are sorted, so this is a binary search. Ties are
 * counted as half, which keeps the percentile continuous when many reference
 * samples land on the same score — the outputs are rounded to four decimals
 * before shipping, so exact ties are common.
 *
 * @param {number[]} sorted - Sorted reference scores.
 * @param {number} value
 * @returns {number} Percentile in 0..100.
 */
export function percentileOf(sorted, value) {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (sorted[mid] < value) low = mid + 1;
    else high = mid;
  }
  const below = low;

  let lowTie = 0;
  let highTie = sorted.length;
  while (lowTie < highTie) {
    const mid = (lowTie + highTie) >> 1;
    if (sorted[mid] <= value) lowTie = mid + 1;
    else highTie = mid;
  }
  const equal = lowTie - below;

  return ((below + equal / 2) / sorted.length) * 100;
}

/**
 * Rank the sequence positions a sample's score leaned on.
 *
 * Positions are the top-`numSteps` OTUs in the order they were encoded, so an
 * attention weight maps back to an OTU through `sample.features`.
 *
 * @param {Float32Array} attention
 * @param {{features: Int32Array, abundance: Float32Array, mask: Int32Array}} sample
 * @param {number} limit
 * @returns {Array<{index: number, weight: number, abundance: number}>}
 */
export function topContributors(attention, sample, limit = 10) {
  const positions = [];
  for (let i = 0; i < sample.features.length; i += 1) {
    if (sample.mask[i] === 0) continue;
    positions.push({
      index: sample.features[i],
      weight: attention[i],
      abundance: sample.abundance[i],
    });
  }
  positions.sort((a, b) => b.weight - a.weight);
  return positions.slice(0, limit);
}
