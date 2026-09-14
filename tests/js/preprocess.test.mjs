/**
 * The browser path against Python, over real samples.
 *
 * The design document's first verification item. Two claims are checked here,
 * and they fail differently:
 *
 *   1. The sequencing. `preprocess.js` must turn one sample's counts into the
 *      same token, abundance and mask arrays that `preprocessing.py` produced.
 *      A mistake here is invisible on the page — a shifted rank, a padded
 *      position left unmasked, an off-by-one in the vocabulary — and produces a
 *      plausible-looking wrong percentile.
 *   2. The inference. Gathering rows from the float16 embedding and feeding the
 *      ONNX graph under node must reproduce the Python logits.
 *
 * Both sides read the same fixture, which carries raw counts and the arrays
 * Python derived from them, so a failure names the step rather than only the
 * final number.
 *
 * The fixture holds the reference-cohort picks and the six one-click examples
 * the dysbiosis page offers, so the samples a visitor can actually run are
 * covered by the same comparison rather than by nothing at all.
 *
 * Run:
 *   cd tests/js && npm install && node preprocess.test.mjs
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import ort from 'onnxruntime-node';

import { rankNormalizeSample, encodeSample, PAD_INDEX, UNK_INDEX }
  from '../../web/assets/js/preprocess.js';
import { halfToFloat } from '../../web/assets/js/binary.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..', '..');

// What we are willing to accept, and why.
//
// The sequence arrays must match exactly: they are integers, and a shifted
// element is a bug rather than noise. Abundance agrees to float64 rounding.
//
// The logit has two tolerances because there are two regimes.
//
// A sample with no more than `numSteps` non-zero taxa keeps all of them, so
// both implementations select the same set and the only difference is
// floating-point ordering. 1e-4 is three orders above the 1e-7 measured
// there, and tight enough that any real preprocessing mistake fails it.
//
// A deeper sample has to drop taxa, and Python selects with
// `np.argsort(row)[::-1][:numSteps]`. When the value at the cutoff is tied,
// which of the tied taxa survive is decided by introsort's internal ordering
// -- an artefact, not a rule, and not stable across numpy versions either.
// JavaScript sorts by (abundance descending, index ascending), which is
// deterministic. Neither is more correct; the site's is at least reproducible.
//
// Measured on the fixture: one sample, PD/ERR2730328, has 661 taxa with 41
// tied at the cutoff abundance of 0.1649017. The two implementations keep
// different 41-element subsets and the logits differ by 1.3e-3. That is the
// only such sample in the fixture, and the reference cohort contains one
// sample over 600 taxa in total, so this is the whole of the divergence
// rather than an example of it.
const LOGIT_TOLERANCE = 1e-4;
const DEEP_LOGIT_TOLERANCE = 5e-3;
const ARRAY_TOLERANCE = 1e-6;

let failures = 0;
let checks = 0;

function check(condition, message) {
  checks += 1;
  if (!condition) {
    failures += 1;
    console.error(`  FAIL ${message}`);
  }
}

const fixture = JSON.parse(
  readFileSync(join(here, '..', 'fixtures', 'golden.json'), 'utf8'));

const vocabulary = JSON.parse(
  readFileSync(join(root, 'data', 'web', 'vocab.json'), 'utf8'));

const vocabIndex = new Map();
vocabulary.ids.forEach((id, position) => vocabIndex.set(id, position + 2));

const embeddingBytes = readFileSync(
  join(root, 'data', 'web', 'dysbiosis_embed.f16.bin'));
const embedding = halfToFloat(new Uint16Array(
  embeddingBytes.buffer, embeddingBytes.byteOffset,
  embeddingBytes.byteLength / 2));

const session = await ort.InferenceSession.create(
  join(root, 'data', 'web', 'dysbiosis_encoder.onnx'));

const { d_model: dModel, num_steps: numSteps } = vocabulary;

function gather(features) {
  const out = new Float32Array(features.length * dModel);
  for (let i = 0; i < features.length; i += 1) {
    out.set(embedding.subarray(features[i] * dModel,
                               (features[i] + 1) * dModel), i * dModel);
  }
  return out;
}

const exampleCount = fixture.samples
  .filter((sample) => sample.name.startsWith('example/')).length;
console.log(`${fixture.samples.length} samples: `
  + `${fixture.samples.length - exampleCount} reference-cohort picks and `
  + `${exampleCount} one-click examples`);

let worstLogit = 0;
let worstAbundance = 0;
const deepSamples = [];

for (const sample of fixture.samples) {
  const ids = Object.keys(sample.counts);
  const counts = ids.map((id) => sample.counts[id]);

  // The browser ranks over the sample's own feature space, so the ids are
  // passed in as they arrive and the vocabulary is applied afterwards.
  const dense = rankNormalizeSample(ids.length, ids.map((_, i) => i), counts);
  const encoded = encodeSample(dense, numSteps, vocabIndex, ids);

  const expected = sample.expected;

  let maskMismatches = 0;
  for (let i = 0; i < numSteps; i += 1) {
    if (encoded.mask[i] !== expected.mask[i]) maskMismatches += 1;
  }
  check(maskMismatches === 0,
    `${sample.name}: ${maskMismatches} mask positions differ`);

  const deep = sample.n_otus > numSteps;

  if (!deep) {
    // Every non-zero taxon fits, so the same taxa must come out in the same
    // places. A single differing position is a bug.
    let featureMismatches = 0;
    let firstMismatch = -1;
    for (let i = 0; i < numSteps; i += 1) {
      if (encoded.features[i] !== expected.features[i]) {
        featureMismatches += 1;
        if (firstMismatch < 0) firstMismatch = i;
      }
    }
    check(featureMismatches === 0,
      `${sample.name}: ${featureMismatches} of ${numSteps} token positions `
      + `differ, first at ${firstMismatch}`);
  } else {
    // More taxa than positions, so some are dropped, and both the order and
    // the membership of the tied blocks are arbitrary. Comparing position by
    // position is meaningless here -- the model has no positional encoding, so
    // a permutation of equal-abundance taxa is the same input. Compare what
    // actually matters instead:
    //
    //   1. the abundances chosen are the same multiset, so the ranking agrees
    //      even where the tie-break does not;
    //   2. the selection really is the top `numSteps`: no taxon left out has
    //      a higher abundance than one taken. This is the property the whole
    //      step exists to have, and unlike the tie order it is checkable.
    const chosen = [];
    const left = [];
    for (let k = 0; k < numSteps; k += 1) {
      if (encoded.mask[k] === 1) chosen.push(encoded.abundance[k]);
      else if (encoded.abundance[k] === 0) break;
    }
    const takenIndices = new Set();
    for (let k = 0; k < numSteps; k += 1) {
      if (encoded.mask[k] === 1) takenIndices.add(encoded.features[k]);
    }
    for (let k = 0; k < numSteps; k += 1) {
      if (encoded.mask[k] === 1 || encoded.abundance[k] === 0) continue;
      left.push(encoded.abundance[k]);
    }

    const mine = chosen.slice().sort((a, b) => b - a);
    const theirs = expected.abundance
      .filter((_, i) => expected.mask[i] === 1).sort((a, b) => b - a);
    check(mine.length === theirs.length,
      `${sample.name}: selected ${mine.length} taxa against Python's `
      + `${theirs.length}`);

    let abundanceMismatches = 0;
    for (let i = 0; i < Math.min(mine.length, theirs.length); i += 1) {
      if (Math.abs(mine[i] - theirs[i]) > ARRAY_TOLERANCE) abundanceMismatches += 1;
    }
    check(abundanceMismatches === 0,
      `${sample.name}: the selected abundances differ as a multiset at `
      + `${abundanceMismatches} positions, so the ranking itself is wrong`);

    // Independent of Python, and of the tie order: the smallest abundance
    // taken must be at least the largest abundance left behind.
    if (left.length) {
      const lowestTaken = Math.min(...chosen);
      const highestLeft = Math.max(...left);
      check(lowestTaken >= highestLeft - ARRAY_TOLERANCE,
        `${sample.name}: kept a taxon at abundance `
        + `${lowestTaken.toFixed(7)} while dropping one at `
        + `${highestLeft.toFixed(7)}, so this is not the top ${numSteps}`);
    }

    const differing = expected.features
      .filter((feature, i) => expected.mask[i] === 1
        && !takenIndices.has(feature)).length;
    if (differing) {
      console.log(`  note ${sample.name}: ${sample.n_otus} taxa, ${differing} `
        + `of the ${numSteps} drawn from elsewhere in the tied block`);
    }
  }

  let abundanceDifference = 0;
  for (let i = 0; i < numSteps; i += 1) {
    abundanceDifference = Math.max(abundanceDifference,
      Math.abs(encoded.abundance[i] - expected.abundance[i]));
  }
  worstAbundance = Math.max(worstAbundance, abundanceDifference);
  check(abundanceDifference <= ARRAY_TOLERANCE,
    `${sample.name}: abundances differ by ${abundanceDifference.toExponential(2)}`);

  // Sanity on the encoding itself, independent of the fixture: padding must
  // be zero-abundance and masked, and never mistaken for a real token.
  for (let i = 0; i < numSteps; i += 1) {
    if (encoded.features[i] === PAD_INDEX) {
      check(encoded.mask[i] === 0 && encoded.abundance[i] === 0,
        `${sample.name}: position ${i} is padding but is not zeroed and masked`);
      break;
    }
  }
  for (let i = 0; i < numSteps; i += 1) {
    if (encoded.features[i] === UNK_INDEX) {
      check(encoded.mask[i] === 0,
        `${sample.name}: position ${i} is out of vocabulary but not masked`);
      break;
    }
  }

  const inputs = gather(encoded.features);
  const mask = new BigInt64Array(numSteps);
  for (let i = 0; i < numSteps; i += 1) mask[i] = BigInt(encoded.mask[i]);

  const outputs = await session.run({
    inputs: new ort.Tensor('float32', inputs, [1, numSteps, dModel]),
    weight: new ort.Tensor('float32', encoded.abundance, [1, numSteps]),
    mask: new ort.Tensor('int64', mask, [1, numSteps]),
  });

  const logit = outputs.logit.data[0];
  const difference = Math.abs(logit - expected.logit);
  worstLogit = Math.max(worstLogit, difference);

  if (deep) deepSamples.push(sample.name);
  const tolerance = deep ? DEEP_LOGIT_TOLERANCE : LOGIT_TOLERANCE;
  check(difference <= tolerance,
    `${sample.name}: logit ${logit.toFixed(6)} against Python's `
    + `${expected.logit.toFixed(6)}, a difference of `
    + `${difference.toExponential(2)}, over the `
    + `${tolerance.toExponential(0)} allowed for `
    + `${deep ? 'a sample deeper than numSteps' : 'a sample that fits'}`);
}

console.log(`\nchecks: ${checks}, failures: ${failures}`);
console.log(`worst abundance difference: ${worstAbundance.toExponential(2)}`);
console.log(`worst logit difference:     ${worstLogit.toExponential(2)}`);
console.log(`tolerance:                  ${LOGIT_TOLERANCE.toExponential(0)} `
  + `(${DEEP_LOGIT_TOLERANCE.toExponential(0)} for samples deeper than `
  + `${numSteps})`);
if (deepSamples.length) {
  console.log(`deeper than ${numSteps} taxa: ${deepSamples.join(', ')}`);
}

if (failures) {
  console.error('\nThe browser path does not reproduce the Python pipeline.');
  process.exit(1);
}
console.log('\nThe browser path reproduces the Python pipeline.');
