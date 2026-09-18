/**
 * The dysbiosis page.
 *
 * The model runs here, not on the server. That is a deliberate choice with
 * three consequences the page has to be honest about: the visitor pays the
 * download once, the score never leaves their machine, and the numbers shown
 * are only as trustworthy as the preprocessing below, which is why
 * `preprocess.js` is pinned against Python by a regression test rather than
 * written from the description.
 *
 * Two input routes: an OTU table already keyed by SILVA 138.2 ids, and
 * rep-seqs plus counts. Only the second touches the network, and only to
 * translate sequences into OTU ids. The reference-cohort examples are written
 * into the table route's file input, so they are scored as an upload.
 */

import { fetchWithProgress, halfToFloat } from './binary.js?v=5';
import { element } from './dom.js?v=5';
import { preprocessSample, UNK_INDEX } from './preprocess.js?v=5';
import { makeGather, scoreSample, percentileOf, topContributors } from './inference.js?v=5';
import { renderBand } from './band.js?v=5';
import { readTable } from './table.js?v=5';

const DATA = '/data';
const WASM_PATH = '/assets/vendor/ort/';

const state = {
  vocab: null,
  metrics: null,
  refScores: null,
  embedding: null,
  session: null,
  gather: null,
  examples: [],
  sampleList: [],
  selection: 0,
  mappingNote: '',
  loaded: false,
};

const elements = {
  modes: document.getElementById('modes'),
  table: document.getElementById('mode-table'),
  fasta: document.getElementById('mode-fasta'),
  examples: document.getElementById('examples'),
  tableFile: document.getElementById('table-file'),
  tableSamples: document.getElementById('table-samples'),
  fastaFile: document.getElementById('fasta-file'),
  countsFile: document.getElementById('counts-file'),
  fastaSamples: document.getElementById('fasta-samples'),
  run: document.getElementById('run'),
  runNote: document.getElementById('run-note'),
  runStatus: document.getElementById('run-status'),
  resultSection: document.getElementById('result-section'),
  result: document.getElementById('result'),
};

/**
 * Put one message in the run area, optionally with a progress bar.
 *
 * `kind` picks the frame: a plain card while something is running, a warn box
 * when it failed. Both replace whatever was there, because two messages at
 * once in a single-line status area read as one.
 */
function say(text, { fraction, kind = 'card' } = {}) {
  elements.runStatus.replaceChildren();
  const box = element('div', kind);
  const line = element('p', null, text);
  line.style.margin = fraction === undefined ? '0' : '0 0 8px';
  box.appendChild(line);
  if (fraction !== undefined) {
    const progress = element('div', 'progress');
    const bar = element('div', 'progress__bar');
    bar.style.width = `${Math.round(fraction * 100)}%`;
    progress.appendChild(bar);
    box.appendChild(progress);
  }
  elements.runStatus.appendChild(box);
}

const status = (text, fraction) => say(text, { fraction });
const problem = (message) => say(message, { kind: 'warn' });

/* ------------------------------------------------------------------ inputs */

function currentMode() {
  return elements.modes.querySelector('input:checked').value;
}

function showMode() {
  const mode = currentMode();
  elements.table.hidden = mode !== 'table';
  elements.fasta.hidden = mode !== 'fasta';
  elements.runStatus.replaceChildren();
}

function renderSamplePicker(container, samples, onPick) {
  container.replaceChildren();
  if (!samples.length) return;
  if (samples.length === 1) {
    const note = element('p', 'small muted');
    note.textContent = `${samples[0].name} — ${samples[0].otuCounts.size} taxa`;
    container.appendChild(note);
    return;
  }
  const label = element('label');
  label.textContent = `${samples.length} samples detected`;
  const select = element('select');
  select.style.marginLeft = '8px';
  samples.forEach((sample, index) => {
    const option = element('option');
    option.value = String(index);
    option.textContent = `${sample.name} (${sample.otuCounts.size} taxa)`;
    // Rebuilding the picker between runs must not silently move the visitor
    // back to the first sample: the run buttons re-render it, and a select
    // that forgets its choice scores the wrong sample with no visible change.
    option.selected = index === state.selection;
    select.appendChild(option);
  });
  select.addEventListener('change', () => onPick(Number(select.value)));
  label.appendChild(select);
  container.appendChild(label);
}

/* --------------------------------------------------------------- inference */

/**
 * Make sure the onnxruntime global is there, fetching it once more if not.
 *
 * The runtime is a plain script tag in the page head, so a request that fails
 * -- a dropped connection, a redeploy mid-load, an extension that blocks
 * bundled scripts -- leaves `ort` undefined and every later line throwing
 * "ort is not defined". One retry fixes the transient case; the rest get a
 * message that says what failed instead of a bare ReferenceError.
 */
async function ensureRuntime() {
  if (typeof ort !== 'undefined') return;
  status('Loading the inference runtime…', 0);
  await new Promise((resolve) => {
    const script = document.createElement('script');
    script.src = `${WASM_PATH}ort.min.js`;
    script.onload = resolve;
    script.onerror = resolve;
    document.head.appendChild(script);
  });
  if (typeof ort === 'undefined') {
    throw new Error('The inference runtime (onnxruntime-web) could not be '
      + 'loaded from this site, so the model cannot run. Reload the page; if '
      + 'that does not help, a browser extension or a network filter is '
      + 'likely blocking /assets/vendor/ort/ort.min.js.');
  }
}

async function ensureModel() {
  if (state.loaded) return;
  await ensureRuntime();

  status('Loading embedding table…', 0);
  const embeddingBuffer = await fetchWithProgress(`${DATA}/dysbiosis_embed.f16.bin`,
    (loaded, size) => status('Loading embedding table…', loaded / size));
  state.embedding = halfToFloat(new Uint16Array(embeddingBuffer));
  state.gather = makeGather(state.embedding, state.vocab.d_model);
  // Vocabulary rows with a non-zero embedding. About 5,000 OTUs are in the
  // model's vocabulary but not in the pretrained embedding; their rows are
  // zero, the model knows nothing about them, and none has a SILVA 138.2
  // lineage, so the attention list does not name them.
  const d = state.vocab.d_model;
  state.informative = new Uint8Array(state.embedding.length / d);
  for (let row = 0; row < state.informative.length; row += 1) {
    for (let j = 0; j < d; j += 1) {
      if (state.embedding[row * d + j] !== 0) { state.informative[row] = 1; break; }
    }
  }

  // Lineages for the contributing taxa, keyed by OTU id.
  state.taxonomy = await (await fetch(`${DATA}/taxonomy.json`, { cache: 'no-cache' })).json();

  status('Loading model…', 0);
  const modelBuffer = await fetchWithProgress(`${DATA}/dysbiosis_encoder.onnx`,
    (loaded, size) => status('Loading model…', loaded / size));

  ort.env.wasm.wasmPaths = WASM_PATH;
  // One thread: the multi-threaded build wants COOP/COEP headers, and a
  // 600x100 encoder does not need them.
  ort.env.wasm.numThreads = 1;
  try {
    state.session = await ort.InferenceSession.create(modelBuffer,
      { executionProviders: ['wasm'] });
  } catch (error) {
    // The runtime picks its WebAssembly build at this point: ort-wasm-simd.wasm
    // where SIMD is available and ort-wasm.wasm where it is not, both of which
    // this site serves. A failure here is therefore the browser, not a missing
    // file, and the runtime's own message ("no available backend found") does
    // not say so.
    throw new Error('The model could not be started in this browser '
      + `(${error.message}). Scoring runs on WebAssembly, which needs a `
      + 'current version of Chrome, Firefox, Edge or Safari.');
  }
  state.loaded = true;
}

/**
 * Score one sample.
 *
 * Ranking uses every count the sample carries, including OTUs outside the
 * vocabulary: dropping those first would shift every other rank. The
 * vocabulary is applied afterwards, when a feature id becomes an index, and
 * an unrecognised id becomes '<unk>' -- which the mask then hides. That is
 * what the training pipeline did with a held-out cohort.
 */
async function score(otuCounts) {
  const ids = [...otuCounts.keys()];
  const counts = new Float64Array(ids.length);
  const positions = new Int32Array(ids.length);
  for (let i = 0; i < ids.length; i += 1) {
    counts[i] = otuCounts.get(ids[i]);
    positions[i] = i;
  }

  const sample = preprocessSample({
    nFeatures: ids.length,
    featureIds: ids,
    vocabIndex: state.vocab.index,
    numSteps: state.vocab.num_steps,
  }, positions, counts);

  let unknown = 0;
  let known = 0;
  for (let i = 0; i < sample.features.length; i += 1) {
    if (sample.abundance[i] <= 0) continue;
    if (sample.features[i] === UNK_INDEX) unknown += 1;
    else known += 1;
  }

  // Refuse to score a sample the model cannot see. With every position masked,
  // the pooled representation is zero, the head returns its bias, and the page
  // would print a confident percentile derived from nothing. A table keyed by
  // ASV ids instead of SILVA OTU ids gets here, and the message has to say so
  // rather than let the number through.
  if (known === 0) {
    throw new Error(
      `None of the ${unknown} taxa read from this sample (${sample.nOtus} in `
      + `total) are in the model vocabulary, so all positions were masked and `
      + `no score can be computed. The model requires SILVA 138.2 OTU `
      + `identifiers (97% identity) of the form accession.start.stop. Tables `
      + `with ASV or exact-sequence identifiers must be submitted through the `
      + `FASTA option, which maps them to OTUs.`);
  }

  const scored = await scoreSample(state.session, ort, sample,
    state.vocab.num_steps, state.vocab.d_model, state.gather);
  return { sample, unknown, known, ...scored };
}

/* ----------------------------------------------------------------- results */

/** "st", "nd", "rd" or "th" for a whole number. */
function ordinalSuffix(value) {
  const rounded = Math.round(value);
  const tens = rounded % 100;
  if (tens >= 11 && tens <= 13) return 'th';
  return { 1: 'st', 2: 'nd', 3: 'rd' }[rounded % 10] || 'th';
}

function renderResult(scored) {
  const { logit, sample, attention, unknown, known } = scored;
  const controls = state.refScores.controls;
  const cases = state.refScores.cases;
  const all = controls.concat(cases).slice().sort((a, b) => a - b);
  const percentile = percentileOf(all, logit);

  const fragment = document.createDocumentFragment();

  // A score built from a handful of recognised taxa is not the same claim as
  // one built from the whole community, and the difference is invisible in the
  // percentile. Say it before the number, not after it.
  if (known < sample.nOtus * 0.5) {
    const warning = element('div', 'warn');
    warning.style.marginBottom = '16px';
    const line = element('p');
    line.style.margin = '0';
    line.textContent = `Only ${known} of the ${sample.nOtus} taxa in this `
      + `sample are in the model vocabulary. The score below is based on these `
      + `${known} taxa; the remainder were masked. Interpret the score with `
      + `caution.`;
    warning.appendChild(line);
    fragment.appendChild(warning);
  }

  const sentence = element('p', 'result__sentence');
  sentence.style.margin = '0 0 4px';
  sentence.appendChild(document.createTextNode('This sample is at the '));
  const number = element('span', 'result__number');
  number.textContent = String(Math.round(percentile));
  sentence.appendChild(number);
  sentence.appendChild(document.createTextNode(
    `${ordinalSuffix(percentile)} percentile of the reference cohort `
    + `(n = ${all.length}).`));
  fragment.appendChild(sentence);

  // The two groups are not the same size and do not sit in the same place, so
  // where their medians fall is what makes a bare percentile readable.
  // Without it, "60th" sounds worse than it is.
  const controlMedian = percentileOf(all, controls[Math.floor(controls.length / 2)]);
  const caseMedian = percentileOf(all, cases[Math.floor(cases.length / 2)]);
  const sub = element('p', 'small muted');
  sub.textContent = `${Math.round(percentile)}% of reference samples have `
    + `a lower score. The cohort comprises ${controls.length} controls and `
    + `${cases.length} cases, with medians at the `
    + `${Math.round(controlMedian)}${ordinalSuffix(controlMedian)} and `
    + `${Math.round(caseMedian)}${ordinalSuffix(caseMedian)} percentile.`;
  fragment.appendChild(sub);

  const plot = element('div', 'band');
  plot.style.marginTop = '20px';
  fragment.appendChild(plot);

  const low = Math.min(all[0], logit);
  const high = Math.max(all[all.length - 1], logit);
  const pad = (high - low) * 0.04 || 1;
  renderBand(plot, {
    ctrl: controls,
    case: cases,
    you: logit,
    className: 'model scores',
    ctrlLabel: 'reference controls',
    caseLabel: 'reference cases',
    youLabel: 'this sample',
    domain: [low - pad, high + pad],
    ticks: [all[0], all[Math.floor(all.length / 2)], all[all.length - 1]],
    axisFormat: '.1f',
    axisLabel: 'Model score (logit) → more case-like',
    width: 720,
  });

  const heading = element('h3');
  heading.textContent = 'Taxa with the highest attention weights';
  heading.style.margin = '24px 0 4px';
  fragment.appendChild(heading);

  const note = element('p', 'small muted');
  note.textContent = 'Attention weights are averaged over folds, heads and '
    + 'query positions. They describe what the model attends to and do not '
    + 'imply a biological mechanism. OTUs without a pretrained embedding carry '
    + 'no information for the model and are not listed.';
  fragment.appendChild(note);

  const list = element('ol', 'attn-list');
  const listed = topContributors(attention, sample, Infinity)
    .filter((contributor) => state.informative[contributor.index])
    .slice(0, 12);
  for (const contributor of listed) {
    const item = element('li');
    // topContributors only returns positions the mask kept, so this is always
    // a real vocabulary index and the id is always present.
    const id = state.vocab.ids[contributor.index - 2];
    const taxon = element('div', 'attn-list__taxon');
    if (id) {
      const link = element('a');
      link.href = `/atlas/?otu=${encodeURIComponent(id)}`;
      link.textContent = id;
      link.className = 'mono';
      taxon.appendChild(link);
      const lineage = state.taxonomy[id];
      taxon.appendChild(element('div', 'lineage small muted', lineage
        ? lineage.split(';').filter(Boolean).join(' › ')
        : 'taxonomy not available in SILVA 138.2'));
    } else {
      taxon.appendChild(element('span', 'mono muted',
        `vocabulary index ${contributor.index}`));
    }
    item.appendChild(taxon);
    item.appendChild(element('span', 'attn-list__abundance',
      `percentile abundance ${contributor.abundance.toFixed(2)}`));
    const weight = element('span', 'weight');
    weight.textContent = `attention weight ${contributor.weight.toFixed(4)}`;
    item.appendChild(weight);
    list.appendChild(item);
  }
  fragment.appendChild(list);

  const coverage = element('p', 'small muted');
  coverage.style.marginTop = '16px';
  coverage.textContent = `This sample contains ${sample.nOtus} taxa with `
    + `non-zero counts; the model uses at most ${state.vocab.num_steps}.`;
  if (unknown > 0) {
    coverage.textContent += ` ${unknown} of these positions are OTUs not `
      + `in the vocabulary. They are masked: their abundances contribute to `
      + `rank normalization but not to the embedding.`;
  }
  fragment.appendChild(coverage);

  const disclaimer = element('p', 'disclaimer');
  disclaimer.style.marginTop = '20px';
  disclaimer.textContent = 'For research use only. This score is not a '
    + 'diagnosis or a probability of disease, and the tool is not a medical '
    + 'device. It is the relative position of one sample under one model; '
    + 'within-individual variation over a month can exceed the between-'
    + 'individual differences the score measures.';
  fragment.appendChild(disclaimer);

  elements.result.replaceChildren(fragment);
  elements.resultSection.hidden = false;
  elements.resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* -------------------------------------------------------------------- flow */

async function loadStatics() {
  const [vocab, metrics, refScores, examples] = await Promise.all([
    fetch(`${DATA}/vocab.json`, { cache: 'no-cache' }).then((r) => r.json()),
    fetch(`${DATA}/metrics.json`, { cache: 'no-cache' }).then((r) => r.json()),
    fetch(`${DATA}/ref_scores.json`, { cache: 'no-cache' }).then((r) => r.json()),
    fetch(`${DATA}/examples.json`, { cache: 'no-cache' }).then((r) => r.json()),
  ]);
  state.vocab = vocab;
  state.metrics = metrics;
  state.refScores = refScores;
  state.vocab.index = new Map();
  vocab.ids.forEach((id, position) => state.vocab.index.set(id, position + 2));

  state.examples = examples;
  examples.forEach((record, index) => {
    if (index) elements.examples.appendChild(document.createTextNode(' · '));
    const button = element('button', 'link', `${record.label} (${record.sample_id})`);
    button.type = 'button';
    button.addEventListener('click', () => loadExample(record));
    elements.examples.appendChild(button);
  });

  elements.runNote.textContent = `Vocabulary: ${vocab.ids.length} OTUs, `
    + `${metrics.n_informative_otus} with a trained embedding.`;
}

/**
 * Put one reference-cohort sample into the upload box, as a table file.
 *
 * The counts are written out in the same TSV the page asks visitors to upload
 * and handed to the file input, so an example takes the upload's code path
 * rather than one of its own: what it scores is what a downloaded and
 * re-uploaded table would score.
 */
async function loadExample(record) {
  try {
    const data = await (await fetch(`${DATA}/examples/${record.file}`,
      { cache: 'no-cache' })).json();
    const column = `${data.sample_id}_${record.label.replace(/ /g, '_')}`;
    const rows = [`#OTU ID\t${column}`];
    for (const [otu, count] of Object.entries(data.counts)) rows.push(`${otu}\t${count}`);
    const file = new File([`${rows.join('\n')}\n`], `${column}.tsv`,
      { type: 'text/tab-separated-values' });
    const transfer = new DataTransfer();
    transfer.items.add(file);
    elements.tableFile.files = transfer.files;
    state.selection = 0;
    elements.tableFile.dispatchEvent(new Event('change'));
  } catch (error) {
    problem(`The example could not be loaded: ${error.message}`);
  }
}

async function collectSamples() {
  const mode = currentMode();
  state.mappingNote = '';

  if (mode === 'table') {
    const file = elements.tableFile.files[0];
    if (!file) throw new Error('Please select a table file.');
    const text = await file.text();
    if (text.slice(0, 4) === '\x89HDF') {
      throw new Error('HDF5 BIOM files cannot be read in the browser. Please '
        + 'convert the file to TSV '
        + '(biom convert -i table.biom -o table.tsv --to-tsv), or use the '
        + 'FASTA option.');
    }
    if (text.trim().startsWith('{')) {
      throw new Error('This appears to be a JSON BIOM file. Please convert '
        + 'it to TSV (biom convert).');
    }
    const samples = readTable(text, file.name);
    if (!samples.length) throw new Error(`${file.name}: no non-zero counts found.`);
    return samples;
  }

  const fasta = elements.fastaFile.files[0];
  const countsFile = elements.countsFile.files[0];
  if (!fasta || !countsFile) {
    throw new Error('This option requires both a FASTA file and a count table.');
  }

  status('Mapping sequences to reference OTUs…');
  const body = new FormData();
  body.append('rep_seqs', fasta);
  const response = await fetch('/map', { method: 'POST', body });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail || `the mapping service returned ${response.status}`;
    const ratio = payload.total
      ? ` — ${payload.mapped} of ${payload.total} sequences mapped` : '';
    throw new Error(`${detail}${ratio}`);
  }

  const mapping = payload.mapping || {};
  const samples = readTable(await countsFile.text(), countsFile.name);
  if (!samples.length) throw new Error(`${countsFile.name}: no non-zero counts found.`);

  for (const sample of samples) {
    const translated = new Map();
    for (const [id, value] of sample.otuCounts) {
      const otu = mapping[id] || id;
      translated.set(otu, (translated.get(otu) || 0) + value);
    }
    sample.otuCounts = translated;
  }
  // Rendered by run() once the sample picker is in place: that picker clears
  // its container, so appending here would be wiped a moment later and the
  // mapping rate -- the one number that says whether the FASTA was comparable
  // to the cohort at all -- would never be seen.
  state.mappingNote = `${payload.mapped} of ${payload.total} sequences were `
    + `mapped to reference OTUs at 97% identity. Unmapped sequences are `
    + `retained and masked, as in the held-out cohort analysis.`;

  return samples;
}

async function run() {
  elements.run.disabled = true;
  elements.resultSection.hidden = true;
  try {
    const samples = await collectSamples();
    state.sampleList = samples;
    // Keep the visitor's choice across runs; only fall back when the new file
    // has fewer samples than the old one.
    if (state.selection >= samples.length) state.selection = 0;

    const target = currentMode() === 'table' ? elements.tableSamples
      : elements.fastaSamples;
    renderSamplePicker(target, samples, (index) => { state.selection = index; });
    if (state.mappingNote) {
      const note = element('p', 'small muted');
      note.textContent = state.mappingNote;
      target.appendChild(note);
    }

    await ensureModel();

    const chosen = samples[state.selection] || samples[0];
    status(`Scoring ${chosen.name}…`);
    const scored = await score(chosen.otuCounts);
    elements.runStatus.replaceChildren();
    renderResult(scored);
  } catch (error) {
    problem(error.message || String(error));
    console.error(error);
  } finally {
    elements.run.disabled = false;
  }
}

elements.modes.addEventListener('change', showMode);
elements.run.addEventListener('click', run);

elements.tableFile.addEventListener('change', async () => {
  const file = elements.tableFile.files[0];
  if (!file) return;
  try {
    renderSamplePicker(elements.tableSamples, readTable(await file.text(), file.name),
      (index) => { state.selection = index; });
  } catch (error) {
    problem(error.message);
  }
});

elements.countsFile.addEventListener('change', async () => {
  const file = elements.countsFile.files[0];
  if (!file) return;
  try {
    renderSamplePicker(elements.fastaSamples, readTable(await file.text(), file.name),
      (index) => { state.selection = index; });
  } catch (error) {
    problem(error.message);
  }
});

showMode();
loadStatics().catch((error) => {
  problem(error.message || String(error));
  console.error(error);
});
