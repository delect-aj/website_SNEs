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
 * Three input routes, in decreasing order of how often they will be used:
 * a ready-made example, an OTU table already keyed by SILVA 138.2 ids, and
 * rep-seqs plus counts. Only the last one touches the network, and only to
 * translate sequences into OTU ids.
 */

import { fetchWithProgress, halfToFloat } from './binary.js';
import { element } from './dom.js';
import { preprocessSample, UNK_INDEX } from './preprocess.js';
import { makeGather, scoreSample, percentileOf, topContributors } from './inference.js';
import { renderBand } from './band.js';

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
  loaded: false,
};

const elements = {
  modes: document.getElementById('modes'),
  example: document.getElementById('mode-example'),
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
  elements.example.hidden = mode !== 'example';
  elements.table.hidden = mode !== 'table';
  elements.fasta.hidden = mode !== 'fasta';
  elements.runStatus.replaceChildren();
}

/** Turn a feature-by-sample grid into one sparse map per sample. */
function samplesFromGrid(rows, header, firstColumn) {
  const names = header.slice(firstColumn).map(String);
  const samples = names.map((name) => ({ name, otuCounts: new Map() }));

  for (const row of rows) {
    const id = String(row[0] ?? '').trim();
    if (!id) continue;
    for (let column = firstColumn; column < row.length; column += 1) {
      const value = Number(row[column]);
      if (!Number.isFinite(value) || value <= 0) continue;
      const sample = samples[column - firstColumn];
      sample.otuCounts.set(id, (sample.otuCounts.get(id) || 0) + value);
    }
  }
  return samples.filter((sample) => sample.otuCounts.size > 0);
}

/**
 * Read a count table in either orientation.
 *
 * QIIME2 and DADA2 both write features down the side. A table written the
 * other way round is detected by checking whether the header cells reappear
 * among the row labels.
 */
function readTable(text, label) {
  const parsed = Papa.parse(text.trim(), { skipEmptyLines: true });
  if (parsed.errors.length && parsed.errors[0].type === 'Delimiter') {
    throw new Error(`${label}: could not find a delimiter; a tab-separated `
      + `file is expected.`);
  }
  const rows = parsed.data;
  if (rows.length < 2) throw new Error(`${label}: no data rows`);

  const header = rows[0].map(String);
  const body = rows.slice(1);

  const rowLabels = new Set(body.map((row) => String(row[0])));
  const headerOverlap = header.slice(1)
    .filter((cell) => rowLabels.has(cell)).length;
  const transposed = headerOverlap >= Math.max(1, (header.length - 1) / 2);

  if (!transposed) {
    return samplesFromGrid(body, header, 1);
  }

  // Samples down the side: flip into feature rows first.
  const featureIds = header.slice(1);
  const sampleNames = body.map((row) => String(row[0]));
  const flipped = featureIds.map((id, column) =>
    [id, ...body.map((row) => row[column + 1])]);
  return samplesFromGrid(flipped, ['feature', ...sampleNames], 1);
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
  label.textContent = `${samples.length} samples found`;
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

async function ensureModel() {
  if (state.loaded) return;

  status('Loading the embedding table…', 0);
  const embeddingBuffer = await fetchWithProgress(`${DATA}/dysbiosis_embed.f16.bin`,
    (loaded, size) => status('Loading the embedding table…', loaded / size));
  state.embedding = halfToFloat(new Uint16Array(embeddingBuffer));
  state.gather = makeGather(state.embedding, state.vocab.d_model);

  status('Loading the model…', 0);
  const modelBuffer = await fetchWithProgress(`${DATA}/dysbiosis_encoder.onnx`,
    (loaded, size) => status('Loading the model…', loaded / size));

  ort.env.wasm.wasmPaths = WASM_PATH;
  // One thread: the multi-threaded build wants COOP/COEP headers, and a
  // 600x100 encoder does not need them.
  ort.env.wasm.numThreads = 1;
  state.session = await ort.InferenceSession.create(modelBuffer,
    { executionProviders: ['wasm'] });
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
      `None of the ${unknown} taxa in this sample are in the model's `
      + `vocabulary, so every position was masked and there is nothing to `
      + `score. The model reads SILVA 138.2 97% OTU ids of the form `
      + `accession.start.stop; a table of ASV or exact-sequence ids has to go `
      + `through the FASTA route first, which maps them.`);
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
    line.textContent = `Only ${known} of this sample's ${sample.nOtus} taxa `
      + `are in the model's vocabulary. The score below comes from those `
      + `${known}; the rest were masked out. Treat it as indicative at best.`;
    warning.appendChild(line);
    fragment.appendChild(warning);
  }

  const sentence = element('p', 'result__sentence');
  sentence.style.margin = '0 0 4px';
  sentence.appendChild(document.createTextNode('This sample sits at the '));
  const number = element('span', 'result__number');
  number.textContent = String(Math.round(percentile));
  sentence.appendChild(number);
  sentence.appendChild(document.createTextNode(
    `${ordinalSuffix(percentile)} percentile of the reference cohort of `
    + `${all.length} samples.`));
  fragment.appendChild(sentence);

  // The two groups are not the same size and do not sit in the same place, so
  // where their medians fall is what makes a bare percentile readable.
  // Without it, "60th" sounds worse than it is.
  const controlMedian = percentileOf(all, controls[Math.floor(controls.length / 2)]);
  const caseMedian = percentileOf(all, cases[Math.floor(cases.length / 2)]);
  const sub = element('p', 'small muted');
  sub.textContent = `${Math.round(percentile)}% of reference samples score `
    + `below this one. The cohort is ${controls.length} controls and `
    + `${cases.length} cases; their medians sit at the `
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
    className: 'the model score',
    ctrlLabel: 'reference controls',
    caseLabel: 'reference cases',
    youLabel: 'this sample',
    domain: [low - pad, high + pad],
    ticks: [all[0], all[Math.floor(all.length / 2)], all[all.length - 1]],
    axisFormat: '.1f',
    axisLabel: 'model score (logit) → more case-like',
    width: 720,
  });

  const heading = element('h3');
  heading.textContent = 'Taxa this score leaned on';
  heading.style.margin = '24px 0 4px';
  fragment.appendChild(heading);

  const note = element('p', 'small muted');
  note.textContent = 'Attention averaged over folds, heads and query '
    + 'positions, so a taxon ranks high when the sample as a whole attended '
    + 'to it. This describes the model, not a biological mechanism.';
  fragment.appendChild(note);

  const list = element('ol', 'attn-list');
  for (const contributor of topContributors(attention, sample, 12)) {
    const item = element('li');
    const id = state.vocab.ids[contributor.index - 2];
    if (id) {
      const link = element('a');
      link.href = `/atlas/?otu=${encodeURIComponent(id)}`;
      link.textContent = id;
      link.className = 'mono';
      item.appendChild(link);
    } else {
      const unknownNode = element('span', 'mono muted');
      unknownNode.textContent = '<unk>';
      item.appendChild(unknownNode);
    }
    item.appendChild(document.createTextNode(
      `abundance ${contributor.abundance.toFixed(2)}`));
    const weight = element('span', 'weight');
    weight.textContent = `attention ${contributor.weight.toFixed(4)}`;
    item.appendChild(weight);
    list.appendChild(item);
  }
  fragment.appendChild(list);

  const coverage = element('p', 'small muted');
  coverage.style.marginTop = '16px';
  coverage.textContent = `${sample.nOtus} non-zero taxa in this sample; the `
    + `model reads at most ${state.vocab.num_steps} of them.`;
  if (unknown > 0) {
    coverage.textContent += ` ${unknown} of the positions the model read are `
      + `OTUs absent from its vocabulary, which the attention mask hides — `
      + `their abundance still counted towards the ranking but contributed no `
      + `embedding.`;
  }
  fragment.appendChild(coverage);

  const disclaimer = element('p', 'disclaimer');
  disclaimer.style.marginTop = '20px';
  disclaimer.textContent = 'Research use only. Not a medical device, not a '
    + 'diagnosis, and not a probability of disease. This is one model\'s '
    + 'relative placement of one sample; gut communities vary more within a '
    + 'person over a month than this score measures between people.';
  fragment.appendChild(disclaimer);

  elements.result.replaceChildren(fragment);
  elements.resultSection.hidden = false;
  elements.resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* -------------------------------------------------------------------- flow */

async function loadStatics() {
  const [vocab, metrics, refScores, examples] = await Promise.all([
    fetch(`${DATA}/vocab.json`).then((r) => r.json()),
    fetch(`${DATA}/metrics.json`).then((r) => r.json()),
    fetch(`${DATA}/ref_scores.json`).then((r) => r.json()),
    fetch(`${DATA}/examples.json`).then((r) => r.json()),
  ]);
  state.vocab = vocab;
  state.metrics = metrics;
  state.refScores = refScores;
  state.vocab.index = new Map();
  vocab.ids.forEach((id, position) => state.vocab.index.set(id, position + 2));

  state.examples = examples;
  for (const record of examples) {
    const label = element('label');
    const radio = element('input');
    radio.type = 'radio';
    radio.name = 'example';
    radio.value = record.file;
    const span = element('span');
    span.textContent = record.label;
    label.appendChild(radio);
    label.appendChild(span);
    elements.examples.appendChild(label);
  }
  if (examples.length) elements.examples.querySelector('input').checked = true;
  elements.examples.addEventListener('change', () => { state.selection = 0; });

  elements.runNote.textContent = `${vocab.ids.length} OTUs in the vocabulary, `
    + `${metrics.n_informative_otus} with a trained embedding.`;
}

async function collectSamples() {
  const mode = currentMode();

  if (mode === 'example') {
    const picked = elements.examples.querySelector('input:checked');
    if (!picked) throw new Error('Pick an example first.');
    const record = await (await fetch(`${DATA}/examples/${picked.value}`)).json();
    return [{
      name: `${record.label} (${record.sample_id})`,
      otuCounts: new Map(Object.entries(record.counts)),
    }];
  }

  if (mode === 'table') {
    const file = elements.tableFile.files[0];
    if (!file) throw new Error('Choose a table file first.');
    const text = await file.text();
    if (text.slice(0, 4) === '\x89HDF') {
      throw new Error('That is an HDF5 BIOM file, which this page cannot read '
        + 'in the browser. Convert it to TSV first '
        + '(biom convert -i table.biom -o table.tsv --to-tsv), or use the '
        + 'FASTA route, which the server maps for you.');
    }
    if (text.trim().startsWith('{')) {
      throw new Error('That looks like classic JSON BIOM. Export it as TSV '
        + 'instead (biom convert).');
    }
    const samples = readTable(text, file.name);
    if (!samples.length) throw new Error(`${file.name}: no non-zero counts.`);
    return samples;
  }

  const fasta = elements.fastaFile.files[0];
  const countsFile = elements.countsFile.files[0];
  if (!fasta || !countsFile) {
    throw new Error('This route needs both the FASTA and the count table.');
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
  if (!samples.length) throw new Error(`${countsFile.name}: no non-zero counts.`);

  for (const sample of samples) {
    const translated = new Map();
    for (const [id, value] of sample.otuCounts) {
      const otu = mapping[id] || id;
      translated.set(otu, (translated.get(otu) || 0) + value);
    }
    sample.otuCounts = translated;
  }
  const note = element('p', 'small muted');
  note.textContent = `${payload.mapped} of ${payload.total} sequences mapped `
    + `to reference OTUs at 97% identity. Unmapped sequences are kept and `
    + `masked, the same way a held-out cohort was.`;
  elements.fastaSamples.appendChild(note);

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
    if (currentMode() !== 'example') {
      renderSamplePicker(target, samples, (index) => { state.selection = index; });
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
