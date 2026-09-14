/**
 * The atlas page: load the precomputed arrays, draw the scatter, drive the
 * card.
 *
 * Everything here is array lookup. The UMAP coordinates, both neighbour lists
 * and the trait predictions were computed offline and ship as binaries, so the
 * page does no analysis of its own — which is also why a 1-core server can
 * serve it.
 */

import { loadArray, fetchWithProgress, halfToFloat } from './binary.js';
import { element } from './dom.js';
import { Scatter } from './scatter.js';
import { renderCard, cardSummary } from './card.js';

// Okabe-Ito. Colour is a convenience here, never the only carrier of a
// distinction: the card labels everything in words.
const OKABE_ITO = [
  [0.000, 0.447, 0.698], [0.835, 0.369, 0.000], [0.000, 0.620, 0.451],
  [0.800, 0.475, 0.655], [0.941, 0.894, 0.259], [0.337, 0.706, 0.914],
  [0.902, 0.624, 0.000], [0.600, 0.600, 0.600],
];
const MAX_CATEGORIES = 8;

// A pasted 16S read rather than a name: nucleotide letters only, and longer
// than any genus or OTU id that happens to spell one.
const NUCLEOTIDES = /^[ACGTUN]{20,}$/i;
// The mapping service refuses anything shorter, so say it without a request.
const MIN_READ = 100;
// `ATLAS_MAX_HITS` in server/app/main.py: a list this long may be truncated.
const MAX_HITS = 64;

const status = document.getElementById('status');
const progressBar = document.getElementById('bar');
const progressText = document.getElementById('progress-text');
const panel = document.getElementById('panel');
const legend = document.getElementById('legend');
const results = document.getElementById('results');
const searchInput = document.getElementById('search');
const readsInput = document.getElementById('reads');

let data = null;
let scatter = null;
let currentColors = null;
// Bumped by every search, so a sequence search that answers after the visitor
// has moved on cannot overwrite what they are looking at now.
let searchToken = 0;
let lastRead = '';

function setStatus(title, detail, fraction) {
  status.hidden = false;
  status.querySelector('p').textContent = title;
  progressText.textContent = detail || '';
  progressBar.style.width = fraction !== undefined
    ? `${Math.round(fraction * 100)}%` : '0%';
}

/** Assign each level a colour, folding everything past the eighth into grey. */
function paletteFor(levels) {
  const assignment = new Map();
  levels.forEach((level, index) => {
    assignment.set(String(level),
      index < MAX_CATEGORIES - 1 ? OKABE_ITO[index] : OKABE_ITO[MAX_CATEGORIES - 1]);
  });
  return assignment;
}

function colorsFor(field) {
  const levels = data.meta.color_by[field] || [];
  const palette = paletteFor(levels);
  const fallback = OKABE_ITO[MAX_CATEGORIES - 1];
  const colors = new Float32Array(data.otus.length * 3);
  data.otus.forEach((record, index) => {
    let value;
    if (field === 'phylum') value = record.phylum;
    else value = record.traits[field] ? record.traits[field].value : null;
    const color = palette.get(String(value)) || fallback;
    colors[index * 3] = color[0];
    colors[index * 3 + 1] = color[1];
    colors[index * 3 + 2] = color[2];
  });
  return colors;
}

function renderLegend(field) {
  const levels = (data.meta.color_by[field] || []).slice(0, MAX_CATEGORIES);
  const palette = paletteFor(data.meta.color_by[field] || []);
  legend.replaceChildren();
  const heading = element('div');
  heading.style.marginBottom = '6px';
  heading.textContent = data.meta.color_by[field].length > MAX_CATEGORIES
    ? `${data.meta.color_by[field].length} categories; the smallest are grouped as one colour`
    : `${levels.length} categories`;
  legend.appendChild(heading);

  for (const level of levels) {
    const item = element('span', 'legend__item');
    const swatch = element('span', 'legend__swatch');
    const color = palette.get(String(level)) || OKABE_ITO[MAX_CATEGORIES - 1];
    swatch.style.background =
      `rgb(${color.map((c) => Math.round(c * 255)).join(',')})`;
    item.appendChild(swatch);
    const label = field === 'phylum' ? level
      : (data.traits.traits[field].value_labels[String(level)] || String(level));
    item.appendChild(document.createTextNode(label));
    legend.appendChild(item);
  }
  legend.hidden = false;
}

function select(index, focus) {
  scatter.setSelected(index);
  renderCard(panel, index, data);
  if (focus) scatter.focus(index);
  const summary = cardSummary(index, data);
  history.replaceState(null, '', `?otu=${encodeURIComponent(summary.id)}`);
}

function clearResults() {
  results.replaceChildren();
}

function showNote(text, className = 'small muted') {
  clearResults();
  results.appendChild(element('p', className, text));
}

function displayName(record) {
  return record.species || record.genus || record.family || record.id;
}

function otuButton(record, detail = '') {
  const button = element('button', 'button--quiet',
    `${displayName(record)} — ${record.id}${detail}`);
  button.type = 'button';
  button.style.display = 'block';
  button.style.width = '100%';
  button.style.textAlign = 'left';
  button.addEventListener('click', () => select(record.i, true));
  return button;
}

/** Atlas records for the OTU ids the mapping service returned. */
function recordsFor(ids) {
  return (ids || []).map((id) => data.otus[data.byId.get(id)]).filter(Boolean);
}

/** "3 OTUs, all Bacteroides" -- whether a tie matters for the traits. */
function describeTie(records) {
  const genera = new Set(records.map((record) => record.genus || 'unclassified'));
  const count = records.length >= MAX_HITS ? `${MAX_HITS} or more` : records.length;
  return genera.size === 1
    ? `${count} OTUs, all ${[...genera][0]}`
    : `${count} OTUs across ${genera.size} genera`;
}

/**
 * Send reads to the mapping service's atlas database.
 *
 * @param {File} file - FASTA.
 * @returns {Promise<{hits: Object<string, string[]>,
 *   identity: Object<string, number>, mapped: number, total: number}>}
 * @throws {Error} With the service's own explanation when it has one.
 */
async function mapReads(file) {
  const body = new FormData();
  body.append('rep_seqs', file);
  let response;
  try {
    response = await fetch('/map?db=atlas', { method: 'POST', body });
  } catch {
    throw new Error('The sequence search service is unavailable.');
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(typeof payload.detail === 'string' ? payload.detail
      : `The sequence search service returned an error (${response.status}).`);
  }
  return payload;
}

/** One pasted read: open its OTU, or list the OTUs it cannot tell apart. */
async function searchRead(read, token) {
  lastRead = read;
  if (read.length < MIN_READ) {
    lastRead = '';
    showNote(`The sequence has ${read.length} bases. At least ${MIN_READ} `
      + `bases are required for matching at 97% identity.`);
    return;
  }

  showNote('Searching 14,093 reference sequences…');
  let payload;
  try {
    payload = await mapReads(new File([`>read\n${read}\n`], 'read.fasta'));
  } catch (error) {
    if (token !== searchToken) return;
    lastRead = '';
    showNote(error.message);
    return;
  }
  if (token !== searchToken) return;

  const records = recordsFor(payload.hits.read);
  if (!records.length) {
    showNote('No atlas OTU matches this sequence at 97% identity or higher. '
      + 'The atlas contains only human gut OTUs from SILVA 138.2.');
    return;
  }
  const identity = payload.identity.read.toFixed(1);
  if (records.length === 1) {
    showNote(`Best-matching atlas OTU (${identity}% identity):`);
    results.appendChild(otuButton(records[0]));
    select(records[0].i, true);
    return;
  }
  // Not opened: picking the first of equals would present one of them as the
  // answer, and their trait cards need not agree.
  showNote(`${describeTie(records)} match equally well (${identity}% identity). `
    + 'The sequence is too short to distinguish them; select one to view its details.');
  const list = element('div');
  list.style.maxHeight = '320px';
  list.style.overflowY = 'auto';
  for (const record of records) list.appendChild(otuButton(record));
  results.appendChild(list);
}

/** One row of a FASTA upload's result list. */
function readRow(name, records, identity) {
  const row = element('div');
  row.style.padding = '8px 0';
  row.style.borderTop = '1px solid var(--line)';
  const label = element('div', 'small mono', name || '(unnamed)');
  label.style.overflowWrap = 'anywhere';
  row.appendChild(label);

  if (!records.length) {
    row.appendChild(element('div', 'small muted', 'no atlas OTU at ≥97% identity'));
  } else if (records.length === 1) {
    row.appendChild(otuButton(records[0], ` · ${identity.toFixed(1)}%`));
  } else {
    const details = element('details');
    details.style.padding = '4px 0 0';
    details.style.borderTop = '0';
    details.appendChild(element('summary', 'small',
      `${describeTie(records)}, equal matches · ${identity.toFixed(1)}%`));
    for (const record of records) details.appendChild(otuButton(record));
    row.appendChild(details);
  }
  return row;
}

async function searchFasta(file) {
  const token = ++searchToken;
  lastRead = '';
  showNote(`Searching ${file.name} against 14,093 reference sequences…`);
  // The service answers by header, and leaves out the reads it could not
  // place; the headers are read here so those rows can still be listed.
  const names = (await file.text()).split('\n')
    .filter((line) => line.startsWith('>'))
    .map((line) => line.slice(1).trim().split(/\s+/)[0]);

  let payload;
  try {
    payload = await mapReads(file);
  } catch (error) {
    if (token === searchToken) showNote(error.message);
    return;
  }
  if (token !== searchToken) return;

  showNote(`${payload.mapped} of ${payload.total} sequences matched an atlas `
    + 'OTU at 97% identity or higher.', 'small');
  const list = element('div');
  list.style.maxHeight = '420px';
  list.style.overflowY = 'auto';
  for (const name of names) {
    list.appendChild(readRow(name, recordsFor(payload.hits[name]),
      payload.identity[name]));
  }
  results.appendChild(list);
}

function runSearch(query) {
  const read = query.replace(/\s+/g, '');
  if (NUCLEOTIDES.test(read)) {
    if (read !== lastRead) searchRead(read, ++searchToken);
    return;
  }
  searchToken += 1;
  lastRead = '';
  clearResults();
  const needle = query.trim().toLowerCase();
  if (needle.length < 2) return;

  const matches = [];
  for (const record of data.otus) {
    const haystack = `${record.id} ${record.genus ?? ''} ${record.species ?? ''} `
      + `${record.family ?? ''}`;
    if (haystack.toLowerCase().includes(needle)) {
      matches.push(record);
      if (matches.length >= 12) break;
    }
  }

  if (!matches.length) {
    const empty = element('p', 'small muted', 'No matching OTU. Try a genus name or an OTU identifier.');
    results.appendChild(empty);
    return;
  }

  for (const record of matches) results.appendChild(otuButton(record));
}

async function load() {
  setStatus('Loading manifest…');
  const meta = await (await fetch('/data/meta.json')).json();
  const spec = (name) => meta.arrays[name];

  const track = (label) => (loaded, size) =>
    setStatus(label, `${(loaded / 1048576).toFixed(1)} of `
      + `${(size / 1048576).toFixed(1)} MB`, loaded / size);

  const umap = await loadArray('/data/umap.f32.bin', spec('umap.f32.bin'),
                               track('Loading map coordinates…'));

  setStatus('Loading taxonomy and trait predictions…');
  const otus = await (await fetch('/data/otus.json')).json();
  const traits = await (await fetch('/data/traits.json')).json();
  const probaBuffer = await fetchWithProgress('/data/traits_proba.f16.bin',
                                              track('Loading trait probabilities…'));
  const bacdive = await (await fetch('/data/bacdive.json')).json();

  setStatus('Loading neighbour lists…');
  const loadHalf = async (name) => halfToFloat(new Uint16Array(
    await fetchWithProgress(`/data/${name}`)));
  const nbrSneIdx = await loadArray('/data/nbr_sne_idx.i16.bin',
                                    spec('nbr_sne_idx.i16.bin'));
  const nbrSneSim = await loadHalf('nbr_sne_sim.f16.bin');
  const nbrPhyloIdx = await loadArray('/data/nbr_phylo_idx.i16.bin',
                                      spec('nbr_phylo_idx.i16.bin'));
  const nbrPhyloSim = await loadHalf('nbr_phylo_sim.f16.bin');

  data = {
    meta,
    otus,
    byId: new Map(otus.map((record) => [record.id, record.i])),
    traits,
    // Every table on the wire that is not a raw index is float16: three
    // significant digits is more than a bar chart, a percentile or a cosine
    // printed to three decimals can use.
    traitsProba: halfToFloat(new Uint16Array(probaBuffer)),
    bacdive,
    k: meta.k_neighbours,
    nbrSneIdx: nbrSneIdx.data,
    nbrPhyloIdx: nbrPhyloIdx.data,
    nbrSneSim,
    nbrPhyloSim,
  };

  setStatus('Drawing…');
  scatter = new Scatter(document.getElementById('canvas'), {
    onSelect: (index) => select(index, false),
  });
  scatter.setPoints(umap.data);
  currentColors = colorsFor('phylum');
  scatter.setColors(currentColors);

  const selector = document.getElementById('color-by');
  for (const field of Object.keys(meta.color_by)) {
    const option = element('option');
    option.value = field;
    option.textContent = field === 'phylum' ? 'phylum'
      : (traits.traits[field] ? traits.traits[field].label : field);
    selector.appendChild(option);
  }
  selector.value = 'phylum';
  selector.addEventListener('change', () => {
    currentColors = colorsFor(selector.value);
    scatter.setColors(currentColors);
    renderLegend(selector.value);
  });
  renderLegend('phylum');

  document.getElementById('reset').addEventListener('click', () => {
    scatter.reset();
    history.replaceState(null, '', location.pathname);
  });

  let timer = null;
  searchInput.addEventListener('input', () => {
    clearTimeout(timer);
    // A sequence search spends one of ten requests a minute, so it waits for
    // the paste or the typing to settle; a name search is free.
    const read = NUCLEOTIDES.test(searchInput.value.replace(/\s+/g, ''));
    timer = setTimeout(() => runSearch(searchInput.value), read ? 600 : 120);
  });

  // Example queries under the search box. The two reads live in the same
  // example FASTA the page offers for download, so there is one copy of them.
  let exampleReads = null;
  document.getElementById('search-examples').addEventListener('click', async (event) => {
    const button = event.target.closest('button');
    if (!button) return;
    let query = button.dataset.query;
    if (button.dataset.asv !== undefined) {
      exampleReads ??= fetch('/data/examples/atlas_asv_example.fasta')
        .then((response) => response.text())
        .then((text) => text.split('>').slice(1)
          .map((record) => record.split('\n').slice(1).join('')));
      query = (await exampleReads)[Number(button.dataset.asv)];
    }
    searchInput.value = query;
    clearTimeout(timer);
    runSearch(query);
  });

  readsInput.addEventListener('change', () => {
    const file = readsInput.files[0];
    readsInput.value = '';          // so choosing the same file again reruns it
    if (file) searchFasta(file);
  });

  status.hidden = true;

  const requested = new URLSearchParams(location.search).get('otu');
  if (requested) {
    const found = data.otus.find((record) => record.id === requested);
    if (found) {
      searchInput.value = found.genus || found.id;
      select(found.i, true);
    }
  }
}

load().catch((error) => {
  setStatus('The atlas failed to load.', String(error.message || error));
  console.error(error);
});
