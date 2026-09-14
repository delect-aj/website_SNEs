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

const status = document.getElementById('status');
const progressBar = document.getElementById('bar');
const progressText = document.getElementById('progress-text');
const panel = document.getElementById('panel');
const legend = document.getElementById('legend');
const results = document.getElementById('results');
const searchInput = document.getElementById('search');

let data = null;
let scatter = null;
let currentColors = null;

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
    ? `${data.meta.color_by[field].length} levels, the smallest folded into one colour`
    : `${levels.length} levels`;
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

function runSearch(query) {
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
    const empty = element('p', 'small muted', 'No OTU matches that. Try a genus, or paste an OTU id.');
    results.appendChild(empty);
    return;
  }

  for (const record of matches) {
    const button = element('button');
    button.type = 'button';
    button.className = 'button--quiet';
    button.style.display = 'block';
    button.style.width = '100%';
    button.style.textAlign = 'left';
    const name = record.species || record.genus || record.family || record.id;
    button.textContent = `${name} — ${record.id}`;
    button.addEventListener('click', () => select(record.i, true));
    results.appendChild(button);
  }
}

async function load() {
  setStatus('Reading the manifest…');
  const meta = await (await fetch('/data/meta.json')).json();
  const spec = (name) => meta.arrays[name];

  const track = (label) => (loaded, size) =>
    setStatus(label, `${(loaded / 1048576).toFixed(1)} of `
      + `${(size / 1048576).toFixed(1)} MB`, loaded / size);

  const umap = await loadArray('/data/umap.f32.bin', spec('umap.f32.bin'),
                               track('Loading the map…'));

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
    timer = setTimeout(() => runSearch(searchInput.value), 120);
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
  setStatus('The atlas could not be loaded.', String(error.message || error));
  console.error(error);
});
