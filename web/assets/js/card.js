/**
 * The microbe card: taxonomy, the two neighbour columns, and the trait list.
 *
 * The card is where the site says something the paper cannot: what an
 * uncultured OTU probably does. Two rules run through all of it.
 *
 * Every inferred value carries the cross-validated AUC it earned, at the same
 * size as the value itself. A 0.87 from an AUC-0.55 model and a 0.87 from an
 * AUC-0.90 model look identical otherwise, and presenting them identically is
 * the single easiest way for this page to mislead someone.
 *
 * Confidence is never carried by colour alone. The three steps are a glyph
 * (filled, half, hollow) plus fill density, so a greyscale screenshot or a
 * colour-blind reader loses nothing. Measured and inferred values differ by
 * their left border as well.
 */

import { element } from './dom.js';
import { renderBand } from './band.js';

const RANKS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'];


/** Confidence glyph and wording for a cross-validated AUC. */
function confidence(auc, thresholds) {
  if (!Number.isFinite(auc)) return { glyph: '?', label: 'not scored', level: 'weak' };
  if (auc >= thresholds.trusted) return { glyph: '●', label: 'trusted', level: 'strong' };
  if (auc >= thresholds.hidden_below) return { glyph: '◐', label: 'moderate', level: 'medium' };
  return { glyph: '○', label: 'weak', level: 'weak' };
}

function taxonName(record) {
  if (record.species) return record.species;
  if (record.genus) return record.genus;
  if (record.family) return record.family;
  if (record.order) return record.order;
  if (record.class) return record.class;
  return record.phylum || record.id;
}

/**
 * The two neighbour columns.
 *
 * Same width, same order of rows, same typography. Anything that makes one
 * column look like the lesser one -- a different size, colour, or row count --
 * weakens the point the pair is making, so the only difference is the heading
 * and a background tint on rows present in both.
 */
function renderNeighbours(container, index, data) {
  const wrapper = element('div', 'neighbours');
  const lists = [
    { title: 'Ecological neighbours', idx: data.nbrSneIdx, sim: data.nbrSneSim,
      other: new Set(Array.from(data.nbrPhyloIdx.subarray(index * data.k, (index + 1) * data.k))) },
    { title: 'Phylogenetic neighbours', idx: data.nbrPhyloIdx, sim: data.nbrPhyloSim,
      other: new Set(Array.from(data.nbrSneIdx.subarray(index * data.k, (index + 1) * data.k))) },
  ];

  for (const list of lists) {
    const column = element('div');
    column.appendChild(element('h3', null, list.title));
    const ordered = element('ol');
    const start = index * data.k;
    for (let k = 0; k < data.k; k += 1) {
      const neighbour = list.idx[start + k];
      const record = data.otus[neighbour];
      const row = element('li');
      if (list.other.has(neighbour)) row.className = 'overlap';
      const name = element('span', 'taxon', taxonName(record));
      name.title = record.id;
      row.appendChild(name);
      // Three decimals, not two: a phylogenetic neighbour list can sit
      // entirely between 0.992 and 1.000, and two decimals turns that into a
      // column of identical numbers.
      row.appendChild(element('span', 'num',
        list.sim[start + k].toFixed(3)));
      ordered.appendChild(row);
    }
    column.appendChild(ordered);
    wrapper.appendChild(column);
  }

  const overlap = data.otus[index].nbr_overlap;
  const note = element('p', 'small muted');
  note.style.marginTop = '12px';
  note.textContent = `${overlap} of ${data.k} ecological neighbours are also `
    + `phylogenetic neighbours. Highlighted rows appear in both lists.`;
  wrapper.appendChild(note);

  container.replaceChildren(wrapper);
}

/** Probability of one class for one labelled OTU, from the flat block. */
function labelledProbability(data, trait, position, classIndex) {
  const meta = data.traits.traits[trait];
  return data.traitsProba[meta.offset + position * meta.n_classes + classIndex];
}

/**
 * OTU indices that carry a Traitar label for a trait, in ascending order.
 *
 * The export wrote the probability block in this order, so a label's position
 * in this list is its row in the block. Cached per trait: the card asks for it
 * once per band, and rebuilding it is a walk over 14,093 records.
 */
function labelledPositions(data, trait) {
  const cached = data._labelled && data._labelled[trait];
  if (cached) return cached;
  const positions = [];
  for (let i = 0; i < data.otus.length; i += 1) {
    if (data.otus[i].traits[trait].source === 'Traitar') positions.push(i);
  }
  if (!data._labelled) data._labelled = {};
  data._labelled[trait] = positions;
  return positions;
}

/**
 * The value the card presents for one trait.
 *
 * A BacDive measurement overrides the trait table's value, here and in the
 * row's badge; the two have to agree or the card would label an inference as
 * a measurement. About a fifth of the measured pairs hold a different value
 * from the inference they override.
 */
function displayedValue(record, trait, data) {
  const measured = (data.bacdive[record.id] || {})[trait];
  return measured === undefined ? record.traits[trait].value : measured;
}

/**
 * The forest's probability for one class of one trait, for one OTU.
 *
 * An inferred row stores the probability of the class it holds, so that is
 * returned directly -- but only when it is the class asked for. A row with a
 * genome label has none stored: `traits_predict.ipynb` leaves it empty
 * because an in-sample probability would be near one and mean nothing. Those
 * are read from the block the export wrote by refitting the same forest.

 * Returns null when the class is not one the forest knows, or when the OTU is
 * not among the labelled rows the block covers -- which is the case for a
 * curated measurement on an OTU with no genome label for that trait.
 */
function probabilityFor(trait, record, data,
                        value = record.traits[trait].value) {
  const meta = data.traits.traits[trait];
  const classIndex = meta.classes.indexOf(String(value));
  if (classIndex < 0) return null;

  const entry = record.traits[trait];
  if (String(entry.value) === String(value) && Number.isFinite(entry.prob)) {
    return entry.prob;
  }

  const position = labelledPositions(data, trait).indexOf(record.i);
  if (position < 0) return null;
  return labelledProbability(data, trait, position, classIndex);
}

/**
 * Everything the band needs for one trait and one OTU.
 *
 * Returns null when the queried OTU's own probability is unknown, since a
 * band cannot place a marker it does not have a position for.
 */
function renderBandFor(trait, record, data) {
  const meta = data.traits.traits[trait];
  const labels = labelledPositions(data, trait);
  const measured = (data.bacdive[record.id] || {})[trait] !== undefined;
  // The band is drawn for the class the card claims. When BacDive has a
  // record that is the measured value: an axis labelled with a class the row
  // above does not show reads as the card contradicting itself.
  const queryValue = String(displayedValue(record, trait, data));
  const classIndex = meta.classes.indexOf(queryValue);
  const you = probabilityFor(trait, record, data, queryValue);
  if (you === null || classIndex < 0) return null;

  const ctrl = [];
  const cases = [];
  for (let position = 0; position < labels.length; position += 1) {
    const other = data.otus[labels[position]];
    const value = labelledProbability(data, trait, position, classIndex);
    if (String(other.traits[trait].value) === queryValue) cases.push(value);
    else ctrl.push(value);
  }

  const otherValues = meta.classes.filter((c) => c !== queryValue)
    .map((c) => meta.value_labels[c] || c);
  const labelled = record.traits[trait].source === 'Traitar';
  // `ctrl` and `case` are the names `band.js` reads and the ones the CSS
  // classes use; the two modules have to agree on them.
  return {
    ctrl, case: cases, you,
    className: meta.value_labels[queryValue] || queryValue,
    ctrlLabel: `labelled ${otherValues.join(' / ')}`,
    caseLabel: `labelled ${meta.value_labels[queryValue] || queryValue}`,
    youLabel: measured ? `${record.id} (BacDive measured)`
      : labelled ? `${record.id} (genome label)` : `${record.id} (inferred)`,
  };
}

/** The band, in a fold-out, drawn the first time it is opened. */
function bandDetails(trait, record, data) {
  if (!renderBandFor(trait, record, data)) return null;
  const details = element('details');
  details.appendChild(element('summary', null, 'Show the labelled distribution'));
  const body = element('div', 'band');
  details.appendChild(body);
  details.addEventListener('toggle', () => {
    if (details.open && !body.dataset.drawn) {
      renderBand(body, renderBandFor(trait, record, data));
      body.dataset.drawn = '1';
    }
  });
  return details;
}

function renderTrait(trait, record, data, thresholds) {
  const meta = data.traits.traits[trait];
  const entry = record.traits[trait];
  const measurement = (data.bacdive[record.id] || {})[trait];
  const curated = measurement !== undefined;
  const labelled = entry.source === 'Traitar';
  const row = element('div', curated ? 'trait trait--measured' : 'trait');

  const head = element('div', 'trait__head');
  head.appendChild(element('span', 'trait__name', meta.label));
  const shown = displayedValue(record, trait, data);
  const valueText = meta.value_labels[String(shown)] || String(shown);
  const value = element('span', 'trait__value', valueText);
  if (curated || labelled) {
    const badge = element('span', 'badge',
      curated ? 'BacDive measured' : 'Traitar genome label');
    badge.style.marginLeft = '8px';
    value.appendChild(badge);
  }
  head.appendChild(value);
  row.appendChild(head);

  if (curated) {
    const note = element('p', 'small muted');
    note.style.margin = '4px 0 0';
    note.textContent = 'A curated wet-lab record, shown instead of the '
      + 'inference for this OTU.';
    row.appendChild(note);
    const details = bandDetails(trait, record, data);
    if (details) row.appendChild(details);
    return row;
  }

  const line = element('div', 'trait__row');
  const probability = probabilityFor(trait, record, data);

  if (labelled) {
    // The value is a genome prediction, so there is no probability of *this
    // OTU* to show. What the row can carry is the trait's own discrimination:
    // how well anyone can call this trait from the embedding. Showing the
    // trait-wide AUC keeps the row informative without dressing a label up as
    // an inference, and the band below still places the OTU among the
    // labelled members of each class.
    const level = confidence(meta.auc, thresholds);
    line.appendChild(element('span', `conf conf--${level.level}`,
      `${level.glyph} ${level.label}`));
    line.appendChild(element('span', 'trait__auc',
      `trait-wide leave-one-phylum AUC ${meta.auc.toFixed(2)}`));
  } else {
    if (Number.isFinite(probability)) {
      const bar = element('span', 'bar');
      const fill = element('span', 'bar__fill');
      fill.style.width = `${Math.round(probability * 100)}%`;
      bar.appendChild(fill);
      line.appendChild(bar);
      line.appendChild(element('span', 'num', probability.toFixed(2)));
    }
    const level = confidence(entry.auc, thresholds);
    line.appendChild(element('span', `conf conf--${level.level}`,
      `${level.glyph} ${level.label}`));
    line.appendChild(element('span', 'trait__auc',
      `leave-one-phylum AUC ${Number.isFinite(entry.auc) ? entry.auc.toFixed(2) : '—'}`));
  }
  row.appendChild(line);

  const details = bandDetails(trait, record, data);
  if (details) row.appendChild(details);

  return row;
}

/**
 * Render one OTU's card.
 *
 * @param {HTMLElement} container
 * @param {number} index - Row number in `otus.json`.
 * @param {object} data - Everything the page loaded.
 */
export function renderCard(container, index, data) {
  const record = data.otus[index];
  const thresholds = data.traits.thresholds;
  const fragment = document.createDocumentFragment();

  const title = element('div', 'card__title');
  title.appendChild(element('h2', 'otu-id', record.id));
  if (!record.genome_linked) {
    title.appendChild(element('span', 'badge', 'no genome'));
  }
  fragment.appendChild(title);

  const lineage = RANKS.map((rank) => record[rank]).filter(Boolean);
  const lineageNode = element('p', 'small muted');
  lineageNode.style.margin = '0 0 4px';
  lineageNode.textContent = lineage.join(' › ');
  fragment.appendChild(lineageNode);

  const provenance = element('p', 'small muted');
  provenance.style.margin = '0 0 16px';
  provenance.textContent = record.genome_linked
    ? 'Maps to a representative genome; its trait values below are genome-based labels.'
    : 'Uncultured: no representative genome, so every trait value below is inferred from the embedding.';
  fragment.appendChild(provenance);

  const neighbourBlock = element('div');
  neighbourBlock.style.marginBottom = '20px';
  renderNeighbours(neighbourBlock, index, data);
  fragment.appendChild(neighbourBlock);

  const heading = element('h3', null, 'Ecological traits');
  heading.style.marginBottom = '4px';
  fragment.appendChild(heading);

  const caveat = element('p', 'small muted');
  caveat.textContent = 'Inferred from co-occurrence, not measured. These describe '
    + 'the role a taxon plays in the gut community, which is not the same as its '
    + 'physiology in pure culture.';
  fragment.appendChild(caveat);

  const shown = [];
  const hidden = [];
  for (const trait of data.traits.order) {
    if (data.traits.traits[trait].displayed) shown.push(trait);
    else hidden.push(trait);
  }

  for (const trait of shown) {
    fragment.appendChild(renderTrait(trait, record, data, thresholds));
  }

  if (hidden.length) {
    const details = element('details');
    const summary = element('summary', null,
      `${hidden.length} traits withheld — cross-validated AUC below `
      + `${thresholds.hidden_below}`);
    details.appendChild(summary);
    const body = element('div');
    body.style.marginTop = '12px';
    for (const trait of hidden) {
      body.appendChild(renderTrait(trait, record, data, thresholds));
    }
    details.appendChild(body);
    fragment.appendChild(details);
  }

  container.replaceChildren(fragment);
  container.classList.remove('fade-in');
  void container.offsetWidth;
  container.classList.add('fade-in');
}

/** A short summary line for a search result or a hovered point. */
export function cardSummary(index, data) {
  const record = data.otus[index];
  const lineage = [record.phylum, record.genus || record.family]
    .filter(Boolean).join(' · ');
  return { id: record.id, lineage };
}
