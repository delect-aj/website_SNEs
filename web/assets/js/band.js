/**
 * The confidence band: one probability axis, both labelled groups on it, and
 * the queried OTU marked.
 *
 * A bare 0.87 does not tell a visitor anything. Placing the query between the
 * known members of each class does: "it sits inside the known anaerobes, so we
 * call it anaerobic" is readable without knowing what a random forest is. The
 * two groups are always drawn -- an axis with one dot on it explains nothing,
 * and the separation (or lack of it) between the groups is the AUC made
 * visible, which is why the same figure also justifies hiding weak traits.
 *
 * Deliberately absent: any gradient or shading along the axis. Colouring the
 * right-hand side "risky" would put a judgement into a figure whose whole
 * purpose is to show evidence.
 */

import { svgElement } from './dom.js';

/**
 * Deterministic vertical jitter.
 *
 * A hash keeps a point in the same row across renders, so expanding one trait
 * and then another does not make the first one's dots jump.
 */
function jitter(index, rows) {
  let hash = (index * 2654435761) >>> 0;
  hash ^= hash >>> 13;
  return (hash % rows) / (rows - 1 || 1);
}

/**
 * Draw the band into a container.
 *
 * @param {HTMLElement} container
 * @param {object} spec
 * @param {number[]} spec.ctrl - Probabilities of the reference class for the
 *   OTUs labelled as not holding it.
 * @param {number[]} spec.case - The same for OTUs labelled as holding it.
 * @param {number} spec.you - The queried OTU's probability.
 * @param {string} spec.className - The class the axis is the probability of.
 * @param {string} spec.ctrlLabel
 * @param {string} spec.caseLabel
 * @param {string} spec.youLabel
 * @param {number} [spec.width=380]
 * @param {number[]} [spec.domain=[0, 1]] - Axis range. Trait probabilities span
 *   (0, 1]; the cohort's model scores are logits and span whatever they span.
 * @param {number[]} [spec.ticks] - Tick positions, defaulting to the quartiles
 *   of the domain.
 * @param {string} [spec.axisFormat='.1f'] - Fixed decimals per tick.
 * @param {number} [spec.bins=26] - Columns the dots are bucketed into. Without
 *   bucketing, a thousand points at the same probability stack into one
 *   unreadable pillar; bucketing makes the group read as a density.
 */
export function renderBand(container, spec) {
  const width = spec.width || 380;
  const pad = { left: 12, right: 12, top: 22, bottom: 40 };
  const plot = width - pad.left - pad.right;
  const laneHeight = 22;
  const rows = 7;
  const height = pad.top + laneHeight * 2 + 26 + pad.bottom;

  const domain = spec.domain || [0, 1];
  const span = domain[1] - domain[0] || 1;
  // Columns the points are bucketed into. Left to the caller only because the
  // trait bands and the cohort band want different granularity; a missing
  // value here is a NaN in every circle's cx, which the browser reports once
  // per point.
  const bins = spec.bins || 26;
  const format = spec.axisFormat || '.1f';
  const decimals = Number(String(format).replace('.', '').replace('f', '')) || 1;
  const ticks = spec.ticks
    || [0, 0.25, 0.5, 0.75, 1].map((t) => domain[0] + t * span);

  const svg = svgElement('svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label',
    `Distribution of ${spec.className} over the reference groups, with the `
    + `queried sample marked at ${spec.you.toFixed(decimals)}`);

  const x = (p) => pad.left
    + Math.min(Math.max((p - domain[0]) / span, 0), 1) * plot;

  // Axis
  const axis = svgElement('line');
  axis.setAttribute('x1', pad.left);
  axis.setAttribute('x2', pad.left + plot);
  axis.setAttribute('y1', pad.top + laneHeight * 2 + 6);
  axis.setAttribute('y2', pad.top + laneHeight * 2 + 6);
  axis.setAttribute('class', 'band__axis');
  svg.appendChild(axis);

  for (const tick of ticks) {
    const line = svgElement('line');
    line.setAttribute('x1', x(tick));
    line.setAttribute('x2', x(tick));
    line.setAttribute('y1', pad.top + laneHeight * 2 + 3);
    line.setAttribute('y2', pad.top + laneHeight * 2 + 9);
    line.setAttribute('class', 'band__tick');
    svg.appendChild(line);

    const label = svgElement('text');
    label.setAttribute('x', x(tick));
    label.setAttribute('y', pad.top + laneHeight * 2 + 22);
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('class', 'band__label');
    label.textContent = tick.toFixed(decimals);
    svg.appendChild(label);
  }

  const groups = [
    { name: 'ctrl', values: spec.ctrl, lane: 0, label: spec.ctrlLabel },
    { name: 'case', values: spec.case, lane: 1, label: spec.caseLabel },
  ];

  for (const group of groups) {
    const counts = new Map();
    for (const value of group.values) {
      if (!Number.isFinite(value)) continue;
      const position = Math.min(Math.max((value - domain[0]) / span, 0), 1);
      const bin = Math.round(position * bins);
      counts.set(bin, (counts.get(bin) || 0) + 1);
    }
    for (const [bin, count] of counts) {
      // Height carries the count, so a group of 800 and a group of 40 are
      // distinguishable without either being a solid block.
      const stack = Math.max(1, Math.round(Math.sqrt(count)));
      for (let k = 0; k < stack; k += 1) {
        const dot = svgElement('circle');
        const t = jitter(bin * 31 + k, rows);
        dot.setAttribute('cx', (pad.left + (bin / bins) * plot).toFixed(1));
        dot.setAttribute('cy',
          (pad.top + group.lane * laneHeight + t * (laneHeight - 8) + 4)
            .toFixed(1));
        dot.setAttribute('r', 2.1);
        dot.setAttribute('class', `band__dot--${group.name}`);
        svg.appendChild(dot);
      }
    }

    const caption = svgElement('text');
    caption.setAttribute('x', group.lane === 0 ? pad.left : pad.left + plot);
    caption.setAttribute('y', height - pad.bottom + 20);
    caption.setAttribute('text-anchor', group.lane === 0 ? 'start' : 'end');
    caption.setAttribute('class', 'band__count');
    caption.textContent = `${group.label} (n=${group.values.length})`;
    svg.appendChild(caption);
  }

  // The query: a solid triangle rising from below the axis, heavier than
  // either group, because it is the only mark a visitor is looking for.
  const marker = svgElement('path');
  const at = x(spec.you);
  const base = pad.top + laneHeight * 2 + 6;
  marker.setAttribute('d',
    `M ${at} ${base - 16} L ${at - 4.5} ${base - 5} L ${at + 4.5} ${base - 5} Z`);
  marker.setAttribute('class', 'band__you');
  svg.appendChild(marker);

  const callout = svgElement('text');
  callout.setAttribute('x', Math.min(Math.max(at, pad.left + 40),
                                     pad.left + plot - 40));
  callout.setAttribute('y', pad.top - 8);
  callout.setAttribute('text-anchor', 'middle');
  callout.setAttribute('class', 'band__label');
  callout.textContent = `${spec.youLabel} · ${spec.you.toFixed(decimals)}`;
  svg.appendChild(callout);

  if (spec.axisLabel) {
    const label = svgElement('text');
    label.setAttribute('x', pad.left);
    label.setAttribute('y', pad.top - 8);
    label.setAttribute('class', 'band__label');
    label.textContent = spec.axisLabel;
    svg.appendChild(label);
  }

  container.replaceChildren(svg);
}
