/**
 * The trait probability block lines up with the OTUs the card reads it for.
 *
 * The card reads `traits_proba.f16.bin` at `offset + position * n_classes +
 * class`, where `position` counts the Traitar-labelled OTUs in `otus.json`
 * order. The forest behind the block was fitted on exactly those OTUs, so
 * reading it correctly gives a training accuracy near 1. A misaligned read
 * lands near chance, which is how offsets written in rows instead of values
 * went unnoticed: every band still drew, from another trait's numbers.
 *
 * Run: cd tests/js && node traits.test.mjs
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { halfToFloat } from '../../web/assets/js/binary.js';

const web = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'data', 'web');
const traits = JSON.parse(readFileSync(join(web, 'traits.json'), 'utf8'));
const otus = JSON.parse(readFileSync(join(web, 'otus.json'), 'utf8'));
const bytes = readFileSync(join(web, 'traits_proba.f16.bin'));
const proba = halfToFloat(new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2));

let failures = 0;
let end = 0;
for (const name of traits.order) {
  const meta = traits.traits[name];
  const labelled = otus.filter((record) => record.traits[name].source === 'Traitar');
  let correct = 0;
  labelled.forEach((record, position) => {
    const row = proba.subarray(meta.offset + position * meta.n_classes,
                               meta.offset + (position + 1) * meta.n_classes);
    const best = row.indexOf(Math.max(...row));
    if (meta.classes[best] === String(record.traits[name].value)) correct += 1;
  });
  const accuracy = correct / labelled.length;
  const ok = labelled.length === meta.count && accuracy > 0.9;
  if (!ok) failures += 1;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name.padEnd(18)} ${labelled.length} rows, `
    + `training accuracy ${accuracy.toFixed(3)}`);
  end = Math.max(end, meta.offset + meta.count * meta.n_classes);
}
if (end !== proba.length) {
  failures += 1;
  console.log(`  FAIL offsets cover ${end} values, the file holds ${proba.length}`);
}
console.log(`checks: ${traits.order.length + 1}, failures: ${failures}`);
process.exit(failures ? 1 : 0);
