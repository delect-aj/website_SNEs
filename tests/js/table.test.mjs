/**
 * The count-table reader, against the files real pipelines write.
 *
 * `table.js` is the first thing a visitor touches on the dysbiosis page and
 * the one place where a malformed file has to produce a sentence rather than
 * a TypeError. Each case below is a shape that arrives in practice: the
 * banner `biom convert --to-tsv` and `qiime tools export` write, the header
 * that follows it, a table transposed in a spreadsheet, a row longer than its
 * header, and a file with no delimiter at all.
 *
 * Run:
 *   cd tests/js && npm install && node table.test.mjs
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));

// PapaParse is a page global (`<script src=papaparse>`), so the module under
// test reads it from scope. This stub covers what `readTable` uses: delimiter
// detection, skipped empty lines, and the `errors[0].type === 'Delimiter'`
// signal for text with no delimiter in it.
globalThis.Papa = {
  parse(text) {
    const delimiter = text.includes('\t') ? '\t' : text.includes(',') ? ',' : null;
    if (!delimiter) {
      return { data: [], errors: [{ type: 'Delimiter' }] };
    }
    return {
      data: text.split('\n').filter((line) => line.trim() !== '')
        .map((line) => line.split(delimiter)),
      errors: [],
    };
  },
};

const { readTable } = await import(
  join(here, '..', '..', 'web', 'assets', 'js', 'table.js'));

let failures = 0;
let checks = 0;

function check(condition, message) {
  checks += 1;
  if (!condition) {
    failures += 1;
    console.error(`  FAIL ${message}`);
  }
}

/** The samples as `{name: {taxon: count}}`, for comparing whole results. */
function shape(samples) {
  return Object.fromEntries(samples.map((sample) =>
    [sample.name, Object.fromEntries(sample.otuCounts)]));
}

function read(name, text) {
  try {
    return { samples: readTable(text, name) };
  } catch (error) {
    return { error };
  }
}

const FEATURES = ['AAAA02020714.1.1202', 'AAFJ01000001.39328.40836',
                  'AAQK01001555.694.2198'];
const EXPECTED = {
  sample_1: { [FEATURES[0]]: 5, [FEATURES[2]]: 3 },
  sample_2: { [FEATURES[1]]: 7, [FEATURES[2]]: 2 },
};

// 1. What `biom convert --to-tsv` and `qiime tools export` write: a banner
//    line, then a header whose first cell is `#OTU ID`.
{
  const text = [
    '# Constructed from biom file',
    `#OTU ID\tsample_1\tsample_2`,
    `${FEATURES[0]}\t5\t0`,
    `${FEATURES[1]}\t0\t7`,
    `${FEATURES[2]}\t3\t2`,
  ].join('\n');
  const { samples, error } = read('biom.tsv', text);
  check(!error, `biom banner + header: threw ${error && error.message}`);
  check(samples && samples.length === 2,
    `biom banner + header: ${samples ? samples.length : '-'} samples, expected 2`);
  check(samples && JSON.stringify(shape(samples)) === JSON.stringify(EXPECTED),
    `biom banner + header: read as ${samples && JSON.stringify(shape(samples))}`);
}

// 2. The same table transposed by hand: samples as rows, features across.
{
  const text = [
    `SampleID\t${FEATURES.join('\t')}`,
    'sample_1\t5\t0\t3',
    'sample_2\t0\t7\t2',
  ].join('\n');
  const { samples, error } = read('transposed.tsv', text);
  check(!error, `transposed: threw ${error && error.message}`);
  check(samples && JSON.stringify(shape(samples)) === JSON.stringify(EXPECTED),
    `transposed: read as ${samples && JSON.stringify(shape(samples))}`);
}

// 3. A row with more cells than the header -- what a stray edit in a
//    spreadsheet produces.
{
  const text = [
    `#OTU ID\tsample_1\tsample_2`,
    `${FEATURES[0]}\t5\t0`,
    `${FEATURES[1]}\t0\t7\t999`,
  ].join('\n');
  const { samples, error } = read('ragged.tsv', text);
  check(!error, `ragged row: threw ${error && error.message}`);
  check(samples && JSON.stringify(shape(samples)) === JSON.stringify({
    sample_1: { [FEATURES[0]]: 5 },
    sample_2: { [FEATURES[1]]: 7 },
  }), `ragged row: read as ${samples && JSON.stringify(shape(samples))}`);
}

// 4. Comma separated, with the banner: dropping `#` lines must not eat a
//    header that carries no tabs.
{
  const text = [
    '# Constructed from biom file',
    `#OTU ID,sample_1,sample_2`,
    `${FEATURES[0]},5,0`,
    `${FEATURES[1]},0,7`,
  ].join('\n');
  const { samples, error } = read('table.csv', text);
  check(!error, `comma separated: threw ${error && error.message}`);
  check(samples && JSON.stringify(shape(samples)) === JSON.stringify({
    sample_1: { [FEATURES[0]]: 5 },
    sample_2: { [FEATURES[1]]: 7 },
  }), `comma separated: read as ${samples && JSON.stringify(shape(samples))}`);
}

// 5. A table with nothing in it is empty, not an exception: the caller
//    decides what to say about it.
{
  const text = `#OTU ID\tsample_1\tsample_2\n${FEATURES[0]}\t0\t0\n`;
  const { samples, error } = read('empty.tsv', text);
  check(!error, `all-zero table: threw ${error && error.message}`);
  check(samples && samples.length === 0,
    `all-zero table: ${samples && samples.length} samples, expected 0`);
}

// 6. Text with no delimiter is a sentence, not a TypeError.
{
  const { error } = read('prose.txt', 'this is not a table\njust some words\n');
  check(error && /delimiter/.test(error.message),
    `prose: got ${error ? JSON.stringify(error.message) : 'no error'}`);
}

console.log(`\nchecks: ${checks}, failures: ${failures}`);
if (failures) {
  console.error('\nThe count-table reader does not read the files it is for.');
  process.exit(1);
}
console.log('The count-table reader reads every shape the pipelines write.');
