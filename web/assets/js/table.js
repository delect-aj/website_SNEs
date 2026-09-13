/**
 * Reading a count table into one sparse map per sample.
 *
 * The formats this has to swallow are the ones real pipelines write, and they
 * are not the tidy version of themselves:
 *
 *   - `biom convert --to-tsv` and `qiime tools export` both put a comment line
 *     (`# Constructed from biom file`) above the header, and the header's
 *     first cell is itself `#OTU ID`;
 *   - a table written with samples down the side has to be detected by shape,
 *     because name overlap says nothing about orientation;
 *   - rows can be longer than the header, which Excel produces routinely.
 *
 * None of those may throw: this is the first thing a visitor touches, and a
 * TypeError from inside the parser is not something they can act on. The
 * module is DOM-free so `tests/js/table.test.mjs` can run it under node --
 * PapaParse is a global the page loads with a <script> tag, and the test
 * installs a stub for it.
 */

/** A `#` line with no delimiter in it: the banner, not the `#OTU ID` header. */
const COMMENT_LINE = /^#[^\t,;]*$/;

/**
 * Turn a feature-by-sample grid into one sparse map per sample.
 *
 * @param {Array<Array<string>>} rows
 * @param {Array<string>} header - Column labels, first cell included.
 * @param {number} firstColumn - Index of the first data column.
 * @returns {Array<{name: string, otuCounts: Map<string, number>}>} Samples
 *   with at least one non-zero count.
 */
export function samplesFromGrid(rows, header, firstColumn) {
  const names = header.slice(firstColumn).map(String);
  const samples = names.map((name) => ({ name, otuCounts: new Map() }));

  for (const row of rows) {
    const id = String(row[0] ?? '').trim();
    if (!id) continue;
    for (let column = firstColumn; column < row.length; column += 1) {
      const sample = samples[column - firstColumn];
      if (!sample) continue;        // row longer than the header, not a crash
      const value = Number(row[column]);
      if (!Number.isFinite(value) || value <= 0) continue;
      sample.otuCounts.set(id, (sample.otuCounts.get(id) || 0) + value);
    }
  }
  return samples.filter((sample) => sample.otuCounts.size > 0);
}

/**
 * Read a count table in either orientation.
 *
 * QIIME2 and DADA2 both write features down the side, which is what the page
 * asks for. The other way round is read too, because a visitor who transposed
 * a table in a spreadsheet has no way to know it matters.
 *
 * @param {string} text
 * @param {string} label - Name of the source file, for error messages.
 * @throws {Error} When the text has no delimiter, no data rows, or no sample
 *   column with a non-zero count.
 */
export function readTable(text, label) {
  const cleaned = text.split('\n')
    .filter((line) => !COMMENT_LINE.test(line)).join('\n');
  const parsed = Papa.parse(cleaned.trim(), { skipEmptyLines: true });
  if (parsed.errors.length && parsed.errors[0].type === 'Delimiter') {
    throw new Error(`${label}: could not find a delimiter; a tab-separated `
      + `file is expected.`);
  }
  const rows = parsed.data;
  if (rows.length < 2) throw new Error(`${label}: no data rows`);

  const header = rows[0].map(String);
  const body = rows.slice(1);

  // Orientation comes from the shape: a feature table has far more features
  // than samples, so the shorter axis is the sample axis. (Testing whether
  // header cells reappear among the row labels, the first attempt, only ever
  // fires on a square matrix -- a transposed table was read with its samples
  // as taxa and its taxa as sample names.)
  const transposed = body.length < header.length - 1;

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
