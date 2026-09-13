/**
 * Loading the precomputed arrays.
 *
 * Every asset the pages read is a headerless little-endian binary whose shape
 * and dtype are declared in `data/web/meta.json`, so parsing is
 * `new Float32Array(buffer)` and nothing else. Keeping the shapes in one
 * manifest is why the pages can be plain ES with no build step.
 */

/**
 * Fetch a binary file, reporting real byte progress.
 *
 * The site ships an 2.8 MB matrix and a 4.7 MB model, so "still spinning" is
 * not an acceptable loading state: the caller gets `Content-Length`-based
 * progress and can say how much is left.
 *
 * @param {string} url
 * @param {(loaded: number, total: number) => void} [onProgress]
 * @returns {Promise<ArrayBuffer>}
 */
export async function fetchWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: HTTP ${response.status}`);
  }

  const total = Number(response.headers.get('Content-Length') || 0);
  if (!onProgress || !response.body || !total) {
    return response.arrayBuffer();
  }

  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }

  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }
  return buffer.buffer;
}

/**
 * Read an array declared in the manifest.
 *
 * @param {string} url
 * @param {{dtype: string, shape: number[]}} spec
 * @returns {Promise<{data: TypedArray, dtype: string, shape: number[]}>}
 */
export async function loadArray(url, spec, onProgress) {
  const buffer = await fetchWithProgress(url, onProgress);
  const expected = spec.shape.reduce((a, b) => a * b, 1);
  const bytes = expected * bytesPerElement(spec.dtype);
  if (buffer.byteLength !== bytes) {
    throw new Error(
      `${url}: ${buffer.byteLength} bytes on the wire but the manifest ` +
      `declares ${spec.shape.join('x')} ${spec.dtype} (${bytes} bytes)`);
  }
  return { data: view(buffer, spec.dtype), dtype: spec.dtype, shape: spec.shape };
}

/**
 * Wrap a buffer in the right typed array.
 *
 * @param {ArrayBuffer} buffer
 * @param {string} dtype
 */
function view(buffer, dtype) {
  switch (dtype) {
    case 'float32': return new Float32Array(buffer);
    case 'float16': return new Uint16Array(buffer);
    case 'int32': return new Int32Array(buffer);
    case 'int16': return new Int16Array(buffer);
    case 'int8': return new Int8Array(buffer);
    case 'uint8': return new Uint8Array(buffer);
    default: throw new Error(`unsupported dtype ${dtype}`);
  }
}

/** Bytes per element of a manifest dtype. */
function bytesPerElement(dtype) {
  switch (dtype) {
    case 'float32':
    case 'int32': return 4;
    case 'float16':
    case 'int16': return 2;
    case 'int8':
    case 'uint8': return 1;
    default: throw new Error(`unsupported dtype ${dtype}`);
  }
}

/**
 * Expand a float16 buffer to float32.
 *
 * The embedding table and the SNE matrix ship as float16 to halve their size,
 * and JavaScript has no native half type, so the bits are decoded once through
 * a 64K-entry lookup table rather than per element.
 *
 * @param {Uint16Array} half
 * @returns {Float32Array}
 */
export function halfToFloat(half) {
  const table = halfLookup();
  const out = new Float32Array(half.length);
  for (let i = 0; i < half.length; i += 1) out[i] = table[half[i]];
  return out;
}

let halfTable = null;

function halfLookup() {
  if (halfTable) return halfTable;
  halfTable = new Float32Array(65536);
  for (let bits = 0; bits < 65536; bits += 1) {
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits & 0x7C00) >> 10;
    const fraction = bits & 0x03FF;

    if (exponent === 0) {
      // Subnormal, and the exact zero the padding rows use.
      halfTable[bits] = sign * (fraction / 1024) * 2 ** -14;
    } else if (exponent === 0x1F) {
      halfTable[bits] = fraction ? NaN : sign * Infinity;
    } else {
      halfTable[bits] = sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
    }
  }
  return halfTable;
}

