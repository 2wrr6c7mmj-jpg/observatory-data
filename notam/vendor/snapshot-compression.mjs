// Vendored unchanged from 2wrr6c7mmj-jpg/notam-auto-update shared/snapshot-compression.mjs (commit b54a8d2).
// NOTAM Watch publishes its snapshot with this transport; update both together.
/** Optional lossless delivery alongside the existing immutable snapshot assets. */
export const COMPRESSED_FILE = /^snapshot-[a-f0-9]{16}\.gzip-[a-f0-9]{16}\.bin$/;
const MAX_COMPRESSED = 25 * 1024 * 1024;
const MAX_DECODED = 64 * 1024 * 1024;
const check = (value, message) => { if (!value) throw Error(message); };
const hash = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), x => x.toString(16).padStart(2, '0')).join('');

export function validateCompression(manifest) {
  if (!Object.hasOwn(manifest, 'compressed')) return null;
  const value = manifest.compressed;
  check(value && Object.keys(value).sort().join(',') === 'byteLength,decodedByteLength,file,kind,schemaVersion,sha256', 'Invalid compressed snapshot descriptor');
  check(value.schemaVersion === 1 && value.kind === 'gzip' && /^[a-f0-9]{64}$/.test(value.sha256), 'Invalid compression identity');
  check(value.file === `snapshot-${manifest.sha256.slice(0, 16)}.gzip-${value.sha256.slice(0, 16)}.bin`, 'Invalid compression filename');
  check(Number.isSafeInteger(value.byteLength) && value.byteLength > 0 && value.byteLength <= MAX_COMPRESSED, 'Invalid compressed size');
  check(Number.isSafeInteger(value.decodedByteLength) && value.decodedByteLength > 0 && value.decodedByteLength <= MAX_DECODED, 'Invalid decompressed size');
  if (manifest.transport) check(value.decodedByteLength === manifest.transport.byteLength, 'Conflicting snapshot sizes');
  return value;
}

async function collectBounded(stream, limit, {expected, signal} = {}) {
  const reader = stream.getReader(), chunks = [];
  let size = 0, complete = false;
  const abort = () => { void reader.cancel().catch(() => {}); };
  signal?.addEventListener('abort', abort, {once: true});
  try {
    while (true) {
      signal?.throwIfAborted();
      const {done, value} = await reader.read();
      signal?.throwIfAborted();
      if (done) break;
      check(value instanceof Uint8Array && value.byteLength <= limit - size, 'Decompressed snapshot exceeds bound');
      chunks.push(value); size += value.byteLength;
    }
    check(expected === undefined || size === expected, 'Decompressed snapshot length mismatch');
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    complete = true;
    return bytes;
  } finally {
    signal?.removeEventListener('abort', abort);
    if (!complete) abort();
    reader.releaseLock();
  }
}

export async function compressSnapshot(bytes, canonicalHash) {
  if (bytes.byteLength < 1024 * 1024 || typeof CompressionStream !== 'function') return null;
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'));
  let packed;
  try { packed = await collectBounded(stream, MAX_COMPRESSED); } catch { return null; }
  if (packed.byteLength >= bytes.byteLength * 0.9) return null;
  const sha256 = await hash(packed);
  return {bytes: packed, descriptor: {schemaVersion: 1, kind: 'gzip',
    file: `snapshot-${canonicalHash.slice(0, 16)}.gzip-${sha256.slice(0, 16)}.bin`,
    sha256, byteLength: packed.byteLength, decodedByteLength: bytes.byteLength}};
}

export async function decompressSnapshot(bytes, descriptor, {signal} = {}) {
  check(bytes.byteLength === descriptor.byteLength && await hash(bytes) === descriptor.sha256, 'Compressed snapshot integrity mismatch');
  signal?.throwIfAborted();
  return collectBounded(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip')),
    descriptor.decodedByteLength, {expected: descriptor.decodedByteLength, signal});
}

export async function validateCompressedAsset(file, bytes) {
  check(COMPRESSED_FILE.test(file), 'Invalid compressed asset');
  const sha256 = await hash(bytes);
  check(sha256.slice(0, 16) === file.slice(31, 47), 'Compressed asset hash mismatch');
  const decoded = await collectBounded(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip')), MAX_DECODED);
  check((await hash(decoded)).slice(0, 16) === file.slice(9, 25), 'Compressed asset snapshot mismatch');
  return {file: `snapshot-${file.slice(9, 25)}.json`, bytes: decoded};
}
