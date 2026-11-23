/**
 * Image acquisition and preview handling.
 *
 * Acquisition sources (all funneled through `acquireImage`):
 *   - the literal string "clipboard" → OS clipboard (see clipboard.ts)
 *   - an http(s) URL → streamed download, zero token cost
 *   - a local filesystem path → copied verbatim (source paths are OS
 *     paths, NOT sandbox-root paths: a picture in ~/Pictures is meant to
 *     be archivable; only the DESTINATION inside the root is sandboxed)
 *
 * Previews (`makePreview`): the on-disk original is never touched. Only
 * the in-memory copy handed to the model is downscaled, and only when it
 * exceeds IMAGE_PREVIEW_MAX_BYTES. Formats the pure-JS decoder (jimp)
 * cannot read (WebP, SVG) are passed through untouched when under the cap
 * and reported with a note when over it — never crashed on.
 */
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import type http from 'node:http';
import os from 'node:os';
import path from 'node:path';

import { Jimp } from 'jimp';

import { readClipboardImage } from './clipboard.js';
import {
  IMAGE_MIME_BY_EXTENSION,
  IMAGE_PREVIEW_MAX_BYTES,
  IMAGE_PREVIEW_MAX_PIXELS,
  IMAGE_PREVIEW_MIME,
  IMAGE_PREVIEW_MIN_DIMENSION,
  IMAGE_PREVIEW_QUALITIES,
  IMAGE_PREVIEW_QUALITY,
  IMAGE_PREVIEW_SCALE_STEP,
  SAVE_IMAGES_MAX_DOWNLOAD_BYTES,
  SAVE_IMAGES_MAX_TOTAL_BYTES,
  URL_DOWNLOAD_MAX_BYTES,
  URL_DOWNLOAD_TIMEOUT_MS,
} from './config.js';
import {
  BlockedUrlError,
  blockedUrlReason,
  isPublicIp,
  precheckUrlReason,
  safeRequest,
  type SafeRequestOptions,
} from './net-guard.js';
import { cleanPath } from './sandbox.js';

export { blockedUrlReason, isPublicIp, safeRequest, type SafeRequestOptions };

export interface AcquiredImage {
  data: Buffer;
  mimeType: string;
  /** Suggested file stem (no extension), derived from the source. */
  baseName: string;
}

export type ImageAcquisition =
  | { ok: true; image: AcquiredImage }
  | { ok: false; note: string };

/**
 * Per-save image byte budgets, shared across every image of one
 * save_file call. Mutated as images are acquired; when one runs
 * out, further images are refused with a note instead of failing
 * the save. Two counters: `storedRemainingBytes` bounds the bytes
 * actually kept (URL downloads confirmed as images, local copies,
 * clipboard images), while `downloadRemainingBytes` bounds the
 * bytes fetched from URLs. A failed or non-image download spends
 * download traffic, never stored budget.
 */
export interface ImageSaveBudget {
  storedRemainingBytes: number;
  downloadRemainingBytes: number;
}

/** Reason text when the per-save stored-image budget is hit. */
function storedBudgetNote(): string {
  return `stored image budget for this save (${SAVE_IMAGES_MAX_TOTAL_BYTES / 1_000_000} MB) was reached`;
}

/** Reason text when the per-save download-traffic budget is hit. */
function downloadBudgetNote(): string {
  return `download limit for this save (${SAVE_IMAGES_MAX_DOWNLOAD_BYTES / 1_000_000} MB) was reached`;
}

const EXTENSION_BY_MIME: Record<string, string> = {
  'image/png': 'png',
  'image/jpeg': 'jpg',
  'image/webp': 'webp',
  'image/gif': 'gif',
  'image/bmp': 'bmp',
  'image/tiff': 'tiff',
};

export function mimeFromExtension(filePath: string): string | null {
  const ext = path.extname(filePath).toLowerCase();
  return Object.hasOwn(IMAGE_MIME_BY_EXTENSION, ext) ? (IMAGE_MIME_BY_EXTENSION[ext] ?? null) : null;
}

export function extensionForMime(mimeType: string): string {
  return Object.hasOwn(EXTENSION_BY_MIME, mimeType) ? (EXTENSION_BY_MIME[mimeType] ?? 'png') : 'png';
}

const BMP_HEADER_SIZES = new Set([12, 40, 52, 56, 64, 108, 124]);

/** Detect an image MIME type from magic bytes; null when not an image. */
export function detectImageMime(data: Buffer): string | null {
  if (data.length >= 8 && data.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) {
    return 'image/png';
  }
  if (data.length >= 3 && data[0] === 0xff && data[1] === 0xd8 && data[2] === 0xff) {
    return 'image/jpeg';
  }
  if (data.length >= 6) {
    const gifHead = data.subarray(0, 6).toString('ascii');
    if (gifHead === 'GIF87a' || gifHead === 'GIF89a') {
      return 'image/gif';
    }
  }
  if (
    data.length >= 18 &&
    data[0] === 0x42 &&
    data[1] === 0x4d &&
    data.readUInt32LE(6) === 0 &&
    BMP_HEADER_SIZES.has(data.readUInt32LE(14))
  ) {
    return 'image/bmp';
  }
  if (
    data.length >= 12 &&
    data.subarray(0, 4).toString('ascii') === 'RIFF' &&
    data.subarray(8, 12).toString('ascii') === 'WEBP'
  ) {
    return 'image/webp';
  }
  if (
    data.length >= 4 &&
    ((data[0] === 0x49 && data[1] === 0x49 && data[2] === 0x2a && data[3] === 0x00) ||
      (data[0] === 0x4d && data[1] === 0x4d && data[2] === 0x00 && data[3] === 0x2a))
  ) {
    return 'image/tiff';
  }
  return null;
}

function baseNameOf(filePath: string): string {
  const base = path.basename(filePath, path.extname(filePath));
  const cleaned = base.replace(/[^\p{L}\p{N}_\- ]+/gu, '').trim().replace(/\s+/g, '-');
  return cleaned.length > 0 ? cleaned : 'image';
}

/** Redirect statuses HTTP(S) downloads follow; anything else is a final answer. */
const REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308]);

/** Maximum redirects followed for one image URL. */
const MAX_REDIRECTS = 5;

/** Switch large downloads to a temporary disk spool to avoid doubling memory on Buffer.concat. */
const DOWNLOAD_SPOOL_THRESHOLD_BYTES = 8 * 1024 * 1024;
/** Leading bytes needed by `detectImageMime`. */
const HEAD_SNIFF_BYTES = 32;
/** Minimum JPEG container header bytes emitted by jpeg-js; smaller budgets can never fit. */
const MIN_JPEG_HEADER_BYTES = 512;

function yieldLoop(): Promise<void> {
  return new Promise((resolve) => {
    setImmediate(resolve);
  });
}

async function writeAll(fh: Awaited<ReturnType<typeof fs.open>>, buf: Buffer): Promise<void> {
  let offset = 0;
  while (offset < buf.byteLength) {
    const { bytesWritten } = await fh.write(buf, offset, buf.byteLength - offset);
    if (bytesWritten === 0) {
      throw new Error('Short write to temporary download file');
    }
    offset += bytesWritten;
  }
}

/** Destroy an HTTP response after attaching a defensive no-op 'error' listener. */
function destroyResponse(res: http.IncomingMessage): void {
  res.once('error', () => {});
  res.destroy();
}

/** Download an image URL, streaming to memory with a safety ceiling. */
async function downloadImage(
  url: string,
  budget?: ImageSaveBudget,
  netOpts?: SafeRequestOptions,
): Promise<ImageAcquisition> {
  // Either per-save counter exhausted refuses the fetch outright:
  // no download traffic is spent and no stored budget is needed.
  if (budget !== undefined && budget.downloadRemainingBytes <= 0) {
    return { ok: false, note: `${downloadBudgetNote()}.` };
  }
  if (budget !== undefined && budget.storedRemainingBytes <= 0) {
    return { ok: false, note: `${storedBudgetNote()}.` };
  }
  // One deadline for the whole redirect chain and body stream.
  const timeout = AbortSignal.timeout(URL_DOWNLOAD_TIMEOUT_MS);
  let current = url;
  let response: http.IncomingMessage | null = null;
  for (let hop = 0; hop <= MAX_REDIRECTS; hop++) {
    // Layer 1 (pre-check): URL syntax, scheme, localhost, and IP literals.
    // Layer 2 (authoritative connection-time check in safeRequest):
    // single DNS resolution in custom `lookup` with `agent: false`.
    const blocked = precheckUrlReason(
      current,
      netOpts?.isAllowedAddress ?? isPublicIp,
    );
    if (blocked !== null) {
      return { ok: false, note: `Refusing image URL ${url}: ${blocked}.` };
    }
    try {
      response = await safeRequest(current, {
        ...netOpts,
        signal: timeout,
      });
    } catch (error) {
      if (error instanceof BlockedUrlError) {
        return { ok: false, note: `Refusing image URL ${url}: ${error.reason}.` };
      }
      const message = error instanceof Error ? error.message : String(error);
      return { ok: false, note: `Could not download image URL ${url}: ${message}` };
    }
    const status = response.statusCode ?? 0;
    if (REDIRECT_STATUSES.has(status)) {
      const locationHeader = response.headers.location;
      const location = Array.isArray(locationHeader)
        ? locationHeader[0]
        : locationHeader;
      destroyResponse(response);
      response = null;
      if (location === undefined || location === '') {
        return {
          ok: false,
          note: `Could not download image URL ${url}: redirect without a location.`,
        };
      }
      if (hop === MAX_REDIRECTS) {
        return {
          ok: false,
          note: `Image URL ${url} exceeds the ${MAX_REDIRECTS} redirect limit.`,
        };
      }
      try {
        current = new URL(location, current).toString();
      } catch {
        return {
          ok: false,
          note: `Could not download image URL ${url}: redirect to an invalid URL.`,
        };
      }
      continue;
    }
    break;
  }
  if (response === null) {
    return { ok: false, note: `Could not download image URL ${url}.` };
  }
  const status = response.statusCode ?? 0;
  if (status < 200 || status >= 300) {
    destroyResponse(response);
    return {
      ok: false,
      note: `Could not download image URL ${url}: HTTP ${status}`,
    };
  }
  const chunks: Buffer[] = [];
  const headChunks: Buffer[] = [];
  let headBytes = 0;
  let total = 0;
  let tmpPath: string | null = null;
  let tmpHandle: Awaited<ReturnType<typeof fs.open>> | null = null;
  // The stream stops at whichever ceiling is lowest: the per-download
  // ceiling, the remaining download-traffic budget, or the remaining
  // stored-image budget (so an image too large to store is aborted
  // immediately instead of downloaded in full first).
  const downloadCap = budget?.downloadRemainingBytes ?? URL_DOWNLOAD_MAX_BYTES;
  const storedCap = budget?.storedRemainingBytes ?? URL_DOWNLOAD_MAX_BYTES;
  const ceiling = Math.min(URL_DOWNLOAD_MAX_BYTES, downloadCap, storedCap);
  let data: Buffer;
  try {
    try {
      for await (const chunk of response) {
        const buf = chunk as Buffer;
        total += buf.byteLength;
        if (budget !== undefined) {
          budget.downloadRemainingBytes = Math.max(
            0,
            budget.downloadRemainingBytes - buf.byteLength,
          );
        }
        if (total > ceiling) {
          destroyResponse(response);
          if (budget !== undefined && ceiling < URL_DOWNLOAD_MAX_BYTES) {
            return {
              ok: false,
              note:
                downloadCap <= storedCap
                  ? `${downloadBudgetNote()}.`
                  : `${storedBudgetNote()}.`,
            };
          }
          return {
            ok: false,
            note: `Image URL ${url} exceeds the ${URL_DOWNLOAD_MAX_BYTES} byte download safety ceiling.`,
          };
        }
        if (headBytes < HEAD_SNIFF_BYTES) {
          const take = Math.min(buf.byteLength, HEAD_SNIFF_BYTES - headBytes);
          headChunks.push(Buffer.from(buf.subarray(0, take)));
          headBytes += take;
        }
        if (tmpHandle !== null) {
          await writeAll(tmpHandle, buf);
        } else {
          chunks.push(buf);
          if (total > DOWNLOAD_SPOOL_THRESHOLD_BYTES) {
            const candidatePath = path.join(
              os.tmpdir(),
              `memdrive-dl-${crypto.randomBytes(8).toString('hex')}`,
            );
            tmpHandle = await fs.open(candidatePath, 'wx+', 0o600);
            tmpPath = candidatePath;
            for (const c of chunks) {
              await writeAll(tmpHandle, c);
            }
            chunks.length = 0;
          }
        }
      }
    } catch (error) {
      destroyResponse(response);
      const message = error instanceof Error ? error.message : String(error);
      return { ok: false, note: `Download of image URL ${url} failed: ${message}` };
    }
    const rawContentType = response.headers['content-type'];
    const contentTypeStr = Array.isArray(rawContentType)
      ? rawContentType[0]
      : rawContentType;
    const headerType = contentTypeStr?.split(';')[0]?.trim().toLowerCase();
    if (tmpHandle !== null && tmpPath !== null) {
      await tmpHandle.close();
      tmpHandle = null;
      const head = Buffer.concat(headChunks, headBytes);
      if (detectImageMime(head) === null) {
        return {
          ok: false,
          note: `URL ${url} did not return image data (content-type: ${headerType ?? 'unknown'}).`,
        };
      }
      if (budget !== undefined && total > budget.storedRemainingBytes) {
        return { ok: false, note: `${storedBudgetNote()}.` };
      }
      try {
        data = await fs.readFile(tmpPath);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return { ok: false, note: `Download of image URL ${url} failed: ${message}` };
      }
    } else {
      data = Buffer.concat(chunks, total);
      chunks.length = 0;
    }
    // Magic bytes are authoritative: a `200 OK` HTML error page or a
    // 0-byte body served as `image/png` is not an image. detectImageMime
    // recognizes all six raster formats this code supports, so dropping
    // the Content-Type fallback excludes nothing legitimate.
    const mimeType = detectImageMime(data);
    if (mimeType === null) {
      return {
        ok: false,
        note: `URL ${url} did not return image data (content-type: ${headerType ?? 'unknown'}).`,
      };
    }
    // Only now, with a confirmed image, does the stored budget move:
    // a failed or non-image download has already spent traffic, but
    // must not block later legitimate images.
    if (budget !== undefined && data.length > budget.storedRemainingBytes) {
      return { ok: false, note: `${storedBudgetNote()}.` };
    }
    if (budget !== undefined) {
      budget.storedRemainingBytes = Math.max(0, budget.storedRemainingBytes - data.length);
    }
    let stem = 'image';
    try {
      stem = baseNameOf(new URL(current).pathname);
    } catch {
      // Keep the default stem.
    }
    return { ok: true, image: { data, mimeType, baseName: stem } };
  } finally {
    if (tmpHandle !== null) {
      await tmpHandle.close().catch(() => undefined);
    }
    if (tmpPath !== null) {
      await fs.unlink(tmpPath).catch(() => undefined);
    }
  }
}

const UNC_OR_DEVICE_PATH_NOTE =
  'network/UNC or device paths are not supported for local images; copy the file to a local drive or a mapped drive letter.';

/**
 * True when a user-supplied local path is a UNC share or Windows device
 * path (`\\...`, `//...`, `\\?\...`, `\\.\...`, `\\?\UNC\...`). Checked
 * before `cleanPath` because `path.posix.normalize` collapses leading `//`.
 */
function isUncOrDevicePath(spec: string): boolean {
  let p = spec.trim();
  for (;;) {
    if (
      p.length >= 2 &&
      ((p.startsWith('"') && p.endsWith('"')) ||
        (p.startsWith("'") && p.endsWith("'")))
    ) {
      p = p.slice(1, -1).trim();
      continue;
    }
    break;
  }
  return p.replace(/\\/g, '/').startsWith('//');
}

const MAX_LOCAL_SYMLINK_HOPS = 40;

/**
 * Walk `absolutePath` segment by segment using `lstat` and `readlink`
 * (never `stat` or `realpath`) so a symlink at any segment pointing to
 * a UNC share or device path is rejected before the OS follows it.
 */
async function resolveLocalImagePathNoUnc(
  absolutePath: string,
): Promise<{ path: string } | { uncOrDevice: true } | { errorMessage: string }> {
  if (isUncOrDevicePath(absolutePath)) {
    return { uncOrDevice: true };
  }
  const segSepRe = process.platform === 'win32' ? /[\\/]+/ : /\/+/;
  const initialParsed = path.parse(absolutePath);
  let resolvedPrefix = initialParsed.root;
  if (isUncOrDevicePath(resolvedPrefix)) {
    return { uncOrDevice: true };
  }
  const pending = absolutePath
    .slice(initialParsed.root.length)
    .split(segSepRe)
    .filter((s) => s.length > 0);

  let hops = 0;
  while (pending.length > 0) {
    const seg = pending.shift() as string;
    if (seg === '.') {
      continue;
    }
    if (seg === '..') {
      const parent = path.dirname(resolvedPrefix);
      if (parent.length >= path.parse(resolvedPrefix).root.length) {
        resolvedPrefix = parent;
      }
      continue;
    }
    const nextPath = path.join(resolvedPrefix, seg);
    if (isUncOrDevicePath(nextPath)) {
      return { uncOrDevice: true };
    }
    let lst: Awaited<ReturnType<typeof fs.lstat>>;
    try {
      lst = await fs.lstat(nextPath);
    } catch (error) {
      return { errorMessage: error instanceof Error ? error.message : String(error) };
    }
    if (!lst.isSymbolicLink()) {
      resolvedPrefix = nextPath;
      continue;
    }
    hops += 1;
    if (hops > MAX_LOCAL_SYMLINK_HOPS) {
      return { errorMessage: `Too many levels of symbolic links: ${absolutePath}` };
    }
    let target: string;
    try {
      target = await fs.readlink(nextPath);
    } catch (error) {
      return { errorMessage: error instanceof Error ? error.message : String(error) };
    }
    if (isUncOrDevicePath(target)) {
      return { uncOrDevice: true };
    }
    if (path.isAbsolute(target)) {
      const targetParsed = path.parse(target);
      const targetRest = target
        .slice(targetParsed.root.length)
        .split(segSepRe)
        .filter((s) => s.length > 0);
      // On Windows, a root-relative target (`\foo`) keeps the current drive root.
      if (targetParsed.root === '\\' || targetParsed.root === '/') {
        const prefixRoot = path.parse(resolvedPrefix).root;
        resolvedPrefix = prefixRoot || targetParsed.root;
      } else {
        resolvedPrefix = targetParsed.root;
      }
      pending.unshift(...targetRest);
    } else {
      pending.unshift(...target.split(segSepRe).filter((s) => s.length > 0));
    }
  }
  if (isUncOrDevicePath(resolvedPrefix)) {
    return { uncOrDevice: true };
  }
  return { path: resolvedPrefix };
}

/** Copy an image from a local OS path (outside the sandbox root is fine). */
async function copyLocalImage(spec: string, budget?: ImageSaveBudget): Promise<ImageAcquisition> {
  if (isUncOrDevicePath(spec)) {
    return { ok: false, note: UNC_OR_DEVICE_PATH_NOTE };
  }
  let cleaned: string;
  try {
    cleaned = cleanPath(spec);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, note: `Could not read local image ${spec}: ${message}` };
  }
  if (isUncOrDevicePath(cleaned)) {
    return { ok: false, note: UNC_OR_DEVICE_PATH_NOTE };
  }
  const cwd = process.cwd();
  if (!path.isAbsolute(cleaned) && isUncOrDevicePath(cwd)) {
    return { ok: false, note: UNC_OR_DEVICE_PATH_NOTE };
  }
  const absolute = path.isAbsolute(cleaned)
    ? path.resolve(cleaned)
    : path.resolve(cwd, cleaned);
  if (isUncOrDevicePath(absolute)) {
    return { ok: false, note: UNC_OR_DEVICE_PATH_NOTE };
  }
  const resolved = await resolveLocalImagePathNoUnc(absolute);
  if ('uncOrDevice' in resolved) {
    return { ok: false, note: UNC_OR_DEVICE_PATH_NOTE };
  }
  if ('errorMessage' in resolved) {
    return { ok: false, note: `Could not read local image ${spec}: ${resolved.errorMessage}` };
  }
  // Refuse non-regular files outright: /dev/zero, FIFOs, and device
  // nodes report size 0, so a size check alone cannot stop an
  // unbounded read — and readFile on them never returns. A stat
  // failure is reported directly instead of falling through to a
  // read that would report a different, misleading error.
  let st: Awaited<ReturnType<typeof fs.stat>>;
  try {
    st = await fs.stat(resolved.path);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, note: `Could not read local image ${spec}: ${message}` };
  }
  if (!st.isFile()) {
    return { ok: false, note: `Not a regular file: ${spec}` };
  }
  // A local file that alone would exhaust the save's stored budget is
  // refused without reading it.
  if (budget !== undefined && st.size > budget.storedRemainingBytes) {
    return { ok: false, note: `${storedBudgetNote()}.` };
  }
  let data: Buffer;
  try {
    data = await fs.readFile(resolved.path);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, note: `Could not read local image ${spec}: ${message}` };
  }
  // The file may have grown between stat and read; re-check after read.
  if (budget !== undefined && data.length > budget.storedRemainingBytes) {
    return { ok: false, note: `${storedBudgetNote()}.` };
  }
  // Magic bytes are authoritative here too: a file whose extension
  // claims an image but whose bytes are not one (or are empty) is not
  // an image. This also covers the extensionless case, which already
  // used detectImageMime alone.
  const mimeType = detectImageMime(data);
  if (mimeType === null) {
    return {
      ok: false,
      note: `File ${spec} does not look like an image (unrecognized format).`,
    };
  }
  if (budget !== undefined) {
    budget.storedRemainingBytes = Math.max(0, budget.storedRemainingBytes - data.length);
  }
  return { ok: true, image: { data, mimeType, baseName: baseNameOf(absolute) } };
}

/**
 * Acquire one image from a save_file `images` entry: "clipboard", an
 * http(s) URL, or a local file path. Never throws for expected failures;
 * returns a note instead so the caller can save the text and report.
 */
export async function acquireImage(
  spec: string,
  budget?: ImageSaveBudget,
  netOpts?: SafeRequestOptions,
): Promise<ImageAcquisition> {
  const trimmed = spec.trim();
  if (trimmed === 'clipboard') {
    const result = await readClipboardImage();
    if (result.kind === 'image') {
      if (budget !== undefined && result.data.length > budget.storedRemainingBytes) {
        return { ok: false, note: `${storedBudgetNote()}.` };
      }
      if (budget !== undefined) {
        budget.storedRemainingBytes = Math.max(0, budget.storedRemainingBytes - result.data.length);
      }
      return {
        ok: true,
        image: { data: result.data, mimeType: result.mimeType, baseName: 'clipboard' },
      };
    }
    if (result.kind === 'text') {
      return {
        ok: false,
        note: 'Clipboard holds text, not an image. The text was saved; re-copy a screenshot to attach an image.',
      };
    }
    if (result.kind === 'empty') {
      return {
        ok: false,
        note: 'Clipboard is empty. The text was saved; take a screenshot and retry to attach an image.',
      };
    }
    return { ok: false, note: result.reason };
  }
  if (/^https?:\/\//i.test(trimmed)) {
    return downloadImage(trimmed, budget, netOpts);
  }
  return copyLocalImage(trimmed, budget);
}

/** Base64 length (bytes) of `byteLength` binary bytes: ceil(n/3)*4. */
export function base64Length(byteLength: number): number {
  return Math.ceil(byteLength / 3) * 4;
}

/**
 * Largest binary size whose base64 encoding fits in `base64Budget`
 * bytes: floor(budget/4)*3.
 */
export function maxBinaryForBase64Budget(base64Budget: number): number {
  if (base64Budget <= 0) {
    return 0;
  }
  return Math.floor(base64Budget / 4) * 3;
}

/** Pixel dimensions parsed from an image container header. */
export interface ImageDimensions {
  width: number;
  height: number;
}

const JPEG_SOF_MARKERS = new Set([
  0xc0, 0xc1, 0xc2, 0xc3,
  0xc5, 0xc6, 0xc7,
  0xc9, 0xca, 0xcb,
  0xcd, 0xce, 0xcf,
]);

function parsePngDimensions(data: Buffer): ImageDimensions | null {
  if (data.length < 24 || data.subarray(12, 16).toString('ascii') !== 'IHDR') {
    return null;
  }
  const width = data.readUInt32BE(16);
  const height = data.readUInt32BE(20);
  return width > 0 && height > 0 ? { width, height } : null;
}

function parseGifDimensions(data: Buffer): ImageDimensions | null {
  if (data.length < 10) {
    return null;
  }
  const width = data.readUInt16LE(6);
  const height = data.readUInt16LE(8);
  return width > 0 && height > 0 ? { width, height } : null;
}

function parseBmpDimensions(data: Buffer): ImageDimensions | null {
  if (data.length < 18) {
    return null;
  }
  const dibSize = data.readUInt32LE(14);
  if (dibSize === 12) {
    if (data.length < 22) {
      return null;
    }
    const width = data.readUInt16LE(18);
    const height = data.readUInt16LE(20);
    return width > 0 && height > 0 ? { width, height } : null;
  }
  if (dibSize >= 40) {
    if (data.length < 26) {
      return null;
    }
    const width = data.readInt32LE(18);
    const height = Math.abs(data.readInt32LE(22));
    return width > 0 && height > 0 ? { width, height } : null;
  }
  return null;
}

function parseJpegDimensions(data: Buffer): ImageDimensions | null {
  if (data.length < 4 || data.readUInt8(0) !== 0xff || data.readUInt8(1) !== 0xd8) {
    return null;
  }
  let offset = 2;
  while (offset < data.length) {
    if (data.readUInt8(offset) !== 0xff) {
      return null;
    }
    while (offset < data.length && data.readUInt8(offset) === 0xff) {
      offset += 1;
    }
    if (offset >= data.length) {
      return null;
    }
    const marker = data.readUInt8(offset);
    offset += 1;
    // Standalone markers with no 2-byte length field: TEM (0x01) and RST0-RST7 (0xD0-0xD7).
    if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      continue;
    }
    // Stuffed 0x00, SOI (0xD8), EOI (0xD9), or SOS (0xDA) before SOF means no frame header.
    if (marker === 0x00 || marker === 0xd8 || marker === 0xd9 || marker === 0xda) {
      return null;
    }
    if (offset + 2 > data.length) {
      return null;
    }
    const segmentLength = data.readUInt16BE(offset);
    if (segmentLength < 2) {
      return null;
    }
    if (JPEG_SOF_MARKERS.has(marker)) {
      if (segmentLength < 8 || offset + 7 > data.length) {
        return null;
      }
      const height = data.readUInt16BE(offset + 3);
      const width = data.readUInt16BE(offset + 5);
      return width > 0 && height > 0 ? { width, height } : null;
    }
    offset += segmentLength;
  }
  return null;
}

function parseTiffDimensions(data: Buffer): ImageDimensions | null {
  if (data.length < 8) {
    return null;
  }
  const b0 = data.readUInt8(0);
  const b1 = data.readUInt8(1);
  let littleEndian: boolean;
  if (b0 === 0x49 && b1 === 0x49) {
    littleEndian = true;
  } else if (b0 === 0x4d && b1 === 0x4d) {
    littleEndian = false;
  } else {
    return null;
  }
  const readU16 = (off: number): number =>
    littleEndian ? data.readUInt16LE(off) : data.readUInt16BE(off);
  const readU32 = (off: number): number =>
    littleEndian ? data.readUInt32LE(off) : data.readUInt32BE(off);
  if (readU16(2) !== 42) {
    return null;
  }
  const ifdOffset = readU32(4);
  if (ifdOffset < 8 || ifdOffset + 2 > data.length) {
    return null;
  }
  const entryCount = readU16(ifdOffset);
  if (ifdOffset + 2 + entryCount * 12 > data.length) {
    return null;
  }
  let width: number | null = null;
  let height: number | null = null;
  for (let i = 0; i < entryCount; i += 1) {
    const entryOffset = ifdOffset + 2 + i * 12;
    const tag = readU16(entryOffset);
    if (tag !== 256 && tag !== 257) {
      continue;
    }
    const fieldType = readU16(entryOffset + 2);
    const count = readU32(entryOffset + 4);
    if (count !== 1) {
      return null;
    }
    let value: number;
    if (fieldType === 3) {
      // SHORT (16-bit) is left-justified in the 4-byte value/offset slot.
      value = readU16(entryOffset + 8);
    } else if (fieldType === 4) {
      value = readU32(entryOffset + 8);
    } else if (fieldType === 1) {
      value = data.readUInt8(entryOffset + 8);
    } else {
      return null;
    }
    if (tag === 256) {
      width = value;
    } else {
      height = value;
    }
    if (width !== null && height !== null) {
      break;
    }
  }
  if (width === null || height === null || width === 0 || height === 0) {
    return null;
  }
  return { width, height };
}

/**
 * Parse pixel dimensions from an image file header without decoding
 * pixel data. Supports PNG, JPEG, GIF, BMP, and TIFF; returns null
 * for unrecognized, truncated/corrupt, or undecoded (WebP) headers.
 * Never throws.
 */
export function parseImageDimensions(data: Buffer): ImageDimensions | null {
  try {
    switch (detectImageMime(data)) {
      case 'image/png':
        return parsePngDimensions(data);
      case 'image/jpeg':
        return parseJpegDimensions(data);
      case 'image/gif':
        return parseGifDimensions(data);
      case 'image/bmp':
        return parseBmpDimensions(data);
      case 'image/tiff':
        return parseTiffDimensions(data);
      default:
        return null;
    }
  } catch {
    return null;
  }
}

/**
 * Build a model-facing preview for an image buffer.
 *
 * Returns the original bytes untouched when under `maxBytes`. Before
 * decoding with jimp, inspects the container header for non-WebP images
 * and returns null immediately when dimensions cannot be read or exceed
 * `IMAGE_PREVIEW_MAX_PIXELS`. When over `maxBytes`,
 * decodes with jimp and tries, in order from mildest to most aggressive:
 * a JPEG re-encode at full resolution (scale 1.0 — for PNG screenshots
 * this alone usually drops below the cap with no detail lost to
 * resizing), trying JPEG qualities 95, 90, then 80 before any
 * shrinking, then progressively smaller resizes in 0.9 steps at
 * quality 80. The FIRST attempt that fits `maxBytes` wins,
 * so the preview stays close to the largest size the budget allows
 * instead of stopping far below it. At the minimum dimension, lower
 * JPEG qualities are tried before giving up. Returns null when no
 * preview can be produced (format the decoder cannot read, pixel count
 * over IMAGE_PREVIEW_MAX_PIXELS, or the
 * image stays over the cap even at minimum size/quality) — callers
 * then report a note instead. The on-disk original is never modified
 * by this function.
 *
 * `maxBytes` defaults to IMAGE_PREVIEW_MAX_BYTES (the per-image
 * ceiling). Callers pass a smaller value when the total response
 * budget leaves less room; the preview then fits the remaining budget
 * instead of the per-image ceiling.
 */
export async function makePreview(
  data: Buffer,
  mimeType: string,
  maxBytes: number = IMAGE_PREVIEW_MAX_BYTES,
): Promise<{ data: Buffer; mimeType: string } | null> {
  if (maxBytes <= 0) {
    return null;
  }
  const detectedMime = detectImageMime(data);
  if (detectedMime === null) {
    return null;
  }
  if (detectedMime !== 'image/webp') {
    const dims = parseImageDimensions(data);
    if (dims === null || dims.width * dims.height > IMAGE_PREVIEW_MAX_PIXELS) {
      return null;
    }
  }
  if (
    data.length > maxBytes &&
    (detectedMime === 'image/webp' || maxBytes < MIN_JPEG_HEADER_BYTES)
  ) {
    return null;
  }
  let image: Awaited<ReturnType<typeof Jimp.read>> | null = null;
  if (detectedMime !== 'image/webp') {
    try {
      image = await Jimp.read(data);
      if (image.width <= 0 || image.height <= 0) {
        return null;
      }
    } catch {
      return null;
    }
  }
  if (data.length <= maxBytes) {
    return { data, mimeType: detectedMime };
  }
  if (image === null) {
    return null;
  }
  try {
    // Scale 1.0: spend budget on quality first (95, 90, 80) before shrinking.
    for (const quality of IMAGE_PREVIEW_QUALITIES) {
      const preview = await image.getBuffer(IMAGE_PREVIEW_MIME, { quality });
      if (preview.length <= maxBytes) {
        return { data: preview, mimeType: IMAGE_PREVIEW_MIME };
      }
    }
    const tryMinQualities = async (
      target: Awaited<ReturnType<typeof Jimp.read>>,
    ): Promise<{ data: Buffer; mimeType: string } | null> => {
      for (const quality of [60, 40, 20]) {
        const smaller = await target.getBuffer(IMAGE_PREVIEW_MIME, { quality });
        if (smaller.length <= maxBytes) {
          return { data: smaller, mimeType: IMAGE_PREVIEW_MIME };
        }
      }
      return null;
    };
    if (
      image.width <= IMAGE_PREVIEW_MIN_DIMENSION &&
      image.height <= IMAGE_PREVIEW_MIN_DIMENSION
    ) {
      return await tryMinQualities(image);
    }
    const steps: Array<{ width: number; height: number }> = [];
    let scale = IMAGE_PREVIEW_SCALE_STEP;
    while (scale > 1e-6) {
      const width = Math.max(
        IMAGE_PREVIEW_MIN_DIMENSION,
        Math.round(image.width * scale),
      );
      const height = Math.max(
        IMAGE_PREVIEW_MIN_DIMENSION,
        Math.round(image.height * scale),
      );
      const prev = steps[steps.length - 1];
      if (prev === undefined || prev.width !== width || prev.height !== height) {
        steps.push({ width, height });
      }
      if (
        width <= IMAGE_PREVIEW_MIN_DIMENSION &&
        height <= IMAGE_PREVIEW_MIN_DIMENSION
      ) {
        break;
      }
      scale *= IMAGE_PREVIEW_SCALE_STEP;
    }
    if (steps.length === 0) {
      return await tryMinQualities(image);
    }
    const lastStep = steps[steps.length - 1] as { width: number; height: number };
    await yieldLoop();
    const minWorking = image.clone();
    minWorking.resize({ w: lastStep.width, h: lastStep.height });
    const minBuf = await minWorking.getBuffer(IMAGE_PREVIEW_MIME, {
      quality: IMAGE_PREVIEW_QUALITY,
    });
    if (minBuf.length > maxBytes) {
      return await tryMinQualities(minWorking);
    }
    let best: Buffer = minBuf;
    let lo = 0;
    let hi = steps.length - 2;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const step = steps[mid] as { width: number; height: number };
      await yieldLoop();
      const working = image.clone();
      working.resize({ w: step.width, h: step.height });
      const candidate = await working.getBuffer(IMAGE_PREVIEW_MIME, {
        quality: IMAGE_PREVIEW_QUALITY,
      });
      if (candidate.length <= maxBytes) {
        best = candidate;
        hi = mid - 1;
      } else {
        lo = mid + 1;
      }
    }
    return { data: best, mimeType: IMAGE_PREVIEW_MIME };
  } catch {
    return null;
  }
}

/** A local image reference found inside markdown text. */
export interface MarkdownImageRef {
  /** The raw src string as written in the document. */
  src: string;
  /**
   * Where the reference came from: `obsidian` embeds resolve by
   * filename across the whole library, the others relative to the
   * note's directory.
   */
  kind: 'inline' | 'img' | 'obsidian' | 'reference';
}

/**
 * Scan markdown for locally-referenced images: inline `![alt](src)`,
 * `<img src="...">`, Obsidian embeds `![[src]]` / `![[src|alt]]`, and
 * reference-style `![alt][id]` with `[id]: src` definitions. Remote
 * URLs (http/https), data: URIs, and fragments are skipped — only
 * files that could live next to the document.
 */
/**
 * Percent-encode characters in a Markdown image src that would break
 * the inline scan regex below (ASCII and non-ASCII whitespace, BOM,
 * parentheses), be parsed by Markdown renderers as a URL fragment or
 * query string ('#', '?'), or collide with percent-decoding ('%'),
 * while keeping non-whitespace CJK characters directly readable in
 * the note. Only newly saved notes carry encoded refs; pre-existing
 * notes with unencoded refs are unchanged.
 */
export function encodeImageSrc(src: string): string {
  return src.replace(/[\s\uFEFF\x00-\x1f\x7f%()#?<>\[\]"'`\\]/g, (ch) => {
    const code = ch.codePointAt(0)!;
    return code < 0x80
      ? `%${code.toString(16).toUpperCase().padStart(2, '0')}`
      : encodeURIComponent(ch);
  });
}

/** Strip CommonMark fenced code blocks (3+ backticks or tildes, <=3 spaces indent, CRLF-safe). */
function stripFencedCode(text: string): string {
  const out: string[] = [];
  let fence: { ch: string; len: number } | null = null;
  for (const line of text.split(/\r?\n/)) {
    const m = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line);
    const marker = m?.[1];
    const info = m?.[2] ?? '';
    if (fence === null) {
      // A backtick fence's info string may not contain a backtick.
      if (marker !== undefined && !(marker.startsWith('`') && info.includes('`'))) {
        fence = { ch: marker.charAt(0), len: marker.length };
        continue;
      }
    } else if (
      marker !== undefined &&
      marker.charAt(0) === fence.ch &&
      marker.length >= fence.len &&
      info.trim() === ''
    ) {
      fence = null;
      continue;
    }
    if (fence === null) {
      out.push(line);
    }
  }
  return out.join('\n');
}

/** Strip fenced code blocks and inline code spans before scanning markdown syntax. */
function stripMarkdownCode(text: string): string {
  return stripFencedCode(text).replace(
    /(?<!`)(`+)(?!`)[^\n]*?(?<!`)\1(?!`)/g,
    ' ',
  );
}

export function scanMarkdownImages(markdown: string): MarkdownImageRef[] {
  const refs: MarkdownImageRef[] = [];
  const seen = new Set<string>();
  const push = (src: string, kind: MarkdownImageRef['kind']): void => {
    let trimmed = src.trim();
    // A raw `#fragment` or `?query` is not part of the on-disk path
    // — but ONLY for Obsidian embeds (`![[pic.png#center]]` refers
    // to `pic.png`; `#` there is an anchor, not a filename). Inline,
    // `<img>`, and reference-style refs keep 0.7.0 behavior: the path
    // is literal, so `notes/C#/pic.png` stays `notes/C#/pic.png`.
    // Strip BEFORE decoding: a `#` that belongs to the filename
    // arrives percent-encoded (`%23`) and must survive the strip.
    if (kind === 'obsidian') {
      const hashAt = trimmed.indexOf('#');
      if (hashAt !== -1) {
        trimmed = trimmed.slice(0, hashAt).trim();
      }
      const queryAt = trimmed.indexOf('?');
      if (queryAt !== -1) {
        trimmed = trimmed.slice(0, queryAt).trim();
      }
    }
    // A raw `#fragment` is a same-document anchor, not a file — check
    // BEFORE decoding, so a percent-encoded `#` in a filename (`%23`)
    // survives (it decodes to `#` only after this check).
    if (trimmed.startsWith('#')) {
      return;
    }
    // Newly saved notes carry percent-encoded refs; decode them back
    // to the on-disk path. A bad escape falls back to the raw src.
    try {
      trimmed = decodeURIComponent(trimmed);
    } catch {
      // Keep the raw src.
    }
    // Obsidian embeds can point at notes and PDFs too (`![[Meeting
    // Notes]]`, `![[spec.pdf]]`): only image extensions are image
    // references, the rest must not reach read_file's image path.
    if (kind === 'obsidian' && mimeFromExtension(trimmed) === null) {
      return;
    }
    if (
      trimmed.length === 0 ||
      seen.has(`${kind}:${trimmed}`) ||
      /^https?:\/\//i.test(trimmed) ||
      /^data:/i.test(trimmed)
    ) {
      return;
    }
    seen.add(`${kind}:${trimmed}`);
    refs.push({ src: trimmed, kind });
  };
  const stripped = stripMarkdownCode(markdown);
  const matches: Array<{
    index: number;
    end: number;
    priority: number;
    src: string;
    kind: MarkdownImageRef['kind'];
  }> = [];
  const inlinePattern = /!\[(?:\\.|[^\]\\])*\]\(\s*(?:<([^>]*)>|([^\s)]+))(?:\s+(?:"[^"]*"|'[^']*'|\([^)]*\)))?\s*\)/g;
  for (const match of stripped.matchAll(inlinePattern)) {
    const src = match[1] ?? match[2];
    if (src !== undefined) {
      matches.push({
        index: match.index,
        end: match.index + match[0].length,
        priority: 0,
        src,
        kind: 'inline',
      });
    }
  }
  const imgPattern = /<img\b[^>]*\bsrc\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s>]+))/gi;
  for (const match of stripped.matchAll(imgPattern)) {
    const src = match[1] ?? match[2] ?? match[3];
    if (src !== undefined) {
      matches.push({
        index: match.index,
        end: match.index + match[0].length,
        priority: 1,
        src,
        kind: 'img',
      });
    }
  }
  // Obsidian embeds: `![[path.png]]` and `![[folder/img.png|alt]]`.
  const obsidianPattern = /!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g;
  for (const match of stripped.matchAll(obsidianPattern)) {
    const src = match[1];
    if (src !== undefined) {
      matches.push({
        index: match.index,
        end: match.index + match[0].length,
        priority: 2,
        src,
        kind: 'obsidian',
      });
    }
  }
  // Reference-style: collect `[id]: src` definitions first, then
  // resolve `![alt][id]` (and collapsed `![id][]`, where id = alt).
  // A definition may carry an optional title: `[id]: src "title"`.
  const refDefs = new Map<string, string>();
  for (const match of stripped.matchAll(
    /^\s*\[([^\]]+)\]:\s*(\S+)(?:\s+(?:"[^"]*"|'[^']*'|\([^)]*\)))?\s*$/gm,
  )) {
    const id = match[1];
    const src = match[2];
    if (id !== undefined && src !== undefined) {
      refDefs.set(id.toLowerCase(), src);
    }
  }
  const refPattern = /!\[(?:\\.|[^\]\\])*\]\[([^\]]*)\]/g;
  for (const match of stripped.matchAll(refPattern)) {
    const idRaw = match[1] as string;
    const altMatch = /^!\[((?:\\.|[^\]\\])*)\]/.exec(match[0]);
    const id = idRaw.length > 0 ? idRaw : (altMatch?.[1] ?? '');
    const src = refDefs.get(id.toLowerCase());
    if (src !== undefined) {
      matches.push({
        index: match.index,
        end: match.index + match[0].length,
        priority: 3,
        src,
        kind: 'reference',
      });
    }
  }
  matches.sort((a, b) => a.index - b.index || a.priority - b.priority);
  let lastEnd = 0;
  for (const item of matches) {
    if (item.index < lastEnd) {
      continue;
    }
    lastEnd = item.end;
    push(item.src, item.kind);
  }
  return refs;
}
