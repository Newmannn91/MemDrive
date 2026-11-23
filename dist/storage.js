/**
 * Storage backends.
 *
 * The tools talk only to the `StorageBackend` interface, using root-relative
 * POSIX-style paths ('' means the root itself). This MVP ships one backend,
 * `LocalStorageBackend` (plain filesystem). Local only, by design.
 *
 * The backend performs NO sandbox checks: callers must validate paths with
 * the sandbox module before calling in. The backend trusts its inputs.
 * The one exception is `search`: the walk discovers entries the caller
 * never validated, so it skips any entry whose real location escapes the
 * root (a symlink inside the root pointing outside it) instead of
 * reading its content.
 */
import crypto from 'node:crypto';
import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { atomicWrite } from './atomic.js';
import { SEARCH_CONTENT_MAX_BYTES } from './config.js';
import { MemDriveError, NotFoundError } from './errors.js';
import { assertInsideRealRoot, isInside } from './sandbox.js';
/**
 * Filesystem backend. `rootDir` must be an absolute path; callers guarantee
 * every relative path stays inside it (sandbox module).
 */
export class LocalStorageBackend {
    rootDir;
    constructor(rootDir) {
        this.rootDir = rootDir;
    }
    absolute(relPath) {
        return relPath === '' ? this.rootDir : path.join(this.rootDir, relPath);
    }
    async listDirectory(relDir) {
        const dirAbs = this.absolute(relDir);
        const dirents = await fs.readdir(dirAbs, { withFileTypes: true });
        const entries = [];
        for (const dirent of dirents) {
            // Skip orphaned atomic-write temp files: a crash between
            // write and rename leaves `.mdtmp-<pid>-<n>` behind, and it
            // must never be listed (or searched) as a note.
            if (dirent.name.startsWith('.mdtmp-')) {
                continue;
            }
            const entryAbs = path.join(dirAbs, dirent.name);
            let st;
            try {
                st = await fs.stat(entryAbs);
            }
            catch {
                // Entry vanished between readdir and stat (or broken symlink); skip it.
                continue;
            }
            const isDir = st.isDirectory();
            entries.push({
                name: dirent.name,
                relPath: (relDir === '' ? '' : `${relDir}/`) + dirent.name,
                type: isDir ? 'directory' : 'file',
                size: isDir ? 0 : st.size,
                modifiedAt: st.mtime.toISOString(),
                isSymlink: dirent.isSymbolicLink(),
            });
        }
        entries.sort((a, b) => {
            if (a.type !== b.type) {
                return a.type === 'directory' ? -1 : 1;
            }
            return a.name.localeCompare(b.name);
        });
        return entries;
    }
    async readFile(relPath) {
        return fs.readFile(this.absolute(relPath));
    }
    async hashFile(relPath) {
        const hash = crypto.createHash('sha256');
        const stream = createReadStream(this.absolute(relPath));
        for await (const chunk of stream) {
            hash.update(chunk);
        }
        return hash.digest('hex');
    }
    async stat(relPath) {
        try {
            const st = await fs.stat(this.absolute(relPath));
            return {
                size: st.size,
                isDirectory: st.isDirectory(),
                modifiedAt: st.mtime.toISOString(),
                blocks: st.blocks,
            };
        }
        catch (error) {
            if (error.code === 'ENOENT' || error.code === 'ENOTDIR') {
                return null;
            }
            throw error;
        }
    }
    async exists(relPath) {
        return (await this.stat(relPath)) !== null;
    }
    async writeFile(relPath, data, mode) {
        const target = this.absolute(relPath);
        let rootRealDir;
        try {
            rootRealDir = await fs.realpath(this.rootDir);
        }
        catch {
            rootRealDir = this.rootDir;
        }
        await assertInsideRealRoot(rootRealDir, target);
        await fs.mkdir(path.dirname(target), { recursive: true });
        if (mode === 'new') {
            try {
                const handle = await fs.open(target, 'wx');
                await writeAndCloseHandle(handle, data, { path: target });
                return { created: true };
            }
            catch (error) {
                if (error.code === 'EEXIST') {
                    throw new MemDriveError('ALREADY_EXISTS', `File already exists: ${relPath}. Use mode "overwrite" to replace it or "append" to add to it.`);
                }
                throw error;
            }
        }
        if (mode === 'append') {
            // Check read-only before open (non-root open('a') throws raw EACCES
            // before handle.stat() runs), and distinguish new vs existing open so
            // `created` is race-free.
            let handle;
            let created = false;
            const cleanupTarget = { path: null };
            for (let attempt = 0; attempt < 16; attempt++) {
                let preStat = null;
                try {
                    preStat = await fs.stat(target);
                }
                catch (error) {
                    if (error.code !== 'ENOENT') {
                        throw error;
                    }
                }
                if (preStat !== null) {
                    if (!preStat.isDirectory() && !(preStat.mode & 0o200)) {
                        throw new MemDriveError('READ_ONLY', `File is read-only: ${relPath}`);
                    }
                    try {
                        handle = await fs.open(target, fs.constants.O_WRONLY | fs.constants.O_APPEND);
                        created = false;
                        cleanupTarget.path = null;
                        break;
                    }
                    catch (error) {
                        if (error.code === 'ENOENT' && attempt < 15) {
                            continue;
                        }
                        throw error;
                    }
                }
                try {
                    handle = await fs.open(target, 'wx');
                    created = true;
                    cleanupTarget.path = target;
                    break;
                }
                catch (error) {
                    if (error.code === 'EEXIST') {
                        // Distinguish a concurrent file creation from a dangling symlink
                        // (O_CREAT|O_EXCL fails with EEXIST on a dangling symlink).
                        try {
                            const lst = await fs.lstat(target);
                            if (lst.isSymbolicLink()) {
                                handle = await fs.open(target, 'a');
                                created = true;
                                cleanupTarget.path = null;
                                break;
                            }
                        }
                        catch (lstatError) {
                            if (lstatError.code !== 'ENOENT') {
                                throw lstatError;
                            }
                        }
                        if (attempt < 15) {
                            continue;
                        }
                    }
                    throw error;
                }
            }
            if (handle === undefined) {
                throw new Error(`Could not open ${relPath} for append`);
            }
            const appendHandle = handle;
            await writeAndCloseHandle(appendHandle, data, cleanupTarget, async () => {
                const st = await appendHandle.stat();
                if (!(st.mode & 0o200)) {
                    throw new MemDriveError('READ_ONLY', `File is read-only: ${relPath}`);
                }
                if (created && cleanupTarget.path === null && st.size === 0) {
                    cleanupTarget.path = await fs.realpath(target).catch(() => null);
                }
            });
            return { created };
        }
        // Atomic replace via the shared helper (resolves symlinks with
        // a sandbox check, preserves permissions, refuses read-only
        // targets, fsyncs, retries Windows locks). Returns whether the
        // file existed, so overwrite of an existing file reports
        // created:false. ('new' uses 'wx' and 'append' is untouched.)
        const { existed } = await atomicWrite(target, data, { rootRealDir });
        return { created: !existed };
    }
    async ensureDir(relDir) {
        if (relDir === '') {
            return;
        }
        await fs.mkdir(this.absolute(relDir), { recursive: true });
    }
    async moveFile(srcRel, destRel) {
        const srcAbs = this.absolute(srcRel);
        const destAbs = this.absolute(destRel);
        let rootRealDir;
        try {
            rootRealDir = await fs.realpath(this.rootDir);
        }
        catch {
            rootRealDir = this.rootDir;
        }
        await assertInsideRealRoot(rootRealDir, srcAbs);
        await assertInsideRealRoot(rootRealDir, destAbs);
        try {
            const lst = await fs.lstat(srcAbs);
            await fs.mkdir(path.dirname(destAbs), { recursive: true });
            let linkTarget;
            if (lst.isSymbolicLink()) {
                linkTarget = await fs.readlink(srcAbs);
                let newTarget = linkTarget;
                if (!path.isAbsolute(linkTarget)) {
                    const realSrcParent = await fs.realpath(path.dirname(srcAbs));
                    const realDestParent = await fs.realpath(path.dirname(destAbs));
                    if (realSrcParent !== realDestParent) {
                        const resolvedTarget = path.resolve(realSrcParent, linkTarget);
                        newTarget = path.relative(realDestParent, resolvedTarget) || '.';
                    }
                }
                if (process.platform === 'win32') {
                    let targetIsDir = false;
                    try {
                        targetIsDir = (await fs.stat(srcAbs)).isDirectory();
                    }
                    catch {
                        targetIsDir = false;
                    }
                    await fs.symlink(newTarget, destAbs, targetIsDir ? 'dir' : 'file');
                }
                else {
                    await fs.symlink(newTarget, destAbs);
                }
            }
            else {
                try {
                    await fs.link(srcAbs, destAbs);
                }
                catch (linkError) {
                    const linkCode = linkError.code;
                    if (linkCode === 'EPERM' ||
                        linkCode === 'ENOSYS' ||
                        linkCode === 'EOPNOTSUPP' ||
                        linkCode === 'ENOTSUP' ||
                        linkCode === 'EXDEV' ||
                        linkCode === 'EMLINK') {
                        try {
                            await fs.copyFile(srcAbs, destAbs, fs.constants.COPYFILE_EXCL);
                        }
                        catch (copyError) {
                            if (copyError.code !== 'EEXIST') {
                                await fs.unlink(destAbs).catch(() => undefined); // only clean up a partial target we created
                            }
                            throw copyError;
                        }
                    }
                    else {
                        throw linkError;
                    }
                }
            }
            const abortSourceChanged = async (reason) => {
                const message = `${srcRel} changed during the move and was left in place (${reason}).`;
                try {
                    await fs.unlink(destAbs);
                }
                catch (cleanupError) {
                    if (cleanupError.code !== 'ENOENT') {
                        const cleanupMessage = cleanupError instanceof Error ? cleanupError.message : String(cleanupError);
                        throw new MemDriveError('SOURCE_CHANGED', `${message} Also failed to remove destination ${destRel}: ${cleanupMessage}`);
                    }
                }
                throw new MemDriveError('SOURCE_CHANGED', message);
            };
            try {
                // Verify srcAbs is still the same entry we started moving before
                // unlinking it: if a sync client or another process replaced it
                // after destAbs was created, unlinking would destroy the new
                // content. Exclude ctimeMs because fs.link updates ctime/nlink.
                const currentLst = await fs.lstat(srcAbs);
                const sameEntry = currentLst.dev === lst.dev &&
                    currentLst.ino === lst.ino &&
                    currentLst.isSymbolicLink() === lst.isSymbolicLink() &&
                    (currentLst.mode & fs.constants.S_IFMT) === (lst.mode & fs.constants.S_IFMT) &&
                    currentLst.size === lst.size &&
                    currentLst.mtimeMs === lst.mtimeMs;
                if (!sameEntry) {
                    return await abortSourceChanged('file metadata changed');
                }
                if (linkTarget !== undefined) {
                    let currentLinkTarget;
                    try {
                        currentLinkTarget = await fs.readlink(srcAbs);
                    }
                    catch (readlinkErr) {
                        if (readlinkErr.code === 'EINVAL') {
                            // Replaced by a non-symlink between lstat and readlink.
                            return await abortSourceChanged('symlink was replaced');
                        }
                        throw readlinkErr;
                    }
                    if (currentLinkTarget !== linkTarget) {
                        return await abortSourceChanged('symlink target changed');
                    }
                }
                await fs.unlink(srcAbs);
            }
            catch (e) {
                if (e instanceof MemDriveError && e.code === 'SOURCE_CHANGED') {
                    throw e;
                }
                if (e.code === 'ENOENT') {
                    // The source vanished after the target was created (a sync
                    // client or another process moved it). The target is now
                    // the only complete copy: the move has effectively
                    // succeeded, and rolling back would destroy the data.
                    return;
                }
                await fs.unlink(destAbs).catch(() => undefined); // rollback: delete the created target, source file remains
                throw e;
            }
        }
        catch (error) {
            const code = error.code;
            if (code === 'EEXIST') {
                throw new MemDriveError('ALREADY_EXISTS', `Destination already exists: ${destRel}. Refusing to overwrite.`);
            }
            if (code === 'ENOENT') {
                throw new NotFoundError(`File not found: ${srcRel}`);
            }
            throw error;
        }
    }
    /**
     * macOS only: a cloud file with no local content (evicted by
     * iCloud Drive / OneDrive) reports zero allocated blocks.
     * Protected so tests can simulate a dataless file on Linux,
     * where the platform gate never triggers.
     */
    async isDatalessFile(relPath, size) {
        if (process.platform !== 'darwin' || size === 0) {
            return false;
        }
        try {
            const st = await this.stat(relPath);
            return st !== null && st.blocks === 0;
        }
        catch {
            return false;
        }
    }
    async search(keyword, relDir, maxResults, fileTypes) {
        const needle = keyword.toLowerCase();
        const wantedExts = fileTypes !== undefined && fileTypes.length > 0
            ? new Set(fileTypes.map((e) => (e.startsWith('.') ? e.toLowerCase() : `.${e.toLowerCase()}`)))
            : null;
        const hits = [];
        // [Issue 4 & Issue 6] Refuse direct scan inside top-level archive/ so its
        // behavior matches the indexed search path.
        const lowerRelDir = relDir.toLowerCase();
        if (lowerRelDir === 'archive' || lowerRelDir.startsWith('archive/')) {
            return hits;
        }
        // Match the indexed walk: dot directories/files and `.icloud`
        // placeholders are never searched.
        if (relDir !== '' &&
            relDir !== '.' &&
            relDir
                .split('/')
                .some((seg) => (seg.startsWith('.') && seg !== '.') || seg.toLowerCase().endsWith('.icloud'))) {
            return hits;
        }
        // The search root was sandbox-checked by the caller, but the walk
        // discovers entries the caller never validated. A symlink inside
        // the root may point at a file outside it, and fs.stat follows
        // symlinks, so without this check its content would be read and
        // matched. Resolve the real root once and skip any entry whose
        // real location escapes it. Within-root symlinks behave like the
        // files they point at, matching read_file's sandbox rule.
        let rootRealDir;
        try {
            rootRealDir = await fs.realpath(this.rootDir);
        }
        catch {
            rootRealDir = this.rootDir;
        }
        // Resolve top-level archive dir realpaths case-insensitively (e.g.,
        // `Archive/` on case-sensitive Linux) so symlinks into any archive/
        // variant are excluded, not just the literal lowercase name.
        const archiveReals = [];
        try {
            const topEntries = await fs.readdir(this.rootDir, { withFileTypes: true });
            for (const dirent of topEntries) {
                if (dirent.name.toLowerCase() !== 'archive') {
                    continue;
                }
                const abs = path.join(this.rootDir, dirent.name);
                try {
                    const st = await fs.stat(abs);
                    if (st.isDirectory()) {
                        archiveReals.push(await fs.realpath(abs));
                    }
                }
                catch {
                    // Vanished or broken link: ignore.
                }
            }
        }
        catch {
            // Unreadable root: listDirectory below will handle it.
        }
        // When a relDir scope is given, confine the walk to its realpath.
        let scopeReal = null;
        if (relDir !== '') {
            try {
                scopeReal = await fs.realpath(this.absolute(relDir));
            }
            catch {
                scopeReal = null;
            }
        }
        const visitedDirs = new Set([rootRealDir]);
        if (scopeReal !== null) {
            visitedDirs.add(scopeReal);
        }
        const queue = [relDir];
        while (queue.length > 0 && hits.length < maxResults) {
            const current = queue.shift();
            let entries;
            try {
                entries = await this.listDirectory(current);
            }
            catch {
                continue;
            }
            for (const entry of entries) {
                if (hits.length >= maxResults) {
                    break;
                }
                // Skip dot entries (`.git`, `.obsidian`, dotfiles) and iCloud
                // placeholders (`.icloud`) to match LocalSearchIndex and never
                // trigger cloud downloads or leak hidden metadata.
                if (entry.name.startsWith('.') || entry.name.toLowerCase().endsWith('.icloud')) {
                    continue;
                }
                let real;
                try {
                    real = await fs.realpath(this.absolute(entry.relPath));
                    if (!isInside(rootRealDir, real)) {
                        continue;
                    }
                    if (archiveReals.some((ar) => isInside(ar, real))) {
                        continue;
                    }
                    if (scopeReal !== null && !isInside(scopeReal, real)) {
                        continue;
                    }
                }
                catch {
                    // Vanished or broken link: nothing readable here.
                    continue;
                }
                if (entry.type === 'directory') {
                    // [Issue 6] Case-insensitive check for top-level archive/
                    if (current === '' && entry.name.toLowerCase() === 'archive') {
                        continue;
                    }
                    if (!visitedDirs.has(real)) {
                        visitedDirs.add(real);
                        queue.push(entry.relPath);
                    }
                    continue;
                }
                if (wantedExts !== null && !wantedExts.has(path.extname(entry.name).toLowerCase())) {
                    continue;
                }
                if (entry.name.toLowerCase().includes(needle)) {
                    hits.push(entry.relPath);
                    continue;
                }
                // Content match for small files only; binary content and
                // macOS dataless cloud placeholders are skipped without reading.
                if (entry.size > 0 && entry.size <= SEARCH_CONTENT_MAX_BYTES) {
                    if (await this.isDatalessFile(entry.relPath, entry.size)) {
                        continue;
                    }
                    try {
                        const data = await this.readFile(entry.relPath);
                        if (looksLikeText(data) && data.toString('utf8').toLowerCase().includes(needle)) {
                            hits.push(entry.relPath);
                        }
                    }
                    catch {
                        // Unreadable file: name match already failed, move on.
                    }
                }
            }
        }
        return hits;
    }
}
/**
 * Call `handle.sync()`, ignoring filesystem-unsupported errors.
 * Some FUSE / network mounts throw EINVAL / ENOTSUP / EOPNOTSUPP after
 * the data is already written; the write + close path is sufficient
 * durability for a local note store. Any other error propagates.
 */
async function syncIgnoreUnsupported(handle) {
    try {
        await handle.sync();
    }
    catch (error) {
        const code = error.code;
        if (code !== 'EINVAL' && code !== 'ENOTSUP' && code !== 'EOPNOTSUPP') {
            throw error;
        }
    }
}
/**
 * Write `data`, fsync, and close an open file handle. On any failure,
 * close the handle first (so unlink does not fail on Windows due to an
 * open handle), remove any newly created target file, and re-throw.
 */
async function writeAndCloseHandle(handle, data, cleanupTarget, beforeWrite) {
    try {
        if (beforeWrite !== undefined) {
            await beforeWrite();
        }
        await handle.writeFile(data);
        await syncIgnoreUnsupported(handle);
        await handle.close();
    }
    catch (error) {
        await handle.close().catch(() => undefined);
        if (cleanupTarget.path !== null) {
            await fs.unlink(cleanupTarget.path).catch(() => undefined);
        }
        throw error;
    }
}
/** Heuristic: no NUL byte in the first 8 KB means probably text. */
function looksLikeText(data) {
    const sample = data.subarray(0, 8192);
    return !sample.includes(0);
}
