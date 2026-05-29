import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { promisify } from 'node:util';
import { config } from '../config';
import { createEmbeddings } from '../embeddings';
import type { QdrantPoint } from '../qdrant/client';
import { createQdrantClient } from '../qdrant/index';

const execFileAsync = promisify(execFile);
const DEFAULT_PDF_ROOT = path.join(process.cwd(), 'pdf');
const TEXT_CACHE_ROOT = path.join(process.cwd(), '.rag-cache', 'pdf-text');
const MANIFEST_VERSION = 1;
const DEFAULT_EXTRACTOR_VERSION = 1;
const COLUMN_EXTRACTOR_VERSION = 5;
const CHUNK_SIZE = 1_400;
const CHUNK_OVERLAP = 180;
const EMBEDDING_BATCH_SIZE = 64;
const UPSERT_BATCH_SIZE = 64;

interface PdfDocument {
  filePath: string;
  fileName: string;
  relativePath: string;
  category: string;
  dictionaryType: string;
  title: string;
  cacheKey: string;
  sha256: string;
  extractionMode: 'layout' | 'columns';
  extractorVersion: number;
}

interface TextChunk {
  text: string;
  pageNumber: number;
  chunkIndex: number;
  document: PdfDocument;
}

interface IndexedManifest {
  version: number;
  documents: Record<string, IndexedManifestEntry>;
}

interface IndexedManifestEntry {
  sha256: string;
  pages: number;
  chunks: number;
  category: string;
  dictionaryType: string;
  qdrantCollection: string;
  embeddingModel: string;
  embeddingDimensions: number;
  extractionMode?: 'layout' | 'raw' | 'columns';
  extractorVersion?: number;
  indexedAt: string;
}

interface PreparedDocument {
  document: PdfDocument;
  pageCount: number;
  chunks: TextChunk[];
}

const args = new Set(process.argv.slice(2));
const dryRun = args.has('--dry-run');
const force = args.has('--force');
const syncManifest = args.has('--sync-manifest');
const ocrEnabled = !args.has('--no-ocr');
const rootArg = process.argv.find((arg) => arg.startsWith('--root='));
const includeArg = process.argv.find((arg) => arg.startsWith('--include='));
const maxPagesArg = process.argv.find((arg) => arg.startsWith('--max-pages='));
const pdfRoot = rootArg ? path.resolve(rootArg.slice('--root='.length)) : DEFAULT_PDF_ROOT;
const manifestPath = path.join(pdfRoot, '.indexed.json');
const includePathPart = includeArg?.slice('--include='.length);
const maxPages = maxPagesArg ? Number(maxPagesArg.slice('--max-pages='.length)) : undefined;

async function main(): Promise<void> {
  if (maxPages && !dryRun) {
    throw new Error('--max-pages can only be used with --dry-run.');
  }

  const documents = (await listPdfDocuments(pdfRoot)).filter((document) =>
    includePathPart ? document.relativePath.includes(includePathPart) : true
  );
  if (documents.length === 0) {
    throw new Error(`No PDF files found under ${pdfRoot}`);
  }

  console.log(`[index-pdf] Found ${documents.length} PDF files.`);
  const manifest = await readManifest(manifestPath);
  if (syncManifest) {
    await syncManifestFromQdrant(documents, manifest);
    await writeManifest(manifestPath, manifest);
    return;
  }

  const chunks: TextChunk[] = [];
  const preparedDocuments: PreparedDocument[] = [];
  let skippedCount = 0;

  for (const document of documents) {
    const pageCount = await readPageCount(document.filePath);
    const existing = manifest.documents[document.relativePath];
    if (!force && !maxPages && isAlreadyIndexed(document, existing)) {
      skippedCount += 1;
      console.log(
        `[index-pdf] Skipping ${document.relativePath}; already indexed (${existing.chunks} chunks).`
      );
      continue;
    }

    console.log(`[index-pdf] Extracting ${document.relativePath} (${pageCount} pages).`);
    const documentChunks: TextChunk[] = [];

    const pagesToRead = maxPages && Number.isFinite(maxPages) ? Math.min(pageCount, maxPages) : pageCount;
    for (let pageNumber = 1; pageNumber <= pagesToRead; pageNumber += 1) {
      const text = normalizeExtractedText(await readCachedPageText(document, pageNumber, 'text'));
      if (!text) continue;

      documentChunks.push(
        ...splitIntoChunks(text, CHUNK_SIZE, CHUNK_OVERLAP).map((chunkText, chunkIndex) => ({
          text: chunkText,
          pageNumber,
          chunkIndex,
          document,
        }))
      );
    }

    if (documentChunks.length === 0 && ocrEnabled) {
      console.log(`[index-pdf] OCR fallback for ${document.relativePath}.`);
      for (let pageNumber = 1; pageNumber <= pagesToRead; pageNumber += 1) {
        if (pageNumber === 1 || pageNumber % 25 === 0 || pageNumber === pagesToRead) {
          console.log(`[index-pdf] OCR ${document.relativePath}: page ${pageNumber}/${pagesToRead}`);
        }

        const text = normalizeExtractedText(await readCachedPageText(document, pageNumber, 'ocr'));
        if (!text) continue;

        documentChunks.push(
          ...splitIntoChunks(text, CHUNK_SIZE, CHUNK_OVERLAP).map((chunkText, chunkIndex) => ({
            text: chunkText,
            pageNumber,
            chunkIndex,
            document,
          }))
        );
      }
    }

    chunks.push(...documentChunks);
    preparedDocuments.push({ document, pageCount: pagesToRead, chunks: documentChunks });
  }

  if (skippedCount > 0) {
    console.log(`[index-pdf] Skipped ${skippedCount} unchanged PDF files. Use --force to reindex.`);
  }
  console.log(`[index-pdf] Prepared ${chunks.length} text chunks.`);
  if (dryRun) {
    printDryRunSummary(chunks, preparedDocuments);
    return;
  }

  if (chunks.length === 0) {
    console.log('[index-pdf] Nothing to index.');
    return;
  }

  const qdrant = createQdrantClient();
  const embeddings = createEmbeddings();
  await qdrant.ensureCollection(config.qdrant.collection, config.embeddings.dimensions);

  for (const item of preparedDocuments) {
    console.log(`[index-pdf] Deleting old Qdrant points for ${item.document.relativePath}.`);
    await qdrant.deletePoints(config.qdrant.collection, {
      must: [{ key: 'source', match: { value: item.document.relativePath } }],
    });
  }

  for (let index = 0; index < chunks.length; index += EMBEDDING_BATCH_SIZE) {
    const batch = chunks.slice(index, index + EMBEDDING_BATCH_SIZE);
    const vectors = await embeddings.embedDocuments(batch.map((chunk) => chunk.text));
    const points = batch.map((chunk, batchIndex): QdrantPoint => ({
      id: stablePointId(chunk),
      vector: vectors[batchIndex],
      payload: payloadForChunk(chunk),
    }));

    for (let pointIndex = 0; pointIndex < points.length; pointIndex += UPSERT_BATCH_SIZE) {
      await qdrant.upsertPoints(
        config.qdrant.collection,
        points.slice(pointIndex, pointIndex + UPSERT_BATCH_SIZE)
      );
    }

    console.log(
      `[index-pdf] Indexed ${Math.min(index + batch.length, chunks.length)} / ${chunks.length} chunks.`
    );
  }

  for (const item of preparedDocuments) {
    manifest.documents[item.document.relativePath] = {
      sha256: item.document.sha256,
      pages: item.pageCount,
      chunks: item.chunks.length,
      category: item.document.category,
      dictionaryType: item.document.dictionaryType,
      qdrantCollection: config.qdrant.collection,
      embeddingModel: config.embeddings.model,
      embeddingDimensions: config.embeddings.dimensions,
      extractionMode: item.document.extractionMode,
      extractorVersion: item.document.extractorVersion,
      indexedAt: new Date().toISOString(),
    };
  }
  await writeManifest(manifestPath, manifest);

  console.log(`[index-pdf] Done. Collection: ${config.qdrant.collection}`);
}

async function listPdfDocuments(root: string): Promise<PdfDocument[]> {
  const files = await walk(root);
  const pdfFiles = files.filter((filePath) => filePath.toLowerCase().endsWith('.pdf'));
  const documents = await Promise.all(pdfFiles.map((filePath) => documentMetadata(root, filePath)));

  return documents.sort((left, right) => left.relativePath.localeCompare(right.relativePath));
}

async function walk(directory: string): Promise<string[]> {
  const entries = await readdir(directory);
  const files: string[] = [];

  for (const entry of entries) {
    if (entry.startsWith('.')) continue;

    const fullPath = path.join(directory, entry);
    const entryStat = await stat(fullPath);
    if (entryStat.isDirectory()) {
      files.push(...(await walk(fullPath)));
      continue;
    }

    files.push(fullPath);
  }

  return files;
}

async function documentMetadata(root: string, filePath: string): Promise<PdfDocument> {
  const relativePath = path.relative(root, filePath);
  const [folder = 'general'] = relativePath.split(path.sep);
  const fileName = path.basename(filePath);
  const normalizedFileName = normalizeForMatch(fileName);
  const dictionaryType =
    folder === 'vocabulary'
      ? 'orthographic'
      : folder === 'translate'
      ? 'translation'
      : folder === 'tlumach'
      ? 'explanatory'
      : normalizedFileName.includes('vusacki') || normalizedFileName.includes('baradulin')
      ? 'dialect'
      : folder === 'proverbs'
      ? 'proverbs'
      : 'general';

  return {
    filePath,
    fileName,
    relativePath,
    category: folder,
    dictionaryType,
    title: fileName.replace(/\.pdf$/i, ''),
    cacheKey: createHash('sha256').update(relativePath).digest('hex').slice(0, 16),
    sha256: await fileSha256(filePath),
    extractionMode: dictionaryType === 'explanatory' ? 'columns' : 'layout',
    extractorVersion:
      dictionaryType === 'explanatory' ? COLUMN_EXTRACTOR_VERSION : DEFAULT_EXTRACTOR_VERSION,
  };
}

async function fileSha256(filePath: string): Promise<string> {
  return createHash('sha256').update(await readFile(filePath)).digest('hex');
}

function isAlreadyIndexed(document: PdfDocument, entry?: IndexedManifestEntry): entry is IndexedManifestEntry {
  return Boolean(
    entry &&
      entry.sha256 === document.sha256 &&
      entry.qdrantCollection === config.qdrant.collection &&
      entry.embeddingModel === config.embeddings.model &&
      entry.embeddingDimensions === config.embeddings.dimensions &&
      normalizeManifestExtractionMode(entry.extractionMode) === document.extractionMode &&
      (entry.extractorVersion || DEFAULT_EXTRACTOR_VERSION) === document.extractorVersion
  );
}

function normalizeManifestExtractionMode(mode: IndexedManifestEntry['extractionMode']): PdfDocument['extractionMode'] {
  if (mode === 'columns') return 'columns';
  return 'layout';
}

async function readManifest(filePath: string): Promise<IndexedManifest> {
  const text = await readTextIfExists(filePath);
  if (!text) {
    return { version: MANIFEST_VERSION, documents: {} };
  }

  try {
    const parsed = JSON.parse(text) as Partial<IndexedManifest>;
    return {
      version: parsed.version || MANIFEST_VERSION,
      documents: parsed.documents || {},
    };
  } catch (error) {
    throw new Error(`Could not parse index manifest ${filePath}: ${error}`);
  }
}

async function syncManifestFromQdrant(
  documents: PdfDocument[],
  manifest: IndexedManifest
): Promise<void> {
  const qdrant = createQdrantClient();
  await qdrant.ensureCollection(config.qdrant.collection, config.embeddings.dimensions);

  for (const document of documents) {
    const [pageCount, chunkCount] = await Promise.all([
      readPageCount(document.filePath),
      countQdrantPointsForSource(document.relativePath),
    ]);

    if (chunkCount === 0) {
      console.log(`[index-pdf] Manifest sync skipped ${document.relativePath}; no Qdrant points.`);
      continue;
    }

    manifest.documents[document.relativePath] = {
      sha256: document.sha256,
      pages: pageCount,
      chunks: chunkCount,
      category: document.category,
      dictionaryType: document.dictionaryType,
      qdrantCollection: config.qdrant.collection,
      embeddingModel: config.embeddings.model,
      embeddingDimensions: config.embeddings.dimensions,
      extractionMode: document.extractionMode,
      extractorVersion: document.extractorVersion,
      indexedAt: new Date().toISOString(),
    };
    console.log(`[index-pdf] Manifest synced ${document.relativePath}: ${chunkCount} chunks.`);
  }
}

async function countQdrantPointsForSource(source: string): Promise<number> {
  const qdrant = createQdrantClient();
  let offset: string | number | undefined;
  let count = 0;

  do {
    const page = await qdrant.scrollPayloads(config.qdrant.collection, 256, offset, {
      must: [{ key: 'source', match: { value: source } }],
    });
    count += page.points.length;
    offset = page.nextOffset;
  } while (offset);

  return count;
}

async function writeManifest(filePath: string, manifest: IndexedManifest): Promise<void> {
  await writeFile(
    filePath,
    `${JSON.stringify({ ...manifest, version: MANIFEST_VERSION }, null, 2)}\n`,
    'utf8'
  );
}

async function readPageCount(filePath: string): Promise<number> {
  const { stdout } = await execFileAsync('pdfinfo', [filePath], { maxBuffer: 1024 * 1024 });
  const match = stdout.match(/^Pages:\s+(\d+)/im);
  if (!match) {
    throw new Error(`Could not read page count from ${filePath}`);
  }

  return Number(match[1]);
}

async function extractPageText(document: PdfDocument, pageNumber: number): Promise<string> {
  if (document.extractionMode === 'columns') {
    return extractPageTextByColumns(document.filePath, pageNumber);
  }

  const { stdout } = await execFileAsync(
    'pdftotext',
    [
      '-f',
      String(pageNumber),
      '-l',
      String(pageNumber),
      '-layout',
      document.filePath,
      '-',
    ],
    { maxBuffer: 16 * 1024 * 1024 }
  );

  return stdout;
}

async function extractPageTextByColumns(filePath: string, pageNumber: number): Promise<string> {
  const { stdout } = await execFileAsync(
    'pdftotext',
    ['-f', String(pageNumber), '-l', String(pageNumber), '-bbox-layout', filePath, '-'],
    { maxBuffer: 32 * 1024 * 1024 }
  );
  const pageMatch = stdout.match(/<page[^>]*width="([^"]+)"[^>]*height="([^"]+)"/);
  const pageWidth = pageMatch ? Number(pageMatch[1]) : 0;
  const words = parseBboxWords(stdout);
  const columns = groupWordsIntoColumns(words, pageWidth);

  return columns
    .map((column) => column.map((line) => line.map((word) => word.text).join(' ')).join('\n'))
    .filter(Boolean)
    .join('\n\n');
}

interface BboxWord {
  xMin: number;
  yMin: number;
  text: string;
}

function parseBboxWords(xml: string): BboxWord[] {
  const words: BboxWord[] = [];
  const wordPattern = /<word\b([^>]*)>([\s\S]*?)<\/word>/g;
  let match: RegExpExecArray | null;

  while ((match = wordPattern.exec(xml))) {
    const attrs = match[1];
    const xMin = numberAttr(attrs, 'xMin');
    const yMin = numberAttr(attrs, 'yMin');
    if (xMin === null || yMin === null) continue;

    const text = decodeXmlText(match[2]).trim();
    if (!text) continue;

    words.push({ xMin, yMin, text });
  }

  return words;
}

function groupWordsIntoColumns(words: BboxWord[], pageWidth: number): BboxWord[][][] {
  const contentWords = words.filter((word) => !isLikelyPageNumber(word, pageWidth));
  const usableWidth = pageWidth || Math.max(...contentWords.map((word) => word.xMin), 1);
  const boundaries = [usableWidth * 0.34, usableWidth * 0.665];
  const columnWords: BboxWord[][] = [[], [], []];

  for (const word of contentWords) {
    const columnIndex = word.xMin < boundaries[0] ? 0 : word.xMin < boundaries[1] ? 1 : 2;
    columnWords[columnIndex].push(word);
  }

  return columnWords.map(groupWordsIntoLines);
}

function groupWordsIntoLines(words: BboxWord[]): BboxWord[][] {
  const sorted = [...words].sort((left, right) => left.yMin - right.yMin || left.xMin - right.xMin);
  const lines: BboxWord[][] = [];

  for (const word of sorted) {
    const currentLine = lines.at(-1);
    if (!currentLine || Math.abs(currentLine[0].yMin - word.yMin) > 3.2) {
      lines.push([word]);
      continue;
    }

    currentLine.push(word);
  }

  return lines.map((line) => line.sort((left, right) => left.xMin - right.xMin));
}

function isLikelyPageNumber(word: BboxWord, pageWidth: number): boolean {
  if (!/^\d+$/.test(word.text)) return false;
  if (word.yMin > 45) return false;
  if (!pageWidth) return true;

  return word.xMin > pageWidth * 0.35 && word.xMin < pageWidth * 0.65;
}

function numberAttr(attrs: string, name: string): number | null {
  const match = attrs.match(new RegExp(`${name}="([^"]+)"`));
  if (!match) return null;

  const value = Number(match[1]);
  return Number.isFinite(value) ? value : null;
}

function decodeXmlText(value: string): string {
  return value
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, '&');
}

async function readCachedPageText(
  document: PdfDocument,
  pageNumber: number,
  mode: 'text' | 'ocr'
): Promise<string> {
  const cachePrefix =
    mode === 'text'
      ? `text-${document.extractionMode}-v${document.extractorVersion}`
      : `ocr-v${DEFAULT_EXTRACTOR_VERSION}`;
  const cachePath = path.join(TEXT_CACHE_ROOT, document.cacheKey, `${cachePrefix}-${pageNumber}.txt`);
  const cached = await readTextIfExists(cachePath);
  if (cached !== null) return cached;

  const text =
    mode === 'text'
      ? await extractPageText(document, pageNumber)
      : await ocrPage(document.filePath, pageNumber);
  await mkdir(path.dirname(cachePath), { recursive: true });
  await writeFile(cachePath, text);
  return text;
}

async function readTextIfExists(filePath: string): Promise<string | null> {
  try {
    return await readFile(filePath, 'utf8');
  } catch (error) {
    if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') {
      return null;
    }
    throw error;
  }
}

async function ocrPage(filePath: string, pageNumber: number): Promise<string> {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), 'telegram-bot-pdf-ocr-'));
  const prefix = path.join(tempDir, 'page');

  try {
    await execFileAsync(
      'pdftoppm',
      ['-f', String(pageNumber), '-l', String(pageNumber), '-r', '220', '-png', filePath, prefix],
      { maxBuffer: 16 * 1024 * 1024, timeout: 45_000 }
    );
    const images = (await readdir(tempDir))
      .filter((entry) => entry.endsWith('.png'))
      .sort();
    if (images.length === 0) return '';

    const { stdout } = await execFileAsync(
      'tesseract',
      [path.join(tempDir, images[0]), 'stdout', '-l', 'bel+rus+pol+eng', '--psm', '6'],
      { maxBuffer: 16 * 1024 * 1024, timeout: 45_000 }
    );
    return stdout;
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}

function splitIntoChunks(text: string, chunkSize: number, overlap: number): string[] {
  if (text.length <= chunkSize) return [text];

  const chunks: string[] = [];
  let start = 0;

  while (start < text.length) {
    const hardEnd = Math.min(text.length, start + chunkSize);
    const softEnd = findChunkBoundary(text, start, hardEnd);
    const chunk = text.slice(start, softEnd).trim();
    if (chunk) chunks.push(chunk);
    if (softEnd >= text.length) break;
    start = Math.max(softEnd - overlap, start + 1);
  }

  return chunks;
}

function findChunkBoundary(text: string, start: number, hardEnd: number): number {
  const slice = text.slice(start, hardEnd);
  const paragraphBreak = slice.lastIndexOf('\n\n');
  if (paragraphBreak > CHUNK_SIZE * 0.45) return start + paragraphBreak;

  const sentenceBreak = Math.max(slice.lastIndexOf('. '), slice.lastIndexOf('; '), slice.lastIndexOf('! '));
  if (sentenceBreak > CHUNK_SIZE * 0.55) return start + sentenceBreak + 1;

  const space = slice.lastIndexOf(' ');
  return space > CHUNK_SIZE * 0.65 ? start + space : hardEnd;
}

function normalizeExtractedText(text: string): string {
  return text
    .replace(/\r/g, '\n')
    .replace(/([\p{L}])-\n([\p{Ll}])/gu, '$1$2')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{4,}/g, '\n\n')
    .replace(/[ \t]{2,}/g, ' ')
    .trim();
}

function payloadForChunk(chunk: TextChunk): Record<string, unknown> {
  return {
    text: chunk.text,
    source: chunk.document.relativePath,
    fileName: chunk.document.fileName,
    category: chunk.document.category,
    dictionaryType: chunk.document.dictionaryType,
    title: chunk.document.title,
    pdfSha256: chunk.document.sha256,
    extractionMode: chunk.document.extractionMode,
    extractorVersion: chunk.document.extractorVersion,
    chunkIndex: chunk.chunkIndex,
    loc: {
      pageNumber: chunk.pageNumber,
    },
  };
}

function stablePointId(chunk: TextChunk): string {
  const hash = createHash('sha256')
    .update(`${chunk.document.relativePath}:${chunk.pageNumber}:${chunk.chunkIndex}:${chunk.text}`)
    .digest('hex')
    .slice(0, 32);

  return `${hash.slice(0, 8)}-${hash.slice(8, 12)}-${hash.slice(12, 16)}-${hash.slice(16, 20)}-${hash.slice(20)}`;
}

function normalizeForMatch(value: string): string {
  return value
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '');
}

function printDryRunSummary(chunks: TextChunk[], preparedDocuments: PreparedDocument[]): void {
  const byType = new Map<string, number>();
  for (const chunk of chunks) {
    byType.set(chunk.document.dictionaryType, (byType.get(chunk.document.dictionaryType) || 0) + 1);
  }

  console.log('[index-pdf] Dry run summary:');
  for (const [type, count] of [...byType.entries()].sort()) {
    console.log(`  ${type}: ${count} chunks`);
  }
  if (preparedDocuments.length > 0) {
    console.log('[index-pdf] Files to index:');
    for (const item of preparedDocuments) {
      console.log(`  ${item.document.relativePath}: ${item.chunks.length} chunks`);
    }
  }
}

main().catch((error) => {
  console.error('[index-pdf] Failed:', error);
  process.exitCode = 1;
});
