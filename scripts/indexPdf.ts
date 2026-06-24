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
const OCR_EXTRACTOR_VERSION = 2;
const CHUNK_SIZE = 1_400;
const CHUNK_OVERLAP = 180;
const EMBEDDING_BATCH_SIZE = 64;
const UPSERT_BATCH_SIZE = 64;

interface PdfDocument {
  filePath: string;
  fileName: string;
  relativePath: string;
  sourceBook: string;
  category: string;
  dictionaryType: string;
  title: string;
  cacheKey: string;
  sha256: string;
  extractionMode: 'layout' | 'columns' | 'ocr';
  extractorVersion: number;
}

interface TextChunk {
  text: string;
  pageNumber: number;
  chunkIndex: number;
  sectionTitle?: string;
  document: PdfDocument;
}

interface TextRecord {
  text: string;
  pageNumber: number;
  recordIndex: number;
  sectionTitle?: string;
  recordType: string;
  tags: string[];
  document: PdfDocument;
}

type IndexItem =
  | { kind: 'chunk'; chunk: TextChunk }
  | { kind: 'record'; record: TextRecord };

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
  extractionMode?: 'layout' | 'raw' | 'columns' | 'ocr';
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
  const records: TextRecord[] = [];
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
    const primaryReadMode = document.extractionMode === 'ocr' ? 'ocr' : 'text';
    for (let pageNumber = 1; pageNumber <= pagesToRead; pageNumber += 1) {
      if (
        primaryReadMode === 'ocr' &&
        (pageNumber === 1 || pageNumber % 25 === 0 || pageNumber === pagesToRead)
      ) {
        console.log(`[index-pdf] OCR ${document.relativePath}: page ${pageNumber}/${pagesToRead}`);
      }

      const text = normalizeExtractedText(await readCachedPageText(document, pageNumber, primaryReadMode));
      if (!text) continue;

      documentChunks.push(
        ...splitIntoChunks(text, CHUNK_SIZE, CHUNK_OVERLAP).map((chunkText, chunkIndex) => ({
          text: chunkText,
          pageNumber,
          chunkIndex,
          sectionTitle: sectionTitleForPage(document, pageNumber),
          document,
        }))
      );
    }

    if (documentChunks.length === 0 && ocrEnabled && primaryReadMode !== 'ocr') {
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
            sectionTitle: sectionTitleForPage(document, pageNumber),
            document,
          }))
        );
      }
    }

    const documentRecords = extractRecordsFromChunks(documentChunks);
    chunks.push(...documentChunks);
    records.push(...documentRecords);
    preparedDocuments.push({ document, pageCount: pagesToRead, chunks: documentChunks });
  }

  if (skippedCount > 0) {
    console.log(`[index-pdf] Skipped ${skippedCount} unchanged PDF files. Use --force to reindex.`);
  }
  console.log(`[index-pdf] Prepared ${chunks.length} text chunks and ${records.length} records.`);
  if (dryRun) {
    printDryRunSummary(chunks, records, preparedDocuments);
    return;
  }

  const indexItems: IndexItem[] = [
    ...chunks.map((chunk): IndexItem => ({ kind: 'chunk', chunk })),
    ...records.map((record): IndexItem => ({ kind: 'record', record })),
  ];

  if (indexItems.length === 0) {
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

  for (let index = 0; index < indexItems.length; index += EMBEDDING_BATCH_SIZE) {
    const batch = indexItems.slice(index, index + EMBEDDING_BATCH_SIZE);
    const vectors = await embeddings.embedDocuments(batch.map(indexItemText));
    const points = batch.map((item, batchIndex): QdrantPoint => ({
      id: stablePointId(item),
      vector: vectors[batchIndex],
      payload: payloadForIndexItem(item),
    }));

    for (let pointIndex = 0; pointIndex < points.length; pointIndex += UPSERT_BATCH_SIZE) {
      await qdrant.upsertPoints(
        config.qdrant.collection,
        points.slice(pointIndex, pointIndex + UPSERT_BATCH_SIZE)
      );
    }

    console.log(
      `[index-pdf] Indexed ${Math.min(index + batch.length, indexItems.length)} / ${indexItems.length} items.`
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
  const sourceBook = sourceBookForDocument(folder, normalizedFileName);
  const dictionaryType =
    isDialectSourceBook(sourceBook)
      ? 'dialect'
      : folder === 'vocabulary'
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
    sourceBook,
    category: folder,
    dictionaryType,
    title: fileName.replace(/\.pdf$/i, ''),
    cacheKey: createHash('sha256').update(relativePath).digest('hex').slice(0, 16),
    sha256: await fileSha256(filePath),
    extractionMode: extractionModeForDocument(sourceBook, dictionaryType),
    extractorVersion:
      sourceBook === 'maskouska_kryvicki_phrasebook'
        ? OCR_EXTRACTOR_VERSION
        : isOcrOnlySourceBook(sourceBook)
        ? OCR_EXTRACTOR_VERSION
        : dictionaryType === 'explanatory'
        ? COLUMN_EXTRACTOR_VERSION
        : DEFAULT_EXTRACTOR_VERSION,
  };
}

function sourceBookForDocument(folder: string, normalizedFileName: string): string {
  if (normalizedFileName.includes('vusacki') || normalizedFileName.includes('baradulin')) {
    return 'vushatski_slovazbor';
  }
  if (
    normalizedFileName.includes('maskouska') ||
    normalizedFileName.includes('kryvicki') ||
    normalizedFileName.includes('perakladchyk') ||
    normalizedFileName.includes('маскоу') ||
    normalizedFileName.includes('крывіц') ||
    normalizedFileName.includes('перакладчык')
  ) {
    return 'maskouska_kryvicki_phrasebook';
  }
  if (normalizedFileName.includes('viciebski') || normalizedFileName.includes('krajovy')) {
    return 'viciebski_krajovy_slounik';
  }
  if (
    normalizedFileName.includes('havorak') ||
    normalizedFileName.includes('paunoc') ||
    normalizedFileName.includes('zachodniaj') ||
    normalizedFileName.includes('pahranic')
  ) {
    if (normalizedFileName.includes('tom 1')) return 'pnz_havorki_tom_1';
    if (normalizedFileName.includes('tom 2')) return 'pnz_havorki_tom_2';
    if (normalizedFileName.includes('tom 3')) return 'pnz_havorki_tom_3';
    return 'pnz_havorki';
  }
  if (folder === 'proverbs') return 'proverbs_dictionary';
  if (folder === 'vocabulary') return 'orthographic_dictionary';
  if (folder === 'translate') return 'translation_dictionary';
  if (folder === 'tlumach') return 'explanatory_dictionary';
  return 'unknown';
}

function extractionModeForDocument(
  sourceBook: string,
  dictionaryType: string
): PdfDocument['extractionMode'] {
  if (isOcrOnlySourceBook(sourceBook)) return 'ocr';
  if (dictionaryType === 'explanatory') return 'columns';
  return 'layout';
}

function isOcrOnlySourceBook(sourceBook: string): boolean {
  return (
    sourceBook === 'maskouska_kryvicki_phrasebook' ||
    sourceBook === 'viciebski_krajovy_slounik' ||
    sourceBook === 'pnz_havorki_tom_1' ||
    sourceBook === 'pnz_havorki_tom_2' ||
    sourceBook === 'pnz_havorki_tom_3' ||
    sourceBook === 'pnz_havorki'
  );
}

function isDialectSourceBook(sourceBook: string): boolean {
  return (
    sourceBook === 'viciebski_krajovy_slounik' ||
    sourceBook === 'pnz_havorki_tom_1' ||
    sourceBook === 'pnz_havorki_tom_2' ||
    sourceBook === 'pnz_havorki_tom_3' ||
    sourceBook === 'pnz_havorki'
  );
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
  if (mode === 'ocr') return 'ocr';
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
      : `ocr-v${document.extractorVersion}`;
  const cachePath = path.join(TEXT_CACHE_ROOT, document.cacheKey, `${cachePrefix}-${pageNumber}.txt`);
  const cached = await readTextIfExists(cachePath);
  if (cached !== null) return cached;

  const legacyCached = await readLegacyCachedPageText(document, pageNumber, mode);
  if (legacyCached !== null) return legacyCached;

  const text =
    mode === 'text'
      ? await extractPageText(document, pageNumber)
      : await ocrPage(document.filePath, pageNumber);
  await mkdir(path.dirname(cachePath), { recursive: true });
  await writeFile(cachePath, text);
  return text;
}

async function readLegacyCachedPageText(
  document: PdfDocument,
  pageNumber: number,
  mode: 'text' | 'ocr'
): Promise<string | null> {
  const candidates =
    mode === 'ocr'
      ? [`ocr-${pageNumber}.txt`]
      : [`text-${pageNumber}.txt`, `text-raw-v2-${pageNumber}.txt`];

  for (const candidate of candidates) {
    const cached = await readTextIfExists(path.join(TEXT_CACHE_ROOT, document.cacheKey, candidate));
    if (cached !== null) return cached;
  }

  return null;
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
  const attempts = [
    { resolution: '220', psm: '6' },
    { resolution: '160', psm: '6' },
    { resolution: '160', psm: '11' },
  ];
  let lastError: unknown;

  for (const attempt of attempts) {
    try {
      return await ocrPageWithOptions(filePath, pageNumber, attempt.resolution, attempt.psm);
    } catch (error) {
      lastError = error;
      console.warn(
        `[index-pdf] OCR retry for ${path.basename(filePath)} page ${pageNumber} after failure at ${attempt.resolution}dpi/psm${attempt.psm}: ${error}`
      );
    }
  }

  console.warn(`[index-pdf] OCR skipped ${path.basename(filePath)} page ${pageNumber}: ${lastError}`);
  return '';
}

async function ocrPageWithOptions(
  filePath: string,
  pageNumber: number,
  resolution: string,
  psm: string
): Promise<string> {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), 'telegram-bot-pdf-ocr-'));
  const prefix = path.join(tempDir, 'page');

  try {
    await execFileAsync(
      'pdftoppm',
      ['-f', String(pageNumber), '-l', String(pageNumber), '-r', resolution, '-png', filePath, prefix],
      { maxBuffer: 16 * 1024 * 1024, timeout: 120_000 }
    );
    const images = (await readdir(tempDir))
      .filter((entry) => entry.endsWith('.png'))
      .sort();
    if (images.length === 0) return '';

    const { stdout } = await execFileAsync(
      'tesseract',
      [path.join(tempDir, images[0]), 'stdout', '-l', 'bel+rus+pol+eng', '--psm', psm],
      { maxBuffer: 16 * 1024 * 1024, timeout: 120_000 }
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

function extractRecordsFromChunks(chunks: TextChunk[]): TextRecord[] {
  const records: TextRecord[] = [];

  for (const chunk of chunks) {
    for (const candidate of splitChunkIntoRecordCandidates(chunk.text)) {
      const recordType = classifyRecord(candidate, chunk);
      if (!recordType) continue;

      records.push({
        text: candidate,
        pageNumber: chunk.pageNumber,
        recordIndex: records.length,
        sectionTitle: chunk.sectionTitle,
        recordType,
        tags: tagsForRecord(recordType, chunk.sectionTitle),
        document: chunk.document,
      });
    }
  }

  return deduplicateRecords(records);
}

function splitChunkIntoRecordCandidates(text: string): string[] {
  const candidates: string[] = [];
  let buffer = '';

  for (const rawLine of text.split('\n')) {
    const line = rawLine.trim();
    if (!line || isNonRecordLine(line)) {
      if (buffer) {
        candidates.push(buffer.trim());
        buffer = '';
      }
      continue;
    }

    const cleaned = cleanRecordLine(line);
    if (!cleaned) continue;

    if (!buffer) {
      buffer = cleaned;
      continue;
    }

    if (continuesPreviousLine(buffer, cleaned)) {
      buffer = `${buffer.replace(/-$/, '')}${buffer.endsWith('-') ? '' : ' '}${cleaned}`;
      continue;
    }

    candidates.push(buffer.trim());
    buffer = cleaned;
  }

  if (buffer) candidates.push(buffer.trim());

  return candidates
    .map((candidate) => candidate.replace(/\s+/g, ' ').trim())
    .filter((candidate) => candidate.length >= 4 && candidate.length <= 420);
}

function isNonRecordLine(line: string): boolean {
  return (
    /^---\s*PAGE\s+\d+\s*---$/i.test(line) ||
    /^\d{1,4}$/.test(line) ||
    /^\([^)]{1,80}\)[.!?]?$/.test(line) ||
    /^[*жx\s.·-]{2,}$/iu.test(line) ||
    /^запісана ад\b/iu.test(line) ||
    /^занатавана\b/iu.test(line)
  );
}

function cleanRecordLine(line: string): string {
  return line
    .replace(/\s{2,}/g, ' ')
    .replace(/^[•*—-]\s*/, '')
    .trim();
}

function continuesPreviousLine(previous: string, current: string): boolean {
  if (previous.endsWith('-')) return true;
  if (/[,:;—-]$/.test(previous)) return true;
  if (/^[а-яёіўўa-z]/iu.test(current) && !/[.!?)»]$/.test(previous)) return true;
  return false;
}

function classifyRecord(text: string, chunk: TextChunk): string | undefined {
  const normalizedText = normalizeRecordText(text);
  const normalizedSection = normalizeRecordText(chunk.sectionTitle || '');
  const sourceBook = chunk.document.sourceBook;

  if (sourceBook === 'vushatski_slovazbor') {
    if (!chunk.sectionTitle) return undefined;

    if (normalizedSection.includes('прыкмет')) return 'weather_sign';
    if (normalizedSection.includes('праклен') || normalizedSection.includes('грозьб')) return 'curse';
    if (normalizedSection.includes('пад ялдыч') || normalizedSection.includes('цвяліл')) return 'insult';
    if (normalizedSection.includes('добрыя пажадан')) return 'wish';
    if (normalizedSection.includes('прыказк') || normalizedSection.includes('прымаўк')) return 'proverb';

    if (matchesAny(normalizedText, ['праклен', 'грозьб', 'гразьб', 'пагроз', 'кляцьб'])) {
      return 'curse';
    }
    if (matchesAny(normalizedText, ['пад ялдыч', 'цвяліл', 'абраз', 'лаянк', 'зняваг'])) {
      return 'insult';
    }
    if (matchesAny(normalizedText, ['дай бог', 'каб бог', 'будзь здар', 'хай жа бог', 'добра вам жыць'])) {
      return 'wish';
    }
    if (matchesAny(normalizedText, ['пачуемся', 'развітан', 'бывай', 'з богам дамоў', 'адыходзячы з гасцей'])) {
      return 'farewell';
    }
    if (matchesAny(normalizedText, ['добры дзень', 'добры вечар', 'госць у хату', 'заходзьце', 'дабрыдзень', 'прывет'])) {
      return 'greeting';
    }
    if (matchesAny(normalizedText, ['прыкмет', 'жыццевыя назіран', 'калі ', 'як ', 'на ', 'дождж', 'мароз', 'вецер'])) {
      if ((chunk.sectionTitle || '').toLowerCase().includes('прыкмет')) return 'weather_sign';
    }
  }

  if (sourceBook === 'maskouska_kryvicki_phrasebook') {
    if (!chunk.sectionTitle || normalizedSection.includes('дадатк')) return undefined;
    if (/^[IVXІ]+\.?\s+/u.test(text.trim())) return undefined;
    if (normalizedSection.includes('прывітан') || matchesAny(normalizedText, ['дабрыдзень', 'добры вечар'])) {
      return 'greeting';
    }
    if (normalizedSection.includes('зычэнь') || matchesAny(normalizedText, ['дай бог', 'жадаю', 'зычу'])) {
      return 'wish';
    }
    if (/[—-]/u.test(text) && /[а-яёіўў]/iu.test(text)) {
      return 'phrase_equivalent';
    }
    return undefined;
  }

  if (chunk.document.dictionaryType === 'proverbs' || /прыказк|прымаўк/iu.test(chunk.sectionTitle || '')) {
    return 'proverb';
  }

  return undefined;
}

function matchesAny(value: string, needles: string[]): boolean {
  return needles.some((needle) => value.includes(normalizeRecordText(needle)));
}

function tagsForRecord(recordType: string, sectionTitle?: string): string[] {
  const base = RECORD_TYPE_TAGS[recordType] || [];
  return [...new Set([...base, ...(sectionTitle ? [sectionTitle] : [])])];
}

const RECORD_TYPE_TAGS: Record<string, string[]> = {
  greeting: ['вітанне', 'вітанні', 'прывітанне', 'здароўканне'],
  farewell: ['развітанне', 'бывай', 'пачуемся'],
  wish: ['пажаданне', 'зычэнне', 'добрыя пажаданні'],
  curse: ['праклён', 'праклёны', 'праклены', 'гразьбы', 'грозьбы', 'пагрозы'],
  insult: ['абраза', 'абразы', 'лаянка', 'знявага', 'цвялілкі'],
  threat: ['пагроза', 'пагрозы', 'грозьбы'],
  proverb: ['прыказка', 'прыказкі', 'прымаўка', 'прымаўкі'],
  weather_sign: ['прыкмета', 'прыкметы', 'надвор\'е', 'дождж', 'мароз', 'вецер'],
  phrase_equivalent: ['адпаведнік', 'адпаведнікі', 'фразэалёгія', 'перакладчык прыказак'],
};

function deduplicateRecords(records: TextRecord[]): TextRecord[] {
  const byKey = new Map<string, TextRecord>();
  for (const record of records) {
    const key = `${record.document.relativePath}:${record.recordType}:${normalizeRecordText(record.text)}`;
    if (!byKey.has(key)) byKey.set(key, record);
  }

  return [...byKey.values()].map((record, index) => ({ ...record, recordIndex: index }));
}

function sectionTitleForPage(document: PdfDocument, pageNumber: number): string | undefined {
  if (document.sourceBook === 'maskouska_kryvicki_phrasebook') {
    if (pageNumber >= 8 && pageNumber <= 20) return 'Фразы і прыказкі';
    if (pageNumber >= 20 && pageNumber <= 24) return 'Прывітаньні і зычэньні';
    if (pageNumber >= 25) return 'Дадаткі і пасьляслоўе';
    return undefined;
  }

  if (document.sourceBook !== 'vushatski_slovazbor') {
    if (document.dictionaryType === 'proverbs') return 'Прыказкі і прымаўкі';
    return undefined;
  }

  const printedPage = pageNumber - 2;
  if (printedPage >= 192 && printedPage <= 220) return 'Устойлівыя выразы';
  if (printedPage >= 221 && printedPage <= 227) return 'Параўнанні';
  if (printedPage >= 228 && printedPage <= 258) return 'Прыказкі й прымаўкі';
  if (printedPage >= 259 && printedPage <= 265) return 'Народны каляндар';
  if (printedPage >= 266 && printedPage <= 275) return 'Прыкметы, жыццёвыя назіранні, парады';
  if (printedPage >= 276 && printedPage <= 277) return 'Прыгаворкі й зычэнні';
  if (printedPage >= 278 && printedPage <= 290) return 'Звычай, госці, зычэнні і развітанні';
  if (printedPage >= 291 && printedPage <= 293) return 'Пытанні й воклічы';
  if (printedPage >= 294 && printedPage <= 295) return 'Праклёны й грозьбы';
  if (printedPage >= 296 && printedPage <= 297) return 'Пад’ялдычкі ды цвялілкі';
  if (printedPage >= 298 && printedPage <= 299) return 'Добрыя пажаданні';
  return undefined;
}

function normalizeRecordText(value: string): string {
  return value
    .toLowerCase()
    .replace(/[ё]/g, 'е')
    .replace(/[ў]/g, 'у')
    .replace(/[^\p{L}\p{N}\s'’]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function payloadForIndexItem(item: IndexItem): Record<string, unknown> {
  return item.kind === 'chunk' ? payloadForChunk(item.chunk) : payloadForRecord(item.record);
}

function payloadForChunk(chunk: TextChunk): Record<string, unknown> {
  return {
    payloadKind: 'chunk',
    text: chunk.text,
    source: chunk.document.relativePath,
    fileName: chunk.document.fileName,
    sourceBook: chunk.document.sourceBook,
    category: chunk.document.category,
    dictionaryType: chunk.document.dictionaryType,
    sectionTitle: chunk.sectionTitle,
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

function payloadForRecord(record: TextRecord): Record<string, unknown> {
  return {
    payloadKind: 'record',
    text: record.text,
    recordText: record.text,
    normalizedText: normalizeRecordText(record.text),
    recordType: record.recordType,
    tags: record.tags,
    source: record.document.relativePath,
    fileName: record.document.fileName,
    sourceBook: record.document.sourceBook,
    category: record.document.category,
    dictionaryType: record.document.dictionaryType,
    sectionTitle: record.sectionTitle,
    title: record.document.title,
    pdfSha256: record.document.sha256,
    recordIndex: record.recordIndex,
    loc: {
      pageNumber: record.pageNumber,
    },
  };
}

function stablePointId(item: IndexItem): string {
  const identity =
    item.kind === 'chunk'
      ? `${item.chunk.document.relativePath}:chunk:${item.chunk.pageNumber}:${item.chunk.chunkIndex}:${item.chunk.text}`
      : `${item.record.document.relativePath}:record:${item.record.pageNumber}:${item.record.recordIndex}:${item.record.text}`;
  const hash = createHash('sha256')
    .update(identity)
    .digest('hex')
    .slice(0, 32);

  return `${hash.slice(0, 8)}-${hash.slice(8, 12)}-${hash.slice(12, 16)}-${hash.slice(16, 20)}-${hash.slice(20)}`;
}

function indexItemText(item: IndexItem): string {
  return item.kind === 'chunk' ? item.chunk.text : item.record.text;
}

function normalizeForMatch(value: string): string {
  return value
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '');
}

function printDryRunSummary(chunks: TextChunk[], records: TextRecord[], preparedDocuments: PreparedDocument[]): void {
  const byType = new Map<string, number>();
  for (const chunk of chunks) {
    byType.set(chunk.document.dictionaryType, (byType.get(chunk.document.dictionaryType) || 0) + 1);
  }
  const recordsByType = new Map<string, number>();
  for (const record of records) {
    recordsByType.set(record.recordType, (recordsByType.get(record.recordType) || 0) + 1);
  }

  console.log('[index-pdf] Dry run summary:');
  for (const [type, count] of [...byType.entries()].sort()) {
    console.log(`  ${type}: ${count} chunks`);
  }
  for (const [type, count] of [...recordsByType.entries()].sort()) {
    console.log(`  record:${type}: ${count}`);
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
