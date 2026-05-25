import { config } from '../config';
import type { QdrantClient, Payload } from '../qdrant/client';
import type { RetrievedSource } from './types';

export class LexicalRetriever {
  constructor(private readonly qdrant: QdrantClient) {}

  async retrieve(
    query: string,
    limit = config.server.topK,
    options?: { fileNameIncludes?: string }
  ): Promise<RetrievedSource[]> {
    const terms = extractTerms(query);
    if (terms.length === 0) return [];

    const matches: RetrievedSource[] = [];
    let offset: string | number | undefined;

    do {
      const page = await this.qdrant.scrollPayloads(config.qdrant.collection, 256, offset);

      for (const point of page.points) {
        const payload = point.payload || {};
        if (!matchesFile(payload, options?.fileNameIncludes)) {
          continue;
        }

        const text = String(payload.text || '');
        const score = lexicalScore(text, terms);

        if (score > 0) {
          matches.push(toSource(payload, score));
        }
      }

      offset = page.nextOffset;
    } while (offset);

    return matches.sort((left, right) => right.score - left.score).slice(0, limit);
  }

  async retrievePageRange(options: {
    fileNameIncludes: string;
    startPage: number;
    endPage: number;
    limit: number;
  }): Promise<RetrievedSource[]> {
    const matches: RetrievedSource[] = [];
    let offset: string | number | undefined;

    do {
      const page = await this.qdrant.scrollPayloads(config.qdrant.collection, 256, offset);

      for (const point of page.points) {
        const payload = point.payload || {};
        if (!matchesFile(payload, options.fileNameIncludes)) {
          continue;
        }

        const pageNumber = readPageNumber(payload);
        if (!pageNumber || pageNumber < options.startPage || pageNumber > options.endPage) {
          continue;
        }

        matches.push(toSource(payload, 10 - Math.abs(pageNumber - options.startPage) * 0.01));
      }

      offset = page.nextOffset;
    } while (offset);

    return matches.sort((left, right) => (left.page || 0) - (right.page || 0)).slice(0, options.limit);
  }

  async retrieveAdjacent(options: {
    anchor: RetrievedSource;
    fileNameIncludes: string;
    forwardPages: number;
    backwardPages: number;
    limit: number;
  }): Promise<RetrievedSource[]> {
    const anchorPage = options.anchor.page;
    if (!anchorPage) {
      return [options.anchor];
    }

    return this.retrievePageRange({
      fileNameIncludes: options.fileNameIncludes,
      startPage: Math.max(1, anchorPage - options.backwardPages),
      endPage: anchorPage + options.forwardPages,
      limit: options.limit,
    });
  }
}

function matchesFile(payload: Payload, fileNameIncludes?: string): boolean {
  if (!fileNameIncludes) return true;

  const fileName = typeof payload.fileName === 'string' ? payload.fileName : '';
  return fileName.toLowerCase().includes(fileNameIncludes.toLowerCase());
}

function extractTerms(query: string): string[] {
  const normalized = normalize(query);
  const words = normalized
    .split(/\s+/)
    .map((word) => word.trim())
    .filter((word) => word.length >= 3);

  return [...new Set(words)];
}

function lexicalScore(text: string, terms: string[]): number {
  const normalizedText = normalize(text);
  let score = 0;

  for (const term of terms) {
    if (normalizedText.includes(term)) {
      score += term.length > 5 ? 2 : 1;
    }
  }

  return score / terms.length;
}

function normalize(value: string): string {
  return value
    .toLowerCase()
    .replace(/[ё]/g, 'е')
    .replace(/[ў]/g, 'у')
    .replace(/[^\p{L}\p{N}\s]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function toSource(payload: Payload, score: number): RetrievedSource {
  const pageNumber = readPageNumber(payload);

  return {
    text: String(payload.text || ''),
    score,
    source: stringValue(payload.source),
    fileName: stringValue(payload.fileName),
    page: numberValue(pageNumber),
  };
}

function readPageNumber(payload: Payload): number | undefined {
  const loc = payload.loc;
  const pageNumber = loc && typeof loc === 'object' && 'pageNumber' in loc ? loc.pageNumber : undefined;

  return typeof pageNumber === 'number' ? pageNumber : undefined;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === 'number' ? value : undefined;
}
