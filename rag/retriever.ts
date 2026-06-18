import { config } from '../config';
import type { EmbeddingsClient } from '../embeddings';
import type { Payload, PayloadFilter, QdrantClient } from '../qdrant/client';
import type { RetrievedSource } from './types';

export interface QdrantRetrieveOptions {
  filter?: PayloadFilter;
}

export class QdrantRetriever {
  constructor(
    private readonly qdrant: QdrantClient,
    private readonly embeddings: EmbeddingsClient
  ) {}

  async retrieve(
    query: string,
    limit = config.server.topK,
    options?: QdrantRetrieveOptions
  ): Promise<RetrievedSource[]> {
    const vector = await this.embeddings.embedQuery(query);
    if (vector.length !== config.embeddings.dimensions) {
      throw new Error(
        `Query embedding dimension mismatch: expected ${config.embeddings.dimensions}, got ${vector.length}`
      );
    }

    const results = await this.qdrant.search(config.qdrant.collection, vector, limit, options?.filter);

    return results
      .filter((result) => result.score >= config.search.minScore)
      .map((result) => toSource(result.payload, result.score))
      .filter((source) => source.text.trim().length > 0);
  }
}

function toSource(payload: Payload, score: number): RetrievedSource {
  const loc = payload.loc;
  const pageNumber =
    loc && typeof loc === 'object' && 'pageNumber' in loc ? loc.pageNumber : undefined;

  return {
    text: String(payload.text || ''),
    score,
    payloadKind: stringValue(payload.payloadKind),
    source: stringValue(payload.source),
    fileName: stringValue(payload.fileName),
    category: stringValue(payload.category),
    dictionaryType: stringValue(payload.dictionaryType),
    sourceBook: stringValue(payload.sourceBook),
    sectionTitle: stringValue(payload.sectionTitle),
    recordType: stringValue(payload.recordType),
    tags: stringArrayValue(payload.tags),
    title: stringValue(payload.title),
    page: numberValue(pageNumber),
  };
}

function stringValue(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === 'number' ? value : undefined;
}

function stringArrayValue(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const strings = value.filter((item): item is string => typeof item === 'string' && item.length > 0);
  return strings.length > 0 ? strings : undefined;
}
