import { config } from '../config';
import type { EmbeddingsClient } from '../embeddings';
import type { QdrantClient, Payload } from '../qdrant/client';
import type { RetrievedSource } from './types';

export class QdrantRetriever {
  constructor(
    private readonly qdrant: QdrantClient,
    private readonly embeddings: EmbeddingsClient
  ) {}

  async retrieve(query: string, limit = config.server.topK): Promise<RetrievedSource[]> {
    const vector = await this.embeddings.embedQuery(query);
    if (vector.length !== config.embeddings.dimensions) {
      throw new Error(
        `Query embedding dimension mismatch: expected ${config.embeddings.dimensions}, got ${vector.length}`
      );
    }

    const results = await this.qdrant.search(config.qdrant.collection, vector, limit);

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
    source: stringValue(payload.source),
    fileName: stringValue(payload.fileName),
    page: numberValue(pageNumber),
  };
}

function stringValue(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === 'number' ? value : undefined;
}
