import { config } from '../config';
import type { PayloadFilter } from '../qdrant/client';
import type { RetrievedSource } from './types';
import type { LexicalRetriever } from './lexicalRetriever';
import type { QdrantRetriever } from './retriever';

export interface HybridRetrieveOptions {
  fileNameIncludes?: string;
  filter?: PayloadFilter;
}

// Standard k value from the RRF paper. Larger k dampens rank differences,
// smaller k makes top ranks dominate more strongly.
const RRF_K = 60;

export class HybridRetriever {
  constructor(
    private readonly vectorRetriever: QdrantRetriever,
    private readonly lexicalRetriever: LexicalRetriever
  ) {}

  async retrieve(
    query: string,
    limit = config.server.topK,
    options?: HybridRetrieveOptions
  ): Promise<RetrievedSource[]> {
    const [vectorSources, lexicalSources] = await Promise.allSettled([
      this.vectorRetriever.retrieve(query, limit, options ? { filter: options.filter } : undefined),
      this.lexicalRetriever.retrieve(query, limit, options),
    ]);

    if (vectorSources.status === 'rejected') {
      console.warn(`Vector retrieval failed: ${vectorSources.reason}`);
    }

    const vectorList = vectorSources.status === 'fulfilled' ? vectorSources.value : [];
    const lexicalList = lexicalSources.status === 'fulfilled' ? lexicalSources.value : [];

    const fused = reciprocalRankFusion(
      [
        { label: 'vector', sources: vectorList },
        { label: 'lexical', sources: lexicalList },
      ],
      RRF_K
    );

    return fused
      .filter((source) => matchesSourceFile(source, options?.fileNameIncludes))
      .slice(0, limit);
  }

  async retrievePageRange(options: {
    fileNameIncludes: string;
    startPage: number;
    endPage: number;
    limit: number;
  }): Promise<RetrievedSource[]> {
    return this.lexicalRetriever.retrievePageRange(options);
  }

  async retrieveAdjacent(options: {
    anchor: RetrievedSource;
    fileNameIncludes: string;
    forwardPages: number;
    backwardPages: number;
    limit: number;
  }): Promise<RetrievedSource[]> {
    return this.lexicalRetriever.retrieveAdjacent(options);
  }
}

interface RankedList {
  label: 'vector' | 'lexical';
  sources: RetrievedSource[];
}

function reciprocalRankFusion(lists: RankedList[], k: number): RetrievedSource[] {
  const byKey = new Map<string, RetrievedSource & { rrf: number }>();

  for (const list of lists) {
    for (const [index, source] of list.sources.entries()) {
      const rank = index + 1;
      const contribution = 1 / (k + rank);
      const key = sourceKey(source);
      const existing = byKey.get(key);

      if (existing) {
        existing.rrf += contribution;
        if (list.label === 'vector') existing.vectorRank = rank;
        if (list.label === 'lexical') existing.lexicalRank = rank;
        continue;
      }

      byKey.set(key, {
        ...source,
        rrf: contribution,
        vectorRank: list.label === 'vector' ? rank : undefined,
        lexicalRank: list.label === 'lexical' ? rank : undefined,
      });
    }
  }

  return [...byKey.values()]
    .sort((left, right) => right.rrf - left.rrf)
    .map(({ rrf, ...rest }) => ({ ...rest, score: rrf }));
}

function matchesSourceFile(source: RetrievedSource, fileNameIncludes?: string): boolean {
  if (!fileNameIncludes) return true;
  return (source.fileName || '').toLowerCase().includes(fileNameIncludes.toLowerCase());
}

function sourceKey(source: RetrievedSource): string {
  return `${source.fileName || 'unknown'}:${source.page || 'unknown'}:${source.text.slice(0, 160)}`;
}
