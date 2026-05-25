import type { RetrievedSource } from './types';

export interface QueryResultSet {
  query: string;
  sources: RetrievedSource[];
  weight?: number;
}

export interface QueryBreakdown {
  query: string;
  retrievedCount: number;
  keptCount: number;
}

export interface CollectedSearchResults {
  sources: RetrievedSource[];
  queryBreakdown: QueryBreakdown[];
}

export function collectSearchResults(options: {
  queryResults: QueryResultSet[];
  limit: number;
  perQueryKeep: number;
  diversityBonus?: number;
}): CollectedSearchResults {
  const perQueryCandidates = options.queryResults.flatMap((result) =>
    result.sources.slice(0, options.perQueryKeep).map((source, index) => ({
      source,
      query: result.query,
      rank: index + 1,
      weight: result.weight ?? 1,
    }))
  );

  const byKey = new Map<string, RetrievedSource & { matchedQueries?: string[] }>();

  for (const candidate of perQueryCandidates) {
    const key = sourceKey(candidate.source);
    const existing = byKey.get(key);
    const rankScore = 1 / (candidate.rank + 1);
    const weightedScore = candidate.source.score * candidate.weight + rankScore;

    if (!existing) {
      byKey.set(key, {
        ...candidate.source,
        score: weightedScore,
        matchedQueries: [candidate.query],
      });
      continue;
    }

    existing.matchedQueries = [...new Set([...(existing.matchedQueries || []), candidate.query])];
    existing.score = Math.max(existing.score, weightedScore) + (options.diversityBonus ?? 0.08);
  }

  const sources = [...byKey.values()]
    .sort((left, right) => right.score - left.score)
    .slice(0, options.limit);
  const sourceKeys = new Set(sources.map(sourceKey));

  const queryBreakdown = options.queryResults.map((result) => ({
    query: result.query,
    retrievedCount: result.sources.length,
    keptCount: result.sources
      .slice(0, options.perQueryKeep)
      .filter((source) => sourceKeys.has(sourceKey(source))).length,
  }));

  return { sources, queryBreakdown };
}

export function resultLimitForMode(options: {
  desiredResultCount?: number;
  fallbackLimit: number;
  maxLimit?: number;
}): number {
  const maxLimit = options.maxLimit ?? 80;
  const desired = options.desiredResultCount || options.fallbackLimit;

  return Math.min(Math.max(desired, options.fallbackLimit), maxLimit);
}

export function perQueryLimitForMode(options: {
  finalLimit: number;
  fallbackLimit: number;
  broadMode: boolean;
}): number {
  if (!options.broadMode) {
    return options.fallbackLimit;
  }

  return Math.max(8, Math.ceil(options.finalLimit / 2));
}

function sourceKey(source: RetrievedSource): string {
  return [
    source.fileName || 'unknown',
    source.page || 'unknown',
    normalizeText(source.text).slice(0, 220),
  ].join(':');
}

function normalizeText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[ё]/g, 'е')
    .replace(/[ў]/g, 'у')
    .replace(/[^\p{L}\p{N}\s]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}
