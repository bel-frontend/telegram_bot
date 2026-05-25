import { config } from '../config';
import type { HybridRetriever } from '../rag/hybridRetriever';
import {
  collectSearchResults,
  perQueryLimitForMode,
  resultLimitForMode,
  type QueryResultSet,
} from '../rag/resultCollector';
import { fallbackPlan } from './queryPlannerAgent';
import type { RagSearchOutput, SearchPlan } from './schemas';

export class RagSearchTool {
  readonly name = 'rag_search' as const;

  readonly description = 'Searches the indexed PDF collection for general document lookup.';

  constructor(private readonly retriever: HybridRetriever) {}

  async invoke(query: string): Promise<RagSearchOutput> {
    return this.invokePlan(fallbackPlan(query, 'rag_search'));
  }

  async invokePlan(plan: SearchPlan): Promise<RagSearchOutput> {
    if (!config.qdrant.url) {
      throw new Error('QDRANT_URL is required for rag_search');
    }

    const queries = buildRagQueries(plan);
    const finalLimit = resultLimitForMode({
      desiredResultCount: plan.desiredResultCount,
      fallbackLimit: config.server.topK,
      maxLimit: 50,
    });
    const perQueryLimit = perQueryLimitForMode({
      finalLimit,
      fallbackLimit: config.server.topK,
      broadMode: plan.resultMode === 'list' || plan.resultMode === 'explore',
    });
    const searchResults = await Promise.all(
      queries.map(async (searchQuery): Promise<QueryResultSet> => ({
        query: searchQuery.query,
        weight: searchQuery.weight,
        sources: await this.retriever.retrieve(searchQuery.query, perQueryLimit),
      }))
    );
    const { sources, queryBreakdown } = collectSearchResults({
      queryResults: searchResults,
      limit: finalLimit,
      perQueryKeep: perQueryLimit,
      diversityBonus: 0.08,
    });

    return {
      query: queries.map((item) => item.query).join(' | '),
      found: sources.length > 0,
      sources,
      sourceCount: sources.length,
      queryBreakdown,
    };
  }
}

interface WeightedQuery {
  query: string;
  weight: number;
}

function buildRagQueries(plan: SearchPlan): WeightedQuery[] {
  const queryStrings = [
    plan.coreQuery,
    ...plan.expandedQueries,
    ...(plan.semanticFacets || []),
  ];
  const uniqueQueries = [...new Set(queryStrings.map((query) => query.trim()).filter(Boolean))].slice(0, 8);

  return uniqueQueries.map((query) => ({
    query,
    weight: query === plan.coreQuery ? 1.15 : 1,
  }));
}
