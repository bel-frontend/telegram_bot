import { config } from '../config';
import type { PayloadFilter } from '../qdrant/client';
import type { HybridRetriever } from '../rag/hybridRetriever';
import {
  collectSearchResults,
  perQueryLimitForMode,
  resultLimitForMode,
  type QueryResultSet,
} from '../rag/resultCollector';
import { focusSourcesOnLookupTerms } from './dictionarySearchTools';
import { fallbackPlan } from './queryPlannerAgent';
import type { RagSearchOutput, SearchPlan } from './schemas';

const DIALECT_HINTS = [
  'Вушацкі словазбор',
  'Рыгор Барадулін',
  'дыялектныя словы',
  'мясцовыя выразы',
  'гразьбы',
  'праклёны',
  'праклены',
  'пагрозы',
  'кляцьба',
  'праклінаць',
];

export class DialectDictionarySearchTool {
  readonly name = 'dialect_dictionary_search' as const;

  readonly description =
    'Searches Vushatski Slovazbor and related dialect sections, including curses, threats, local expressions, and section-like lists.';

  constructor(private readonly retriever: HybridRetriever) {}

  async invoke(query: string): Promise<RagSearchOutput> {
    return this.invokePlan(fallbackPlan(query, 'dialect_dictionary_search'));
  }

  async invokePlan(plan: SearchPlan): Promise<RagSearchOutput> {
    if (!config.qdrant.url) {
      throw new Error('QDRANT_URL is required for dialect_dictionary_search');
    }

    const queries = buildDialectQueries(plan);
    const finalLimit = resultLimitForMode({
      desiredResultCount: plan.desiredResultCount,
      fallbackLimit: 30,
      maxLimit: 70,
    });
    const perQueryLimit = perQueryLimitForMode({
      finalLimit,
      fallbackLimit: 12,
      broadMode: plan.resultMode === 'list' || plan.resultMode === 'section' || plan.resultMode === 'explore',
    });
    const filter: PayloadFilter = {
      must: [{ key: 'dictionaryType', match: { value: 'dialect' } }],
    };
    const searchResults = await Promise.all(
      queries.map(async (searchQuery): Promise<QueryResultSet> => ({
        query: searchQuery.query,
        weight: searchQuery.weight,
        sources: await this.retriever.retrieve(searchQuery.query, perQueryLimit, { filter }),
      }))
    );
    const { sources, queryBreakdown } = collectSearchResults({
      queryResults: searchResults,
      limit: finalLimit,
      perQueryKeep: perQueryLimit,
      diversityBonus: 0.12,
    });
    const lookupTerms = plan.lookupTerm ? [plan.lookupTerm] : [];

    return {
      query: queries.map((item) => item.query).join(' | '),
      found: sources.length > 0,
      sources: focusSourcesOnLookupTerms(sources, lookupTerms),
      sourceCount: sources.length,
      queryBreakdown,
    };
  }
}

interface WeightedQuery {
  query: string;
  weight: number;
}

function buildDialectQueries(plan: SearchPlan): WeightedQuery[] {
  const lookupTerms = plan.lookupTerm ? [plan.lookupTerm] : [];
  const queryStrings = [
    ...lookupTerms,
    plan.coreQuery,
    ...plan.expandedQueries,
    ...(plan.semanticFacets || []),
    ...dialectFacetQueries(plan),
    `${plan.coreQuery} Вушацкі словазбор`,
    `${plan.coreQuery} Рыгор Барадулін`,
  ];
  const uniqueQueries = [...new Set(queryStrings.map((query) => query.trim()).filter(Boolean))].slice(0, 10);

  return [
    ...uniqueQueries.map((query) => ({
      query,
      weight: lookupTerms.includes(query) ? 2.1 : query === plan.coreQuery ? 1.2 : 1,
    })),
    { query: DIALECT_HINTS.join(' '), weight: 0.68 },
  ];
}

function dialectFacetQueries(plan: SearchPlan): string[] {
  const query = `${plan.coreQuery} ${plan.expandedQueries.join(' ')}`.toLowerCase();

  if (/(пракл[её]н|праклін|гразьб|пагроз|кляць|curse|curses|threat)/iu.test(query)) {
    return [
      'гразьбы праклёны праклены пагрозы',
      'праклінаць кляцьба ліхія словы',
      'мясцовыя выразы гразьбы праклёны',
    ];
  }

  if (/(устойлів|выраз|прымаў|прыказ|expression|phrase)/iu.test(query)) {
    return ['устойлівыя выразы', 'мясцовыя выразы прымаўкі'];
  }

  return [];
}
