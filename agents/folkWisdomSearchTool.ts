import { config } from '../config';
import type { PayloadFilter } from '../qdrant/client';
import type { HybridRetriever } from '../rag/hybridRetriever';
import {
  collectSearchResults,
  perQueryLimitForMode,
  resultLimitForMode,
  type QueryResultSet,
} from '../rag/resultCollector';
import { fallbackPlan } from './queryPlannerAgent';
import type { RagSearchOutput, SearchPlan } from './schemas';

const FOLK_WISDOM_HINTS = [
  'прыказка',
  'прыказкі',
  'прымаўка',
  'прымаўкі',
  'народная мудрасць',
  'народныя мудрасці',
  'прыслоўе',
  'прыслоўі',
  'выслоўе',
  'выслоўі',
  'proverb',
  'proverbs',
  'saying',
  'sayings',
  'folk wisdom',
];

export class FolkWisdomSearchTool {
  readonly name = 'folk_wisdom_search' as const;

  readonly description =
    'Searches the indexed PDF collection specifically for proverbs, sayings, aphorisms, and other folk wisdom.';

  constructor(private readonly retriever: HybridRetriever) {}

  async invoke(query: string): Promise<RagSearchOutput> {
    return this.invokePlan(fallbackPlan(query, 'folk_wisdom_search'));
  }

  async invokePlan(plan: SearchPlan): Promise<RagSearchOutput> {
    if (!config.qdrant.url) {
      throw new Error('QDRANT_URL is required for folk_wisdom_search');
    }

    const queries = buildFolkWisdomQueries(plan);
    const finalLimit = resultLimitForMode({
      desiredResultCount: plan.desiredResultCount,
      fallbackLimit: config.search.folkWisdomTopK,
      maxLimit: 60,
    });
    const perQueryLimit = perQueryLimitForMode({
      finalLimit,
      fallbackLimit: Math.min(config.search.folkWisdomTopK, 12),
      broadMode: plan.resultMode === 'list' || plan.resultMode === 'explore',
    });
    const filter: PayloadFilter = {
      must: [{ key: 'category', match: { value: 'proverbs' } }],
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

function buildFolkWisdomQueries(plan: SearchPlan): WeightedQuery[] {
  const topicFacets = topicFacetQueries(plan);
  const queryStrings = [
    ...plan.expandedQueries,
    ...(plan.semanticFacets || []),
    ...topicFacets,
    `${plan.coreQuery} прыказкі прымаўкі`,
    `${plan.coreQuery} народная мудрасць выслоўі`,
    `${plan.coreQuery} прыкметы народныя назіранні`,
    `${plan.coreQuery} proverbs sayings folk wisdom`,
  ];
  const fallbackHint = FOLK_WISDOM_HINTS.join(' ');
  const uniqueQueries = [...new Set(queryStrings.map((query) => query.trim()).filter(Boolean))].slice(0, 14);

  return [
    ...uniqueQueries.map((query) => ({
      query,
      weight: query === plan.coreQuery ? 1.15 : 1,
    })),
    { query: fallbackHint, weight: 0.72 },
  ];
}

function topicFacetQueries(plan: SearchPlan): string[] {
  const query = `${plan.coreQuery} ${plan.expandedQueries.join(' ')}`.toLowerCase();
  const facets: string[] = [];

  if (/(прыкмет|надвор|пагод|дождж|снег|вецер|мароз|сонц|weather|rain|snow|wind)/iu.test(query)) {
    facets.push(
      'прыкметы надвор\u2019е пагода',
      'народныя прыкметы дождж снег вецер мароз',
      'прыкметы прырода сонца хмары'
    );
  }

  if (/(прац|работ|гультай|лент|work|labor)/iu.test(query)) {
    facets.push('прыказкі пра працу', 'прымаўкі праца лянота работлівасць');
  }

  if (/(жыцц|чалавек|людз|розум|дурн|life|people|wisdom)/iu.test(query)) {
    facets.push('прыказкі пра жыццё чалавека', 'народная мудрасць розум дурнота людзі');
  }

  if (/(сям|род|бацьк|мац|family)/iu.test(query)) {
    facets.push('прыказкі пра сям\u2019ю род бацькоў', 'прымаўкі маці бацька дзеці');
  }

  return facets;
}
