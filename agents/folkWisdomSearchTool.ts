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
    const rerankedSources = rerankFolkWisdomSources(sources, plan);

    return {
      query: queries.map((item) => item.query).join(' | '),
      found: rerankedSources.length > 0,
      sources: rerankedSources,
      sourceCount: rerankedSources.length,
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
  const topicQueries = extractTopicQueries(plan, topicFacets);
  const queryStrings = [
    ...topicQueries,
    ...(plan.semanticFacets || []),
    ...topicFacets,
    ...plan.expandedQueries,
    `${plan.coreQuery} прыказкі прымаўкі`,
    `${plan.coreQuery} народная мудрасць выслоўі`,
  ];
  const fallbackHint = FOLK_WISDOM_HINTS.join(' ');
  const uniqueQueries = [...new Set(queryStrings.map((query) => query.trim()).filter(Boolean))].slice(0, 14);

  return [
    ...uniqueQueries.map((query) => ({
      query,
      weight: topicQueries.includes(query) ? 1.55 : query === plan.coreQuery ? 1.05 : 1,
    })),
    { query: fallbackHint, weight: 0.28 },
  ];
}

function extractTopicQueries(plan: SearchPlan, topicFacets: string[]): string[] {
  const candidates = [
    ...(plan.semanticFacets || []),
    ...topicFacets,
    stripFolkWisdomServiceWords(plan.coreQuery),
    ...plan.expandedQueries.map(stripFolkWisdomServiceWords),
  ];

  return [...new Set(candidates.map((query) => query.trim()).filter((query) => query.length >= 3))].slice(0, 8);
}

function stripFolkWisdomServiceWords(query: string): string {
  return query
    .replace(/\b(proverbs?|sayings?|folk wisdom)\b/giu, ' ')
    .replace(
      /(знайдзі|пашукай|падбяры|пакажы|падобныя|некалькі|прыказк[іаўу]*|прымаўк[іаўу]*|прыслоў[еіяў]*|выслоў[еіяў]*|народн\w*\s+мудрасц\w*)/giu,
      ' '
    )
    .replace(/\bпра\b/giu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function rerankFolkWisdomSources<T extends { text: string; score: number; page?: number; fileName?: string }>(
  sources: T[],
  plan: SearchPlan
): T[] {
  const topicTerms = extractTopicTerms(plan);
  const hasTopic = topicTerms.length > 0;

  return sources
    .map((source) => {
      let score = source.score;
      const normalizedText = normalizeForFolkSearch(source.text);

      for (const term of topicTerms) {
        if (normalizedText.includes(term)) {
          score += 0.8;
        }
      }

      if (hasTopic && isIntroductoryProverbPage(source)) {
        score -= 1.2;
      }

      return { ...source, score };
    })
    .sort((left, right) => right.score - left.score);
}

function extractTopicTerms(plan: SearchPlan): string[] {
  const text = [
    plan.coreQuery,
    ...plan.expandedQueries,
    ...(plan.semanticFacets || []),
    ...topicFacetQueries(plan),
  ].join(' ');
  const normalized = stripFolkWisdomServiceWords(normalizeForFolkSearch(text));
  const stopWords = new Set(['для', 'або', 'і', 'ды', 'па', 'на', 'у', 'ў', 'з', 'са', 'ад', 'да']);

  return [...new Set(normalized.split(/\s+/).filter((word) => word.length >= 4 && !stopWords.has(word)))].slice(0, 10);
}

function isIntroductoryProverbPage(source: { page?: number; fileName?: string }): boolean {
  return (source.fileName || '').includes('слоу') && typeof source.page === 'number' && source.page <= 15;
}

function normalizeForFolkSearch(value: string): string {
  return value
    .toLowerCase()
    .replace(/[ё]/g, 'е')
    .replace(/[ў]/g, 'у')
    .replace(/[^\p{L}\p{N}\s]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
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
