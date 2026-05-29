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
import type { RagSearchOutput, SearchPlan, ToolName } from './schemas';

interface DictionarySearchToolOptions {
  name: Exclude<ToolName, 'chat' | 'rag_search' | 'folk_wisdom_search' | 'dialect_dictionary_search'>;
  description: string;
  dictionaryType: string;
  hintQueries: string[];
  fallbackLimit: number;
  maxLimit: number;
}

export class FilteredDictionarySearchTool {
  readonly name: DictionarySearchToolOptions['name'];
  readonly description: string;

  constructor(
    private readonly retriever: HybridRetriever,
    private readonly options: DictionarySearchToolOptions
  ) {
    this.name = options.name;
    this.description = options.description;
  }

  async invoke(query: string): Promise<RagSearchOutput> {
    return this.invokePlan(fallbackPlan(query, this.name));
  }

  async invokePlan(plan: SearchPlan): Promise<RagSearchOutput> {
    if (!config.qdrant.url) {
      throw new Error(`QDRANT_URL is required for ${this.name}`);
    }

    const queries = buildDictionaryQueries(plan, this.options.hintQueries);
    const finalLimit = resultLimitForMode({
      desiredResultCount: plan.desiredResultCount,
      fallbackLimit: this.options.fallbackLimit,
      maxLimit: this.options.maxLimit,
    });
    const perQueryLimit = perQueryLimitForMode({
      finalLimit,
      fallbackLimit: Math.min(this.options.fallbackLimit, 12),
      broadMode: plan.resultMode === 'list' || plan.resultMode === 'section' || plan.resultMode === 'explore',
    });
    const filter: PayloadFilter = {
      must: [{ key: 'dictionaryType', match: { value: this.options.dictionaryType } }],
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
      diversityBonus: 0.1,
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

export function createOrthographicDictionarySearchTool(
  retriever: HybridRetriever
): FilteredDictionarySearchTool {
  return new FilteredDictionarySearchTool(retriever, {
    name: 'orthographic_dictionary_search',
    description: 'Searches Belarusian orthographic dictionaries for spelling and word forms.',
    dictionaryType: 'orthographic',
    hintQueries: ['арфаграфічны слоўнік правапіс напісанне форма слова'],
    fallbackLimit: 18,
    maxLimit: 50,
  });
}

export function createTranslationDictionarySearchTool(
  retriever: HybridRetriever
): FilteredDictionarySearchTool {
  return new FilteredDictionarySearchTool(retriever, {
    name: 'translation_dictionary_search',
    description: 'Searches Russian-Belarusian and Belarusian-Russian dictionary PDFs for translations.',
    dictionaryType: 'translation',
    hintQueries: ['руска беларуская пераклад беларуска руская слоўнік перевод'],
    fallbackLimit: 18,
    maxLimit: 50,
  });
}

export function createExplanatoryDictionarySearchTool(
  retriever: HybridRetriever
): FilteredDictionarySearchTool {
  return new FilteredDictionarySearchTool(retriever, {
    name: 'explanatory_dictionary_search',
    description: 'Searches Belarusian explanatory dictionaries for definitions and usage notes.',
    dictionaryType: 'explanatory',
    hintQueries: ['тлумачальны слоўнік значэнне слова азначэнне'],
    fallbackLimit: 18,
    maxLimit: 50,
  });
}

interface WeightedQuery {
  query: string;
  weight: number;
}

function buildDictionaryQueries(plan: SearchPlan, hints: string[]): WeightedQuery[] {
  const lookupTerms = extractLookupTerms(plan.coreQuery);
  const queryStrings = [
    ...lookupTerms,
    plan.coreQuery,
    ...plan.expandedQueries,
    ...(plan.semanticFacets || []),
    ...hints.map((hint) => `${plan.coreQuery} ${hint}`),
  ];
  const uniqueQueries = [...new Set(queryStrings.map((query) => query.trim()).filter(Boolean))].slice(0, 10);

  return uniqueQueries.map((query) => ({
    query,
    weight: lookupTerms.includes(query) ? 2.1 : query === plan.coreQuery ? 1.2 : 1,
  }));
}

function extractLookupTerms(query: string): string[] {
  const quoted = [...query.matchAll(/[\"“«](.+?)[\"”»]/g)]
    .map((match) => match[1]?.trim())
    .filter(Boolean);
  const afterWordLabel = query.match(/(?:слова|слово|word)\s+["“«]?([\p{L}'’ -]{2,40})["”»]?/iu);
  const direct = afterWordLabel?.[1]?.trim();

  const candidates = [...quoted, direct].filter((item): item is string => Boolean(item));
  return [...new Set(candidates.map(cleanLookupTerm).filter(Boolean))].slice(0, 4);
}

function cleanLookupTerm(term: string): string {
  return term
    .replace(/\s+(у|ў|в)\s+.+$/iu, '')
    .replace(/[^\p{L}'’ -]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}
