import { config } from '../config';
import type { Payload, PayloadFilter, QdrantClient } from '../qdrant/client';
import type { HybridRetriever } from '../rag/hybridRetriever';
import type { RetrievedSource } from '../rag/types';
import { resultLimitForMode, type QueryBreakdown } from '../rag/resultCollector';
import { fallbackPlan } from './queryPlannerAgent';
import type { RagSearchOutput, SearchPlan } from './schemas';

const RECORD_TYPE_TRIGGERS: Record<string, string[]> = {
  greeting: ['вітанне', 'вітанні', 'прывітанне', 'здароўканне', 'добры дзень', 'добры вечар'],
  farewell: ['развітанне', 'развітанні', 'бывай', 'пачуемся', 'да пабачэння'],
  wish: ['пажаданне', 'пажаданні', 'зычэнне', 'зычэнні', 'добрыя пажаданні', 'здароўя'],
  curse: ['праклён', 'праклёны', 'праклены', 'гразьбы', 'грозьбы', 'кляцьба'],
  insult: ['абраза', 'абразы', 'лаянка', 'знявага', 'пад’ялдычкі', 'падялдычкі', 'цвялілкі'],
  threat: ['пагроза', 'пагрозы', 'грозьбы', 'гразьбы'],
  proverb: ['прыказка', 'прыказкі', 'прымаўка', 'прымаўкі', 'народная мудрасць'],
  weather_sign: ['прыкмета', 'прыкметы', 'надвор\'е', 'дождж', 'мароз', 'вецер', 'снег'],
};

const SECTION_ALIASES: Record<string, string[]> = {
  greeting: ['вітанні', 'прывітанні', 'здароўканне', 'госці', 'спатканне'],
  farewell: ['развітанне', 'пачуемся', 'адыходзячы з гасцей'],
  wish: ['добрыя пажаданні', 'прыгаворкі й зычэнні', 'зычэнні'],
  curse: ['праклёны й грозьбы', 'гразьбы', 'пагрозы'],
  insult: ['пад’ялдычкі ды цвялілкі', 'абразы', 'лаянка'],
  threat: ['праклёны й грозьбы', 'пагрозы', 'грозьбы'],
  proverb: ['прыказкі й прымаўкі', 'прыказкі', 'прымаўкі'],
  weather_sign: ['прыкметы, жыццёвыя назіранні, парады', 'прыкметы', 'народны каляндар'],
};

export class RecordSearchTool {
  readonly name = 'record_search' as const;

  readonly description =
    'Searches extracted short records by category: greetings, farewells, wishes, curses, insults, threats, proverbs, and weather signs.';

  constructor(
    private readonly qdrant: QdrantClient,
    private readonly retriever: HybridRetriever
  ) {}

  async invoke(query: string): Promise<RagSearchOutput> {
    return this.invokePlan(fallbackPlan(query, 'record_search'));
  }

  async invokePlan(plan: SearchPlan): Promise<RagSearchOutput> {
    if (!config.qdrant.url) {
      throw new Error('QDRANT_URL is required for record_search');
    }

    const recordTypes = inferRecordTypes(plan);
    const finalLimit = resultLimitForMode({
      desiredResultCount: plan.desiredResultCount,
      fallbackLimit: 25,
      maxLimit: 80,
    });
    const expandedQueries = buildExpandedQueries(plan, recordTypes);
    const queryBreakdown: QueryBreakdown[] = [];
    const recordCandidates: RetrievedSource[] = [];

    for (const recordType of recordTypes) {
      const records = await this.scrollRecords(recordType);
      const ranked = rankRecordSources(records, expandedQueries, recordType);
      recordCandidates.push(...ranked);
      queryBreakdown.push({
        query: `recordType:${recordType}`,
        retrievedCount: records.length,
        keptCount: Math.min(ranked.length, finalLimit),
      });
    }

    const sources = deduplicate(recordCandidates)
      .sort((left, right) => right.score - left.score)
      .slice(0, finalLimit);

    if (sources.length > 0) {
      return {
        query: expandedQueries.join(' | '),
        found: true,
        sources,
        sourceCount: sources.length,
        queryBreakdown,
      };
    }

    const fallbackSources = await this.retrieveChunkFallback(expandedQueries, recordTypes, finalLimit);
    return {
      query: expandedQueries.join(' | '),
      found: fallbackSources.length > 0,
      sources: fallbackSources,
      sourceCount: fallbackSources.length,
      queryBreakdown: [
        ...queryBreakdown,
        {
          query: `section-fallback:${recordTypes.join(',')}`,
          retrievedCount: fallbackSources.length,
          keptCount: fallbackSources.length,
        },
      ],
    };
  }

  private async scrollRecords(recordType: string): Promise<RetrievedSource[]> {
    const filter: PayloadFilter = {
      must: [
        { key: 'payloadKind', match: { value: 'record' } },
        { key: 'recordType', match: { value: recordType } },
      ],
    };
    const sources: RetrievedSource[] = [];
    let offset: string | number | undefined;

    do {
      const page = await this.qdrant.scrollPayloads(config.qdrant.collection, 256, offset, filter);
      for (const point of page.points) {
        if (point.payload) sources.push(toSource(point.payload, 1));
      }
      offset = page.nextOffset;
    } while (offset);

    return sources;
  }

  private async retrieveChunkFallback(
    queries: string[],
    recordTypes: string[],
    limit: number
  ): Promise<RetrievedSource[]> {
    const fallbackQueries = [
      ...queries,
      ...recordTypes.flatMap((recordType) => SECTION_ALIASES[recordType] || []),
    ];
    const results: RetrievedSource[] = [];

    for (const query of [...new Set(fallbackQueries)].slice(0, 8)) {
      const retrieved = await this.retriever.retrieve(query, Math.max(8, Math.ceil(limit / 2)), {
        filter: { must: [{ key: 'sourceBook', match: { value: 'vushatski_slovazbor' } }] },
      });
      results.push(...retrieved.map((source) => ({ ...source, score: source.score * 0.75 })));
    }

    return deduplicate(results).sort((left, right) => right.score - left.score).slice(0, limit);
  }
}

function inferRecordTypes(plan: SearchPlan): string[] {
  if (plan.recordTypes && plan.recordTypes.length > 0) return plan.recordTypes;

  const text = normalize([
    plan.coreQuery,
    plan.lookupTerm || '',
    ...plan.expandedQueries,
    ...(plan.semanticFacets || []),
  ].join(' '));
  const matches = Object.entries(RECORD_TYPE_TRIGGERS)
    .filter(([, triggers]) => triggers.some((trigger) => text.includes(normalize(trigger))))
    .map(([recordType]) => recordType);

  if (matches.includes('curse') && !matches.includes('threat')) matches.push('threat');

  return matches.length > 0 ? [...new Set(matches)] : ['proverb'];
}

function buildExpandedQueries(plan: SearchPlan, recordTypes: string[]): string[] {
  return [
    plan.lookupTerm || '',
    plan.coreQuery,
    ...plan.expandedQueries,
    ...(plan.semanticFacets || []),
    ...recordTypes.flatMap((recordType) => RECORD_TYPE_TRIGGERS[recordType] || []),
    ...recordTypes.flatMap((recordType) => SECTION_ALIASES[recordType] || []),
  ]
    .map((query) => query.trim())
    .filter(Boolean)
    .filter((query, index, all) => all.indexOf(query) === index)
    .slice(0, 18);
}

function rankRecordSources(
  records: RetrievedSource[],
  queries: string[],
  recordType: string
): RetrievedSource[] {
  const terms = [...new Set(queries.flatMap((query) => normalize(query).split(/\s+/)).filter((term) => term.length >= 3))];

  return records
    .map((source) => {
      const haystack = normalize([
        source.text,
        source.sectionTitle || '',
        source.recordType || '',
        ...(source.tags || []),
      ].join(' '));
      let score = source.recordType === recordType ? 10 : 0;
      for (const term of terms) {
        if (haystack.includes(term)) score += term.length > 5 ? 1.5 : 0.8;
      }
      return { ...source, score };
    })
    .filter((source) => source.score > 0);
}

function toSource(payload: Payload, score: number): RetrievedSource {
  const loc = payload.loc;
  const pageNumber = loc && typeof loc === 'object' && 'pageNumber' in loc ? loc.pageNumber : payload.page;

  return {
    text: String(payload.text || payload.recordText || ''),
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
    page: typeof pageNumber === 'number' ? pageNumber : undefined,
  };
}

function deduplicate(sources: RetrievedSource[]): RetrievedSource[] {
  const byKey = new Map<string, RetrievedSource>();
  for (const source of sources) {
    const key = `${source.source || source.fileName}:${source.recordType || source.sectionTitle}:${normalize(source.text)}`;
    const existing = byKey.get(key);
    if (!existing || source.score > existing.score) byKey.set(key, source);
  }
  return [...byKey.values()];
}

function normalize(value: string): string {
  return value
    .toLowerCase()
    .replace(/[ё]/g, 'е')
    .replace(/[ў]/g, 'у')
    .replace(/[^\p{L}\p{N}\s'’]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function stringValue(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function stringArrayValue(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const strings = value.filter((item): item is string => typeof item === 'string' && item.length > 0);
  return strings.length > 0 ? strings : undefined;
}
