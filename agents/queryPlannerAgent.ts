import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { chatModel } from '../common/model';
import { config } from '../config';
import {
  SearchPlanSchema,
  type ChatMessage,
  type SearchPlan,
  type ToolName,
} from './schemas';
import { parseStructuredOutput, schemaInstruction } from './json';

type UnnormalizedSearchPlan = Omit<SearchPlan, 'resultMode'> & {
  resultMode?: SearchPlan['resultMode'];
};

const PLANNER_SYSTEM_PROMPT = [
  'You are a query planning agent for a Belarusian RAG workshop.',
  'Your job is to understand the user request before retrieval.',
  'Extract the core idea, choose the best search tool, and generate expanded search queries.',
  'Use dialect_dictionary_search for Vusacki Slovazbor, Ryhor Baradulin, dialect words, meanings, local expressions, curses, threats, гразьбы, праклёны, sections from the first book.',
  'Use intent dialect_section_lookup when the user asks for a full section, list, all items, continuation, or a title-like query such as Устойлівыя выразы, Прыкметы, Стравы, Лекаванне.',
  'Use folk_wisdom_search for proverbs, sayings, aphorisms, народныя мудрасці, прыказкі, прымаўкі across the collection.',
  'Use rag_search for general document lookup.',
  'Use chat only for greetings or conversation that does not need documents.',
  'Expanded queries should include synonyms, likely section titles, spelling variants, and short exact phrases.',
  'Choose resultMode answer for narrow factual questions, list for requested examples/items, section for full section lookups, and explore for broad semantic discovery.',
  'Use semanticFacets for distinct meanings or topic angles that should each get retrieval coverage.',
  'For list/explore requests set desiredResultCount high enough to preserve multiple results, usually 20-40.',
].join(' ');

export class QueryPlannerAgent {
  async plan(messages: ChatMessage[], latestQuestion: string, fallbackTool: ToolName): Promise<SearchPlan> {
    const model = await chatModel(config.chat.toolModel, {
      ollamaUrl: config.chat.ollamaUrl,
      reasoningEffort: config.chat.toolReasoningEffort,
    });

    const response = await model.invoke([
      new SystemMessage(PLANNER_SYSTEM_PROMPT),
      new SystemMessage(
        schemaInstruction(
          'SearchPlan',
          '{"intent":"direct_chat|general_rag|folk_wisdom|dialect_definition|dialect_section_lookup|exact_phrase","coreQuery":"string","expandedQueries":["string"],"semanticFacets":["string optional"],"resultMode":"answer|list|section|explore","desiredResultCount":number,"targetBook":"any|vushatski_slovazbor|proverbs_dictionary","tool":"chat|rag_search|folk_wisdom_search|dialect_dictionary_search","reason":"string"}'
        )
      ),
      new HumanMessage(
        JSON.stringify({
          latestQuestion,
          fallbackTool,
          recentMessages: messages.slice(-8),
        })
      ),
    ]);

    return normalizePlan(
      parseStructuredOutput(response.content, SearchPlanSchema, fallbackPlan(latestQuestion, fallbackTool))
    );
  }
}

export function fallbackPlan(query: string, tool: ToolName): SearchPlan {
  const resultMode = fallbackResultMode(query, tool);

  return normalizePlan({
    intent: fallbackIntent(query, tool),
    coreQuery: query,
    expandedQueries: [query],
    semanticFacets: fallbackSemanticFacets(query),
    resultMode,
    desiredResultCount: defaultResultCount(resultMode, tool),
    targetBook: tool === 'dialect_dictionary_search' ? 'vushatski_slovazbor' : 'any',
    tool,
    reason: 'Fallback search plan.',
  });
}

function fallbackIntent(query: string, tool: ToolName): SearchPlan['intent'] {
  if (
    tool === 'dialect_dictionary_search' &&
    /(раздзел|спіс|усе|увесь|цалкам|устойлів[\p{L}]*\s+выраз|прыкмет|звыча|страв|лекаван|пытан|вокліч)/iu.test(query)
  ) {
    return 'dialect_section_lookup';
  }

  return intentForTool(tool);
}

function normalizePlan(plan: UnnormalizedSearchPlan): SearchPlan {
  const expandedQueries = [...new Set([plan.coreQuery, ...plan.expandedQueries].map((item) => item.trim()))]
    .filter(Boolean)
    .slice(0, 12);
  const semanticFacets = [...new Set((plan.semanticFacets || []).map((item) => item.trim()))]
    .filter(Boolean)
    .slice(0, 10);
  const resultMode = plan.resultMode || fallbackResultMode(plan.coreQuery, plan.tool);

  return {
    ...plan,
    targetBook: plan.targetBook || 'any',
    resultMode,
    semanticFacets,
    desiredResultCount: plan.desiredResultCount || defaultResultCount(resultMode, plan.tool),
    expandedQueries: expandedQueries.length ? expandedQueries : [plan.coreQuery],
  };
}

function intentForTool(tool: ToolName): SearchPlan['intent'] {
  if (tool === 'dialect_dictionary_search') return 'dialect_definition';
  if (tool === 'folk_wisdom_search') return 'folk_wisdom';
  if (tool === 'chat') return 'direct_chat';
  return 'general_rag';
}

function fallbackResultMode(query: string, tool: ToolName): SearchPlan['resultMode'] {
  if (
    tool === 'dialect_dictionary_search' &&
    /(раздзел|секцы|спіс|усе|увесь|цалкам|далей|працяг)/iu.test(query)
  ) {
    return 'section';
  }

  if (/(спіс|усе|увесь|некальк|падбяр|знайдзі|па сэнс|падобн|прыкмет|прыказк|прымаўк|examples|list|all|several|similar)/iu.test(query)) {
    return tool === 'rag_search' ? 'explore' : 'list';
  }

  return 'answer';
}

function defaultResultCount(resultMode: SearchPlan['resultMode'], tool: ToolName): number {
  if (resultMode === 'section') return 60;
  if (resultMode === 'list' || resultMode === 'explore') {
    return tool === 'dialect_dictionary_search' ? 40 : 30;
  }

  return tool === 'rag_search' ? 8 : 20;
}

function fallbackSemanticFacets(query: string): string[] {
  const facets: string[] = [];

  if (/(прыкмет|надвор|пагод|дождж|снег|вецер|weather)/iu.test(query)) {
    facets.push('прыкметы надвор\u2019е пагода', 'народныя назіранні дождж снег вецер');
  }

  if (/(прац|работ|праца|work)/iu.test(query)) {
    facets.push('праца работлівасць лянота');
  }

  if (/(жыцц|чалавек|людз|life|people)/iu.test(query)) {
    facets.push('жыццё чалавек людзі');
  }

  return facets;
}
