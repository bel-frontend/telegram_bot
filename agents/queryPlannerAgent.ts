import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { chatModel } from '../common/model';
import { config } from '../config';
import { parseStructuredOutput, schemaInstruction } from './json';
import { PLANNER_SYSTEM_PROMPT } from './prompts';
import {
  SearchPlanSchema,
  type ChatMessage,
  type SearchPlan,
  type ToolName,
} from './schemas';

type UnnormalizedSearchPlan = Omit<SearchPlan, 'resultMode'> & {
  resultMode?: SearchPlan['resultMode'];
};

export const TOOL_CATALOG = [
  {
    tool: 'folk_wisdom_search',
    useFor: 'Прыказкі, прымаўкі, прыслоўі, выслоўі, народная мудрасць, народныя прыкметы.',
  },
  {
    tool: 'dialect_dictionary_search',
    useFor: 'Дыялектныя словы і мясцовыя выразы, асабліва Вушацкі словазбор, гразьбы, праклёны, пагрозы.',
    structuredInput: 'Set lookupTerm to the exact dialect word or expression when the user asks for a specific item.',
  },
  {
    tool: 'orthographic_dictionary_search',
    useFor: 'Правапіс, напісанне слова, формы слова ў арфаграфічным слоўніку.',
    structuredInput: 'Set lookupTerm to the exact word whose spelling or form should be checked.',
  },
  {
    tool: 'translation_dictionary_search',
    useFor: 'Пераклад паміж расейскай і беларускай мовамі.',
    structuredInput: 'Set lookupTerm to the exact source word or phrase to translate.',
  },
  {
    tool: 'explanatory_dictionary_search',
    useFor:
      'Тлумачэнні і азначэнні беларускіх слоў. Калі карыстальнік просіць знайсці слова ў слоўніку без удакладнення, гэта звычайна lookup у тлумачальным слоўніку.',
    structuredInput: 'Set lookupTerm to the exact word being looked up, without command words.',
  },
  {
    tool: 'record_search',
    useFor:
      'Катэгарыйны пошук кароткіх элементаў: вітанні, развітанні, пажаданні, праклёны, гразьбы, пагрозы, абразы, пад’ялдычкі, прыкметы.',
    structuredInput:
      'Use recordTypes for category lookup, e.g. greeting, farewell, wish, curse, insult, threat, proverb, weather_sign, phrase_equivalent.',
  },
  {
    tool: 'rag_search',
    useFor: 'Агульны пошук па ўсёй PDF-калекцыі, калі ніводзін спецыялізаваны тул не падыходзіць.',
  },
] as const;

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
          '{"intent":"direct_chat|general_rag|folk_wisdom|dialect_definition|dialect_section_lookup|orthographic_lookup|translation_lookup|explanatory_lookup|record_lookup|exact_phrase","coreQuery":"string","lookupTerm":"exact word optional","expandedQueries":["string"],"semanticFacets":["string optional"],"resultMode":"answer|list|section|explore","desiredResultCount":number,"recordTypes":["greeting|farewell|wish|curse|insult|threat|proverb|weather_sign|phrase_equivalent optional"],"sourceBook":"string optional","sectionAliases":["string optional"],"targetBook":"any|vushatski_slovazbor|proverbs_dictionary|orthographic_dictionary|translation_dictionary|explanatory_dictionary","tool":"chat|rag_search|folk_wisdom_search|dialect_dictionary_search|orthographic_dictionary_search|translation_dictionary_search|explanatory_dictionary_search|record_search","reason":"string"}'
        )
      ),
      new HumanMessage(
        JSON.stringify({
          latestQuestion,
          fallbackTool,
          availableTools: TOOL_CATALOG,
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
    lookupTerm: undefined,
    expandedQueries: [query],
    semanticFacets: fallbackSemanticFacets(query),
    resultMode,
    desiredResultCount: defaultResultCount(resultMode, tool),
    targetBook: fallbackTargetBook(tool),
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
  const lookupTerm = plan.lookupTerm?.trim();
  const recordTypes = [...new Set((plan.recordTypes || []).map((item) => item.trim()).filter(Boolean))].slice(0, 8);
  const sectionAliases = [...new Set((plan.sectionAliases || []).map((item) => item.trim()).filter(Boolean))].slice(0, 10);

  return {
    ...plan,
    lookupTerm: lookupTerm || undefined,
    targetBook: plan.targetBook || 'any',
    resultMode,
    semanticFacets,
    recordTypes: recordTypes.length ? recordTypes : undefined,
    sectionAliases: sectionAliases.length ? sectionAliases : undefined,
    sourceBook: plan.sourceBook?.trim() || undefined,
    desiredResultCount: plan.desiredResultCount || defaultResultCount(resultMode, plan.tool),
    expandedQueries: expandedQueries.length ? expandedQueries : [plan.coreQuery],
  };
}

function intentForTool(tool: ToolName): SearchPlan['intent'] {
  if (tool === 'dialect_dictionary_search') return 'dialect_definition';
  if (tool === 'orthographic_dictionary_search') return 'orthographic_lookup';
  if (tool === 'translation_dictionary_search') return 'translation_lookup';
  if (tool === 'explanatory_dictionary_search') return 'explanatory_lookup';
  if (tool === 'record_search') return 'record_lookup';
  if (tool === 'folk_wisdom_search') return 'folk_wisdom';
  if (tool === 'chat') return 'direct_chat';
  return 'general_rag';
}

function fallbackTargetBook(tool: ToolName): SearchPlan['targetBook'] {
  if (tool === 'dialect_dictionary_search') return 'vushatski_slovazbor';
  if (tool === 'orthographic_dictionary_search') return 'orthographic_dictionary';
  if (tool === 'translation_dictionary_search') return 'translation_dictionary';
  if (tool === 'explanatory_dictionary_search') return 'explanatory_dictionary';
  if (tool === 'record_search') return 'vushatski_slovazbor';
  return 'any';
}

function fallbackResultMode(query: string, tool: ToolName): SearchPlan['resultMode'] {
  if (
    (tool === 'dialect_dictionary_search' || tool === 'record_search') &&
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
    return tool === 'dialect_dictionary_search' || tool === 'record_search' ? 40 : 30;
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
