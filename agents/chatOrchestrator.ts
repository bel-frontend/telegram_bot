import { AIMessage, HumanMessage, SystemMessage } from '@langchain/core/messages';
import { chatModel } from '../common/model';
import { config } from '../config';
import type { DialectDictionarySearchTool } from './dialectDictionarySearchTool';
import type { FilteredDictionarySearchTool } from './dictionarySearchTools';
import type { FolkWisdomSearchTool } from './folkWisdomSearchTool';
import { LlmReranker } from './llmReranker';
import { DECISION_SYSTEM_PROMPT, LIST_FINAL_SYSTEM_PROMPT } from './prompts';
import { QueryPlannerAgent, TOOL_CATALOG } from './queryPlannerAgent';
import { QuestionRewriterAgent } from './questionRewriterAgent';
import { RagResultEvaluator } from './ragResultEvaluator';
import type { RagSearchTool } from './ragSearchTool';
import type { RecordSearchTool } from './recordSearchTool';
import {
  FinalAnswerSchema,
  OrchestratorDecisionSchema,
  type ChatAgentResponse,
  type ChatMessage,
  type EvaluationResult,
  type FinalAnswer,
  type OrchestratorDecision,
  type RagSearchOutput,
  type RerankTrace,
  type RerankedSource,
  type SearchPlan,
  type ToolName,
} from './schemas';
import { parseStructuredOutput, schemaInstruction } from './json';

const NARROW_MODE_TOP_K = 6;


export class ChatOrchestratorAgent {
  private readonly evaluator = new RagResultEvaluator();
  private readonly reranker = new LlmReranker();

  constructor(
    private readonly folkWisdomSearchTool: FolkWisdomSearchTool,
    private readonly ragSearchTool: RagSearchTool,
    private readonly dialectDictionarySearchTool: DialectDictionarySearchTool,
    private readonly orthographicDictionarySearchTool: FilteredDictionarySearchTool,
    private readonly translationDictionarySearchTool: FilteredDictionarySearchTool,
    private readonly explanatoryDictionarySearchTool: FilteredDictionarySearchTool,
    private readonly recordSearchTool: RecordSearchTool,
    private readonly queryPlannerAgent = new QueryPlannerAgent(),
    private readonly questionRewriterAgent = new QuestionRewriterAgent()
  ) {}

  async chat(messages: ChatMessage[]): Promise<ChatAgentResponse> {
    const originalQuestion = latestUserMessage(messages);
    const standaloneQuestion = await this.questionRewriterAgent.rewrite(messages, originalQuestion);
    const effectiveQuestion = standaloneQuestion || originalQuestion;
    const wasRewritten = standaloneQuestion !== originalQuestion;

    const decision = await this.decide(messages, effectiveQuestion);

    if (decision.action === 'answer_directly') {
      return {
        answer: decision.directAnswer || 'Чым магу дапамагчы?',
        usedRag: false,
        sources: [],
        trace: {
          orchestratorDecision: decision,
          usedTool: 'chat',
          citations: [],
          ...(wasRewritten ? { standaloneQuestion } : {}),
        },
      };
    }

    const searchQuery = decision.searchQuery?.trim() || effectiveQuestion;
    const requestedTool = toolForDecision(decision);
    const lockTool = isForcedSearchDecision(decision);

    const firstAttempt = await this.searchRerankEvaluate(
      messages,
      effectiveQuestion,
      searchQuery,
      requestedTool,
      lockTool
    );

    const finalResult = !firstAttempt.evaluation.sufficientForAnswer
      ? await this.retrySearch(messages, effectiveQuestion, searchQuery, firstAttempt, lockTool)
      : firstAttempt;

    const usedTool = finalResult.searchPlan.tool;
    const enforcedPlan = finalResult.searchPlan;
    const bestOutput = finalResult.searchOutput;
    const bestEvaluation = finalResult.evaluation;
    const sourcesForAnswer = finalResult.rerankedSources;
    const rerankTrace = finalResult.rerankTrace;

    if (!bestOutput.found || sourcesForAnswer.length === 0) {
      return {
        answer:
          'У калекцыі не знайшлося дастатковых дадзеных для адказу на гэтае пытанне. Паспрабуйце перафармуляваць пытанне.',
        usedRag: true,
        searchQuery,
        sources: [],
        trace: {
          orchestratorDecision: decision,
          usedTool,
          searchPlan: enforcedPlan,
          queryBreakdown: bestOutput.queryBreakdown,
          citations: [],
          evaluationResult: bestEvaluation,
          rerank: rerankTrace,
          ...(wasRewritten ? { standaloneQuestion } : {}),
        },
      };
    }

    const finalAnswer = await this.answerWithContext(
      messages,
      effectiveQuestion,
      { ...bestOutput, sources: sourcesForAnswer },
      bestEvaluation,
      enforcedPlan
    );

    return {
      answer: finalAnswer.answer,
      usedRag: true,
      searchQuery,
      sources: sourcesForAnswer,
      trace: {
        orchestratorDecision: decision,
        usedTool,
        searchPlan: enforcedPlan,
        queryBreakdown: bestOutput.queryBreakdown,
        citations: finalAnswer.citations,
        evaluationResult: bestEvaluation,
        rerank: rerankTrace,
        ...(wasRewritten ? { standaloneQuestion } : {}),
      },
    };
  }

  private async searchRerankEvaluate(
    messages: ChatMessage[],
    latestQuestion: string,
    searchQuery: string,
    fallbackTool: ToolName,
    lockTool = false
  ): Promise<SearchRerankResult> {
    const searchPlan = await this.queryPlannerAgent.plan(messages, searchQuery, fallbackTool);
    const enforcedPlan = lockTool
      ? { ...searchPlan, tool: fallbackTool }
      : searchPlan.tool === 'chat'
      ? { ...searchPlan, tool: fallbackTool }
      : searchPlan;

    const searchOutput = await this.invokeSearchTool(enforcedPlan);

    const topK = topKForPlan(enforcedPlan);
    const { rankedSources, trace: rerankTrace } = await this.reranker.rerank({
      question: latestQuestion,
      sources: searchOutput.sources,
      topK,
    });

    const evaluation = await this.evaluator.evaluate({
      question: latestQuestion,
      sources: rankedSources,
    });

    return {
      searchOutput,
      searchPlan: enforcedPlan,
      rerankedSources: rankedSources,
      rerankTrace,
      evaluation,
    };
  }

  private async retrySearch(
    messages: ChatMessage[],
    latestQuestion: string,
    searchQuery: string,
    previous: SearchRerankResult,
    lockTool = false
  ): Promise<SearchRerankResult> {
    const retryHint = `${searchQuery} — папярэдні пошук: ${previous.evaluation.evaluationReason}`;
    const result = await this.searchRerankEvaluate(
      messages,
      latestQuestion,
      retryHint,
      previous.searchPlan.tool,
      lockTool
    );

    if (result.evaluation.qualityScore >= previous.evaluation.qualityScore) {
      return result;
    }

    return previous;
  }

  private async invokeSearchTool(plan: SearchPlan): Promise<RagSearchOutput> {
    if (plan.tool === 'dialect_dictionary_search') {
      return this.dialectDictionarySearchTool.invokePlan(plan);
    }

    if (plan.tool === 'rag_search') {
      return this.ragSearchTool.invokePlan(plan);
    }

    if (plan.tool === 'orthographic_dictionary_search') {
      return this.orthographicDictionarySearchTool.invokePlan(plan);
    }

    if (plan.tool === 'translation_dictionary_search') {
      return this.translationDictionarySearchTool.invokePlan(plan);
    }

    if (plan.tool === 'explanatory_dictionary_search') {
      return this.explanatoryDictionarySearchTool.invokePlan(plan);
    }

    if (plan.tool === 'record_search') {
      return this.recordSearchTool.invokePlan(plan);
    }

    return this.folkWisdomSearchTool.invokePlan(plan);
  }

  private async decide(
    messages: ChatMessage[],
    latestQuestion: string
  ): Promise<OrchestratorDecision> {
    const recordDecision = recordSearchDecision(latestQuestion);
    if (recordDecision) return recordDecision;

    const model = await chatModel(config.chat.toolModel, {
      ollamaUrl: config.chat.ollamaUrl,
      reasoningEffort: config.chat.toolReasoningEffort,
    });

    const response = await model.invoke([
      new SystemMessage(DECISION_SYSTEM_PROMPT),
      new SystemMessage(
        schemaInstruction(
          'OrchestratorDecision',
          '{"action":"answer_directly|search_rag|search_folk_wisdom|search_dialect_dictionary|search_orthographic_dictionary|search_translation_dictionary|search_explanatory_dictionary","searchQuery":"string optional","directAnswer":"string optional","reason":"string"}'
        )
      ),
      new HumanMessage(
        JSON.stringify({
          latestQuestion,
          availableTools: TOOL_CATALOG,
          messages,
        })
      ),
    ]);

    return parseStructuredOutput(
      response.content,
      OrchestratorDecisionSchema,
      fallbackDecision(latestQuestion)
    );
  }

  private async answerWithContext(
    messages: ChatMessage[],
    latestQuestion: string,
    searchOutput: RagSearchOutput,
    evaluation: EvaluationResult,
    searchPlan: SearchPlan
  ): Promise<FinalAnswer> {
    const model = await chatModel(config.chat.model, {
      ollamaUrl: config.chat.ollamaUrl,
      reasoningEffort: config.chat.reasoningEffort,
    });

    const response = await model.invoke([
      new SystemMessage(LIST_FINAL_SYSTEM_PROMPT),
      new SystemMessage(
        schemaInstruction(
          'FinalAnswer',
          '{"answer":"string","usedRag":true,"citations":["file names or source labels"]}'
        )
      ),
      ...messages.slice(-8).map(toLangChainMessage),
      new HumanMessage(
        JSON.stringify({
          latestQuestion,
          evaluationQualityScore: evaluation.qualityScore,
          evaluationSufficient: evaluation.sufficientForAnswer,
          resultMode: searchPlan.resultMode,
          lookupTerm: searchPlan.lookupTerm,
          semanticFacets: searchPlan.semanticFacets,
          ragSearch: {
            query: searchOutput.query,
            found: searchOutput.found,
            queryBreakdown: searchOutput.queryBreakdown || [],
            sources: searchOutput.sources.map((source, index) => ({
              id: index + 1,
              fileName: source.fileName,
              category: source.category,
              dictionaryType: source.dictionaryType,
              payloadKind: source.payloadKind,
              sourceBook: source.sourceBook,
              sectionTitle: source.sectionTitle,
              recordType: source.recordType,
              tags: source.tags,
              title: source.title,
              page: source.page,
              score: source.score,
              relevanceScore:
                'relevanceScore' in source
                  ? (source as { relevanceScore: number }).relevanceScore
                  : source.score,
              matchedQueries: source.matchedQueries || [],
              text: source.text,
            })),
          },
        })
      ),
    ]);

    return parseStructuredOutput(response.content, FinalAnswerSchema, {
      answer:
        'Не атрымалася атрымаць структураваны адказ ад мадэлі. Паспрабуйце перафармуляваць пытанне.',
      usedRag: true,
      citations: [],
    });
  }
}

function latestUserMessage(messages: ChatMessage[]): string {
  const latest = [...messages].reverse().find((message) => message.role === 'user');
  if (!latest) {
    throw new Error('At least one user message is required');
  }

  return latest.content;
}

function fallbackDecision(latestQuestion: string): OrchestratorDecision {
  if (/^(hi|hello|вітаю|прывітанне|добры дзень|дзякуй|thanks)/i.test(latestQuestion.trim())) {
    return {
      action: 'answer_directly',
      directAnswer: 'Вітаю! Задайце пытанне пра прыказкі або народную мудрасць.',
      reason: 'Fallback: conversational greeting.',
    };
  }

  return {
    action: 'search_rag',
    searchQuery: latestQuestion,
    reason: 'Fallback: defaulting to general RAG search.',
  };
}

function isForcedSearchDecision(decision: OrchestratorDecision): boolean {
  return decision.reason.startsWith('Forced tool:');
}

function toolForDecision(decision: OrchestratorDecision): ToolName {
  if (decision.action === 'search_records') return 'record_search';
  if (decision.action === 'search_dialect_dictionary') return 'dialect_dictionary_search';
  if (decision.action === 'search_orthographic_dictionary') return 'orthographic_dictionary_search';
  if (decision.action === 'search_translation_dictionary') return 'translation_dictionary_search';
  if (decision.action === 'search_explanatory_dictionary') return 'explanatory_dictionary_search';
  if (decision.action === 'search_rag') return 'rag_search';
  if (decision.action === 'search_folk_wisdom') return 'folk_wisdom_search';
  return 'chat';
}

function recordSearchDecision(question: string): OrchestratorDecision | undefined {
  const normalized = question
    .toLowerCase()
    .replace(/[ё]/g, 'е')
    .replace(/[ў]/g, 'у');
  const asksForList = /(дай|пакажы|знайдзі|пашукай|падбяры|якія|спіс|усе|некалькі)/iu.test(normalized);
  const categoryQuery =
    /(вітан|прывітан|здароўкан|развітан|пажадан|зычэн|праклен|праклён|гразьб|грозьб|пагроз|абраз|лаянк|зняваг|цвяліл|пад.?ялдыч|прыкмет|адпаведнік|фразэал|фразеал|перакладчык прыказак|маскоўска.?беларуск)/iu.test(normalized);

  if (!categoryQuery) return undefined;

  return {
    action: 'search_records',
    searchQuery: question,
    reason: asksForList
      ? 'Forced tool: category list/extract query should use record search.'
      : 'Forced tool: category query should use record search for recall.',
  };
}

interface SearchRerankResult {
  searchOutput: RagSearchOutput;
  searchPlan: SearchPlan;
  rerankedSources: RerankedSource[];
  rerankTrace: RerankTrace;
  evaluation: EvaluationResult;
}

function topKForPlan(plan: SearchPlan): number {
  if (
    plan.resultMode === 'list' ||
    plan.resultMode === 'section' ||
    plan.resultMode === 'explore'
  ) {
    if (plan.desiredResultCount) return plan.desiredResultCount;
    if (plan.resultMode === 'section') return 60;
    return 30;
  }

  return NARROW_MODE_TOP_K;
}

function toLangChainMessage(message: ChatMessage): HumanMessage | AIMessage {
  return message.role === 'user'
    ? new HumanMessage(message.content)
    : new AIMessage(message.content);
}
