import { AIMessage, HumanMessage, SystemMessage } from '@langchain/core/messages';
import { chatModel } from '../common/model';
import { config } from '../config';
import type { FolkWisdomSearchTool } from './folkWisdomSearchTool';
import { LlmReranker } from './llmReranker';
import { QueryPlannerAgent, fallbackPlan } from './queryPlannerAgent';
import { QuestionRewriterAgent } from './questionRewriterAgent';
import { RagResultEvaluator } from './ragResultEvaluator';
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

const DECISION_SYSTEM_PROMPT = [
  'You are the orchestrator for a Belarusian proverbs bot.',
  'You receive a chat history and the latest user question.',
  'Use search_folk_wisdom for any question about proverbs, sayings, folk wisdom, aphorisms, народныя мудрасці, прыказкі, прымаўкі.',
  'Use answer_directly ONLY for greetings or simple conversation that clearly does not need the proverbs collection.',
  'When in doubt, always use search_folk_wisdom.',
].join(' ');

const LIST_FINAL_SYSTEM_PROMPT = [
  'You are the final chat agent presenting search results from a Belarusian proverbs collection.',
  'Answer in Belarusian unless the user clearly asks for another language.',
  'Use only the provided RAG context. Never invent or add items not present in the context.',
  'If the user asks for proverbs, sayings, folk wisdom, or any list of items,',
  'present EVERY found item as a numbered list: "1. ...", "2. ...", etc.',
  'Each list item must be on its own line.',
  'Do not merge, summarise, or paraphrase individual proverbs — quote them exactly from the context.',
  'When the context contains many distinct relevant items, preserve breadth: include every distinct found item that answers the request.',
  'If the search appears partial, say briefly that these are the found items, not a guaranteed complete collection.',
  'After the list you may add a short concluding sentence if useful.',
  'If context is missing or insufficient, say that the documents do not contain enough data.',
].join(' ');

export class ChatOrchestratorAgent {
  private readonly evaluator = new RagResultEvaluator();
  private readonly reranker = new LlmReranker();

  constructor(
    private readonly folkWisdomSearchTool: FolkWisdomSearchTool,
    private readonly queryPlannerAgent = new QueryPlannerAgent(),
    private readonly questionRewriterAgent = new QuestionRewriterAgent()
  ) {}

  async chat(messages: ChatMessage[]): Promise<ChatAgentResponse> {
    const originalQuestion = latestUserMessage(messages);
    const standaloneQuestion = await this.questionRewriterAgent.rewrite(messages, originalQuestion);
    const effectiveQuestion = standaloneQuestion || originalQuestion;
    const wasRewritten = standaloneQuestion !== originalQuestion;

    const decision = requiresFolkWisdomTool(effectiveQuestion)
      ? forcedFolkWisdomDecision(effectiveQuestion)
      : await this.decide(messages, effectiveQuestion);

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
    const usedTool: ToolName = 'folk_wisdom_search';

    const firstAttempt = await this.searchRerankEvaluate(messages, effectiveQuestion, searchQuery);

    const finalResult = !firstAttempt.evaluation.sufficientForAnswer
      ? await this.retrySearch(messages, effectiveQuestion, searchQuery, firstAttempt)
      : firstAttempt;

    const enforcedPlan = { ...finalResult.searchPlan, tool: usedTool };
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
    searchQuery: string
  ): Promise<SearchRerankResult> {
    const searchPlan = await this.queryPlannerAgent.plan(messages, searchQuery, 'folk_wisdom_search');
    const enforcedPlan = { ...searchPlan, tool: 'folk_wisdom_search' as ToolName };

    const searchOutput = await this.folkWisdomSearchTool.invokePlan(enforcedPlan);

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
      searchPlan,
      rerankedSources: rankedSources,
      rerankTrace,
      evaluation,
    };
  }

  private async retrySearch(
    messages: ChatMessage[],
    latestQuestion: string,
    searchQuery: string,
    previous: SearchRerankResult
  ): Promise<SearchRerankResult> {
    const retryHint = `${searchQuery} — папярэдні пошук: ${previous.evaluation.evaluationReason}`;
    const result = await this.searchRerankEvaluate(messages, latestQuestion, retryHint);

    if (result.evaluation.qualityScore >= previous.evaluation.qualityScore) {
      return result;
    }

    return previous;
  }

  private async decide(
    messages: ChatMessage[],
    latestQuestion: string
  ): Promise<OrchestratorDecision> {
    const model = await chatModel(config.chat.model, {
      ollamaUrl: config.chat.ollamaUrl,
    });

    const response = await model.invoke([
      new SystemMessage(DECISION_SYSTEM_PROMPT),
      new SystemMessage(
        schemaInstruction(
          'OrchestratorDecision',
          '{"action":"answer_directly|search_folk_wisdom","searchQuery":"string optional","directAnswer":"string optional","reason":"string"}'
        )
      ),
      new HumanMessage(
        JSON.stringify({
          latestQuestion,
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
          semanticFacets: searchPlan.semanticFacets,
          ragSearch: {
            query: searchOutput.query,
            found: searchOutput.found,
            queryBreakdown: searchOutput.queryBreakdown || [],
            sources: searchOutput.sources.map((source, index) => ({
              id: index + 1,
              fileName: source.fileName,
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
  if (requiresFolkWisdomTool(latestQuestion)) {
    return forcedFolkWisdomDecision(latestQuestion);
  }

  if (/^(hi|hello|вітаю|прывітанне|добры дзень|дзякуй|thanks)/i.test(latestQuestion.trim())) {
    return {
      action: 'answer_directly',
      directAnswer: 'Вітаю! Задайце пытанне пра прыказкі або народную мудрасць.',
      reason: 'Fallback: conversational greeting.',
    };
  }

  return {
    action: 'search_folk_wisdom',
    searchQuery: latestQuestion,
    reason: 'Fallback: defaulting to folk wisdom search.',
  };
}

function forcedFolkWisdomDecision(latestQuestion: string): OrchestratorDecision {
  return {
    action: 'search_folk_wisdom',
    searchQuery: latestQuestion,
    reason: 'Forced tool: the question asks for proverbs, sayings, or other folk wisdom.',
  };
}

function requiresFolkWisdomTool(question: string): boolean {
  return (
    /\b(proverb|proverbs|saying|sayings|folk wisdom|aphorism|aphorisms)\b/i.test(question) ||
    /(прыказк|прымаўк|народн\w*\s+мудрасц|прыслоў|выслоў|афарызм)/i.test(question)
  );
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
