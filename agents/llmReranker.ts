import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { chatModel } from '../common/model';
import { config } from '../config';
import { parseStructuredOutput, schemaInstruction } from './json';
import { RERANKER_SYSTEM_PROMPT } from './prompts';
import type { RetrievedSource } from '../rag/types';
import {
  RerankResultSchema,
  type RerankResult,
  type RerankedSource,
  type RerankTrace,
} from './schemas';

const MAX_CANDIDATE_TEXT = 600;

export interface RerankInput {
  question: string;
  sources: RetrievedSource[];
  topK: number;
}

export interface RerankOutput {
  rankedSources: RerankedSource[];
  trace: RerankTrace;
}

export class LlmReranker {
  async rerank(input: RerankInput): Promise<RerankOutput> {
    if (input.sources.length === 0) {
      return {
        rankedSources: [],
        trace: { inputCount: 0, outputCount: 0, modelLatencyMs: 0 },
      };
    }

    if (input.sources.length === 1) {
      return {
        rankedSources: [{ ...input.sources[0], relevanceScore: 1 }],
        trace: { inputCount: 1, outputCount: 1, modelLatencyMs: 0 },
      };
    }

    const startedAt = Date.now();

    const model = await chatModel(config.chat.toolModel, {
      ollamaUrl: config.chat.ollamaUrl,
      reasoningEffort: config.chat.toolReasoningEffort,
    });

    const candidates = input.sources.map((source, index) => ({
      id: index + 1,
      fileName: source.fileName,
      page: source.page,
      text: source.text.slice(0, MAX_CANDIDATE_TEXT),
    }));

    const response = await model.invoke([
      new SystemMessage(RERANKER_SYSTEM_PROMPT),
      new SystemMessage(
        schemaInstruction(
          'RerankResult',
          '{"ranked":[{"id":number,"relevanceScore":0.0-1.0,"reason":"string optional"}]}'
        )
      ),
      new HumanMessage(
        JSON.stringify({
          question: input.question,
          desiredTopK: input.topK,
          candidates,
        })
      ),
    ]);

    const fallback: RerankResult = {
      ranked: input.sources.map((_, index) => ({
        id: index + 1,
        relevanceScore: clamp01(1 - index / Math.max(input.sources.length, 1)),
      })),
    };

    const parsed = parseStructuredOutput(response.content, RerankResultSchema, fallback);

    const scoreById = new Map<number, { score: number; reason?: string }>();
    for (const entry of parsed.ranked) {
      scoreById.set(entry.id, { score: clamp01(entry.relevanceScore), reason: entry.reason });
    }

    const ranked: RerankedSource[] = input.sources
      .map((source, index) => {
        const id = index + 1;
        const judged = scoreById.get(id);
        return {
          ...source,
          relevanceScore: judged?.score ?? 0,
          relevanceReason: judged?.reason,
        };
      })
      .sort((left, right) => right.relevanceScore - left.relevanceScore)
      .slice(0, input.topK);

    const modelLatencyMs = Date.now() - startedAt;

    return {
      rankedSources: ranked,
      trace: {
        inputCount: input.sources.length,
        outputCount: ranked.length,
        modelLatencyMs,
      },
    };
  }
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}
