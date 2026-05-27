import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { chatModel } from '../common/model';
import { config } from '../config';
import { parseStructuredOutput, schemaInstruction } from './json';
import { EVALUATOR_SYSTEM_PROMPT } from './prompts';
import type { RetrievedSource } from '../rag/types';
import {
  EvaluationResultSchema,
  type EvaluationResult,
  type EvaluatedSource,
} from './schemas';

const MAX_SOURCE_TEXT = 500;

export class RagResultEvaluator {
  async evaluate(options: {
    question: string;
    sources: RetrievedSource[];
  }): Promise<EvaluationResult> {
    if (options.sources.length === 0) {
      return {
        sufficientForAnswer: false,
        qualityScore: 0,
        evaluationReason: 'Крыніц не знойдзена.',
        relevantSources: [],
      };
    }

    const model = await chatModel(config.chat.toolModel, {
      ollamaUrl: config.chat.ollamaUrl,
      reasoningEffort: config.chat.toolReasoningEffort,
    });

    const trimmedSources = options.sources.map((source, index) => ({
      id: index + 1,
      fileName: source.fileName,
      page: source.page,
      text: source.text.slice(0, MAX_SOURCE_TEXT),
    }));

    const response = await model.invoke([
      new SystemMessage(EVALUATOR_SYSTEM_PROMPT),
      new SystemMessage(
        schemaInstruction(
          'EvaluationResult',
          '{"sufficientForAnswer":boolean,"qualityScore":0.0-1.0,"evaluationReason":"string","relevantSources":[{"text":"..","score":0.0,"relevanceScore":0.0-1.0,"relevanceReason":"optional"}]}'
        )
      ),
      new HumanMessage(
        JSON.stringify({
          question: options.question,
          sources: trimmedSources,
        })
      ),
    ]);

    const fallback: EvaluationResult = {
      sufficientForAnswer: options.sources.length > 0,
      qualityScore: 0.5,
      evaluationReason: 'Ацэнка недаступная.',
      relevantSources: options.sources.map((source) => ({
        ...source,
        relevanceScore: source.score ?? 0.5,
      })) as EvaluatedSource[],
    };

    return parseStructuredOutput(response.content, EvaluationResultSchema, fallback);
  }
}
