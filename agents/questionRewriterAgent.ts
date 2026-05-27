import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { z } from 'zod';
import { chatModel } from '../common/model';
import { config } from '../config';
import { parseStructuredOutput, schemaInstruction } from './json';
import { REWRITER_SYSTEM_PROMPT } from './prompts';
import type { ChatMessage } from './schemas';


const StandaloneQuestionSchema = z.object({
  standaloneQuestion: z.string().min(1),
});

const REWRITE_TRIGGER_PATTERN =
  /^(а|і|ці|таксама|яшчэ|што|як|пра|для|у|з|ад|па|больш|менш|потым|пасля|тое|той|тая|гэта|гэтыя|іх|ім|ён|яна|яны|такога|такіх|падобн|той самы)\b/iu;

export class QuestionRewriterAgent {
  async rewrite(messages: ChatMessage[], latestQuestion: string): Promise<string> {
    if (!shouldRewrite(messages, latestQuestion)) {
      return latestQuestion;
    }

    const model = await chatModel(config.chat.toolModel, {
      ollamaUrl: config.chat.ollamaUrl,
      reasoningEffort: config.chat.toolReasoningEffort,
    });

    const response = await model.invoke([
      new SystemMessage(REWRITER_SYSTEM_PROMPT),
      new SystemMessage(
        schemaInstruction('StandaloneQuestion', '{"standaloneQuestion":"string"}')
      ),
      new HumanMessage(
        JSON.stringify({
          latestQuestion,
          recentMessages: messages.slice(-8),
        })
      ),
    ]);

    const parsed = parseStructuredOutput(response.content, StandaloneQuestionSchema, {
      standaloneQuestion: latestQuestion,
    });

    const rewritten = parsed.standaloneQuestion.trim();
    return rewritten.length > 0 ? rewritten : latestQuestion;
  }
}

function shouldRewrite(messages: ChatMessage[], latestQuestion: string): boolean {
  if (messages.length <= 1) return false;

  const trimmed = latestQuestion.trim();
  if (trimmed.length === 0) return false;
  if (trimmed.length < 80) return true;

  return REWRITE_TRIGGER_PATTERN.test(trimmed);
}
