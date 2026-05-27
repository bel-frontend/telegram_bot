import { PromptSDK } from 'goman-live';

const sdk = new PromptSDK({
  applicationId: process.env.GOMAN_APPLICATION_ID ?? '',
  apiKey: process.env.GOMAN_API_KEY ?? '',
});

const PROMPT_IDS = {
  DECISION:  '6a1747a9235647f749963031',
  LIST_FINAL: '6a1747a8235647f749963027',
  PLANNER:   '6a1747a7235647f749963013',
  REWRITER:  '6a1747a7235647f74996301c',
  EVALUATOR: '6a1747a6235647f74996300a',
  RERANKER:  '6a1747ab235647f749963058',
} as const;

export let DECISION_SYSTEM_PROMPT   = '';
export let LIST_FINAL_SYSTEM_PROMPT = '';
export let PLANNER_SYSTEM_PROMPT    = '';
export let REWRITER_SYSTEM_PROMPT   = '';
export let EVALUATOR_SYSTEM_PROMPT  = '';
export let RERANKER_SYSTEM_PROMPT   = '';

// ---------------------------------------------------------------------------
// Remote loader
// ---------------------------------------------------------------------------

export async function initPrompts(): Promise<void> {
  const [decision, listFinal, planner, rewriter, evaluator, reranker] =
    await Promise.all([
      sdk.getPromptFromRemote(PROMPT_IDS.DECISION),
      sdk.getPromptFromRemote(PROMPT_IDS.LIST_FINAL),
      sdk.getPromptFromRemote(PROMPT_IDS.PLANNER),
      sdk.getPromptFromRemote(PROMPT_IDS.REWRITER),
      sdk.getPromptFromRemote(PROMPT_IDS.EVALUATOR),
      sdk.getPromptFromRemote(PROMPT_IDS.RERANKER),
    ]);

  DECISION_SYSTEM_PROMPT   = decision.value;
  LIST_FINAL_SYSTEM_PROMPT = listFinal.value;
  PLANNER_SYSTEM_PROMPT    = planner.value;
  REWRITER_SYSTEM_PROMPT   = rewriter.value;
  EVALUATOR_SYSTEM_PROMPT  = evaluator.value;
  RERANKER_SYSTEM_PROMPT   = reranker.value;

  console.log('[prompts] Loaded from Goman.');
}
