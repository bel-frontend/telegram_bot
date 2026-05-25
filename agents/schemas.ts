import { z } from 'zod';

export const ChatRoleSchema = z.enum(['user', 'assistant']);

export const ChatMessageSchema = z.object({
  role: ChatRoleSchema,
  content: z.string().min(1),
});

export const RetrievedSourceSchema = z.object({
  text: z.string(),
  score: z.number(),
  source: z.string().optional(),
  fileName: z.string().optional(),
  page: z.number().optional(),
  matchedQueries: z.array(z.string()).optional(),
  vectorRank: z.number().optional(),
  lexicalRank: z.number().optional(),
});

export const OrchestratorDecisionSchema = z.object({
  action: z.enum(['answer_directly', 'search_rag', 'search_folk_wisdom', 'search_dialect_dictionary']),
  searchQuery: z.string().optional(),
  directAnswer: z.string().optional(),
  reason: z.string(),
});

export const ToolNameSchema = z.enum([
  'chat',
  'rag_search',
  'folk_wisdom_search',
  'dialect_dictionary_search',
]);

export const SearchIntentSchema = z.enum([
  'direct_chat',
  'general_rag',
  'folk_wisdom',
  'dialect_definition',
  'dialect_section_lookup',
  'exact_phrase',
]);

export const ResultModeSchema = z.enum(['answer', 'list', 'section', 'explore']);

export const SearchPlanSchema = z.object({
  intent: SearchIntentSchema,
  coreQuery: z.string(),
  expandedQueries: z.array(z.string()).min(1),
  semanticFacets: z.array(z.string()).optional(),
  resultMode: ResultModeSchema.default('answer'),
  desiredResultCount: z.number().int().min(1).max(80).optional(),
  targetBook: z.enum(['any', 'vushatski_slovazbor', 'proverbs_dictionary']),
  tool: ToolNameSchema,
  reason: z.string(),
});

export const QueryBreakdownSchema = z.object({
  query: z.string(),
  retrievedCount: z.number(),
  keptCount: z.number(),
});

export const RagSearchOutputSchema = z.object({
  query: z.string(),
  found: z.boolean(),
  sources: z.array(RetrievedSourceSchema),
  sourceCount: z.number(),
  queryBreakdown: z.array(QueryBreakdownSchema).optional(),
});

export const FinalAnswerSchema = z.object({
  answer: z.string(),
  usedRag: z.boolean(),
  citations: z.array(z.string()),
});

export const EvaluatedSourceSchema = RetrievedSourceSchema.extend({
  relevanceScore: z.number().min(0).max(1),
  relevanceReason: z.string().optional(),
});

export const EvaluationResultSchema = z.object({
  sufficientForAnswer: z.boolean(),
  qualityScore: z.number().min(0).max(1),
  evaluationReason: z.string(),
  relevantSources: z.array(EvaluatedSourceSchema),
});

export const RerankedSourceSchema = EvaluatedSourceSchema;

export const RerankResultSchema = z.object({
  ranked: z.array(
    z.object({
      id: z.number().int().min(1),
      relevanceScore: z.number().min(0).max(1),
      reason: z.string().optional(),
    })
  ),
});

export const EvaluatedRagOutputSchema = RagSearchOutputSchema.extend({
  evaluation: EvaluationResultSchema.optional(),
});

export const ChatRequestSchema = z.object({
  question: z.string().min(1).optional(),
  messages: z.array(ChatMessageSchema).optional(),
});

export type ChatMessage = z.infer<typeof ChatMessageSchema>;
export type OrchestratorDecision = z.infer<typeof OrchestratorDecisionSchema>;
export type ToolName = z.infer<typeof ToolNameSchema>;
export type ResultMode = z.infer<typeof ResultModeSchema>;
export type SearchPlan = z.infer<typeof SearchPlanSchema>;
export type RagSearchOutput = z.infer<typeof RagSearchOutputSchema>;
export type EvaluatedSource = z.infer<typeof EvaluatedSourceSchema>;
export type RerankedSource = z.infer<typeof RerankedSourceSchema>;
export type RerankResult = z.infer<typeof RerankResultSchema>;
export type EvaluationResult = z.infer<typeof EvaluationResultSchema>;
export type EvaluatedRagOutput = z.infer<typeof EvaluatedRagOutputSchema>;
export type FinalAnswer = z.infer<typeof FinalAnswerSchema>;
export type ChatRequestInput = z.infer<typeof ChatRequestSchema>;

export interface RerankTrace {
  inputCount: number;
  outputCount: number;
  modelLatencyMs: number;
}

export interface ChatAgentResponse {
  answer: string;
  usedRag: boolean;
  searchQuery?: string;
  sources: z.infer<typeof RetrievedSourceSchema>[];
  trace: {
    orchestratorDecision: OrchestratorDecision;
    usedTool: ToolName;
    searchPlan?: SearchPlan;
    queryBreakdown?: z.infer<typeof QueryBreakdownSchema>[];
    citations: string[];
    evaluationResult?: EvaluationResult;
    standaloneQuestion?: string;
    rerank?: RerankTrace;
  };
}
