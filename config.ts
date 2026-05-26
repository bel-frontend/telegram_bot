import dotenv from 'dotenv';
dotenv.config();

function numberFromEnv(name: string, defaultValue: number): number {
  const val = process.env[name];
  if (!val) return defaultValue;
  const parsed = Number(val);
  return Number.isFinite(parsed) ? parsed : defaultValue;
}

function booleanFromEnv(name: string, defaultValue = false): boolean {
  const val = process.env[name];
  if (!val) return defaultValue;
  return ['1', 'true', 'yes', 'on'].includes(val.toLowerCase());
}

function reasoningEffortFromEnv(name: string, defaultValue: 'minimal' | 'low' | 'medium' | 'high') {
  const val = process.env[name];
  if (val === 'minimal' || val === 'low' || val === 'medium' || val === 'high') {
    return val;
  }

  return defaultValue;
}

export const config = {
  qdrant: {
    url: (process.env.QDRANT_URL || '').replace(/\/$/, ''),
    apiKey: process.env.QDRANT_API_KEY,
    collection: process.env.QDRANT_COLLECTION || 'lesson11_pdf_documents',
  },
  embeddings: {
    model: process.env.EMBEDDINGS_MODEL || 'text-embedding-3-large',
    dimensions: numberFromEnv('EMBEDDINGS_DIM', 3072),
  },
  chat: {
    model: process.env.CHAT_MODEL || 'gpt-5.5',
    reasoningEffort: reasoningEffortFromEnv('CHAT_REASONING_EFFORT', 'medium'),
    toolModel: process.env.CHAT_TOOL_MODEL || process.env.CHAT_MODEL || 'gpt-5.4-mini',
    toolReasoningEffort: reasoningEffortFromEnv('CHAT_TOOL_REASONING_EFFORT', 'minimal'),
    ollamaUrl: process.env.OLLAMA_BASE_URL,
  },
  server: {
    topK: numberFromEnv('TOP_K', 5),
  },
  search: {
    minScore: numberFromEnv('RAG_MIN_SCORE', 0.25),
    folkWisdomTopK: numberFromEnv('FOLK_WISDOM_TOP_K', 30),
  },
  debug: {
    rag: booleanFromEnv('DISCORD_DEBUG_RAG') || booleanFromEnv('RAG_DEBUG'),
  },
};
