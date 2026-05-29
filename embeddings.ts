import { OpenAIEmbeddings } from '@langchain/openai';
import { config } from './config';

export type EmbeddingsClient = Pick<OpenAIEmbeddings, 'embedQuery' | 'embedDocuments'>;

export function createEmbeddings(): EmbeddingsClient {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is required for embeddings.');
  }

  return new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
    model: config.embeddings.model,
    dimensions: config.embeddings.dimensions,
    batchSize: 256,
  });
}
