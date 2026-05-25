import { config } from '../config';
import { QdrantClient } from './client';

export function createQdrantClient(): QdrantClient {
  if (!config.qdrant.url) {
    throw new Error('QDRANT_URL is required. Add it to .env');
  }

  return new QdrantClient(config.qdrant.url, config.qdrant.apiKey);
}
