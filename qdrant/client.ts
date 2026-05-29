export type Vector = number[];
export type Payload = Record<string, unknown>;

export interface QdrantPoint {
  id: string;
  vector: Vector;
  payload: Payload;
}

export interface SearchResult {
  id: string;
  score: number;
  payload: Payload;
}

export interface ScrollResult {
  id: string;
  payload?: Payload;
}

export type PayloadFilter = {
  must?: Array<{ key: string; match: { value: string | number | boolean } }>;
};

export class QdrantClient {
  constructor(
    private readonly baseUrl: string,
    private readonly apiKey?: string
  ) {}

  async createCollection(name: string, vectorSize: number, distance = 'Cosine'): Promise<void> {
    await this.request(`/collections/${name}`, {
      method: 'PUT',
      body: {
        vectors: {
          size: vectorSize,
          distance,
        },
      },
    });
  }

  async ensureCollection(name: string, vectorSize: number): Promise<void> {
    const info = await this.getCollection(name);

    if (!info) {
      await this.createCollection(name, vectorSize);
      return;
    }

    const existingSize = readVectorSize(info);
    if (existingSize && existingSize !== vectorSize) {
      throw new Error(
        `Qdrant collection "${name}" uses vector size ${existingSize}, but EMBEDDINGS_DIM is ${vectorSize}`
      );
    }
  }

  async listCollections(): Promise<string[]> {
    const data = await this.request<{ result: { collections: { name: string }[] } }>('/collections', {
      method: 'GET',
    });

    return data.result.collections.map((collection) => collection.name);
  }

  async upsertPoints(collection: string, points: QdrantPoint[]): Promise<void> {
    await this.request(`/collections/${collection}/points?wait=true`, {
      method: 'PUT',
      body: { points },
    });
  }

  async deletePoints(collection: string, filter: PayloadFilter): Promise<void> {
    await this.request(`/collections/${collection}/points/delete?wait=true`, {
      method: 'POST',
      body: { filter },
    });
  }

  async search(
    collection: string,
    vector: Vector,
    limit: number,
    filter?: PayloadFilter
  ): Promise<SearchResult[]> {
    const data = await this.request<{ result: SearchResult[] }>(
      `/collections/${collection}/points/search`,
      {
        method: 'POST',
        body: {
          vector,
          limit,
          ...(filter ? { filter } : {}),
          with_payload: true,
        },
      }
    );

    return data.result;
  }

  async count(collection: string): Promise<number> {
    const data = await this.request<{ result: { count: number } }>(`/collections/${collection}/points/count`, {
      method: 'POST',
      body: {
        exact: true,
      },
    });

    return data.result.count;
  }

  async scrollPayloads(
    collection: string,
    limit = 256,
    offset?: string | number,
    filter?: PayloadFilter
  ): Promise<{ points: ScrollResult[]; nextOffset?: string | number }> {
    const data = await this.request<{
      result: {
        points: ScrollResult[];
        next_page_offset?: string | number;
      };
    }>(`/collections/${collection}/points/scroll`, {
      method: 'POST',
      body: {
        limit,
        offset,
        ...(filter ? { filter } : {}),
        with_payload: true,
        with_vector: false,
      },
    });

    return {
      points: data.result.points,
      nextOffset: data.result.next_page_offset,
    };
  }

  private async getCollection(name: string): Promise<unknown | null> {
    const response = await fetch(`${this.baseUrl}/collections/${name}`, {
      method: 'GET',
      headers: this.headers(),
    });

    if (response.status === 404) {
      return null;
    }

    if (!response.ok) {
      throw new Error(`Qdrant request failed (${response.status}): ${await response.text()}`);
    }

    const data = (await response.json()) as { result?: unknown };
    return data.result || null;
  }

  private async request<T = unknown>(
    path: string,
    options: { method: string; body?: unknown }
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: options.method,
      headers: this.headers(),
      body: options.body ? JSON.stringify(options.body) : undefined,
    });

    if (!response.ok) {
      throw new Error(`Qdrant request failed (${response.status}): ${await response.text()}`);
    }

    if (response.status === 204) {
      return undefined as T;
    }

    return response.json() as Promise<T>;
  }

  private headers(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['api-key'] = this.apiKey;
    }

    return headers;
  }
}

function readVectorSize(info: unknown): number | null {
  if (!info || typeof info !== 'object') return null;

  const config = 'config' in info ? info.config : undefined;
  if (!config || typeof config !== 'object') return null;

  const params = 'params' in config ? config.params : undefined;
  if (!params || typeof params !== 'object') return null;

  const vectors = 'vectors' in params ? params.vectors : undefined;
  if (!vectors || typeof vectors !== 'object') return null;

  const size = 'size' in vectors ? vectors.size : undefined;
  return typeof size === 'number' ? size : null;
}
