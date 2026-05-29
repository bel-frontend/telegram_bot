export interface RetrievedSource {
  text: string;
  score: number;
  source?: string;
  fileName?: string;
  category?: string;
  dictionaryType?: string;
  title?: string;
  page?: number;
  matchedQueries?: string[];
  vectorRank?: number;
  lexicalRank?: number;
}

export interface RagAnswer {
  answer: string;
  sources: RetrievedSource[];
}
