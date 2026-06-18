export interface RetrievedSource {
  text: string;
  score: number;
  payloadKind?: string;
  source?: string;
  fileName?: string;
  category?: string;
  dictionaryType?: string;
  sourceBook?: string;
  sectionTitle?: string;
  recordType?: string;
  tags?: string[];
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
