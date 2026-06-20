import { createQdrantClient } from "../qdrant/index";
import { createEmbeddings } from "../embeddings";
import { QdrantRetriever } from "../rag/retriever";
import { LexicalRetriever } from "../rag/lexicalRetriever";
import { HybridRetriever } from "../rag/hybridRetriever";
import { FolkWisdomSearchTool } from "../agents/folkWisdomSearchTool";
import { RagSearchTool } from "../agents/ragSearchTool";
import { RecordSearchTool } from "../agents/recordSearchTool";
import { DialectDictionarySearchTool } from "../agents/dialectDictionarySearchTool";
import {
  createExplanatoryDictionarySearchTool,
  createOrthographicDictionarySearchTool,
  createTranslationDictionarySearchTool,
} from "../agents/dictionarySearchTools";
import { ChatOrchestratorAgent } from "../agents/chatOrchestrator";
import { initPrompts } from "../agents/prompts";
import type { ChatMessage } from "../agents/schemas";

async function run() {
  console.log("=== Loading Prompts from Goman ===");
  const startupStart = Date.now();
  await initPrompts();
  console.log(`Prompts loaded in ${Date.now() - startupStart}ms`);

  console.log("\n=== Initializing RAG Stack ===");
  const qdrant = createQdrantClient();
  const embeddings = createEmbeddings();
  const vectorRetriever = new QdrantRetriever(qdrant, embeddings);
  const lexicalRetriever = new LexicalRetriever(qdrant);
  const hybridRetriever = new HybridRetriever(vectorRetriever, lexicalRetriever);
  const folkWisdomTool = new FolkWisdomSearchTool(hybridRetriever);
  const ragSearchTool = new RagSearchTool(hybridRetriever);
  const dialectDictionarySearchTool = new DialectDictionarySearchTool(hybridRetriever);
  const orthographicDictionarySearchTool = createOrthographicDictionarySearchTool(hybridRetriever);
  const translationDictionarySearchTool = createTranslationDictionarySearchTool(hybridRetriever);
  const explanatoryDictionarySearchTool = createExplanatoryDictionarySearchTool(hybridRetriever);
  const recordSearchTool = new RecordSearchTool(qdrant, hybridRetriever);

  const orchestrator = new ChatOrchestratorAgent(
    folkWisdomTool,
    ragSearchTool,
    dialectDictionarySearchTool,
    orthographicDictionarySearchTool,
    translationDictionarySearchTool,
    explanatoryDictionarySearchTool,
    recordSearchTool
  );
  console.log("RAG Stack initialized.");

  // Test queries representation
  const queries = [
    "прывітанне",
    "прыказкі пра працу",
    "што значыць слова аброць",
    "як пішацца карова"
  ];

  console.log("\n=== Running Benchmarks ===");
  const results = [];

  for (const query of queries) {
    console.log(`\nQuery: "${query}"`);
    const messages: ChatMessage[] = [{ role: "user", content: query }];
    
    const start = Date.now();
    try {
      const response = await orchestrator.chat(messages);
      const elapsed = Date.now() - start;
      
      console.log(`Done in ${elapsed}ms`);
      console.log(`Answer: ${response.answer.slice(0, 100)}...`);
      console.log(`Used RAG: ${response.usedRag}`);
      console.log(`Tool used: ${response.trace.usedTool}`);
      console.log(`Sources: ${response.sources.length}`);
      if (response.trace.rerank) {
        console.log(`Rerank latency: ${response.trace.rerank.modelLatencyMs}ms`);
      }
      
      results.push({
        query,
        timeMs: elapsed,
        usedRag: response.usedRag,
        tool: response.trace.usedTool,
        sourcesCount: response.sources.length,
        rerankMs: response.trace.rerank?.modelLatencyMs ?? 0,
        status: "SUCCESS"
      });
    } catch (err: any) {
      const elapsed = Date.now() - start;
      console.error(`Error after ${elapsed}ms:`, err);
      results.push({
        query,
        timeMs: elapsed,
        usedRag: false,
        tool: "error",
        sourcesCount: 0,
        rerankMs: 0,
        status: `ERROR: ${err.message}`
      });
    }
  }

  console.log("\n=== Summary Table ===");
  console.table(results);
}

run().catch(err => {
  console.error("Benchmark run failed:", err);
  process.exit(1);
});
