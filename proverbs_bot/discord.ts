import {
  Client,
  GatewayIntentBits,
  Partials,
  ChannelType,
  type Message,
} from "discord.js";
import { createQdrantClient } from "../qdrant/index";
import { createEmbeddings } from "../embeddings";
import { QdrantRetriever } from "../rag/retriever";
import { LexicalRetriever } from "../rag/lexicalRetriever";
import { HybridRetriever } from "../rag/hybridRetriever";
import { FolkWisdomSearchTool } from "../agents/folkWisdomSearchTool";
import { RagSearchTool } from "../agents/ragSearchTool";
import { DialectDictionarySearchTool } from "../agents/dialectDictionarySearchTool";
import { ChatOrchestratorAgent } from "../agents/chatOrchestrator";
import { config } from "../config";
import type { ChatAgentResponse, ChatMessage } from "../agents/schemas";

const DISCORD_TOKEN = process.env.DISCORD_TOKEN;

if (!DISCORD_TOKEN) {
  throw new Error("DISCORD_TOKEN environment variable is required.");
}

// Initialise the RAG stack once at startup
const qdrant = createQdrantClient();
const embeddings = createEmbeddings();
const vectorRetriever = new QdrantRetriever(qdrant, embeddings);
const lexicalRetriever = new LexicalRetriever(qdrant);
const hybridRetriever = new HybridRetriever(vectorRetriever, lexicalRetriever);
const folkWisdomTool = new FolkWisdomSearchTool(hybridRetriever);
const ragSearchTool = new RagSearchTool(hybridRetriever);
const dialectDictionarySearchTool = new DialectDictionarySearchTool(
  hybridRetriever,
);
const orchestrator = new ChatOrchestratorAgent(
  folkWisdomTool,
  ragSearchTool,
  dialectDictionarySearchTool,
);
const conversations = new Map<string, ChatMessage[]>();
const debugConversations = new Set<string>();
const MAX_CONVERSATION_MESSAGES = 12;
const HELP_MESSAGE = `Прывітанне! Я  прыказкавы бот, які шукае прыказкі, прымаўкі, народныя мудрасці, праклёны, грозьбы, дыялектныя словы і выразы ў калекцыі (папаўняецца).
Можна пісаць звычайным тэкстам: шукаць прыказкі, прымаўкі, народныя мудрасці, праклёны, гразьбы, дыялектныя словы і выразы.

Каманды:
reset / рэзэт — пачаць размову нанова.
clear / ачысці — тое самае.`;

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.DirectMessages,
    GatewayIntentBits.MessageContent,
  ],
  partials: [Partials.Channel, Partials.Message],
});

client.once("clientReady", (readyClient) => {
  console.log(`Discord bot logged in as ${readyClient.user.tag}`);
});

client.on("messageCreate", async (message: Message) => {
  if (message.author.bot) return;

  const isDM = message.channel.type === ChannelType.DM;
  const isMentioned = client.user ? message.mentions.has(client.user) : false;

  if (!isDM && !isMentioned) return;

  // Strip mention from message text
  const question = message.content.replace(/<@!?\d+>/g, "").trim();
  if (!question) return;

  const conversationKey = keyForConversation(message, isDM);
  if (isResetCommand(question)) {
    conversations.delete(conversationKey);
    debugConversations.delete(conversationKey);
    await message.reply(`Добра, пачынаем новую размову.\n\n${HELP_MESSAGE}`);
    return;
  }

  const debugCommand = parseDebugCommand(question);
  if (debugCommand) {
    if (debugCommand === "on") {
      debugConversations.add(conversationKey);
      await message.reply(
        "Debug mode уключаны. Пасля адказу буду паказваць RAG-запыты і знойдзеныя крыніцы.",
      );
      return;
    }

    debugConversations.delete(conversationKey);
    await message.reply("Debug mode выключаны.");
    return;
  }

  const previousMessages = conversations.get(conversationKey) || [];
  if (previousMessages.length === 0) {
    await message.reply(HELP_MESSAGE);
  }

  // Show typing indicator and refresh it every 8s while waiting
  await sendTyping(message.channel);
  const typingInterval = setInterval(
    () => void sendTyping(message.channel),
    8000,
  );

  try {
    const messages: ChatMessage[] = [
      ...previousMessages,
      { role: "user", content: question },
    ];
    const result = await orchestrator.chat(messages);
    const reply = result.answer.slice(0, 2000);
    await message.reply(reply);
    if (isDebugEnabled(conversationKey)) {
      const debugText = formatRagDebug(result);
      console.log(debugText);
      for (const chunk of splitDiscordMessage(debugText)) {
        await message.reply(chunk);
      }
    }
    conversations.set(
      conversationKey,
      trimConversation([...messages, { role: "assistant", content: reply }]),
    );
  } catch (error) {
    console.error("Discord message error:", error);
    await message.reply("Нешта пайшло не так. Паспрабуй пазней.");
  } finally {
    clearInterval(typingInterval);
  }
});

client.login(DISCORD_TOKEN);

async function sendTyping(channel: Message["channel"]): Promise<void> {
  if ("sendTyping" in channel && typeof channel.sendTyping === "function") {
    await channel.sendTyping();
  }
}

function keyForConversation(message: Message, isDM: boolean): string {
  if (isDM) return `dm:${message.author.id}`;
  return `guild:${message.guildId || "unknown"}:${message.channel.id}:${message.author.id}`;
}

function isResetCommand(text: string): boolean {
  return /^(\/?reset|\/?restart|\/?clear|\/?new|рэз[эе]т|скінь|ачысці|новая размова|пачаць нанова)$/iu.test(
    text.trim(),
  );
}

function trimConversation(messages: ChatMessage[]): ChatMessage[] {
  return messages.slice(-MAX_CONVERSATION_MESSAGES);
}

function parseDebugCommand(text: string): "on" | "off" | undefined {
  const normalized = text.trim().toLowerCase();
  if (
    /^(\/?debug|\/?dev|д[эе]баг|дэв)\s+(on|ўкл|уключы|так|true)$/iu.test(
      normalized,
    )
  ) {
    return "on";
  }

  if (
    /^(\/?debug|\/?dev|д[эе]баг|дэв)\s+(off|выкл|выключы|не|false)$/iu.test(
      normalized,
    )
  ) {
    return "off";
  }

  return undefined;
}

function isDebugEnabled(conversationKey: string): boolean {
  return config.debug.rag || debugConversations.has(conversationKey);
}

function formatRagDebug(result: ChatAgentResponse): string {
  const plan = result.trace.searchPlan;
  const lines = [
    "**RAG debug**",
    `tool: ${result.trace.usedTool}`,
    `decision: ${result.trace.orchestratorDecision.action}`,
    `searchQuery: ${result.searchQuery || result.trace.orchestratorDecision.searchQuery || "-"}`,
  ];

  if (result.trace.standaloneQuestion) {
    lines.push(`rewritten: ${result.trace.standaloneQuestion}`);
  }

  if (plan) {
    lines.push(`plan.intent: ${plan.intent}`);
    lines.push(`plan.mode: ${plan.resultMode}`);
    lines.push(`plan.coreQuery: ${plan.coreQuery}`);
    lines.push(`plan.expandedQueries: ${plan.expandedQueries.join(" | ")}`);
    if (plan.semanticFacets?.length) {
      lines.push(`plan.semanticFacets: ${plan.semanticFacets.join(" | ")}`);
    }
  }

  if (result.trace.queryBreakdown?.length) {
    lines.push("", "queries:");
    for (const item of result.trace.queryBreakdown) {
      lines.push(
        `- ${item.query} -> retrieved ${item.retrievedCount}, kept ${item.keptCount}`,
      );
    }
  }

  if (result.trace.rerank) {
    lines.push(
      "",
      `rerank: input ${result.trace.rerank.inputCount}, output ${result.trace.rerank.outputCount}, ${result.trace.rerank.modelLatencyMs}ms`,
    );
  }

  if (result.trace.evaluationResult) {
    const evaluation = result.trace.evaluationResult;
    lines.push(
      `evaluation: sufficient=${evaluation.sufficientForAnswer}, quality=${evaluation.qualityScore}`,
      `evaluationReason: ${evaluation.evaluationReason}`,
    );
  }

  lines.push("", `sources returned to final answer: ${result.sources.length}`);
  for (const [index, source] of result.sources.slice(0, 8).entries()) {
    const score =
      typeof source.score === "number" ? source.score.toFixed(3) : "-";
    const relevanceScore =
      "relevanceScore" in source && typeof source.relevanceScore === "number"
        ? source.relevanceScore.toFixed(2)
        : "-";
    lines.push(
      `${index + 1}. ${source.fileName || "unknown"}:${source.page || "?"} score=${score} rel=${relevanceScore}`,
    );
    if (source.matchedQueries?.length) {
      lines.push(
        `   matched: ${source.matchedQueries.slice(0, 3).join(" | ")}`,
      );
    }
    lines.push(`   ${oneLine(source.text).slice(0, 280)}`);
  }

  return lines.join("\n");
}

function oneLine(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function splitDiscordMessage(text: string): string[] {
  const chunks: string[] = [];
  let current = "";

  for (const line of text.split("\n")) {
    if (`${current}\n${line}`.length > 1900) {
      chunks.push(current);
      current = line;
      continue;
    }

    current = current ? `${current}\n${line}` : line;
  }

  if (current) chunks.push(current);
  return chunks;
}
