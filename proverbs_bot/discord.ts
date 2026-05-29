import {
  Client,
  GatewayIntentBits,
  Partials,
  ChannelType,
  Status,
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
import {
  createExplanatoryDictionarySearchTool,
  createOrthographicDictionarySearchTool,
  createTranslationDictionarySearchTool,
} from "../agents/dictionarySearchTools";
import { ChatOrchestratorAgent } from "../agents/chatOrchestrator";
import { initPrompts } from "../agents/prompts";
import { config } from "../config";
import type { ChatAgentResponse, ChatMessage } from "../agents/schemas";

const DISCORD_TOKEN = process.env.DISCORD_TOKEN;

if (!DISCORD_TOKEN) {
  throw new Error("DISCORD_TOKEN environment variable is required.");
}

// Load prompts from Goman before anything else
await initPrompts();

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
const orthographicDictionarySearchTool = createOrthographicDictionarySearchTool(
  hybridRetriever,
);
const translationDictionarySearchTool = createTranslationDictionarySearchTool(
  hybridRetriever,
);
const explanatoryDictionarySearchTool = createExplanatoryDictionarySearchTool(
  hybridRetriever,
);
const orchestrator = new ChatOrchestratorAgent(
  folkWisdomTool,
  ragSearchTool,
  dialectDictionarySearchTool,
  orthographicDictionarySearchTool,
  translationDictionarySearchTool,
  explanatoryDictionarySearchTool,
);
const conversations = new Map<string, ChatMessage[]>();
const debugConversations = new Set<string>();
const MAX_CONVERSATION_MESSAGES = 12;
const DISCORD_WATCHDOG_INTERVAL_MS = 5 * 60_000;
const DISCORD_NOT_READY_RELOGIN_MS = 5 * 60_000;
const DISCORD_PROACTIVE_RELOGIN_MS = minutesFromEnv("DISCORD_PROACTIVE_RELOGIN_MINUTES", 60) * 60_000;
const DISCORD_LOGIN_TIMEOUT_MS = 60_000;
const DISCORD_GREETING_CHANNEL_ID = process.env.DISCORD_GREETING_CHANNEL_ID;
const HELP_MESSAGE = `Прывітанне! Я  слоўнікавы бот, які шукае прыказкі, прымаўкі, народныя мудрасці, праклёны, грозьбы, дыялектныя словы і выразы, пераклады, правапіс і тлумачэнні слоў у калекцыі PDF.
Можна пісаць звычайным тэкстам: шукаць прыказкі, перакладаць з расейскай на беларускую і назад, правяраць напісанне, тлумачыць значэнні слоў.

Каманды:
reset / рэзэт — пачаць размову нанова.`;

function minutesFromEnv(name: string, defaultValue: number): number {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value >= 0 ? value : defaultValue;
}

let client = createDiscordClient();

let lastDiscordReadyAt = Date.now();
let lastDiscordLoginAt = Date.now();
let lastDiscordReloginAt = 0;
let lastDiscordMessageAt = 0;
let discordReloginInFlight = false;
let startupGreetingSent = false;

function createDiscordClient(): Client {
  const instance = new Client({
    intents: [
      GatewayIntentBits.Guilds,
      GatewayIntentBits.GuildMessages,
      GatewayIntentBits.DirectMessages,
      GatewayIntentBits.MessageContent,
    ],
    partials: [Partials.Channel, Partials.Message],
  });
  registerDiscordHandlers(instance);
  return instance;
}

function registerDiscordHandlers(target: Client): void {
  target.on("clientReady", (readyClient) => {
    lastDiscordReadyAt = Date.now();
    lastDiscordLoginAt = Date.now();
    console.log(`Discord bot logged in as ${readyClient.user.tag}`);
    void sendStartupGreeting();
  });

  target.on("error", (error) => {
    console.error("Discord client error:", error);
  });

  target.on("warn", (warning) => {
    console.warn("Discord client warning:", warning);
  });

  target.on("shardDisconnect", (event, shardId) => {
    console.warn(
      `Discord shard ${shardId} disconnected: code=${event.code}, reason=${event.reason || "none"}, clean=${event.wasClean}`,
    );
  });

  target.on("shardReconnecting", (shardId) => {
    console.warn(`Discord shard ${shardId} reconnecting.`);
  });

  target.on("shardError", (error, shardId) => {
    console.error(`Discord shard ${shardId} error:`, error);
  });

  target.on("shardResume", (shardId, replayedEvents) => {
    console.log(`Discord shard ${shardId} resumed; replayed events: ${replayedEvents}.`);
  });

  target.on("invalidated", () => {
    console.error("Discord session invalidated; forcing relogin.");
    void reloginDiscord("session invalidated");
  });

  target.on("messageCreate", (message: Message) => {
    void handleMessageCreate(message).catch((error) => {
      console.error("Discord unhandled messageCreate error:", error);
    });
  });
}

async function handleMessageCreate(message: Message): Promise<void> {
  if (message.author.bot) return;

  lastDiscordMessageAt = Date.now();
  const isDM = message.channel.type === ChannelType.DM;
  const isMentioned = client.user ? message.mentions.has(client.user) : false;
  const isReplyToBot = await isReplyToCurrentBot(message);

  // Strip mention from message text
  const question = message.content.replace(/<@!?\d+>/g, "").trim();
  if (!question) {
    console.log(
      `Discord ignored empty message: channel=${message.channel.id}, author=${message.author.id}, dm=${isDM}`,
    );
    return;
  }

  const conversationKey = keyForConversation(message, isDM);
  if (isClearChatCommand(question)) {
    conversations.delete(conversationKey);
    debugConversations.delete(conversationKey);
    const clearResult = await clearChatMessages(message);
    await sendText(message.channel, clearResult);
    return;
  }

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

  if (!isDM && !isMentioned && !isReplyToBot) {
    console.log(
      `Discord ignored unmentioned guild message: channel=${message.channel.id}, author=${message.author.id}`,
    );
    return;
  }

  console.log(
    `[${new Date().toISOString()}] Discord handling message: channel=${message.channel.id}, author=${message.author.username}(${message.author.id}), dm=${isDM}, mention=${isMentioned}, replyToBot=${isReplyToBot}`,
  );

  const previousMessages = conversations.get(conversationKey) || [];
  if (isDM && previousMessages.length === 0) {
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
    const reply = result.answer;
    const chunks = splitDiscordMessage(reply);
    let firstChunkSent = false;
    for (const chunk of chunks) {
      if (!firstChunkSent) {
        await message.reply(chunk);
        firstChunkSent = true;
      } else {
        await sendText(message.channel, chunk);
      }
    }
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
}

void loginDiscord();
startDiscordWatchdog();

async function sendTyping(channel: Message["channel"]): Promise<void> {
  if ("sendTyping" in channel && typeof channel.sendTyping === "function") {
    try {
      await channel.sendTyping();
    } catch (error) {
      console.warn("Discord sendTyping failed:", error);
    }
  }
}

async function sendText(channel: Message["channel"], text: string): Promise<void> {
  if ("send" in channel && typeof channel.send === "function") {
    try {
      await channel.send(text);
    } catch (error) {
      console.warn("Discord send failed:", error);
    }
  }
}

async function sendStartupGreeting(): Promise<void> {
  if (startupGreetingSent || !DISCORD_GREETING_CHANNEL_ID) return;

  try {
    const channel = await client.channels.fetch(DISCORD_GREETING_CHANNEL_ID);
    if (channel && "send" in channel && typeof channel.send === "function") {
      await channel.send(HELP_MESSAGE);
      startupGreetingSent = true;
    }
  } catch (error) {
    console.error("Discord startup greeting failed:", error);
  }
}

function keyForConversation(message: Message, isDM: boolean): string {
  if (isDM) return `dm:${message.author.id}`;
  return `guild:${message.guildId || "unknown"}:${message.channel.id}:${message.author.id}`;
}

async function isReplyToCurrentBot(message: Message): Promise<boolean> {
  const botId = client.user?.id;
  const referenceMessageId = message.reference?.messageId;
  if (!botId || !referenceMessageId) return false;

  try {
    const referenced = await message.fetchReference();
    return referenced.author.id === botId;
  } catch (error) {
    console.warn(
      `Discord could not fetch referenced message ${referenceMessageId}:`,
      error,
    );
    return false;
  }
}

function isResetCommand(text: string): boolean {
  return /^(\/?reset|\/?restart|\/?new|рэз[эе]т|скінь|новая размова|пачаць нанова)$/iu.test(
    text.trim(),
  );
}

function isClearChatCommand(text: string): boolean {
  return /^(\/clear|\/ачысці)$/iu.test(text.trim());
}

async function clearChatMessages(message: Message): Promise<string> {
  const channel = message.channel;
  const messages = "messages" in channel ? channel.messages : undefined;
  if (!messages || typeof messages.fetch !== "function") {
    return "Не магу атрымаць паведамленні ў гэтым тыпе чату. Памяць размовы ачышчана.";
  }

  try {
    if (channel.type === ChannelType.DM) {
      const fetched = await messages.fetch({ limit: 100 });
      const botId = message.client.user?.id;
      const deletable = [...fetched.values()].filter(
        (item) => item.author.id === botId,
      );
      const deleted = await deleteOneByOne(deletable);

      return [
        "Памяць размовы ачышчана.",
        `У DM Discord не дазваляе боту выдаляць паведамленні карыстальніка; выдаліў свае паведамленні: ${deleted}.`,
      ].join("\n");
    }

    if ("bulkDelete" in channel && typeof channel.bulkDelete === "function") {
      const deleted = await bulkDeleteRecentMessages(channel);
      return [
        "Памяць размовы ачышчана.",
        `Выдаліў паведамленні ў гэтым чaце: ${deleted}.`,
        "Калі нешта засталося, гэта могуць быць паведамленні старэйшыя за 14 дзён або бракуе permission Manage Messages.",
      ].join("\n");
    }

    const fetched = await messages.fetch({ limit: 100 });
    const deleted = await deleteOneByOne([...fetched.values()]);
    return `Памяць размовы ачышчана. Выдаліў паведамленні, якія бот мае права выдаліць: ${deleted}.`;
  } catch (error) {
    console.error("Discord clear error:", error);
    return "Памяць размовы ачышчана, але паведамленні выдаліць не атрымалася. Правер permission Manage Messages.";
  }
}

async function bulkDeleteRecentMessages(
  channel: Message["channel"] & {
    bulkDelete(messages: readonly Message[], filterOld?: boolean): Promise<unknown>;
  },
): Promise<number> {
  let totalDeleted = 0;

  for (let batch = 0; batch < 10; batch += 1) {
    const fetched = await channel.messages.fetch({ limit: 100 });
    const items = [...fetched.values()];
    if (items.length === 0) break;

    const deleted = await channel.bulkDelete(items, true);
    const deletedCount = collectionSize(deleted);
    totalDeleted += deletedCount;
    if (deletedCount === 0 || items.length < 100) break;
  }

  return totalDeleted;
}

async function deleteOneByOne(messages: Message[]): Promise<number> {
  const results = await Promise.allSettled(messages.map((item) => item.delete()));
  return results.filter((result) => result.status === "fulfilled").length;
}

function collectionSize(value: unknown): number {
  if (value && typeof value === "object" && "size" in value) {
    const size = (value as { size: unknown }).size;
    return typeof size === "number" ? size : 0;
  }

  return 0;
}

async function loginDiscord(): Promise<void> {
  try {
    await loginWithTimeout(client);
  } catch (error) {
    console.error("Discord login failed:", error);
    setTimeout(() => void loginDiscord(), 30_000);
  }
}

async function loginWithTimeout(target: Client): Promise<void> {
  await Promise.race([
    target.login(DISCORD_TOKEN),
    new Promise<never>((_, reject) =>
      setTimeout(
        () =>
          reject(
            new Error(`Discord login timed out after ${DISCORD_LOGIN_TIMEOUT_MS}ms`),
          ),
        DISCORD_LOGIN_TIMEOUT_MS,
      ),
    ),
  ]);
}

function startDiscordWatchdog(): void {
  setInterval(() => {
    const status = client.ws.status;
    const ready = client.isReady();
    const statusName = Status[status] || String(status);

    if (ready) {
      lastDiscordReadyAt = Date.now();
      const lastMessageAge =
        lastDiscordMessageAt > 0 ? Date.now() - lastDiscordMessageAt : undefined;
      console.log(
        `Discord watchdog: ready, status=${statusName}, ping=${client.ws.ping}ms, lastMessageAgeMs=${lastMessageAge ?? "none"}`,
      );
      if (
        DISCORD_PROACTIVE_RELOGIN_MS > 0 &&
        Date.now() - lastDiscordLoginAt >= DISCORD_PROACTIVE_RELOGIN_MS
      ) {
        void reloginDiscord("scheduled proactive reconnect");
      }
      return;
    }

    const notReadyForMs = Date.now() - lastDiscordReadyAt;
    console.warn(
      `Discord watchdog: not ready, status=${statusName}, ping=${client.ws.ping}ms, notReadyForMs=${notReadyForMs}`,
    );

    if (notReadyForMs >= DISCORD_NOT_READY_RELOGIN_MS) {
      void reloginDiscord(`watchdog status=${statusName}`);
    }
  }, DISCORD_WATCHDOG_INTERVAL_MS);
}

async function reloginDiscord(reason: string): Promise<void> {
  if (discordReloginInFlight) return;
  const now = Date.now();
  if (now - lastDiscordReloginAt < DISCORD_NOT_READY_RELOGIN_MS) return;

  discordReloginInFlight = true;
  lastDiscordReloginAt = now;

  const previousClient = client;
  try {
    console.warn(`Discord relogin started: ${reason}`);
    previousClient.removeAllListeners();
    try {
      await previousClient.destroy();
    } catch (destroyError) {
      console.warn("Discord previous client destroy failed:", destroyError);
    }

    const nextClient = createDiscordClient();
    client = nextClient;
    await loginWithTimeout(nextClient);
    lastDiscordReadyAt = Date.now();
    lastDiscordLoginAt = Date.now();
    console.warn("Discord relogin completed.");
  } catch (error) {
    console.error("Discord relogin failed:", error);
    // Allow another relogin attempt sooner than the 5-minute debounce.
    lastDiscordReloginAt = 0;
  } finally {
    discordReloginInFlight = false;
  }
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

const DISCORD_MAX_MESSAGE_LENGTH = 2000;
const DISCORD_CHUNK_TARGET = 1900;

function splitDiscordMessage(text: string): string[] {
  if (!text) return [];
  if (text.length <= DISCORD_MAX_MESSAGE_LENGTH) return [text];

  const chunks: string[] = [];
  let remaining = text;

  while (remaining.length > DISCORD_MAX_MESSAGE_LENGTH) {
    const window = remaining.slice(0, DISCORD_MAX_MESSAGE_LENGTH);
    let cut = findSplitIndex(window);
    if (cut <= 0) cut = DISCORD_MAX_MESSAGE_LENGTH;

    chunks.push(remaining.slice(0, cut).replace(/\s+$/u, ""));
    remaining = remaining.slice(cut).replace(/^\s+/u, "");
  }

  if (remaining.length > 0) chunks.push(remaining);
  return chunks;
}

function findSplitIndex(window: string): number {
  // Prefer paragraph break, then newline, then sentence end, then space.
  const candidates = [
    window.lastIndexOf("\n\n", DISCORD_MAX_MESSAGE_LENGTH),
    window.lastIndexOf("\n", DISCORD_MAX_MESSAGE_LENGTH),
    lastSentenceBoundary(window),
    window.lastIndexOf(" ", DISCORD_MAX_MESSAGE_LENGTH),
  ];

  for (const index of candidates) {
    if (index >= DISCORD_CHUNK_TARGET / 2) {
      return index + (window[index] === "\n" || window[index] === " " ? 1 : 1);
    }
  }
  return -1;
}

function lastSentenceBoundary(window: string): number {
  const match = window.match(/[.!?…][\s")»\]]*(?=\s|$)/gu);
  if (!match) return -1;
  const last = match[match.length - 1];
  return window.lastIndexOf(last) + last.length;
}
