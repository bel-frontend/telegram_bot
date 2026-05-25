import {
  Client,
  GatewayIntentBits,
  Partials,
  ChannelType,
  type Message,
} from 'discord.js';
import { createQdrantClient } from '../qdrant/index';
import { createEmbeddings } from '../embeddings';
import { QdrantRetriever } from '../rag/retriever';
import { LexicalRetriever } from '../rag/lexicalRetriever';
import { HybridRetriever } from '../rag/hybridRetriever';
import { FolkWisdomSearchTool } from '../agents/folkWisdomSearchTool';
import { ChatOrchestratorAgent } from '../agents/chatOrchestrator';
import { config } from '../config';
import type { ChatMessage } from '../agents/schemas';

const DISCORD_TOKEN = process.env.DISCORD_TOKEN;

if (!DISCORD_TOKEN) {
  throw new Error('DISCORD_TOKEN environment variable is required.');
}

// Initialise the RAG stack once at startup
const qdrant = createQdrantClient(config);
const embeddings = createEmbeddings();
const vectorRetriever = new QdrantRetriever(qdrant, embeddings);
const lexicalRetriever = new LexicalRetriever(qdrant);
const hybridRetriever = new HybridRetriever(vectorRetriever, lexicalRetriever);
const folkWisdomTool = new FolkWisdomSearchTool(hybridRetriever);
const orchestrator = new ChatOrchestratorAgent(folkWisdomTool);

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.DirectMessages,
    GatewayIntentBits.MessageContent,
  ],
  partials: [Partials.Channel, Partials.Message],
});

client.once('clientReady', (readyClient) => {
  console.log(`Discord bot logged in as ${readyClient.user.tag}`);
});

client.on('messageCreate', async (message: Message) => {
  if (message.author.bot) return;

  const isDM = message.channel.type === ChannelType.DM;
  const isMentioned = client.user ? message.mentions.has(client.user) : false;

  if (!isDM && !isMentioned) return;

  // Strip mention from message text
  const question = message.content.replace(/<@!?\d+>/g, '').trim();
  if (!question) return;

  // Show typing indicator and refresh it every 8s while waiting
  await message.channel.sendTyping();
  const typingInterval = setInterval(() => message.channel.sendTyping(), 8000);

  try {
    const messages: ChatMessage[] = [{ role: 'user', content: question }];
    const result = await orchestrator.chat(messages);
    const reply = result.answer.slice(0, 2000);
    await message.reply(reply);
  } catch (error) {
    console.error('Discord message error:', error);
    await message.reply('Нешта пайшло не так. Паспрабуй пазней.');
  } finally {
    clearInterval(typingInterval);
  }
});

client.login(DISCORD_TOKEN);
