# Discord Bot Setup Guide

This guide explains how to set up the Discord proverbs bot.

## Prerequisites

- A Discord account with a server where you have admin permissions
- The project deployed with `.env` configured

## Step 1: Create a Discord Application

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Click **New Application** and give it a name (e.g. `Proverbs Bot`)
3. Go to **General Information** — copy the **Application ID** (this is your `DISCORD_CLIENT_ID`)

## Step 2: Create a Bot

1. In your application, go to **Bot** in the left sidebar
2. Click **Add Bot** → **Yes, do it!**
3. Under **Token**, click **Reset Token** and copy the token (this is your `DISCORD_TOKEN`)
4. Enable these **Privileged Gateway Intents**:
   - **Message Content Intent** — required to read messages
   - **Server Members Intent** — optional but recommended

## Step 3: Set Environment Variables

Add these variables to your `.env` file:

```env
DISCORD_TOKEN=your_bot_token_here
DISCORD_CLIENT_ID=your_application_id_here
DISCORD_CHANEL_ID=1508465638329553167

# RAG stack (same as lesson11)
OPENAI_API_KEY=your_openai_key
QDRANT_URL=http://your-qdrant-host:6333
QDRANT_COLLECTION=lesson11_pdf_documents
```

## Step 4: Invite the Bot to Your Server

1. In the Developer Portal, go to **OAuth2 → URL Generator**
2. Under **Scopes**, select: `bot`, `applications.commands`
3. Under **Bot Permissions**, select:
   - `Read Messages/View Channels`
   - `Send Messages`
   - `Read Message History`
4. Copy the generated URL, open it in a browser, and invite the bot to your server

## Step 5: Register Slash Commands

Run once to register the `/proverb` command globally:

```bash
bun run register-discord
```

Global commands may take up to 1 hour to propagate. For instant testing on a specific guild, you can modify `register-commands.ts` to use `Routes.applicationGuildCommands(clientId, guildId)`.

## Step 6: Deploy

```bash
./deploy.sh
```

## Usage

### Slash command (any channel)
```
/proverb               — returns a few random proverbs
/proverb topic:праца   — searches proverbs about work
/proverb topic:сям'я   — searches proverbs about family
```

### Text messages
The bot responds to:
- **Direct Messages** — any message sent directly to the bot
- **Channel `1508465638329553167`** — any message in that specific channel

## Architecture

The Discord bot uses the full RAG stack from lesson11:

```
User message
    ↓
QuestionRewriterAgent  (standalone question)
    ↓
ChatOrchestratorAgent  (decide: answer_directly / search_folk_wisdom)
    ↓
QueryPlannerAgent      (build multi-query search plan)
    ↓
FolkWisdomSearchTool
    ├─ QdrantRetriever  (vector similarity, text-embedding-3-large)
    └─ LexicalRetriever (BM25 keyword scroll)
        ↓ HybridRetriever (RRF fusion)
    ↓
LlmReranker            (score each chunk against the question)
    ↓
RagResultEvaluator     (check if results are sufficient, retry if not)
    ↓
Final answer with LIST_FINAL_SYSTEM_PROMPT → Discord reply
```

All models and settings are identical to lesson11: `gpt-5.4` with `useResponsesApi`, `text-embedding-3-large` (3072 dims), Qdrant collection `lesson11_pdf_documents`.
