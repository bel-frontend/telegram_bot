const TelegramBot = require("node-telegram-bot-api");
const json = require("./index.json");

process.on("unhandledRejection", (reason) => {
  console.error("Unhandled promise rejection:", reason);
});

process.on("uncaughtException", (error) => {
  console.error("Uncaught exception:", error);
  process.exit(1);
});

const token = process.env.TELEGRAM_BOT_TOKEN;
if (!token) {
  throw new Error("TELEGRAM_BOT_TOKEN environment variable is required.");
}

const TARGET_CHAT_IDS = (process.env.TARGET_CHAT_IDS || "")
  .split(",")
  .map((chatId) => chatId.trim())
  .filter(Boolean); // Add your target chat IDs in the .env file, separated by commas
const CHAT_USERNAMES_TO_RESOLVE = (process.env.CHAT_USERNAMES_TO_RESOLVE || "")
  .split(",")
  .map((username) => username.trim())
  .filter(Boolean);
const TARGET_NOTIFICATION_ID = process.env.TARGET_NOTIFICATION_ID; // Add your target notification ID in the .env file

if (!TARGET_CHAT_IDS.length) {
  throw new Error("TARGET_CHAT_IDS environment variable is required.");
}

if (!TARGET_NOTIFICATION_ID) {
  throw new Error("TARGET_NOTIFICATION_ID environment variable is required.");
}

const SITES_TO_CHECK = [
"https://bel-geek.com",
// "https://spa-consolidation-reports-v3-demo-stage.dev.quext.io",
 "https://test.goman.live"];
 // Add your site URLs here

const CHECK_INTERVAL = 5 * 60 * 1000; // 5 minutes in milliseconds
const POLLING_RESTART_AFTER_ERRORS = numberFromEnv(
  "TELEGRAM_POLLING_RESTART_AFTER_ERRORS",
  3
);
const POLLING_RESTART_COOLDOWN_MS =
  numberFromEnv("TELEGRAM_POLLING_RESTART_COOLDOWN_SECONDS", 30) * 1000;
const AVATAR_NUDITY_GUARD_ENABLED = booleanFromEnv(
  "AVATAR_NUDITY_GUARD_ENABLED",
  true
);
const AVATAR_NUDITY_THRESHOLD = numberFromEnv("AVATAR_NUDITY_THRESHOLD", 0.85);
const AVATAR_NUDITY_MODEL =
  process.env.AVATAR_NUDITY_MODEL || "omni-moderation-latest";
const TRACKED_USER_MESSAGES_LIMIT = numberFromEnv(
  "TRACKED_USER_MESSAGES_LIMIT",
  100
);
const TEST_UNBAN_COMMAND = process.env.TEST_UNBAN_COMMAND || "/test_unban_bun";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

const bot = new TelegramBot(token, {
  polling: {
    interval: 300,
    autoStart: true,
    params: {
      timeout: 30,
      allowed_updates: ["message", "channel_post", "chat_member", "my_chat_member"],
    },
  },
});

console.log(`Bot started. Watching chats: ${TARGET_CHAT_IDS.join(", ")}`);
if (CHAT_USERNAMES_TO_RESOLVE.length) {
  console.log(`Resolving chat usernames: ${CHAT_USERNAMES_TO_RESOLVE.join(", ")}`);
}
console.log(
  `Avatar guard is ${AVATAR_NUDITY_GUARD_ENABLED ? "enabled" : "disabled"}; threshold=${AVATAR_NUDITY_THRESHOLD}; OpenAI key=${OPENAI_API_KEY ? "set" : "missing"}`
);

let pollingErrorCount = 0;
let lastPollingRestartAt = 0;
let pollingRestartInFlight = false;
const trackedUserMessages = new Map();

bot.on("polling_error", (error) => {
  console.error(`Polling error: ${error.code || error.message}`);

  pollingErrorCount += 1;
  if (pollingErrorCount >= POLLING_RESTART_AFTER_ERRORS) {
    void restartTelegramPolling(`after ${pollingErrorCount} polling errors`);
  }
});

bot.on("message", (msg) => {
  pollingErrorCount = 0;
  void handleTelegramMessage(msg).catch((error) => {
    console.error("Telegram message handler failed:", error);
  });
});

bot.on("channel_post", (msg) => {
  pollingErrorCount = 0;
  console.log(`Channel post received in ${describeChat(msg.chat)}.`);
});

bot.on("chat_member", (update) => {
  pollingErrorCount = 0;
  void handleChatMemberUpdate(update).catch((error) => {
    console.error("Telegram chat_member handler failed:", error);
  });
});

bot.on("my_chat_member", (update) => {
  pollingErrorCount = 0;
  console.log(
    `Bot membership changed in ${describeChat(update.chat)}: ${update.old_chat_member?.status} -> ${update.new_chat_member?.status}.`
  );
});

async function handleTelegramMessage(msg) {
  const chatId = msg.chat.id;

  const targetMatched = TARGET_CHAT_IDS.includes(chatId.toString());
  console.log(
    `Message received in ${describeChat(msg.chat)}${targetMatched ? "" : " (not watched)"}.`
  );

  if (targetMatched) {
    trackUserMessage(msg);

    if (isTestUnbanCommand(msg)) {
      await handleTestUnbanCommand(msg);
      return;
    }

    const avatarPolicyHandled = await enforceAvatarContentPolicyForMessage(msg);
    if (avatarPolicyHandled) return;

    const text = msg.text;
    const caption = msg.caption;

    // Analyze messages and decide to delete
    if (shouldDeleteMessage([text || "", caption || ""])) {
      try {
        await bot.deleteMessage(chatId, msg.message_id);
      } catch (error) {
        console.error(`Failed to delete forbidden message: ${error.message}`);
        return;
      }

      const randomMessage = json[Math.floor(Math.random() * json.length)];

      if (!randomMessage || !randomMessage.message) {
        console.error("Failed to pick replacement proverb.");
        await safeSendMessage(
          TARGET_NOTIFICATION_ID,
          "Error: Could not retrieve a random message from the JSON file."
        );
        return;
      }

      await safeSendMessage(
        TARGET_NOTIFICATION_ID,
        `Message deleted: ${text || caption} `
      );

      const sentReplacement = await safeSendMessage(
        chatId,
        `Тут было паведамленне якое не суадносіцца з нашай суполкай. Мы яго выдалілі і замест гэтага  трымайце беларускую прыказку ці прымаўку:

          "${randomMessage.message}"
                       `
      );

      if (!sentReplacement) {
        await safeSendMessage(
          TARGET_NOTIFICATION_ID,
          `Failed to send replacement message to chat ${chatId}`
        );
      }
    }
  }
}

async function handleChatMemberUpdate(update) {
  const chatId = update.chat?.id;
  const user = update.new_chat_member?.user;

  console.log(
    `Chat member update in ${describeChat(update.chat)}: user ${user?.id || "unknown"} ${update.old_chat_member?.status} -> ${update.new_chat_member?.status}.`
  );

  if (!chatId || !user) return;
  if (!TARGET_CHAT_IDS.includes(chatId.toString())) return;
  if (!isNewMemberStatusChange(update)) return;

  console.log(`New member detected: ${formatUserForLog(user)} in chat ${chatId}.`);

  if (!AVATAR_NUDITY_GUARD_ENABLED) return;
  if (!OPENAI_API_KEY) {
    console.warn("Skipping avatar check: OpenAI key is missing.");
    return;
  }
  if (user.is_bot) return;

  const moderation = await checkUserAvatarContent(user, {
    chatId,
    source: "chat_member",
  });
  if (!moderation) return;

  if (!shouldBanForAvatarNudity(moderation)) return;

  await banUserFromTargetChats(user, moderation, "channel subscriber");
}

async function enforceAvatarContentPolicyForMessage(msg) {
  if (!AVATAR_NUDITY_GUARD_ENABLED) return false;
  if (!OPENAI_API_KEY) {
    console.warn("Skipping commenter avatar check: OpenAI key is missing.");
    return false;
  }

  const chatId = msg.chat?.id;
  const user = msg.from;
  if (!chatId || !user || user.is_bot) return false;

  let moderation;
  try {
    moderation = await checkUserAvatarContent(user, {
      chatId,
      source: "message",
    });
  } catch (error) {
    console.error(`Failed to check commenter avatar: ${error.message}`);
    return false;
  }
  if (!moderation || !shouldBanForAvatarNudity(moderation)) return false;

  try {
    await bot.deleteMessage(chatId, msg.message_id);
  } catch (error) {
    console.error(`Failed to delete bad-avatar comment: ${error.message}`);
  }

  await banUserFromTargetChats(user, moderation, `commenter in chat ${chatId}`);

  return true;
}

async function handleTestUnbanCommand(msg) {
  const userId = getCommandArg(msg);
  if (!userId) {
    await safeSendMessage(
      TARGET_NOTIFICATION_ID,
      `Usage: ${TEST_UNBAN_COMMAND} <telegram_user_id>`
    );
    return;
  }

  const results = [];
  for (const targetChatId of TARGET_CHAT_IDS) {
    try {
      await bot.unbanChatMember(targetChatId, userId, { only_if_banned: true });
      console.log(`Unbanned user ${userId} in ${targetChatId}.`);
      results.push(`${targetChatId}: unbanned`);
    } catch (error) {
      console.error(`Failed to unban user ${userId} in ${targetChatId}: ${error.message}`);
      results.push(`${targetChatId}: failed`);
    }
  }

  await safeSendMessage(
    TARGET_NOTIFICATION_ID,
    `Test unban command ${TEST_UNBAN_COMMAND}: user ${userId}. Results: ${results.join(", ")}`
  );
}

function isTestUnbanCommand(msg) {
  const text = msg.text || msg.caption || "";
  return text.trim().split(/\s+/)[0] === TEST_UNBAN_COMMAND;
}

function getCommandArg(msg) {
  const text = msg.text || msg.caption || "";
  return text.trim().split(/\s+/)[1];
}

const sendMessage = (chatId, text) => {
  return bot.sendMessage(chatId, text);
};

async function safeSendMessage(chatId, text) {
  try {
    return await sendMessage(chatId, text);
  } catch (error) {
    console.error(`Failed to send message to ${chatId}: ${error.message}`);
    return undefined;
  }
}

async function restartTelegramPolling(reason) {
  const now = Date.now();
  if (pollingRestartInFlight) return;
  if (now - lastPollingRestartAt < POLLING_RESTART_COOLDOWN_MS) return;

  pollingRestartInFlight = true;
  lastPollingRestartAt = now;

  console.warn(`Restarting Telegram polling ${reason}.`);
  try {
    await bot.stopPolling({ cancel: true, reason });
    await bot.startPolling();
    pollingErrorCount = 0;
    console.log("Telegram polling restarted.");
  } catch (error) {
    console.error(`Failed to restart Telegram polling: ${error.message}`);
  } finally {
    pollingRestartInFlight = false;
  }
}

function numberFromEnv(name, defaultValue) {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value >= 0 ? value : defaultValue;
}

function booleanFromEnv(name, defaultValue) {
  const value = process.env[name];
  if (!value) return defaultValue;
  return ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
}

function isNewMemberStatusChange(update) {
  const oldStatus = update.old_chat_member?.status;
  const newStatus = update.new_chat_member?.status;

  return (
    ["left", "kicked"].includes(oldStatus) &&
    ["member", "administrator", "creator"].includes(newStatus)
  );
}

async function getLatestProfilePhoto(userId) {
  const profilePhotos = await bot.getUserProfilePhotos(userId, {
    offset: 0,
    limit: 1,
  });
  const latestPhoto = profilePhotos.photos?.[0];
  if (!latestPhoto?.length) return undefined;

  const largestPhoto = latestPhoto.reduce((largest, photo) => {
    if (!largest) return photo;
    const photoSize =
      photo.file_size || photo.width * photo.height || photo.width || 0;
    const largestSize =
      largest.file_size || largest.width * largest.height || largest.width || 0;
    return photoSize > largestSize ? photo : largest;
  }, undefined);

  if (!largestPhoto) return undefined;

  return {
    fileUniqueId: largestPhoto.file_unique_id || largestPhoto.file_id,
    url: await bot.getFileLink(largestPhoto.file_id),
  };
}

async function checkUserAvatarContent(user, context) {
  const avatar = await getLatestProfilePhoto(user.id);
  if (!avatar) {
    console.log(`Skipping avatar check for ${formatUserForLog(user)}: no profile photo.`);
    return undefined;
  }

  console.log(`Checking avatar for ${formatUserForLog(user)} (${context.source}).`);
  const moderation = await moderateAvatarUrl(avatar.url);

  console.log(
    `Avatar check result for ${formatUserForLog(user)}: sexual=${moderation.sexualScore.toFixed(
      3
    )}, sexual/minors=${moderation.sexualMinorsScore.toFixed(3)}, flagged=${moderation.flagged}.`
  );

  return moderation;
}

async function banUserFromTargetChats(user, moderation, reason) {
  const results = [];

  await deleteTrackedUserMessages(user.id);

  for (const targetChatId of TARGET_CHAT_IDS) {
    try {
      await bot.banChatMember(targetChatId, user.id, { revoke_messages: true });
      console.log(`Banned ${formatUserForLog(user)} in ${targetChatId}.`);
      results.push(`${targetChatId}: banned`);
    } catch (error) {
      results.push(`${targetChatId}: failed`);
      console.error(
        `Failed to ban ${formatUserForLog(user)} in ${targetChatId}: ${error.message}`
      );
    }
  }

  await safeSendMessage(
    TARGET_NOTIFICATION_ID,
    `Banned ${reason} ${formatUserForLog(user)} from target chats: ${results.join(
      ", "
    )}. explicit nudity detected in profile photo. sexual=${moderation.sexualScore.toFixed(
      3
    )}, sexual/minors=${moderation.sexualMinorsScore.toFixed(3)}`
  );
}

function trackUserMessage(msg) {
  const userId = msg.from?.id;
  const chatId = msg.chat?.id;
  const messageId = msg.message_id;
  if (!userId || !chatId || !messageId) return;

  const key = String(userId);
  const messages = trackedUserMessages.get(key) || [];
  messages.push({ chatId, messageId });

  if (messages.length > TRACKED_USER_MESSAGES_LIMIT) {
    messages.splice(0, messages.length - TRACKED_USER_MESSAGES_LIMIT);
  }

  trackedUserMessages.set(key, messages);
}

async function deleteTrackedUserMessages(userId) {
  const key = String(userId);
  const messages = trackedUserMessages.get(key) || [];
  if (!messages.length) return;

  for (const message of messages) {
    try {
      await bot.deleteMessage(message.chatId, message.messageId);
    } catch (error) {
      console.error(
        `Failed to delete tracked message ${message.messageId} in ${message.chatId}: ${error.message}`
      );
    }
  }

  trackedUserMessages.delete(key);
}

async function moderateAvatarUrl(imageUrl) {
  const response = await fetch("https://api.openai.com/v1/moderations", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: AVATAR_NUDITY_MODEL,
      input: [
        {
          type: "image_url",
          image_url: {
            url: imageUrl,
          },
        },
      ],
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`OpenAI moderation failed: ${response.status} ${body}`);
  }

  const result = await response.json();
  const moderation = result.results?.[0] || {};
  const categoryScores = moderation.category_scores || {};

  return {
    flagged: Boolean(moderation.flagged),
    sexualScore: Number(categoryScores.sexual || 0),
    sexualMinorsScore: Number(categoryScores["sexual/minors"] || 0),
    raw: moderation,
  };
}

function shouldBanForAvatarNudity(moderation) {
  return (
    moderation.sexualScore >= AVATAR_NUDITY_THRESHOLD ||
    moderation.sexualMinorsScore >= AVATAR_NUDITY_THRESHOLD
  );
}

function formatUserForLog(user) {
  const username = user.username ? ` @${user.username}` : "";
  const name = [user.first_name, user.last_name].filter(Boolean).join(" ");
  return `${user.id}${username}${name ? ` (${name})` : ""}`;
}

function describeChat(chat) {
  if (!chat) return "unknown chat";
  const title = chat.title || chat.username || chat.first_name || "untitled";
  return `${chat.id} ${chat.type || "chat"} "${title}"`;
}

function shouldDeleteMessage(messagesArray) {
  // Simple example of a condition to delete a message
  // You can implement any analysis logic here
  const forbiddenWords = [
    "крипто",
    "криптовалюта",
    "подписывайтесь на канал",
    "и", // disabled russian text
  ];
  return forbiddenWords.some((word) => {
    return messagesArray.find((text) =>
      text.trim().toLowerCase().includes(word)
    );
  });
}

// Function to check the status of the sites
async function checkSitesStatus() {
  for (const site of SITES_TO_CHECK) {
    try {
      const response = await fetch(site);
      if (!response.ok) {
        // If the response status is not OK (e.g., 404, 500), send a notification
        await safeSendMessage(
          TARGET_NOTIFICATION_ID,
          `Site check failed: ${site} returned status ${response.status}`
        );
      }
    } catch (error) {
      // If there's an error (e.g., network issue), send a notification
      await safeSendMessage(
        TARGET_NOTIFICATION_ID,
        `Site check failed: ${site} is offline or unreachable. Error: ${error.message}`
      );
    }
  }
}

// Start the site status check immediately when the bot starts
checkSitesStatus();
logTargetChats();

// Set up periodic checks every 5-10 minutes
setInterval(checkSitesStatus, CHECK_INTERVAL);

async function logTargetChats() {
  for (const chatId of [...TARGET_CHAT_IDS, ...CHAT_USERNAMES_TO_RESOLVE]) {
    try {
      const chat = await bot.getChat(chatId);
      console.log(`Resolved target chat: ${describeChat(chat)}.`);
    } catch (error) {
      console.error(`Failed to resolve target chat ${chatId}: ${error.message}`);
    }
  }
}
