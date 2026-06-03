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

const bot = new TelegramBot(token, {
  polling: {
    interval: 300,
    autoStart: true,
    params: {
      timeout: 30,
    },
  },
});

let pollingErrorCount = 0;
let lastPollingRestartAt = 0;
let pollingRestartInFlight = false;

bot.on("polling_error", (error) => {
  console.error("Polling error:", error.code, error?.response?.body);

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

async function handleTelegramMessage(msg) {
  const chatId = msg.chat.id;

  if (TARGET_CHAT_IDS.includes(chatId.toString())) {
    console.log(chatId.toString());

    const text = msg.text;
    const caption = msg.caption;

    // Analyze messages and decide to delete
    if (shouldDeleteMessage([text || "", caption || ""])) {
      try {
        await bot.deleteMessage(chatId, msg.message_id);
      } catch (error) {
        console.error("Failed to delete message:", error);
        return;
      }

      const randomMessage = json[Math.floor(Math.random() * json.length)];

      if (!randomMessage || !randomMessage.message) {
        console.error(
          "randomMessage is undefined or missing the 'message' property."
        );
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

      await safeSendMessage(
        chatId,
        `Тут было паведамленне якое не суадносіцца з нашай суполкай. Мы яго выдалілі і замест гэтага  трымайце беларускую прыказку ці прымаўку:

          "${randomMessage.message}"
                       `
      );
    }
  }
}

const sendMessage = (chatId, text) => {
  return bot.sendMessage(chatId, text);
};

async function safeSendMessage(chatId, text) {
  try {
    return await sendMessage(chatId, text);
  } catch (error) {
    console.error("Failed to send message:", error);
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
    console.error("Failed to restart Telegram polling:", error);
  } finally {
    pollingRestartInFlight = false;
  }
}

function numberFromEnv(name, defaultValue) {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value >= 0 ? value : defaultValue;
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

// Set up periodic checks every 5-10 minutes
setInterval(checkSitesStatus, CHECK_INTERVAL);
