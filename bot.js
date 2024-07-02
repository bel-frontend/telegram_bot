const TelegramBot = require("node-telegram-bot-api");

const token = process.env.TELEGRAM_BOT_TOKEN;
const bot = new TelegramBot(token, { polling: true });

const TARGET_CHAT_ID = process.env.TARGET_CHAT_ID; // Add your target chat ID in the .env file

bot.on("message", (msg) => {
  const chatId = msg.chat.id;
  const text = msg.text;
  console.log(text, TARGET_CHAT_ID, chatId.toString());

  if (chatId.toString() === TARGET_CHAT_ID) {
    // Analyze messages and decide to delete
    if (text && shouldDeleteMessage(text)) {
      console.log("delete");
      bot.deleteMessage(chatId, msg.message_id).catch((error) => {
        console.error("Failed to delete message:", error);
      });
    }
  }
});

function shouldDeleteMessage(text) {
  // Simple example of a condition to delete a message
  // You can implement any analysis logic here
  const forbiddenWords = [
    "test",
    "крипто",
    "криптовалюта",
    "подписывайтесь на канал",
    "и", // disabled russian text
  ];
  return forbiddenWords.some((word) =>
    text.trim().toLowerCase().includes(word)
  );
}
