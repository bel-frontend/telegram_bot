const TelegramBot = require("node-telegram-bot-api");
const json = require("./index.json");

const token = process.env.TELEGRAM_BOT_TOKEN;
const bot = new TelegramBot(token, { polling: true });

const TARGET_CHAT_IDS = process.env.TARGET_CHAT_IDS.split(","); // Add your target chat IDs in the .env file, separated by commas
const TARGET_NOTIFICATION_ID = process.env.TARGET_NOTIFICATION_ID; // Add your target notification ID in the .env file

bot.on("polling_error", (error) => {
  console.log(error);
  console.error("Polling error:", error.code, error?.response?.body);
});

bot.on("message", (msg) => {
  const chatId = msg.chat.id;

  if (TARGET_CHAT_IDS.includes(chatId.toString())) {
    const text = msg.text;
    const caption = msg.caption;

    // Analyze messages and decide to delete
    if (shouldDeleteMessage([text || "", caption || ""])) {
      bot
        .deleteMessage(chatId, msg.message_id)
        .then(() => {
          const randomMessage = json[Math.floor(Math.random() * json.length)];

          try {
            sendMessage(
              TARGET_NOTIFICATION_ID,
              `Message deleted: ${text || caption} `
            );
          } catch (error) {
            console.error("Failed to send message:", error);
          }

          sendMessage(
            chatId,
            `Тут было паведамленне якое не суадносіцца з нашай суполкай. Мы яго выдалілі і замест гэтага  трымайце беларускую прыказку ці прымаўку:

          "${randomMessage.message}"
                       `
          ).catch((error) => {
            console.error("Failed to send message:", error);
          });
        })
        .catch((error) => {
          console.error("Failed to delete message:", error);
        });
    }
  }
});

const sendMessage = (chatId, text) => {
  bot.sendMessage(chatId, text);
};

function shouldDeleteMessage(messagesArray) {
  // Simple example of a condition to delete a message
  // You can implement any analysis logic here
  const forbiddenWords = [
    "test",
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
