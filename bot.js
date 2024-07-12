const TelegramBot = require("node-telegram-bot-api");
const json = require("./index.json");

const token = process.env.TELEGRAM_BOT_TOKEN;
const bot = new TelegramBot(token, { polling: true });

const TARGET_CHAT_ID = process.env.TARGET_CHAT_ID; // Add your target chat ID in the .env file
const TARGET_NOTIFICATION_ID = process.env.TARGET_NOTIFICATION_ID; // Add your target chat ID in the .env file
//  get random message from 1  to 770
const randomMessage = json[Math.floor(Math.random() * json.length)];

bot.on("message", (msg) => {
  const chatId = msg.chat.id;
  const text = msg.text;
  console.log(text, TARGET_CHAT_ID, chatId.toString());

  if (chatId.toString() === TARGET_CHAT_ID) {
    // Analyze messages and decide to delete
    if (text && shouldDeleteMessage(text)) {
      console.log("delete");
      bot
        .deleteMessage(chatId, msg.message_id)
        .then(() => {
          const randomMessage = json[Math.floor(Math.random() * json.length)];

          try {
            sendMessage(TARGET_NOTIFICATION_ID, `Message deleted: ${text} `);
          } catch (error) {
            console.error("Failed to send message:", error);
          }

          sendMessage(
            TARGET_CHAT_ID,
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
