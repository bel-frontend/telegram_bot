import dotenv from 'dotenv';
dotenv.config();

import { REST, Routes } from 'discord.js';

const token = process.env.DISCORD_TOKEN;
const clientId = process.env.DISCORD_CLIENT_ID;

if (!token) {
  throw new Error('DISCORD_TOKEN is required in .env');
}

if (!clientId) {
  throw new Error('DISCORD_CLIENT_ID is required in .env');
}

const commands = [
  {
    name: 'proverb',
    description: 'Знайдзі беларускія прыказкі па тэме або атрымай выпадковыя',
    options: [
      {
        type: 3, // STRING
        name: 'topic',
        description: 'Тэма для пошуку прыказак (напрыклад: праца, сям\'я, прырода)',
        required: false,
      },
    ],
  },
];

const rest = new REST({ version: '10' }).setToken(token);

(async () => {
  try {
    console.log('Registering Discord application commands...');
    await rest.put(Routes.applicationCommands(clientId), { body: commands });
    console.log('Successfully registered application commands.');
  } catch (error) {
    console.error('Failed to register commands:', error);
    process.exit(1);
  }
})();
