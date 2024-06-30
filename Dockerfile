# Use the official Node.js image as the base image
FROM node:20

# Create and change to the app directory
WORKDIR /usr/src/app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Expose port (optional, if your bot serves a web server)
EXPOSE 3000

# Command to run the bot
CMD ["node", "--env-file=.env", "bot.js"]
