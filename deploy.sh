#!/bin/bash

docker build -t telegram_bot  .
docker stop telegram_bot || true && docker rm telegram_bot || true
<<<<<<< HEAD
docker run -d -p 3004:3001 --name telegram_bot --restart unless-stopped telegram_bot
=======
docker run -d -p 3004:3000 --name telegram_bot --restart unless-stopped telegram_bot
>>>>>>> main
