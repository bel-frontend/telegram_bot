FROM oven/bun:1

WORKDIR /usr/src/app

COPY package.json bun.lockb* ./

RUN bun install --frozen-lockfile

COPY . .

CMD ["bun", "--env-file=.env", "run", "index.ts"]
