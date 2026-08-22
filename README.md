# PrisMate

Role-playing chat app with custom AI personas, per-user API keys, persistent
long-term memory, and a soul/persona system that lets characters grow over time.

## Stack

- **Frontend:** Next.js 15 · React 19 · TypeScript · Tailwind 4 · Redux Toolkit · Apollo Client (GraphQL)
- **Backend:** Django 5.2 · DRF · Strawberry GraphQL · Celery · PostgreSQL · Redis
- **AI:** OpenAI-compatible + Gemini providers (key configured per user in-app)

## Quick Start

Prereqs: Node 18+, Python 3.10+, PostgreSQL, Redis.

```bash
# Redis
docker-compose up -d redis

# Backend
cd backend
python -m venv venv_stable
source venv_stable/Scripts/activate   # PowerShell: .\venv_stable\Scripts\Activate.ps1
cp .env.template .env                # set DATABASE_*, REDIS_URL, SECRET_KEY
pip install -r requirements.txt
python manage.py migrate

# Run API + worker (separate terminals)
python manage.py runserver
python -m celery -A prismate worker --loglevel=info -c 1

# Frontend
cd ../frontend
cp .env.local.template .env.local    # NEXT_PUBLIC_API_URL=http://localhost:8000/api
npm install
npm run dev                          # http://localhost:3000
```

Sign in → open **Project Settings** → add a model configuration with your API key.

## Features

- Custom character creation (manual form or AI-assisted generation from files)
- **Character reference files** stored in exactly one place (`CharacterKnowledgeAsset`);
  the legacy `Character.file` mirror is no longer written
- **Memory Tools for draft generation** — uploaded files are never inlined into the
  prompt; the model browses/reads them on demand via `list_memory_files` /
  `read_memory_file` (OpenAI-compatible / Anthropic)
- **Reduce pipeline for large uploads** — when 12+ text files are attached,
  `generateCharacterDraft` runs a tiered map-reduce pipeline
  (`chat/character_reduce.py`): tier by screen time → batch close-read (main files
  in full, cameo files as segments) → structured notes with citations → merge into
  a `PrisMateDraft`-aligned profile
- Streamed chat with persistent per-session history
- **Long-term memory** per character, with browse/edit/merge/wipe UI at `/memory`
- **Private Mode** per session to skip long-term memory writes
- i18n (zh-CN / en-US)

## Key Endpoints

- `POST /api/chat/send_message` — send a user message (streamed response in payload)
- `GET/POST /api/characters` · `GET /api/characters/{id}` · `GET /api/sessions`
- `GET /api/characters/{id}/memory` · `POST/PATCH/DELETE /api/characters/{id}/memory[/{id}]`
- `POST /api/characters/{id}/memory/merge` · `DELETE /api/characters/{id}/memory`
- `POST /api/graphql/` — character CRUD (Strawberry GraphQL)

See `backend/chat/urls.py` for the full route map.

## Project Layout

```
backend/
  prismate/            # Django project (settings, celery, urls)
  chat/                # models, views, serializers, graphql, tasks, soul, memory
  chat/character_reduce.py   # reduce pipeline (tier → batch notes → merge → PrisMateDraft)
  chat/memory/filesystem.py  # Memory Tools filesystem backends (character / staged uploads)
  chat/management/commands/run_character_reduce.py  # run reduce with a user's real model config
  scripts/             # pure-rules map prototype (character_file_indexer.py) + reduce prototype
frontend/src/
  app/                 # Next.js routes (/, /create-character, /edit-character/[id], /memory)
  components/          # ChatInterface, ChatWindow, MemoryPanel, SoulPanel, ModelApiSettingsPanel…
  store/               # Redux Toolkit slices
  i18n/                # zh-CN / en-US
  utils/api.ts         # REST client
  lib/apolloClient.ts  # GraphQL client
spec/memory-system-spec.md
ARCHITECTURE.md        # internal data-flow reference
```

## Daily Commands

See [`everydaycommand.md`](./everydaycommand.md) for the full cheat-sheet (setup, run,
migrate, troubleshooting).
