# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Stack

- **Python 3.11+** with **FastAPI** and **uvicorn**
- **Anthropic Claude API** (`claude-sonnet-4-6`) for AI-powered recommendations
- **Jikan API** (`https://api.jikan.moe/v4`) for anime metadata — no auth required
- **httpx** for async HTTP calls, **Pydantic v2** for models and settings

## Commands

```bash
# Install (editable + dev deps)
pip install -e ".[dev]"

# Run dev server (auto-reload)
python -m src.main

# Lint
ruff check src tests

# Format
ruff format src tests

# Type check
mypy src

# Run all tests
pytest

# Run a single test file
pytest tests/test_jikan.py

# Run a single test by name
pytest -k "test_search_anime_returns_none"
```

## Architecture

The app is a multi-endpoint FastAPI service. The request flow is:

```
POST /api/recommend  →  recommender.get_recommendations()
                          │
                          ├─ Claude API (async, tool use): infers 9 anime titles
                          │    ├─ search_anime_by_genre tool × 3-4 (Jikan API)
                          │    └─ search_reviews tool × 2-3 (Tavily API)
                          │
                          └─ jikan.search_anime() × 9 (concurrent via asyncio.gather)
                               │
                               └─ Returns enriched Anime objects (with in-memory LRU cache)

POST /api/explain    →  recommender.explain_recommendation_stream()
                          │
                          └─ Claude API (async streaming): returns match analysis
```

Key files:
- `src/main.py` — FastAPI app, routes, lifespan, in-memory cache
- `src/recommender.py` — orchestrates async Claude + Jikan + Tavily calls
- `src/jikan.py` — async Jikan API client with connection pooling and rate limiting
- `src/models.py` — Pydantic models (`Anime`, `RecommendRequest`, `RecommendResponse`, `ExplainRequest`)
- `src/config.py` — settings loaded from `.env` via `pydantic-settings`
- `static/index.html` — frontend SPA with card grid and streaming explanation modal

## Environment

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` and `TAVILY_API_KEY`. The app will fail to start without them.

## Testing

Tests use `pytest-asyncio` (auto mode) and `respx` for mocking httpx calls. The Anthropic client is mocked via `monkeypatch` — no real API calls are made in tests.
