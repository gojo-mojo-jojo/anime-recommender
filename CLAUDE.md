# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Stack

- **Python 3.11+** with **FastAPI** and **uvicorn**
- **Anthropic Claude API** (`claude-sonnet-4-6`) for AI-powered recommendations
- **Jikan API** (`https://api.jikan.moe/v4`) for anime metadata — no auth required
- **TMDB API** (`https://api.themoviedb.org/3`) for movie, series, and cartoon metadata
- **Tavily API** for web-based review searches
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

The app is a multi-category entertainment recommender with pluggable agents for anime, movies, series, and cartoons.

```
POST /api/recommend  →  recommender.get_recommendations(preferences, mode, category)
                          │
                          ├─ Agent registry selects category-specific:
                          │    system prompt, tools, data client
                          │
                          ├─ Claude API (async, tool use): infers 9 titles
                          │    ├─ search_by_genre tool × 2-3 (Jikan or TMDB)
                          │    └─ search_reviews tool × 1-2 (Tavily)
                          │
                          └─ Category search_fn × 9 (concurrent via asyncio.gather)
                               │
                               └─ Returns enriched ContentItem objects

POST /api/explain    →  recommender.explain_recommendation_stream()
                          │
                          └─ Claude API (async streaming): category-aware analysis
```

Categories and their data sources:
- **anime** → Jikan API (MyAnimeList)
- **movie** → TMDB API (discover/movie, search/movie)
- **series** → TMDB API (discover/tv, search/tv)
- **cartoon** → TMDB API (discover/tv with animation genre filter)

Key files:
- `src/main.py` — FastAPI app, routes, lifespan, in-memory cache
- `src/recommender.py` — agent registry, orchestrates async Claude + tool calls
- `src/jikan.py` — async Jikan API client with connection pooling and rate limiting
- `src/tmdb.py` — async TMDB API client for movies, series, and cartoons
- `src/models.py` — Pydantic models (`ContentItem`, `Anime` alias, request/response)
- `src/config.py` — settings loaded from `.env` via `pydantic-settings`
- `static/index.html` — frontend SPA with category tabs, card grid, explanation modal

## Environment

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, and `TMDB_API_KEY`. The app will fail to start without them.

## Testing

Tests use `pytest-asyncio` (auto mode) and `respx` for mocking httpx calls. The Anthropic client is mocked via `monkeypatch` — no real API calls are made in tests.
