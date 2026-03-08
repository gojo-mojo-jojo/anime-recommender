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

The app is a single-endpoint FastAPI service. The request flow is:

```
POST /api/recommend  →  recommender.get_recommendations()
                          │
                          ├─ Claude API: infers 5 anime titles from free-text preferences
                          │
                          └─ jikan.search_anime() × 5 (concurrent via asyncio.gather)
                               │
                               └─ Returns enriched Anime objects
```

Key files:
- `src/main.py` — FastAPI app and route definition
- `src/recommender.py` — orchestrates Claude + Jikan calls
- `src/jikan.py` — async Jikan API client
- `src/models.py` — Pydantic models (`Anime`, `RecommendRequest`, `RecommendResponse`)
- `src/config.py` — settings loaded from `.env` via `pydantic-settings`

## Environment

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`. The app will fail to start without it.

## Testing

Tests use `pytest-asyncio` (auto mode) and `respx` for mocking httpx calls. The Anthropic client is mocked via `monkeypatch` — no real API calls are made in tests.
