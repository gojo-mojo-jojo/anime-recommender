import asyncio
import logging
from typing import Any

import httpx

from .models import ContentItem

logger = logging.getLogger(__name__)

JIKAN_BASE = "https://api.jikan.moe/v4"
_JIKAN_SEMAPHORE = asyncio.Semaphore(2)
_JIKAN_DELAY = 0.5

_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=15.0)
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


GENRE_IDS: dict[str, int] = {
    "action": 1, "adventure": 2, "comedy": 4, "demons": 6, "drama": 8,
    "ecchi": 9, "fantasy": 10, "historical": 13, "horror": 14,
    "isekai": 62, "magic": 16, "martial arts": 17, "mecha": 18,
    "military": 38, "music": 19, "mystery": 7, "parody": 20,
    "psychological": 40, "romance": 22, "samurai": 21, "school": 23,
    "sci-fi": 24, "sci fi": 24, "science fiction": 24,
    "seinen": 42, "shoujo": 25, "shounen": 27,
    "slice of life": 36, "space": 29, "sports": 30,
    "supernatural": 37, "suspense": 41, "thriller": 41, "vampire": 32,
}


async def _get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """Rate-limited GET with retry on 429."""
    async with _JIKAN_SEMAPHORE:
        client = get_http_client()
        for attempt in range(4):
            try:
                response = await client.get(url, params=params)
            except httpx.HTTPError:
                logger.warning("Jikan request failed (attempt %d): %s", attempt + 1, url)
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            if response.status_code == 429:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            response.raise_for_status()
            await asyncio.sleep(_JIKAN_DELAY)
            return response.json()
    return {}


def _parse_anime(entry: dict[str, Any]) -> ContentItem:
    images = entry.get("images", {}).get("jpg", {})
    studios = entry.get("studios", [])
    return ContentItem(
        id=entry["mal_id"],
        title=entry.get("title_english") or entry["title"],
        synopsis=entry.get("synopsis") or "",
        genres=[g["name"] for g in entry.get("genres", [])],
        themes=[t["name"] for t in entry.get("themes", [])],
        score=entry.get("score"),
        episodes=entry.get("episodes"),
        image_url=images.get("large_image_url") or images.get("image_url"),
        year=entry.get("year"),
        studio=studios[0]["name"] if studios else None,
        content_type=entry.get("type"),
        category="anime",
    )


def _title_similarity(query: str, entry: dict[str, Any]) -> float:
    """Score how well a Jikan result matches the query title."""
    q = query.lower().strip()
    candidates = [
        (entry.get("title") or "").lower(),
        (entry.get("title_english") or "").lower(),
    ]
    for t in entry.get("title_synonyms", []):
        candidates.append(t.lower())
    for t in entry.get("titles", []):
        candidates.append((t.get("title") or "").lower())

    best = 0.0
    for c in candidates:
        if not c:
            continue
        if q == c:
            return 1.0
        if q in c or c in q:
            best = max(best, 0.8)
        else:
            common = set(q.split()) & set(c.split())
            if common:
                best = max(best, len(common) / max(len(q.split()), len(c.split())))
    return best


async def search_anime(title: str) -> ContentItem | None:
    data = await _get(f"{JIKAN_BASE}/anime", {"q": title, "limit": 3})
    entries = data.get("data", [])
    if not entries:
        return None

    best = max(entries, key=lambda e: _title_similarity(title, e))
    return _parse_anime(best)


async def search_anime_by_genres(
    genres: list[str], limit: int = 10
) -> list[dict[str, Any]]:
    genre_ids = [
        str(GENRE_IDS[g.lower()])
        for g in genres
        if g.lower() in GENRE_IDS
    ]
    params: dict[str, Any] = {
        "limit": limit,
        "order_by": "members",
        "sort": "desc",
        "sfw": "true",
    }
    if genre_ids:
        params["genres"] = ",".join(genre_ids)
    data = await _get(f"{JIKAN_BASE}/anime", params)
    return data.get("data", [])
