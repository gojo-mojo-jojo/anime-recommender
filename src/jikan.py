import asyncio
from typing import Any

import httpx

from .models import Anime

JIKAN_BASE = "https://api.jikan.moe/v4"
_JIKAN_SEMAPHORE = asyncio.Semaphore(1)  # one request at a time to respect rate limit

GENRE_IDS: dict[str, int] = {
    "action": 1, "adventure": 2, "comedy": 4, "demons": 6, "drama": 8,
    "fantasy": 10, "historical": 13, "horror": 14, "magic": 16, "mecha": 18,
    "music": 19, "mystery": 7, "psychological": 40, "romance": 22,
    "samurai": 21, "school": 23, "sci-fi": 24, "sci fi": 24,
    "science fiction": 24, "slice of life": 36, "space": 29, "sports": 30,
    "supernatural": 37, "suspense": 41, "thriller": 41, "vampire": 32,
}


async def _get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """Rate-limited GET with retry on 429."""
    async with _JIKAN_SEMAPHORE:
        async with httpx.AsyncClient() as client:
            for attempt in range(3):
                response = await client.get(url, params=params)
                if response.status_code == 429:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response.json()
    return {}


async def search_anime(title: str) -> Anime | None:
    data = await _get(f"{JIKAN_BASE}/anime", {"q": title, "limit": 1})
    entries = data.get("data", [])
    if not entries:
        return None
    entry = entries[0]
    return Anime(
        id=entry["mal_id"],
        title=entry["title"],
        synopsis=entry.get("synopsis") or "",
        genres=[g["name"] for g in entry.get("genres", [])],
        score=entry.get("score"),
        episodes=entry.get("episodes"),
        image_url=entry.get("images", {}).get("jpg", {}).get("image_url"),
    )


async def search_anime_by_genres(genres: list[str], limit: int = 20) -> list[dict[str, Any]]:
    genre_ids = [str(GENRE_IDS[g.lower()]) for g in genres if g.lower() in GENRE_IDS]
    params: dict[str, Any] = {"limit": limit, "order_by": "score", "sort": "desc", "sfw": "true"}
    if genre_ids:
        params["genres"] = ",".join(genre_ids)
    data = await _get(f"{JIKAN_BASE}/anime", params)
    return data.get("data", [])
