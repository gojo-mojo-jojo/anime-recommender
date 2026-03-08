import asyncio
import logging
from typing import Any

import httpx

from .config import settings
from .models import ContentItem

logger = logging.getLogger(__name__)

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"
_TMDB_SEMAPHORE = asyncio.Semaphore(4)
_TMDB_DELAY = 0.15

_http_client: httpx.AsyncClient | None = None
_use_bearer: bool = settings.tmdb_api_key.startswith("eyJ")


def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        headers = {}
        if _use_bearer:
            headers["Authorization"] = f"Bearer {settings.tmdb_api_key}"
        _http_client = httpx.AsyncClient(timeout=15.0, headers=headers)
    return _http_client


async def close_tmdb_client() -> None:
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


MOVIE_GENRE_IDS: dict[str, int] = {
    "action": 28, "adventure": 12, "animation": 16, "comedy": 35,
    "crime": 80, "documentary": 99, "drama": 18, "family": 10751,
    "fantasy": 14, "history": 36, "horror": 27, "music": 10402,
    "mystery": 9648, "romance": 10749, "sci-fi": 878,
    "science fiction": 878, "thriller": 53, "war": 10752,
    "western": 37,
}

TV_GENRE_IDS: dict[str, int] = {
    "action": 10759, "action & adventure": 10759, "adventure": 10759,
    "animation": 16, "comedy": 35, "crime": 80, "documentary": 99,
    "drama": 18, "family": 10751, "kids": 10762, "mystery": 9648,
    "sci-fi": 10765, "sci-fi & fantasy": 10765, "science fiction": 10765,
    "fantasy": 10765, "war": 10768, "western": 37,
}

_MOVIE_GENRE_NAMES: dict[int, str] = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Sci-Fi",
    53: "Thriller",
    10752: "War",
    37: "Western",
}

_TV_GENRE_NAMES: dict[int, str] = {
    10759: "Action & Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    10762: "Kids",
    9648: "Mystery",
    10765: "Sci-Fi & Fantasy",
    10768: "War & Politics",
    37: "Western",
}


async def _get(
    path: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Rate-limited GET against TMDB with retry on 429."""
    async with _TMDB_SEMAPHORE:
        client = _get_client()
        merged: dict[str, Any] = {"language": "en-US"}
        if not _use_bearer:
            merged["api_key"] = settings.tmdb_api_key
        if params:
            merged.update(params)
        for attempt in range(4):
            try:
                resp = await client.get(
                    f"{TMDB_BASE}{path}", params=merged
                )
            except httpx.HTTPError:
                logger.warning(
                    "TMDB request failed (attempt %d): %s",
                    attempt + 1,
                    path,
                )
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            if resp.status_code == 429:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            await asyncio.sleep(_TMDB_DELAY)
            return resp.json()
    return {}


def _genre_names(ids: list[int], lookup: dict[int, str]) -> list[str]:
    return [lookup.get(gid, f"Genre {gid}") for gid in ids]


def _parse_movie(entry: dict[str, Any]) -> ContentItem:
    poster = entry.get("poster_path")
    return ContentItem(
        id=entry["id"],
        title=entry.get("title") or entry.get("original_title", ""),
        synopsis=entry.get("overview") or "",
        genres=_genre_names(
            entry.get("genre_ids", []), _MOVIE_GENRE_NAMES
        ),
        score=entry.get("vote_average"),
        image_url=f"{TMDB_IMG}{poster}" if poster else None,
        year=int(entry["release_date"][:4])
        if entry.get("release_date")
        else None,
        category="movie",
        content_type="Movie",
        runtime=entry.get("runtime"),
    )


def _parse_series(
    entry: dict[str, Any], category: str = "series"
) -> ContentItem:
    poster = entry.get("poster_path")
    first_air = entry.get("first_air_date") or ""
    return ContentItem(
        id=entry["id"],
        title=entry.get("name") or entry.get("original_name", ""),
        synopsis=entry.get("overview") or "",
        genres=_genre_names(
            entry.get("genre_ids", []), _TV_GENRE_NAMES
        ),
        score=entry.get("vote_average"),
        image_url=f"{TMDB_IMG}{poster}" if poster else None,
        year=int(first_air[:4]) if first_air else None,
        category=category,
        content_type="TV",
        seasons=entry.get("number_of_seasons"),
        episodes=entry.get("number_of_episodes"),
    )


def _title_match(query: str, entry: dict[str, Any], is_tv: bool) -> float:
    q = query.lower().strip()
    if is_tv:
        candidates = [
            (entry.get("name") or "").lower(),
            (entry.get("original_name") or "").lower(),
        ]
    else:
        candidates = [
            (entry.get("title") or "").lower(),
            (entry.get("original_title") or "").lower(),
        ]

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
                best = max(
                    best,
                    len(common) / max(len(q.split()), len(c.split())),
                )
    return best


async def search_movie(title: str) -> ContentItem | None:
    data = await _get("/search/movie", {"query": title})
    results = data.get("results", [])
    if not results:
        return None
    best = max(results, key=lambda e: _title_match(title, e, False))
    return _parse_movie(best)


async def search_series(title: str) -> ContentItem | None:
    data = await _get("/search/tv", {"query": title})
    results = data.get("results", [])
    if not results:
        return None
    best = max(results, key=lambda e: _title_match(title, e, True))
    return _parse_series(best)


async def search_cartoon(title: str) -> ContentItem | None:
    data = await _get("/search/tv", {"query": title})
    results = data.get("results", [])
    animated = [r for r in results if 16 in r.get("genre_ids", [])]
    pool = animated or results
    if not pool:
        return None
    best = max(pool, key=lambda e: _title_match(title, e, True))
    return _parse_series(best, category="cartoon")


async def search_movies_by_genres(
    genres: list[str], limit: int = 10
) -> list[dict[str, Any]]:
    ids = [
        str(MOVIE_GENRE_IDS[g.lower()])
        for g in genres
        if g.lower() in MOVIE_GENRE_IDS
    ]
    params: dict[str, Any] = {
        "sort_by": "popularity.desc",
        "page": 1,
        "vote_count.gte": 100,
    }
    if ids:
        params["with_genres"] = ",".join(ids)
    data = await _get("/discover/movie", params)
    return (data.get("results") or [])[:limit]


async def search_series_by_genres(
    genres: list[str], limit: int = 10
) -> list[dict[str, Any]]:
    ids = [
        str(TV_GENRE_IDS[g.lower()])
        for g in genres
        if g.lower() in TV_GENRE_IDS
    ]
    params: dict[str, Any] = {
        "sort_by": "popularity.desc",
        "page": 1,
        "vote_count.gte": 50,
    }
    if ids:
        params["with_genres"] = ",".join(ids)
    data = await _get("/discover/tv", params)
    return (data.get("results") or [])[:limit]


async def search_cartoons_by_genres(
    genres: list[str], limit: int = 10
) -> list[dict[str, Any]]:
    ids = [
        str(TV_GENRE_IDS[g.lower()])
        for g in genres
        if g.lower() in TV_GENRE_IDS
    ]
    ids_set = set(ids)
    ids_set.add(str(TV_GENRE_IDS["animation"]))
    params: dict[str, Any] = {
        "sort_by": "popularity.desc",
        "page": 1,
        "vote_count.gte": 50,
        "with_genres": ",".join(ids_set),
    }
    data = await _get("/discover/tv", params)
    return (data.get("results") or [])[:limit]


# ---------------------------------------------------------------------------
# Watch Providers (streaming availability)
# ---------------------------------------------------------------------------

TMDB_LOGO = "https://image.tmdb.org/t/p/w92"


async def get_watch_providers(
    tmdb_id: int, media_type: str, region: str = "US"
) -> list[dict[str, str]]:
    """Fetch streaming/buy/rent providers for a title in a given region."""
    path = f"/{media_type}/{tmdb_id}/watch/providers"
    data = await _get(path)
    all_results = data.get("results", {})
    country = all_results.get(region.upper(), {})

    providers: list[dict[str, str]] = []
    seen: set[int] = set()

    for bucket in ("flatrate", "free", "ads", "rent", "buy"):
        for p in country.get(bucket, []):
            pid = p.get("provider_id", 0)
            if pid in seen:
                continue
            seen.add(pid)
            logo = p.get("logo_path") or ""
            providers.append({
                "name": p.get("provider_name", ""),
                "logo_url": f"{TMDB_LOGO}{logo}" if logo else "",
                "type": bucket,
            })

    return providers


async def find_tmdb_id_for_anime(title: str) -> int | None:
    """Search TMDB for an anime title to get its TMDB TV ID."""
    data = await _get("/search/tv", {"query": title})
    results = data.get("results", [])
    if not results:
        return None
    best = max(results, key=lambda e: _title_match(title, e, True))
    return best.get("id")
