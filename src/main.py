import logging
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .jikan import close_http_client
from .models import ExplainRequest, RecommendRequest, RecommendResponse
from .recommender import explain_recommendation_stream, get_recommendations
from .tmdb import (
    close_tmdb_client,
    find_tmdb_id_for_anime,
    get_available_platforms,
    get_watch_providers,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CACHE_MAX = 64
_recommendation_cache: OrderedDict[str, list] = OrderedDict()

RATE_LIMIT_PER_DAY = 4
_rate_limit_store: dict[str, tuple[date, int]] = {}


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(ip: str) -> bool:
    """Return True if allowed, False if exceeded. Increments count on allow."""
    today = date.today()
    if ip not in _rate_limit_store:
        _rate_limit_store[ip] = (today, 1)
        return True
    stored_date, count = _rate_limit_store[ip]
    if stored_date != today:
        _rate_limit_store[ip] = (today, 1)
        return True
    if count >= RATE_LIMIT_PER_DAY:
        return False
    _rate_limit_store[ip] = (stored_date, count + 1)
    return True


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    await close_http_client()
    await close_tmdb_client()


app = FastAPI(title="Entertainment Recommender", version="0.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(request: Request, body: RecommendRequest) -> RecommendResponse:
    creativity = max(0.0, min(1.0, body.creativity))
    count = max(3, min(18, body.count))
    plat_key = ",".join(str(p) for p in sorted(body.platforms))
    cache_key = (
        f"{body.category}:{body.mode}:{creativity:.2f}:{count}"
        f":{plat_key}:{body.region}"
        f":{body.preferences.strip().lower()}"
    )

    if cache_key in _recommendation_cache:
        logger.info("Cache hit for: %s", cache_key[:80])
        _recommendation_cache.move_to_end(cache_key)
        return RecommendResponse(recommendations=_recommendation_cache[cache_key])

    ip = _get_client_ip(request)
    if not _check_rate_limit(ip):
        return JSONResponse(
            status_code=429,
            content={
                "detail": "rate_limit_exceeded",
                "message": "You've reached your daily limit of 4 searches. Support the project to keep it running!",
            },
        )

    try:
        items = await get_recommendations(
            body.preferences,
            body.mode,
            body.category,
            creativity,
            count,
            platforms=body.platforms or None,
            region=body.region,
        )
    except Exception:
        logger.exception("Recommendation failed for: %s", body.preferences[:80])
        raise HTTPException(status_code=500, detail="Failed to get recommendations")

    if items:
        _recommendation_cache[cache_key] = items
        if len(_recommendation_cache) > CACHE_MAX:
            _recommendation_cache.popitem(last=False)

    return RecommendResponse(recommendations=items)


@app.post("/api/explain")
async def explain(body: ExplainRequest) -> StreamingResponse:
    return StreamingResponse(
        explain_recommendation_stream(
            body.preference, body.title, body.synopsis, body.genres, body.category
        ),
        media_type="text/plain",
    )


@app.get("/api/platforms")
async def platforms(region: str = "IN") -> list[dict[str, str]]:
    return await get_available_platforms(region)


@app.get("/api/debug/providers")
async def debug_providers(region: str = "IN") -> dict:
    from .tmdb import _get
    movie_data = await _get("/watch/providers/movie", {"watch_region": region})
    tv_data = await _get("/watch/providers/tv", {"watch_region": region})
    all_provs = {}
    for entry in movie_data.get("results", []) + tv_data.get("results", []):
        pid = entry.get("provider_id", 0)
        name = entry.get("provider_name", "")
        all_provs[pid] = name
    return dict(sorted(all_provs.items(), key=lambda x: x[1]))


@app.get("/api/providers")
async def providers(
    id: int,
    category: str = "anime",
    region: str = "IN",
    title: str = "",
) -> list[dict[str, str]]:
    media_type = "tv" if category in ("anime", "series", "cartoon") else "movie"

    if category == "anime":
        tmdb_id = await find_tmdb_id_for_anime(title) if title else None
        if tmdb_id is None:
            return []
        return await get_watch_providers(tmdb_id, media_type, region)

    return await get_watch_providers(id, media_type, region)


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=settings.port, reload=True)
