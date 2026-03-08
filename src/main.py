import logging
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .jikan import close_http_client
from .models import ExplainRequest, RecommendRequest, RecommendResponse
from .recommender import explain_recommendation_stream, get_recommendations

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CACHE_MAX = 64
_recommendation_cache: OrderedDict[str, list] = OrderedDict()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    await close_http_client()


app = FastAPI(title="Anime Recommender", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(body: RecommendRequest) -> RecommendResponse:
    cache_key = f"{body.mode}:{body.preferences.strip().lower()}"

    if cache_key in _recommendation_cache:
        logger.info("Cache hit for: %s", cache_key[:60])
        _recommendation_cache.move_to_end(cache_key)
        return RecommendResponse(recommendations=_recommendation_cache[cache_key])

    try:
        anime_list = await get_recommendations(body.preferences, body.mode)
    except Exception:
        logger.exception("Recommendation failed for: %s", body.preferences[:80])
        raise HTTPException(status_code=500, detail="Failed to get recommendations")

    if anime_list:
        _recommendation_cache[cache_key] = anime_list
        if len(_recommendation_cache) > CACHE_MAX:
            _recommendation_cache.popitem(last=False)

    return RecommendResponse(recommendations=anime_list)


@app.post("/api/explain")
async def explain(body: ExplainRequest) -> StreamingResponse:
    return StreamingResponse(
        explain_recommendation_stream(body.preference, body.title, body.synopsis, body.genres),
        media_type="text/plain",
    )


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=settings.port, reload=True)
