import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .models import ExplainRequest, RecommendRequest, RecommendResponse
from .recommender import explain_recommendation_stream, get_recommendations

app = FastAPI(title="Anime Recommender", version="0.1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(body: RecommendRequest) -> RecommendResponse:
    try:
        anime_list = await get_recommendations(body.preferences)
        return RecommendResponse(recommendations=anime_list)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to get recommendations") from exc


@app.post("/api/explain")
async def explain(body: ExplainRequest) -> StreamingResponse:
    return StreamingResponse(
        explain_recommendation_stream(body.preference, body.title, body.synopsis, body.genres),
        media_type="text/plain",
    )


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=settings.port, reload=True)
