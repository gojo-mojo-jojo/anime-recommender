from pydantic import BaseModel


class Anime(BaseModel):
    id: int
    title: str
    synopsis: str
    genres: list[str]
    score: float | None
    episodes: int | None
    image_url: str | None
    reason: str = ""


class RecommendRequest(BaseModel):
    preferences: str


class RecommendResponse(BaseModel):
    recommendations: list[Anime]
