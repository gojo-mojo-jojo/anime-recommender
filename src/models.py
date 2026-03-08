from pydantic import BaseModel


class Anime(BaseModel):
    id: int
    title: str
    synopsis: str
    genres: list[str]
    themes: list[str] = []
    score: float | None
    episodes: int | None
    image_url: str | None
    year: int | None = None
    studio: str | None = None
    anime_type: str | None = None
    reason: str = ""


class RecommendRequest(BaseModel):
    preferences: str
    mode: str = "personalized"


class RecommendResponse(BaseModel):
    recommendations: list[Anime]


class ExplainRequest(BaseModel):
    preference: str
    title: str
    synopsis: str
    genres: list[str]
