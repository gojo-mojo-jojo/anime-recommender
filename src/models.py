from pydantic import BaseModel


class ContentItem(BaseModel):
    id: int
    title: str
    synopsis: str
    genres: list[str]
    themes: list[str] = []
    score: float | None
    image_url: str | None
    year: int | None = None
    category: str = ""
    reason: str = ""
    episodes: int | None = None
    seasons: int | None = None
    runtime: int | None = None
    studio: str | None = None
    director: str | None = None
    content_type: str | None = None


Anime = ContentItem


class RecommendRequest(BaseModel):
    preferences: str
    mode: str = "personalized"
    category: str = "anime"
    creativity: float = 0.5
    count: int = 9


class RecommendResponse(BaseModel):
    recommendations: list[ContentItem]


class ExplainRequest(BaseModel):
    preference: str
    title: str
    synopsis: str
    genres: list[str]
    category: str = "anime"
