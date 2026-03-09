from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import ContentItem
from src.recommender import get_recommendations

FAKE_ANIME = ContentItem(
    id=5114,
    title="Fullmetal Alchemist: Brotherhood",
    synopsis="Two brothers...",
    genres=["Action", "Adventure"],
    themes=["Military"],
    score=9.1,
    episodes=64,
    image_url="https://example.com/fmab_large.jpg",
    year=2009,
    studio="Bones",
    content_type="TV",
    category="anime",
)

FAKE_MOVIE = ContentItem(
    id=27205,
    title="Inception",
    synopsis="A thief who steals secrets...",
    genres=["Action", "Science Fiction"],
    score=8.4,
    image_url="https://image.tmdb.org/t/p/w500/poster.jpg",
    year=2010,
    content_type="Movie",
    category="movie",
    runtime=148,
)

FAKE_MOVIE_2 = ContentItem(
    id=155,
    title="The Dark Knight",
    synopsis="Batman fights the Joker...",
    genres=["Action", "Crime", "Drama"],
    score=9.0,
    image_url="https://image.tmdb.org/t/p/w500/dark_knight.jpg",
    year=2008,
    content_type="Movie",
    category="movie",
    runtime=152,
)


def _mock_claude_end_turn(json_text: str) -> AsyncMock:
    """Helper: build a mock Claude client that returns end_turn with JSON."""
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = json_text

    mock_message = MagicMock()
    mock_message.stop_reason = "end_turn"
    mock_message.content = [mock_text_block]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)
    return mock_client


@pytest.mark.asyncio
async def test_get_recommendations_anime(monkeypatch):
    monkeypatch.setattr(
        "src.recommender.claude",
        _mock_claude_end_turn(
            '[{"title": "Fullmetal Alchemist: Brotherhood",'
            ' "reason": "Epic action adventure"}]'
        ),
    )

    with patch(
        "src.recommender._search_item",
        new=AsyncMock(return_value=FAKE_ANIME),
    ):
        results = await get_recommendations(
            "action with great story", category="anime"
        )

    assert len(results) == 1
    assert results[0].title == "Fullmetal Alchemist: Brotherhood"
    assert results[0].reason == "Epic action adventure"
    assert results[0].category == "anime"


@pytest.mark.asyncio
async def test_get_recommendations_movie(monkeypatch):
    monkeypatch.setattr(
        "src.recommender.claude",
        _mock_claude_end_turn(
            '[{"title": "Inception",'
            ' "reason": "Mind-bending sci-fi thriller"}]'
        ),
    )

    with patch(
        "src.recommender._search_item",
        new=AsyncMock(return_value=FAKE_MOVIE),
    ):
        results = await get_recommendations(
            "mind-bending sci-fi", category="movie"
        )

    assert len(results) == 1
    assert results[0].title == "Inception"
    assert results[0].reason == "Mind-bending sci-fi thriller"
    assert results[0].category == "movie"


@pytest.mark.asyncio
async def test_platform_filter_removes_unavailable(monkeypatch):
    """When platforms are selected, titles not on those platforms are removed."""
    monkeypatch.setattr(
        "src.recommender.claude",
        _mock_claude_end_turn(
            '[{"title": "Inception", "reason": "Great"},'
            ' {"title": "The Dark Knight", "reason": "Also great"}]'
        ),
    )

    call_count = 0

    async def _fake_search(title, cat):
        nonlocal call_count
        call_count += 1
        if "Inception" in title:
            return FAKE_MOVIE
        return FAKE_MOVIE_2

    providers_inception = [
        {"name": "Netflix", "logo_url": "", "type": "flatrate", "provider_id": "8"},
    ]
    providers_dark_knight = [
        {"name": "Hulu", "logo_url": "", "type": "flatrate", "provider_id": "15"},
    ]

    async def _fake_providers(tmdb_id, media_type, region):
        if tmdb_id == 27205:
            return providers_inception
        return providers_dark_knight

    with patch("src.recommender._search_item", new=_fake_search), \
         patch("src.recommender.get_watch_providers", new=_fake_providers):
        results = await get_recommendations(
            "action thriller",
            category="movie",
            platforms=[8],
            region="IN",
        )

    assert len(results) == 1
    assert results[0].title == "Inception"


@pytest.mark.asyncio
async def test_no_platform_filter_keeps_all(monkeypatch):
    """Without platform selection, all titles are returned."""
    monkeypatch.setattr(
        "src.recommender.claude",
        _mock_claude_end_turn(
            '[{"title": "Inception", "reason": "Great"},'
            ' {"title": "The Dark Knight", "reason": "Also great"}]'
        ),
    )

    async def _fake_search(title, cat):
        if "Inception" in title:
            return FAKE_MOVIE
        return FAKE_MOVIE_2

    with patch("src.recommender._search_item", new=_fake_search):
        results = await get_recommendations(
            "action thriller", category="movie"
        )

    assert len(results) == 2
