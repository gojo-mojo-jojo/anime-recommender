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


@pytest.mark.asyncio
async def test_get_recommendations_anime(monkeypatch):
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = (
        '[{"title": "Fullmetal Alchemist: Brotherhood",'
        ' "reason": "Epic action adventure"}]'
    )

    mock_message = MagicMock()
    mock_message.stop_reason = "end_turn"
    mock_message.content = [mock_text_block]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    monkeypatch.setattr("src.recommender.claude", mock_client)

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
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = (
        '[{"title": "Inception",'
        ' "reason": "Mind-bending sci-fi thriller"}]'
    )

    mock_message = MagicMock()
    mock_message.stop_reason = "end_turn"
    mock_message.content = [mock_text_block]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    monkeypatch.setattr("src.recommender.claude", mock_client)

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
