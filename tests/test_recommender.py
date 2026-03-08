from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import Anime
from src.recommender import get_recommendations

FAKE_ANIME = Anime(
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
    anime_type="TV",
)


@pytest.mark.asyncio
async def test_get_recommendations(monkeypatch):
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = (
        '[{"title": "Fullmetal Alchemist: Brotherhood", "reason": "Epic action adventure"}]'
    )

    mock_message = MagicMock()
    mock_message.stop_reason = "end_turn"
    mock_message.content = [mock_text_block]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    monkeypatch.setattr("src.recommender.claude", mock_client)

    with patch("src.recommender.search_anime", new=AsyncMock(return_value=FAKE_ANIME)):
        results = await get_recommendations("action with great story")

    assert len(results) == 1
    assert results[0].title == "Fullmetal Alchemist: Brotherhood"
    assert results[0].reason == "Epic action adventure"
