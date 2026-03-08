from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import Anime
from src.recommender import get_recommendations

FAKE_ANIME = Anime(
    id=5114,
    title="Fullmetal Alchemist: Brotherhood",
    synopsis="Two brothers...",
    genres=["Action", "Adventure"],
    score=9.1,
    episodes=64,
    image_url="https://example.com/fmab.jpg",
)


@pytest.mark.asyncio
async def test_get_recommendations(monkeypatch):
    mock_message = MagicMock()
    mock_message.content = [MagicMock(type="text", text='["Fullmetal Alchemist: Brotherhood"]')]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message

    monkeypatch.setattr("src.recommender.client", mock_client)

    with patch("src.recommender.search_anime", new=AsyncMock(return_value=FAKE_ANIME)):
        results = await get_recommendations("action with great story")

    assert len(results) == 1
    assert results[0].title == "Fullmetal Alchemist: Brotherhood"
