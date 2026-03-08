import httpx
import pytest
import respx

from src.jikan import search_anime

JIKAN_URL = "https://api.jikan.moe/v4/anime"

FAKE_ENTRY = {
    "mal_id": 5114,
    "title": "Fullmetal Alchemist: Brotherhood",
    "synopsis": "Two brothers...",
    "genres": [{"name": "Action"}, {"name": "Adventure"}],
    "score": 9.1,
    "episodes": 64,
    "images": {"jpg": {"image_url": "https://example.com/fmab.jpg"}},
}


@pytest.mark.asyncio
@respx.mock
async def test_search_anime_returns_mapped_result():
    respx.get(JIKAN_URL).mock(return_value=httpx.Response(200, json={"data": [FAKE_ENTRY]}))
    result = await search_anime("Fullmetal Alchemist Brotherhood")
    assert result is not None
    assert result.id == 5114
    assert result.title == "Fullmetal Alchemist: Brotherhood"
    assert "Action" in result.genres


@pytest.mark.asyncio
@respx.mock
async def test_search_anime_returns_none_when_empty():
    respx.get(JIKAN_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    result = await search_anime("nonexistent xyz")
    assert result is None
