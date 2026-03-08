import httpx
import pytest
import respx

from src.jikan import search_anime

JIKAN_URL = "https://api.jikan.moe/v4/anime"

FAKE_ENTRY = {
    "mal_id": 5114,
    "title": "Fullmetal Alchemist: Brotherhood",
    "title_english": "Fullmetal Alchemist: Brotherhood",
    "title_synonyms": ["FMA:B"],
    "titles": [{"type": "English", "title": "Fullmetal Alchemist: Brotherhood"}],
    "synopsis": "Two brothers...",
    "genres": [{"name": "Action"}, {"name": "Adventure"}],
    "themes": [{"name": "Military"}],
    "demographics": [{"name": "Shounen"}],
    "studios": [{"name": "Bones"}],
    "score": 9.1,
    "episodes": 64,
    "year": 2009,
    "type": "TV",
    "images": {
        "jpg": {
            "image_url": "https://example.com/fmab.jpg",
            "large_image_url": "https://example.com/fmab_large.jpg",
        }
    },
}


@pytest.mark.asyncio
@respx.mock
async def test_search_anime_returns_mapped_result():
    respx.get(JIKAN_URL).mock(
        return_value=httpx.Response(200, json={"data": [FAKE_ENTRY]})
    )
    result = await search_anime("Fullmetal Alchemist Brotherhood")
    assert result is not None
    assert result.id == 5114
    assert result.title == "Fullmetal Alchemist: Brotherhood"
    assert "Action" in result.genres
    assert result.studio == "Bones"
    assert result.year == 2009
    assert "Military" in result.themes
    assert result.image_url == "https://example.com/fmab_large.jpg"
    assert result.category == "anime"


@pytest.mark.asyncio
@respx.mock
async def test_search_anime_picks_best_title_match():
    wrong_entry = {
        **FAKE_ENTRY,
        "mal_id": 9999,
        "title": "Something Else",
        "title_english": "Something Else",
        "titles": [{"type": "English", "title": "Something Else"}],
    }
    respx.get(JIKAN_URL).mock(
        return_value=httpx.Response(
            200, json={"data": [wrong_entry, FAKE_ENTRY]}
        )
    )
    result = await search_anime("Fullmetal Alchemist Brotherhood")
    assert result is not None
    assert result.id == 5114


@pytest.mark.asyncio
@respx.mock
async def test_search_anime_returns_none_when_empty():
    respx.get(JIKAN_URL).mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    result = await search_anime("nonexistent xyz")
    assert result is None
