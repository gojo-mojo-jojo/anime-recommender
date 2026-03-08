import httpx
import pytest
import respx

from src.tmdb import search_cartoon, search_movie, search_series

TMDB_SEARCH_MOVIE = "https://api.themoviedb.org/3/search/movie"
TMDB_SEARCH_TV = "https://api.themoviedb.org/3/search/tv"

FAKE_MOVIE = {
    "id": 27205,
    "title": "Inception",
    "original_title": "Inception",
    "overview": "A thief who steals corporate secrets...",
    "genre_ids": [28, 878, 12],
    "vote_average": 8.4,
    "release_date": "2010-07-15",
    "poster_path": "/poster.jpg",
}

FAKE_SERIES = {
    "id": 1396,
    "name": "Breaking Bad",
    "original_name": "Breaking Bad",
    "overview": "A chemistry teacher diagnosed with cancer...",
    "genre_ids": [18, 80],
    "vote_average": 8.9,
    "first_air_date": "2008-01-20",
    "poster_path": "/bb_poster.jpg",
}

FAKE_CARTOON = {
    "id": 246,
    "name": "Avatar: The Last Airbender",
    "original_name": "Avatar: The Last Airbender",
    "overview": "In a war-torn world...",
    "genre_ids": [16, 10759, 18],
    "vote_average": 8.7,
    "first_air_date": "2005-02-21",
    "poster_path": "/avatar.jpg",
}


@pytest.mark.asyncio
@respx.mock
async def test_search_movie_returns_mapped_result():
    respx.get(TMDB_SEARCH_MOVIE).mock(
        return_value=httpx.Response(
            200, json={"results": [FAKE_MOVIE]}
        )
    )
    result = await search_movie("Inception")
    assert result is not None
    assert result.id == 27205
    assert result.title == "Inception"
    assert result.year == 2010
    assert result.category == "movie"
    assert result.image_url == "https://image.tmdb.org/t/p/w500/poster.jpg"


@pytest.mark.asyncio
@respx.mock
async def test_search_series_returns_mapped_result():
    respx.get(TMDB_SEARCH_TV).mock(
        return_value=httpx.Response(
            200, json={"results": [FAKE_SERIES]}
        )
    )
    result = await search_series("Breaking Bad")
    assert result is not None
    assert result.id == 1396
    assert result.title == "Breaking Bad"
    assert result.year == 2008
    assert result.category == "series"


@pytest.mark.asyncio
@respx.mock
async def test_search_cartoon_prefers_animation():
    non_animated = {
        **FAKE_SERIES,
        "id": 9999,
        "name": "Something Else",
        "genre_ids": [18],
    }
    respx.get(TMDB_SEARCH_TV).mock(
        return_value=httpx.Response(
            200, json={"results": [non_animated, FAKE_CARTOON]}
        )
    )
    result = await search_cartoon("Avatar")
    assert result is not None
    assert result.id == 246
    assert result.category == "cartoon"


@pytest.mark.asyncio
@respx.mock
async def test_search_movie_returns_none_when_empty():
    respx.get(TMDB_SEARCH_MOVIE).mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    result = await search_movie("nonexistent xyz")
    assert result is None
