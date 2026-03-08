import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Any

import anthropic
from tavily import TavilyClient

from .config import settings
from .jikan import search_anime, search_anime_by_genres
from .models import ContentItem
from .tmdb import (
    search_cartoon,
    search_cartoons_by_genres,
    search_movie,
    search_movies_by_genres,
    search_series,
    search_series_by_genres,
)

logger = logging.getLogger(__name__)

claude = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
tavily = TavilyClient(api_key=settings.tavily_api_key)

MAX_TOOL_ROUNDS = 4

# ---------------------------------------------------------------------------
# Shared tool: search_reviews (used by all agents)
# ---------------------------------------------------------------------------

REVIEW_TOOL: dict[str, Any] = {
    "name": "search_reviews",
    "description": (
        "Search the web for audience reviews and opinions. "
        "Call this IN THE SAME response as genre/discover "
        "searches — do NOT wait for those results first. "
        "Use for 1-2 titles you already know about from "
        "the user's preferences."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query, e.g. 'Inception movie "
                    "review reception'"
                ),
            }
        },
        "required": ["query"],
    },
}

# ---------------------------------------------------------------------------
# Simplify helpers for tool results
# ---------------------------------------------------------------------------


def _simplify_anime_entry(r: dict[str, Any]) -> dict[str, Any]:
    studios = r.get("studios", [])
    themes = r.get("themes", [])
    demographics = r.get("demographics", [])
    return {
        "title": r.get("title_english") or r["title"],
        "score": r.get("score"),
        "members": r.get("members"),
        "episodes": r.get("episodes"),
        "type": r.get("type"),
        "year": r.get("year"),
        "studio": studios[0]["name"] if studios else None,
        "genres": [g["name"] for g in r.get("genres", [])],
        "themes": [t["name"] for t in themes],
        "demographics": [d["name"] for d in demographics],
        "synopsis": (r.get("synopsis") or "")[:300],
    }


def _simplify_tmdb_entry(
    r: dict[str, Any], is_tv: bool = False
) -> dict[str, Any]:
    title = r.get("name" if is_tv else "title") or r.get(
        "original_name" if is_tv else "original_title", ""
    )
    date_field = "first_air_date" if is_tv else "release_date"
    date_val = r.get(date_field) or ""
    return {
        "title": title,
        "score": r.get("vote_average"),
        "popularity": r.get("popularity"),
        "year": int(date_val[:4]) if date_val else None,
        "genres": [str(gid) for gid in r.get("genre_ids", [])],
        "synopsis": (r.get("overview") or "")[:300],
    }


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

_SPEED_INSTRUCTIONS = (
    "IMPORTANT: Be FAST — complete in exactly 2 steps:\n"
    "1. FIRST response: call ALL tools at once — genre/discover "
    "searches AND review searches together in parallel.\n"
    "2. SECOND response: return the final JSON immediately.\n"
    "NEVER make a third round of tool calls.\n\n"
)

_QUALITY_INSTRUCTIONS = (
    "A great recommendation set has:\n"
    "- Strong thematic/tonal fit (not just genre match)\n"
    "- Mix of well-known and lesser-known titles\n"
    "- Variety in era and format\n"
    "- No duplicate franchises\n"
)


def _anime_system() -> str:
    return (
        "You are an expert anime curator. You understand nuance "
        "in preferences — 'dark fantasy' means Berserk-style "
        "grimdark vs Made in Abyss-style wonder-with-darkness.\n\n"
        + _SPEED_INSTRUCTIONS
        + _QUALITY_INSTRUCTIONS
        + "- Titles you're confident exist on MyAnimeList"
    )


def _movie_system() -> str:
    return (
        "You are an expert film curator. You know cinema deeply "
        "— from Kurosawa to Villeneuve, from arthouse to "
        "blockbusters.\n\n"
        + _SPEED_INSTRUCTIONS
        + _QUALITY_INSTRUCTIONS
        + "- Titles you're confident exist on TMDB"
    )


def _series_system() -> str:
    return (
        "You are an expert TV series curator. You know the "
        "golden age of TV — from The Wire to Breaking Bad, "
        "from limited series to long-running epics.\n\n"
        + _SPEED_INSTRUCTIONS
        + _QUALITY_INSTRUCTIONS
        + "- Titles you're confident exist on TMDB"
    )


def _cartoon_system() -> str:
    return (
        "You are an expert animation curator specializing in "
        "non-anime animated content — Western cartoons, adult "
        "animation, kids shows, and animated films made for TV. "
        "From Avatar: The Last Airbender to Rick and Morty to "
        "classic Disney Channel shows.\n\n"
        + _SPEED_INSTRUCTIONS
        + _QUALITY_INSTRUCTIONS
        + "- Titles you're confident exist on TMDB\n"
        "- Focus on non-Japanese animation only"
    )


def _anime_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "search_by_genre",
            "description": (
                "Search MyAnimeList for anime by genre, sorted "
                "by popularity. Returns top titles with scores, "
                "studios, year, themes, synopsis. Use AND logic "
                "— 1-2 genres per call. Call 2-3 times in ONE "
                "response with different combos."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "genres": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "1-2 genre names. Supported: action, "
                            "adventure, comedy, drama, fantasy, "
                            "historical, horror, isekai, mecha, "
                            "mystery, psychological, romance, "
                            "sci-fi, seinen, shounen, slice of "
                            "life, sports, supernatural, thriller."
                        ),
                    }
                },
                "required": ["genres"],
            },
        },
        REVIEW_TOOL,
    ]


def _movie_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "search_by_genre",
            "description": (
                "Discover movies on TMDB by genre, sorted by "
                "popularity. Returns titles with scores, year, "
                "synopsis. Use 1-2 genres per call. Call 2-3 "
                "times in ONE response with different combos."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "genres": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "1-2 genre names. Supported: action, "
                            "adventure, animation, comedy, crime, "
                            "documentary, drama, family, fantasy, "
                            "history, horror, music, mystery, "
                            "romance, sci-fi, thriller, war, "
                            "western."
                        ),
                    }
                },
                "required": ["genres"],
            },
        },
        REVIEW_TOOL,
    ]


def _series_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "search_by_genre",
            "description": (
                "Discover TV series on TMDB by genre, sorted by "
                "popularity. Returns titles with scores, year, "
                "synopsis. Use 1-2 genres per call. Call 2-3 "
                "times in ONE response with different combos."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "genres": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "1-2 genre names. Supported: action, "
                            "adventure, animation, comedy, crime, "
                            "documentary, drama, family, kids, "
                            "mystery, sci-fi, fantasy, war, "
                            "western."
                        ),
                    }
                },
                "required": ["genres"],
            },
        },
        REVIEW_TOOL,
    ]


def _cartoon_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "search_by_genre",
            "description": (
                "Discover animated TV shows on TMDB by genre, "
                "sorted by popularity. All results are "
                "animation. Use 1-2 extra genres per call. "
                "Call 2-3 times in ONE response with different "
                "combos."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "genres": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "1-2 genre names. Supported: action, "
                            "adventure, comedy, crime, drama, "
                            "family, kids, mystery, sci-fi, "
                            "fantasy, war, western."
                        ),
                    }
                },
                "required": ["genres"],
            },
        },
        REVIEW_TOOL,
    ]


# ---------------------------------------------------------------------------
# Tool execution per category
# ---------------------------------------------------------------------------


async def _run_reviews(query: str) -> str:
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: tavily.search(
            query, max_results=3, search_depth="basic"
        ),
    )
    snippets = [
        r.get("content", "") for r in result.get("results", [])
    ]
    return "\n\n".join(snippets)


async def _run_anime_tool(
    name: str, inputs: dict[str, Any]
) -> str:
    if name == "search_by_genre":
        results = await search_anime_by_genres(inputs["genres"])
        return json.dumps(
            [_simplify_anime_entry(r) for r in results]
        )
    if name == "search_reviews":
        return await _run_reviews(inputs["query"])
    return json.dumps({"error": "Unknown tool"})


async def _run_movie_tool(
    name: str, inputs: dict[str, Any]
) -> str:
    if name == "search_by_genre":
        results = await search_movies_by_genres(inputs["genres"])
        return json.dumps(
            [_simplify_tmdb_entry(r) for r in results]
        )
    if name == "search_reviews":
        return await _run_reviews(inputs["query"])
    return json.dumps({"error": "Unknown tool"})


async def _run_series_tool(
    name: str, inputs: dict[str, Any]
) -> str:
    if name == "search_by_genre":
        results = await search_series_by_genres(inputs["genres"])
        return json.dumps(
            [_simplify_tmdb_entry(r, is_tv=True) for r in results]
        )
    if name == "search_reviews":
        return await _run_reviews(inputs["query"])
    return json.dumps({"error": "Unknown tool"})


async def _run_cartoon_tool(
    name: str, inputs: dict[str, Any]
) -> str:
    if name == "search_by_genre":
        results = await search_cartoons_by_genres(inputs["genres"])
        return json.dumps(
            [_simplify_tmdb_entry(r, is_tv=True) for r in results]
        )
    if name == "search_reviews":
        return await _run_reviews(inputs["query"])
    return json.dumps({"error": "Unknown tool"})


# ---------------------------------------------------------------------------
# "Any" mixed-category agent
# ---------------------------------------------------------------------------


def _any_system() -> str:
    return (
        "You are an expert entertainment curator who knows "
        "anime, movies, TV series, and cartoons equally well. "
        "Recommend a BALANCED MIX across all four categories.\n\n"
        + _SPEED_INSTRUCTIONS
        + _QUALITY_INSTRUCTIONS
        + "- Include items from at least 3 of: anime, movies, "
        "series, cartoons\n"
        "- Titles must exist on MyAnimeList (anime) or "
        "TMDB (movies/series/cartoons)"
    )


def _any_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "search_anime_by_genre",
            "description": (
                "Search MyAnimeList for anime by genre, sorted "
                "by popularity. Use 1-2 genres per call."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "genres": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "1-2 genre names. Supported: action, "
                            "adventure, comedy, drama, fantasy, "
                            "historical, horror, isekai, mecha, "
                            "mystery, psychological, romance, "
                            "sci-fi, seinen, shounen, slice of "
                            "life, sports, supernatural, thriller."
                        ),
                    }
                },
                "required": ["genres"],
            },
        },
        {
            "name": "search_movies_by_genre",
            "description": (
                "Discover movies on TMDB by genre, sorted by "
                "popularity. Use 1-2 genres per call."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "genres": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "1-2 genre names. Supported: action, "
                            "adventure, animation, comedy, crime, "
                            "documentary, drama, family, fantasy, "
                            "history, horror, music, mystery, "
                            "romance, sci-fi, thriller, war, "
                            "western."
                        ),
                    }
                },
                "required": ["genres"],
            },
        },
        {
            "name": "search_series_by_genre",
            "description": (
                "Discover TV series on TMDB by genre, sorted by "
                "popularity. Use 1-2 genres per call."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "genres": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "1-2 genre names. Supported: action, "
                            "adventure, animation, comedy, crime, "
                            "documentary, drama, family, kids, "
                            "mystery, sci-fi, fantasy, war, "
                            "western."
                        ),
                    }
                },
                "required": ["genres"],
            },
        },
        {
            "name": "search_cartoons_by_genre",
            "description": (
                "Discover animated TV shows on TMDB by genre. "
                "All results are animation. Use 1-2 extra "
                "genres per call."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "genres": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "1-2 genre names. Supported: action, "
                            "adventure, comedy, crime, drama, "
                            "family, kids, mystery, sci-fi, "
                            "fantasy, war, western."
                        ),
                    }
                },
                "required": ["genres"],
            },
        },
        REVIEW_TOOL,
    ]


async def _run_any_tool(
    name: str, inputs: dict[str, Any]
) -> str:
    if name == "search_anime_by_genre":
        results = await search_anime_by_genres(inputs["genres"])
        return json.dumps(
            [_simplify_anime_entry(r) for r in results]
        )
    if name == "search_movies_by_genre":
        results = await search_movies_by_genres(inputs["genres"])
        return json.dumps(
            [_simplify_tmdb_entry(r) for r in results]
        )
    if name == "search_series_by_genre":
        results = await search_series_by_genres(inputs["genres"])
        return json.dumps(
            [_simplify_tmdb_entry(r, is_tv=True) for r in results]
        )
    if name == "search_cartoons_by_genre":
        results = await search_cartoons_by_genres(inputs["genres"])
        return json.dumps(
            [_simplify_tmdb_entry(r, is_tv=True) for r in results]
        )
    if name == "search_reviews":
        return await _run_reviews(inputs["query"])
    return json.dumps({"error": "Unknown tool"})


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

RunToolFn = Callable[
    [str, dict[str, Any]], Coroutine[Any, Any, str]
]

AGENTS: dict[str, dict[str, Any]] = {
    "anime": {
        "system_prompt": _anime_system,
        "tools": _anime_tools,
        "run_tool": _run_anime_tool,
        "label": "anime",
        "db": "MyAnimeList",
    },
    "movie": {
        "system_prompt": _movie_system,
        "tools": _movie_tools,
        "run_tool": _run_movie_tool,
        "label": "movies",
        "db": "TMDB",
    },
    "series": {
        "system_prompt": _series_system,
        "tools": _series_tools,
        "run_tool": _run_series_tool,
        "label": "TV series",
        "db": "TMDB",
    },
    "cartoon": {
        "system_prompt": _cartoon_system,
        "tools": _cartoon_tools,
        "run_tool": _run_cartoon_tool,
        "label": "animated shows",
        "db": "TMDB",
    },
    "any": {
        "system_prompt": _any_system,
        "tools": _any_tools,
        "run_tool": _run_any_tool,
        "label": "entertainment",
        "db": "MyAnimeList/TMDB",
    },
}


async def _search_item(
    title: str, category: str
) -> ContentItem | None:
    if category == "anime":
        return await search_anime(title)
    if category == "movie":
        return await search_movie(title)
    if category == "series":
        return await search_series(title)
    if category == "cartoon":
        return await search_cartoon(title)
    return None

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _creativity_instruction(creativity: float) -> str:
    if creativity < 0.25:
        return (
            "SELECTION STYLE: Be SAFE and PREDICTABLE. "
            "Stick to the most popular, highly-rated, "
            "universally praised titles. Choose mainstream "
            "crowd-pleasers that almost everyone enjoys.\n\n"
        )
    if creativity < 0.45:
        return (
            "SELECTION STYLE: Lean toward well-known, "
            "reliable picks with a few interesting choices. "
            "Mostly popular titles with 1-2 slightly less "
            "obvious selections.\n\n"
        )
    if creativity < 0.65:
        return ""
    if creativity < 0.85:
        return (
            "SELECTION STYLE: Be ADVENTUROUS. Mix popular "
            "titles with hidden gems, cult favorites, and "
            "underrated picks. Surprise the user with "
            "unexpected cross-genre recommendations.\n\n"
        )
    return (
        "SELECTION STYLE: Be BOLD and UNEXPECTED. "
        "Prioritize hidden gems, cult classics, obscure "
        "picks, and daring cross-genre choices. Avoid "
        "obvious mainstream titles — dig deep into the "
        "catalog for surprising, unique recommendations.\n\n"
    )


def _build_user_prompt(
    preferences: str,
    mode: str,
    category: str,
    creativity: float = 0.5,
    count: int = 9,
) -> str:
    agent = AGENTS.get(category, AGENTS["anime"])
    label = agent["label"]
    db = agent["db"]
    creativity_hint = _creativity_instruction(creativity)
    fetch_count = count + 5

    category_field = ""
    if category == "any":
        category_field = (
            '"category": "anime"|"movie"|"series"|"cartoon", '
        )

    tail = (
        "After seeing results, IMMEDIATELY return the "
        "final JSON — no more tool calls.\n"
        "Combine results with your own knowledge. "
        "If the input is a title name, include it first.\n"
        f"Return ONLY a JSON array of exactly {fetch_count} objects:\n"
        "[{" + category_field + '"title": "Full Title", '
        '"reason": "why this fits"}, ...]\n'
    )

    if category == "any":
        tool_instruction = (
            "In THIS response, call ALL tools at once:\n"
            "- Use a mix of search_anime_by_genre, "
            "search_movies_by_genre, search_series_by_genre, "
            "and search_cartoons_by_genre — 3-4 calls total "
            "with genre combos matching the user's taste\n"
            "- search_reviews 1-2 times for titles relevant "
            "to the user's taste\n\n"
        )
    else:
        tool_instruction = (
            "In THIS response, call ALL tools at once:\n"
            "- search_by_genre 2-3 times with different "
            "genre combos\n"
            "- search_reviews 1-2 times for titles relevant "
            "to the user's taste\n\n"
        )

    if mode == "trending":
        return (
            f"Find {fetch_count} trending and popular {label} that match: "
            f'"{preferences}"\n\n'
            + creativity_hint
            + tool_instruction
            + "PRIORITIZE: recent, high popularity, and "
            "widely talked-about titles. "
            "Favor what's hot RIGHT NOW.\n\n"
            + tail
        )

    return (
        f"Find {fetch_count} {label} for these preferences: "
        f'"{preferences}"\n\n'
        + creativity_hint
        + tool_instruction
        + "PRIORITIZE: deep thematic/tonal match to the "
        "user's description. Pick titles that truly fit "
        "what they're asking for, regardless of popularity "
        "or recency.\n\n"
        f"Only recommend titles that exist on {db}.\n\n"
        + tail
    )


# ---------------------------------------------------------------------------
# Core recommendation loop
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> list[dict[str, Any]]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse JSON from Claude response"
            )
    return []


async def get_recommendations(
    preferences: str,
    mode: str = "personalized",
    category: str = "anime",
    creativity: float = 0.5,
    count: int = 9,
) -> list[ContentItem]:
    agent = AGENTS.get(category, AGENTS["anime"])
    system_prompt: str = agent["system_prompt"]()
    tools: list[dict[str, Any]] = agent["tools"]()
    run_tool: RunToolFn = agent["run_tool"]

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": _build_user_prompt(
                preferences, mode, category, creativity, count
            ),
        }
    ]

    picks: list[dict[str, Any]] = []

    for _round in range(MAX_TOOL_ROUNDS):
        response = await claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system_prompt,
            tools=tools,
            messages=messages,
            temperature=creativity,
        )

        messages.append(
            {"role": "assistant", "content": response.content}
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    picks = _extract_json(block.text)
                    break
            break

        if response.stop_reason == "tool_use":
            tool_blocks = [
                b
                for b in response.content
                if b.type == "tool_use"
            ]

            async def _safe_tool(
                name: str, inputs: dict[str, Any]
            ) -> str:
                try:
                    return await run_tool(name, inputs)
                except Exception:
                    logger.exception("Tool %s failed", name)
                    return json.dumps(
                        {"error": f"Tool '{name}' failed"}
                    )

            tool_results = await asyncio.gather(
                *[
                    _safe_tool(b.name, b.input)
                    for b in tool_blocks
                ]
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": b.id,
                            "content": result,
                        }
                        for b, result in zip(
                            tool_blocks, tool_results
                        )
                    ],
                }
            )
    else:
        logger.warning(
            "Tool loop hit max rounds (%d) for: %s",
            MAX_TOOL_ROUNDS,
            preferences,
        )

    if not picks:
        return []

    is_any = category == "any"

    async def _safe_search(
        pick: dict[str, Any],
    ) -> ContentItem | None:
        try:
            cat = pick.get("category", category) if is_any else category
            return await _search_item(pick["title"], cat)
        except Exception:
            logger.warning("Lookup failed for: %s", pick.get("title"))
            return None

    items = await asyncio.gather(
        *[_safe_search(p) for p in picks]
    )

    seen_ids: set[int] = set()
    result: list[ContentItem] = []
    for pick, item in zip(picks, items):
        if item is None or item.id in seen_ids:
            continue
        seen_ids.add(item.id)
        item.reason = pick.get("reason", "")
        if is_any:
            item.category = pick.get("category", "")
        result.append(item)

    return result


# ---------------------------------------------------------------------------
# Explain stream
# ---------------------------------------------------------------------------

_EXPLAIN_SYSTEM: dict[str, str] = {
    "anime": (
        "You are a knowledgeable anime critic who gives "
        "specific, insightful analysis. Reference concrete "
        "plot elements, characters, or themes — never be "
        "generic."
    ),
    "movie": (
        "You are a knowledgeable film critic who gives "
        "specific, insightful analysis. Reference concrete "
        "plot elements, performances, directing choices, or "
        "themes — never be generic."
    ),
    "series": (
        "You are a knowledgeable TV critic who gives "
        "specific, insightful analysis. Reference concrete "
        "plot arcs, character development, writing quality, "
        "or themes — never be generic."
    ),
    "cartoon": (
        "You are a knowledgeable animation critic who gives "
        "specific, insightful analysis. Reference concrete "
        "art style, humor, storytelling, or themes — never "
        "be generic."
    ),
    "any": (
        "You are a knowledgeable entertainment critic who "
        "covers anime, film, TV, and animation. Give "
        "specific, insightful analysis referencing concrete "
        "elements — never be generic."
    ),
}


async def explain_recommendation_stream(
    preference: str,
    title: str,
    synopsis: str,
    genres: list[str],
    category: str = "anime",
) -> AsyncGenerator[str, None]:
    system = _EXPLAIN_SYSTEM.get(
        category, _EXPLAIN_SYSTEM["anime"]
    )
    label = AGENTS.get(category, AGENTS["anime"])["label"]

    async with claude.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=system,
        messages=[
            {
                "role": "user",
                "content": (
                    f'User preferences: "{preference}"\n'
                    f"{label.title()}: {title}\n"
                    f"Genres: {', '.join(genres)}\n"
                    f"Synopsis: {synopsis[:500]}\n\n"
                    "Write a short match analysis in this "
                    "format:\n"
                    "**Why it matches:** one specific sentence\n"
                    "**Tone & style:** one sentence\n"
                    "**Best for you if:** one sentence\n\n"
                    "Be specific — mention characters, themes, "
                    "or plot elements. No intro, no outro."
                ),
            }
        ],
    ) as stream:
        async for text in stream.text_stream:
            yield text
