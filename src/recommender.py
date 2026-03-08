import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

import anthropic
from tavily import TavilyClient

from .config import settings
from .jikan import search_anime, search_anime_by_genres
from .models import Anime

logger = logging.getLogger(__name__)

claude = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
tavily = TavilyClient(api_key=settings.tavily_api_key)

MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = (
    "You are an expert anime curator. You understand nuance in "
    "preferences — 'dark fantasy' means Berserk-style grimdark "
    "vs Made in Abyss-style wonder-with-darkness.\n\n"
    "IMPORTANT: Be FAST — complete in exactly 2 steps:\n"
    "1. FIRST response: call ALL tools at once — genre searches "
    "AND review searches together in parallel.\n"
    "2. SECOND response: return the final JSON immediately.\n"
    "NEVER make a third round of tool calls.\n\n"
    "A great recommendation set has:\n"
    "- Strong thematic/tonal fit (not just genre match)\n"
    "- Mix of well-known and lesser-known titles\n"
    "- Variety in era and format\n"
    "- No duplicate franchises\n"
    "- Titles you're confident exist on MyAnimeList"
)

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_anime_by_genre",
        "description": (
            "Search MyAnimeList for anime by genre, sorted by "
            "popularity. Returns top titles with scores, studios, "
            "year, themes, synopsis. Use AND logic — 1-2 genres "
            "per call. Call 2-3 times in ONE response with "
            "different combos."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "genres": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "1-2 genre names. Supported: "
                        "action, adventure, comedy, drama, "
                        "fantasy, historical, horror, isekai, "
                        "mecha, mystery, psychological, romance, "
                        "sci-fi, seinen, shounen, slice of life, "
                        "sports, supernatural, thriller."
                    ),
                }
            },
            "required": ["genres"],
        },
    },
    {
        "name": "search_reviews",
        "description": (
            "Search the web for audience reviews and opinions "
            "about an anime. Call this IN THE SAME response as "
            "genre searches — do NOT wait for genre results "
            "first. Use for 1-2 titles you already know about "
            "from the user's preferences."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query, e.g. 'Vinland Saga "
                        "anime review reception'"
                    ),
                }
            },
            "required": ["query"],
        },
    },
]


def _simplify_entry(r: dict[str, Any]) -> dict[str, Any]:
    """Extract the most useful fields from a Jikan anime entry."""
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


async def _run_tool(name: str, inputs: dict[str, Any]) -> str:
    try:
        if name == "search_anime_by_genre":
            results = await search_anime_by_genres(inputs["genres"])
            return json.dumps([_simplify_entry(r) for r in results])

        if name == "search_reviews":
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: tavily.search(
                    inputs["query"],
                    max_results=3,
                    search_depth="basic",
                ),
            )
            snippets = [
                r.get("content", "")
                for r in result.get("results", [])
            ]
            return "\n\n".join(snippets)

    except Exception:
        logger.exception("Tool %s failed", name)
        return json.dumps({"error": f"Tool '{name}' failed"})

    return json.dumps({"error": "Unknown tool"})


def _extract_json(text: str) -> list[dict[str, Any]]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from Claude response")
    return []


def _build_user_prompt(preferences: str, mode: str) -> str:
    tail = (
        "After seeing results, IMMEDIATELY return the "
        "final JSON — no more tool calls.\n"
        "Combine results with your own knowledge. "
        "If the input is a title name, include it first.\n"
        "Return ONLY a JSON array of exactly 9 objects:\n"
        '[{"title": "Full Title", '
        '"reason": "why this fits"}, ...]\n'
    )

    if mode == "trending":
        return (
            "Find 9 trending and popular anime that match: "
            f'"{preferences}"\n\n'
            "In THIS response, call ALL tools at once:\n"
            "- search_anime_by_genre 2-3 times with genres "
            "that match the user's taste\n"
            "- search_reviews 1-2 times searching for "
            "'best anime 2024 2025' or 'trending anime "
            "this season'\n\n"
            "PRIORITIZE: currently airing, recent (2023-2025), "
            "high member count, and widely talked-about titles. "
            "Favor what's hot RIGHT NOW over all-time classics.\n\n"
            + tail
        )

    return (
        "Find 9 anime for these preferences: "
        f'"{preferences}"\n\n'
        "In THIS response, call ALL tools at once:\n"
        "- search_anime_by_genre 2-3 times with different "
        "genre combos\n"
        "- search_reviews 1-2 times for titles relevant "
        "to the user's taste\n\n"
        "PRIORITIZE: deep thematic/tonal match to the "
        "user's description. Pick anime that truly fit "
        "what they're asking for, regardless of popularity "
        "or recency.\n\n"
        + tail
    )


async def get_recommendations(
    preferences: str, mode: str = "personalized"
) -> list[Anime]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": _build_user_prompt(preferences, mode),
        }
    ]

    picks: list[dict[str, Any]] = []

    for _round in range(MAX_TOOL_ROUNDS):
        response = await claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    picks = _extract_json(block.text)
                    break
            break

        if response.stop_reason == "tool_use":
            tool_blocks = [
                b for b in response.content if b.type == "tool_use"
            ]
            tool_results = await asyncio.gather(
                *[_run_tool(b.name, b.input) for b in tool_blocks]
            )
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": result,
                    }
                    for b, result in zip(tool_blocks, tool_results)
                ],
            })
    else:
        logger.warning(
            "Tool loop hit max rounds (%d) for: %s",
            MAX_TOOL_ROUNDS,
            preferences,
        )

    if not picks:
        return []

    async def _safe_search(title: str) -> Anime | None:
        try:
            return await search_anime(title)
        except Exception:
            logger.warning("Jikan lookup failed for: %s", title)
            return None

    anime_list = await asyncio.gather(
        *[_safe_search(p["title"]) for p in picks]
    )

    seen_ids: set[int] = set()
    result: list[Anime] = []
    for pick, anime in zip(picks, anime_list):
        if anime is None or anime.id in seen_ids:
            continue
        seen_ids.add(anime.id)
        anime.reason = pick.get("reason", "")
        result.append(anime)

    return result


async def explain_recommendation_stream(
    preference: str, title: str, synopsis: str, genres: list[str]
) -> AsyncGenerator[str, None]:
    async with claude.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=(
            "You are a knowledgeable anime critic who gives "
            "specific, insightful analysis. Reference concrete "
            "plot elements, characters, or themes — never be "
            "generic."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f'User preferences: "{preference}"\n'
                    f"Anime: {title}\n"
                    f"Genres: {', '.join(genres)}\n"
                    f"Synopsis: {synopsis[:500]}\n\n"
                    "Write a short match analysis in this format:\n"
                    "**Why it matches:** one specific sentence\n"
                    "**Tone & style:** one sentence\n"
                    "**Best for you if:** one sentence\n\n"
                    "Be specific — mention characters, themes, or "
                    "plot elements. No intro, no outro."
                ),
            }
        ],
    ) as stream:
        async for text in stream.text_stream:
            yield text
