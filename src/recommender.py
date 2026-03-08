import asyncio
import json
import re
from typing import Any

import anthropic
from tavily import TavilyClient

from .config import settings
from .jikan import search_anime, search_anime_by_genres
from .models import Anime

claude = anthropic.Anthropic(api_key=settings.anthropic_api_key)
tavily = TavilyClient(api_key=settings.tavily_api_key)

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_anime_by_genre",
        "description": (
            "Search MyAnimeList for top-rated anime by genre. "
            "Returns up to 20 results with title, score, episodes, genres, and synopsis. "
            "Call this first to discover candidates that match the user's mood."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "genres": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Genre names to filter by, e.g. ['action', 'fantasy', 'drama']. "
                        "Supported: action, adventure, comedy, demons, drama, fantasy, historical, "
                        "horror, magic, mecha, mystery, psychological, romance, samurai, school, "
                        "sci-fi, slice of life, space, sports, supernatural, thriller, vampire."
                    ),
                }
            },
            "required": ["genres"],
        },
    },
    {
        "name": "search_reviews",
        "description": (
            "Search the internet for reviews, ratings, and audience opinions about a specific anime. "
            "Use this to evaluate how well-received a candidate is before recommending it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, e.g. 'Jujutsu Kaisen anime review Reddit 2024'",
                }
            },
            "required": ["query"],
        },
    },
]


async def _run_tool(name: str, inputs: dict[str, Any]) -> str:
    if name == "search_anime_by_genre":
        results = await search_anime_by_genres(inputs["genres"])
        simplified = [
            {
                "title": r["title"],
                "score": r.get("score"),
                "episodes": r.get("episodes"),
                "genres": [g["name"] for g in r.get("genres", [])],
                "synopsis": (r.get("synopsis") or "")[:300],
            }
            for r in results
        ]
        return json.dumps(simplified)

    if name == "search_reviews":
        result = tavily.search(inputs["query"], max_results=3, search_depth="basic")
        snippets = [r.get("content", "") for r in result.get("results", [])]
        return "\n\n".join(snippets)

    return "Unknown tool"


def _extract_json(text: str) -> list[dict[str, Any]]:
    """Pull a JSON array out of Claude's response even if surrounded by prose."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return []


async def get_recommendations(preferences: str) -> list[Anime]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f'Find the best 9 anime for someone with these preferences: "{preferences}"\n\n'
                "Follow these steps:\n"
                "1. Call search_anime_by_genre with relevant genres to discover candidates.\n"
                "2. Call search_reviews for the most promising candidates to read real audience opinions.\n"
                "3. If the input looks like a title name (even misspelled), include that anime first.\n"
                "4. Return ONLY a JSON array of exactly 9 objects — no explanation outside the array:\n"
                '   [{"title": "Full Correct Title", "reason": "one sentence why this fits"}, ...]\n'
            ),
        }
    ]

    # Agentic loop — runs until Claude stops calling tools
    while True:
        response = claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            picks: list[dict[str, Any]] = []
            for block in response.content:
                if block.type == "text":
                    picks = _extract_json(block.text)
                    break
            break

        if response.stop_reason == "tool_use":
            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            tool_results = await asyncio.gather(
                *[_run_tool(b.name, b.input) for b in tool_blocks]
            )
            messages.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "content": result}
                    for b, result in zip(tool_blocks, tool_results)
                ],
            })

    if not picks:
        return []

    # Enrich Claude's picks with full Jikan metadata
    anime_list = await asyncio.gather(*[search_anime(p["title"]) for p in picks])

    result = []
    for pick, anime in zip(picks, anime_list):
        if anime is not None:
            anime.reason = pick.get("reason", "")
            result.append(anime)

    return result
