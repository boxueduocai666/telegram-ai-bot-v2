from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from ddgs import DDGS

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = "DDGS"


class SearchError(RuntimeError):
    pass


class SearchService:
    def __init__(self, max_results: int = 5, searxng_url: str | None = None) -> None:
        self.max_results = max(1, max_results)
        self.searxng_url = (searxng_url or "").rstrip("/") or None

    async def search(self, query: str) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []
        try:
            raw = await asyncio.to_thread(self._ddgs_search, query)
            if raw:
                return raw
        except Exception:
            logger.exception("DDGS search failed")

        if self.searxng_url:
            try:
                fallback = await self._searxng_search(query)
                if fallback:
                    return fallback
            except Exception:
                logger.exception("SearXNG fallback failed")

        return []

    def _ddgs_search(self, query: str) -> list[SearchResult]:
        results = DDGS().text(query, max_results=self.max_results)
        output: list[SearchResult] = []
        for item in results or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("href") or item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("body") or item.get("snippet") or "").strip()
            if url and (title or snippet):
                output.append(SearchResult(title, url, snippet, "DDGS"))
        return output[: self.max_results]

    async def _searxng_search(self, query: str) -> list[SearchResult]:
        assert self.searxng_url
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{self.searxng_url}/search",
                params={"q": query, "format": "json", "language": "zh-CN"},
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        output = []
        for item in data.get("results", [])[: self.max_results]:
            output.append(
                SearchResult(
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    snippet=str(item.get("content", "")),
                    source="SearXNG",
                )
            )
        return [r for r in output if r.url]


def format_search_results(results: list[SearchResult], max_chars: int = 12000) -> str:
    if not results:
        return ""
    chunks = []
    for index, result in enumerate(results, start=1):
        chunks.append(f"[{index}] {result.title}\nURL: {result.url}\n摘要: {result.snippet}")
    text = "\n\n".join(chunks)
    return text[:max_chars]
