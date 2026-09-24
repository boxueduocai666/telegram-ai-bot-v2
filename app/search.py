from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from ddgs import DDGS

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
    """Web search with per-backend fallbacks for hosted environments.

    DDGS supports several providers. They can fail independently on a cloud
    provider because of rate limits, regional/network restrictions, or a single
    provider changing its response. Calling them one-by-one prevents one failed
    backend from discarding results from another backend.
    """

    BACKENDS = ("bing", "duckduckgo", "brave", "google", "wikipedia")

    def __init__(
        self,
        max_results: int = 5,
        searxng_url: str | None = None,
        *,
        region: str = "wt-wt",
        backend_timeout: int = 8,
    ) -> None:
        self.max_results = max(1, min(int(max_results), 10))
        self.searxng_url = (searxng_url or "").rstrip("/") or None
        self.region = (region or "wt-wt").strip() or "wt-wt"
        self.backend_timeout = max(3, int(backend_timeout))

    async def search(self, query: str) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []

        # First let DDGS choose a healthy backend. If that aggregate request
        # fails, retry providers independently so one broken provider cannot
        # poison the whole search.
        try:
            results = await asyncio.to_thread(self._ddgs_auto_search, query)
            if results:
                return results
        except Exception:
            logger.exception("DDGS auto search failed")

        for backend in self.BACKENDS:
            try:
                results = await asyncio.to_thread(self._ddgs_backend_search, query, backend)
                if results:
                    logger.info("Search succeeded with DDGS backend=%s", backend)
                    return results
            except Exception:
                logger.warning("DDGS backend failed: %s", backend, exc_info=True)

        if self.searxng_url:
            try:
                results = await self._searxng_search(query)
                if results:
                    return results
            except Exception:
                logger.exception("SearXNG fallback failed")

        return []

    def _ddgs_auto_search(self, query: str) -> list[SearchResult]:
        client = DDGS(timeout=self.backend_timeout)
        results = client.text(
            query,
            region=self.region,
            max_results=self.max_results,
            backend="auto",
        )
        return self._normalize_results(results, source="DDGS")

    def _ddgs_backend_search(self, query: str, backend: str) -> list[SearchResult]:
        client = DDGS(timeout=self.backend_timeout)
        results = client.text(
            query,
            region=self.region,
            max_results=self.max_results,
            backend=backend,
        )
        return self._normalize_results(results, source=f"DDGS/{backend}")

    def _normalize_results(self, results: Any, *, source: str) -> list[SearchResult]:
        output: list[SearchResult] = []
        seen_urls: set[str] = set()
        for item in results or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("href") or item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("body") or item.get("snippet") or "").strip()
            if not url or not (title or snippet) or url in seen_urls:
                continue
            seen_urls.add(url)
            output.append(SearchResult(title, url, snippet, source))
            if len(output) >= self.max_results:
                break
        return output

    async def _searxng_search(self, query: str) -> list[SearchResult]:
        assert self.searxng_url
        async with httpx.AsyncClient(
            timeout=self.backend_timeout,
            follow_redirects=True,
            headers={"User-Agent": "telegram-ai-bot-v2/2.0"},
        ) as client:
            response = await client.get(
                f"{self.searxng_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "language": "zh-CN",
                },
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

        raw_results = data.get("results")
        if not isinstance(raw_results, list):
            return []
        output: list[SearchResult] = []
        seen_urls: set[str] = set()
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            snippet = str(item.get("content") or item.get("snippet") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            output.append(SearchResult(title, url, snippet, "SearXNG"))
            if len(output) >= self.max_results:
                break
        return output


def format_search_results(results: list[SearchResult], max_chars: int = 12000) -> str:
    if not results:
        return ""
    chunks: list[str] = []
    for index, result in enumerate(results, start=1):
        chunks.append(
            f"[{index}] {result.title}\n"
            f"URL: {result.url}\n"
            f"摘要: {result.snippet}"
        )
    return "\n\n".join(chunks)[:max_chars]
