from __future__ import annotations

import asyncio
import html
import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import httpx
from ddgs import DDGS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = "DDGS"
    page_content: str = ""


class SearchError(RuntimeError):
    pass


class _PageTextParser(HTMLParser):
    """Small dependency-free HTML -> readable text extractor.

    This intentionally avoids trying to reproduce a full browser. It removes
    navigation/script/style noise and prefers article/main content when the
    page exposes it.
    """

    SKIP_TAGS = {
        "script", "style", "noscript", "svg", "canvas", "iframe",
        "nav", "footer", "form", "button", "template",
    }
    BLOCK_TAGS = {
        "article", "main", "section", "p", "div", "li", "h1", "h2", "h3",
        "h4", "h5", "h6", "blockquote", "pre", "td", "th", "br",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.title_parts: list[str] = []
        self.in_title = False
        self.main_depth = 0
        self.article_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = True
        if tag == "main":
            self.main_depth += 1
        if tag == "article":
            self.article_depth += 1
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            if self.skip_depth:
                self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = False
        if tag == "main":
            self.main_depth = max(0, self.main_depth - 1)
        if tag == "article":
            self.article_depth = max(0, self.article_depth - 1)
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        text = html.unescape(data)
        if self.in_title:
            self.title_parts.append(text)
        if text.strip():
            self.parts.append(text)

    @property
    def title(self) -> str:
        return re.sub(r"\s+", " ", "".join(self.title_parts)).strip()

    @property
    def text(self) -> str:
        text = "".join(self.parts)
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()


class SearchService:
    """Web search + lightweight webpage reading for hosted environments.

    Search results are still obtained through DDGS/SearXNG, but important
    result pages are then fetched and converted into readable text before the
    AI receives them. A failed page fetch never breaks the whole search.
    """

    BACKENDS = ("bing", "duckduckgo", "brave", "google", "wikipedia")

    def __init__(
        self,
        max_results: int = 5,
        searxng_url: str | None = None,
        *,
        region: str = "wt-wt",
        backend_timeout: int = 8,
        page_timeout: int = 12,
        max_page_chars: int = 4500,
        pages_to_read: int = 3,
    ) -> None:
        self.max_results = max(1, min(int(max_results), 10))
        self.searxng_url = (searxng_url or "").rstrip("/") or None
        self.region = (region or "wt-wt").strip() or "wt-wt"
        self.backend_timeout = max(3, int(backend_timeout))
        self.page_timeout = max(5, int(page_timeout))
        self.max_page_chars = max(1000, int(max_page_chars))
        self.pages_to_read = max(0, min(int(pages_to_read), self.max_results))

    async def search(self, query: str) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []

        # If the user supplied a URL directly, read that page first.
        if self._looks_like_url(query):
            direct = await self.read_url(query)
            if direct:
                return [direct]

        results: list[SearchResult] = []

        try:
            results = await asyncio.to_thread(self._ddgs_auto_search, query)
        except Exception:
            logger.exception("DDGS auto search failed")

        if not results:
            for backend in self.BACKENDS:
                try:
                    results = await asyncio.to_thread(
                        self._ddgs_backend_search, query, backend
                    )
                    if results:
                        logger.info("Search succeeded with DDGS backend=%s", backend)
                        break
                except Exception:
                    logger.warning(
                        "DDGS backend failed: %s", backend, exc_info=True
                    )

        if not results and self.searxng_url:
            try:
                results = await self._searxng_search(query)
            except Exception:
                logger.exception("SearXNG fallback failed")

        if not results:
            return []

        # Search result snippets are useful as a fallback, while the first few
        # pages provide actual page text for factual verification/summarization.
        return await self._enrich_results(results)

    async def _enrich_results(self, results: list[SearchResult]) -> list[SearchResult]:
        if self.pages_to_read <= 0:
            return results

        candidates = results[: self.pages_to_read]
        tasks = [self._read_result_page(result) for result in candidates]
        enriched = await asyncio.gather(*tasks, return_exceptions=True)

        output: list[SearchResult] = []
        for index, result in enumerate(results):
            if index < len(enriched):
                item = enriched[index]
                if isinstance(item, SearchResult):
                    output.append(item)
                    continue
            output.append(result)
        return output

    async def _read_result_page(self, result: SearchResult) -> SearchResult:
        try:
            page = await self.read_url(result.url)
            if page and page.page_content:
                return SearchResult(
                    title=page.title or result.title,
                    url=result.url,
                    snippet=result.snippet,
                    source=result.source,
                    page_content=page.page_content,
                )
        except Exception:
            logger.info("Unable to read search result page: %s", result.url, exc_info=True)
        return result

    async def read_url(self, url: str) -> SearchResult | None:
        url = url.strip()
        if not self._is_safe_public_url(url):
            logger.warning("Rejected non-public URL: %s", url)
            return None

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; Telegram-AI-Bot-V2/2.0; "
                "+https://github.com/boxueduocai666/telegram-ai-bot-v2)"
            ),
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.7",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
        }

        async with httpx.AsyncClient(
            timeout=self.page_timeout,
            follow_redirects=True,
            max_redirects=5,
            headers=headers,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if content_type and not any(
                kind in content_type
                for kind in ("text/html", "application/xhtml+xml", "text/plain")
            ):
                return None

            raw = response.content
            if len(raw) > 2_500_000:
                raw = raw[:2_500_000]

            encoding = response.encoding or "utf-8"
            text = raw.decode(encoding, errors="replace")

        parser = _PageTextParser()
        parser.feed(text)
        parser.close()

        page_text = self._clean_page_text(parser.text)
        if not page_text:
            return None

        return SearchResult(
            title=parser.title or urlparse(url).netloc,
            url=url,
            snippet=page_text[:500],
            source="网页阅读",
            page_content=page_text[: self.max_page_chars],
        )

    @staticmethod
    def _clean_page_text(text: str) -> str:
        lines: list[str] = []
        for line in text.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue
            # Avoid sending giant cookie/JS/config blobs to the model.
            if len(line) > 1200:
                line = line[:1200] + "…"
            lines.append(line)

        # Deduplicate adjacent repeated lines commonly produced by menus.
        compact: list[str] = []
        previous = ""
        for line in lines:
            if line == previous:
                continue
            compact.append(line)
            previous = line

        return "\n".join(compact)

    @staticmethod
    def _looks_like_url(value: str) -> bool:
        return bool(re.match(r"^https?://", value, flags=re.IGNORECASE))

    @staticmethod
    def _is_safe_public_url(url: str) -> bool:
        try:
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                return False

            host = parsed.hostname.strip().lower()
            if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
                return False

            # Prevent the bot from becoming a simple SSRF proxy when a URL is
            # supplied directly by a Telegram user.
            try:
                addresses = {
                    info[4][0]
                    for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
                }
            except socket.gaierror:
                return False

            for address in addresses:
                ip = ipaddress.ip_address(address)
                if (
                    ip.is_private
                    or ip.is_loopback
                    or ip.is_link_local
                    or ip.is_reserved
                    or ip.is_multicast
                    or ip.is_unspecified
                ):
                    return False
            return True
        except Exception:
            return False

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


def format_search_results(results: list[SearchResult], max_chars: int = 16000) -> str:
    """Format search results plus readable webpage text for the AI.

    The AI is explicitly told which text came from a page, so it can summarize
    the page instead of treating a search-engine snippet as the whole source.
    """
    if not results:
        return ""

    chunks: list[str] = []
    for index, result in enumerate(results, start=1):
        chunk = (
            f"[{index}] {result.title}\n"
            f"URL: {result.url}\n"
            f"搜索摘要: {result.snippet}"
        )
        if result.page_content:
            chunk += (
                "\n网页正文（已抓取，可能因网站结构而不完整）：\n"
                f"{result.page_content}"
            )
        else:
            chunk += "\n网页正文：未能抓取，不能据此推断该网页的完整内容。"
        chunks.append(chunk)

    return "\n\n".join(chunks)[:max_chars]
