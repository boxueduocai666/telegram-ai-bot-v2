from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from ddgs import DDGS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageSearchResult:
    title: str
    image_url: str


class ImageSearchError(RuntimeError):
    """Raised when image search fails."""


class ImageSearchService:
    def __init__(self, max_results: int = 5) -> None:
        self.max_results = max(1, min(max_results, 5))

    async def search(self, query: str) -> list[ImageSearchResult]:
        query = query.strip()
        if not query:
            return []
        try:
            return await asyncio.to_thread(self._search_sync, query)
        except Exception as exc:
            logger.exception("DDGS image search failed")
            raise ImageSearchError("图片搜索暂时不可用。") from exc

    def _search_sync(self, query: str) -> list[ImageSearchResult]:
        results = DDGS().images(query, max_results=self.max_results)
        output: list[ImageSearchResult] = []
        seen: set[str] = set()

        for item in results or []:
            if not isinstance(item, dict):
                continue
            image_url = str(item.get("image") or item.get("thumbnail") or "").strip()
            parsed = urlparse(image_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                continue
            if image_url in seen:
                continue
            seen.add(image_url)

            title = str(item.get("title") or "图片").strip()
            title = " ".join(title.split())[:200] or "图片"
            output.append(ImageSearchResult(title=title, image_url=image_url))
            if len(output) >= self.max_results:
                break

        return output
