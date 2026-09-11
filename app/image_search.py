from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ddgs import DDGS

logger = logging.getLogger(__name__)

MAX_IMAGE_RESULTS = 5


@dataclass(frozen=True)
class ImageResult:
    title: str
    image_url: str
    thumbnail_url: str = ""
    page_url: str = ""


class ImageSearchError(RuntimeError):
    """图片搜索失败时抛出，供 handler 转成友好提示。"""


class ImageSearchService:
    """DDGS 图片搜索。只提取 URL，不下载、不上传、不缓存。"""

    def __init__(self, max_results: int = MAX_IMAGE_RESULTS) -> None:
        self.max_results = max(1, min(max_results, 10))

    async def search(self, query: str) -> list[ImageResult]:
        query = (query or "").strip()
        if not query:
            return []

        try:
            return await asyncio.to_thread(self._search_sync, query)
        except ImageSearchError:
            raise
        except Exception:
            logger.exception("DDGS image search failed")
            raise ImageSearchError("图片搜索暂时不可用，请稍后再试。") from None

    def _search_sync(self, query: str) -> list[ImageResult]:
        results = DDGS().images(query, max_results=self.max_results)
        output: list[ImageResult] = []

        for item in results or []:
            if not isinstance(item, dict):
                continue

            image_url = str(item.get("image") or "").strip()
            if not image_url or not image_url.startswith(("http://", "https://")):
                continue

            title = str(item.get("title") or "").strip() or image_url
            thumbnail = str(item.get("thumbnail") or "").strip()
            page_url = str(item.get("url") or "").strip()

            output.append(
                ImageResult(
                    title=title,
                    image_url=image_url,
                    thumbnail_url=thumbnail,
                    page_url=page_url,
                )
            )

            if len(output) >= self.max_results:
                break

        return output
