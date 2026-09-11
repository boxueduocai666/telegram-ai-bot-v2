from __future__ import annotations

import logging
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


class ImageGenerationError(RuntimeError):
    """Raised when the image-generation service cannot return an image URL."""


IMAGE_MODELS = {
    "2": "agnes-image-2.0-flash",
    "2.0": "agnes-image-2.0-flash",
    "image2": "agnes-image-2.0-flash",
    "image2.0": "agnes-image-2.0-flash",
    "2.1": "agnes-image-2.1-flash",
    "image2.1": "agnes-image-2.1-flash",
}


def parse_image_request(args: list[str]) -> tuple[str, str]:
    """Parse /image arguments and return (model, prompt)."""
    parts = [part.strip() for part in args if part.strip()]
    if not parts:
        raise ImageGenerationError("缺少图片描述。")

    first = parts[0].lower()
    model = IMAGE_MODELS.get(first)
    if model:
        prompt = " ".join(parts[1:]).strip()
        if not prompt:
            raise ImageGenerationError("请选择模型后填写图片描述。")
        return model, prompt

    return "agnes-image-2.0-flash", " ".join(parts).strip()


class ImageGenerationService:
    def __init__(self, api_key: str, base_url: str, timeout: float = 120.0) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def generate(self, prompt: str, model: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            raise ImageGenerationError("图片描述不能为空。")
        if not self.api_key or not self.base_url:
            raise ImageGenerationError("图片生成服务未正确配置。")

        endpoint = f"{self.base_url}/images/generations"
        payload = {
            "model": model,
            "prompt": prompt,
            "size": "1024x1024",
            "extra_body": {"response_format": "url"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Image generation API returned HTTP %s: %s",
                exc.response.status_code,
                exc.response.text[:500],
            )
            raise ImageGenerationError("图片生成服务暂时不可用。") from exc
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Image generation request failed: %s", exc)
            raise ImageGenerationError("图片生成服务暂时不可用。") from exc
        except Exception as exc:
            logger.exception("Unexpected image generation error")
            raise ImageGenerationError("图片生成服务暂时不可用。") from exc

        try:
            items = data.get("data")
            if not isinstance(items, list) or not items:
                raise ImageGenerationError("图片生成服务未返回有效结果。")
            first = items[0]
            if not isinstance(first, dict):
                raise ImageGenerationError("图片生成服务返回格式异常。")
            image_url = str(first.get("url") or "").strip()
            parsed = urlparse(image_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ImageGenerationError("图片生成服务返回了无效图片地址。")
            return image_url
        except ImageGenerationError:
            raise
        except (AttributeError, IndexError, TypeError) as exc:
            logger.warning("Invalid image generation response: %r", data)
            raise ImageGenerationError("图片生成服务返回格式异常。") from exc
