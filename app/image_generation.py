from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# 用户输入 -> Agnes 图片模型名
IMAGE_MODEL_MAP: dict[str, str] = {
    "2": "agnes-image-2.0-flash",
    "2.0": "agnes-image-2.0-flash",
    "image2": "agnes-image-2.0-flash",
    "image2.0": "agnes-image-2.0-flash",
    "2.1": "agnes-image-2.1-flash",
    "image2.1": "agnes-image-2.1-flash",
}

DEFAULT_IMAGE_MODEL = "agnes-image-2.0-flash"
IMAGE_GENERATION_TIMEOUT = 120.0


class ImageGenerationError(RuntimeError):
    """图片生成失败时抛出，供 handler 转成友好提示。"""


def parse_image_command(args: list[str]) -> tuple[str, str]:
    """解析 /image 参数，返回 (prompt, model)。

    第一个 token 若是模型版本标识则消费它，否则整个参数都作为 prompt。
    """
    if not args:
        return "", DEFAULT_IMAGE_MODEL

    first = args[0].strip().lower()
    if first in IMAGE_MODEL_MAP:
        prompt = " ".join(args[1:]).strip()
        return prompt, IMAGE_MODEL_MAP[first]

    prompt = " ".join(args).strip()
    return prompt, DEFAULT_IMAGE_MODEL


class ImageGenerator:
    """Agnes OpenAI-compatible 图片生成客户端。只取 URL，不下载、不上传。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = IMAGE_GENERATION_TIMEOUT,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def generate(self, prompt: str, model: str = DEFAULT_IMAGE_MODEL) -> str:
        if not prompt or not prompt.strip():
            raise ImageGenerationError("图片描述不能为空。")

        endpoint = f"{self.base_url}/images/generations"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt.strip(),
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
        except httpx.TimeoutException as exc:
            logger.exception("Image generation timed out")
            raise ImageGenerationError("图片生成超时，请稍后再试。") from exc
        except httpx.HTTPStatusError as exc:
            logger.exception(
                "Image generation HTTP error: %s", exc.response.status_code
            )
            raise ImageGenerationError(
                f"图片生成服务返回错误（{exc.response.status_code}）。"
            ) from exc
        except ImageGenerationError:
            raise
        except Exception as exc:
            logger.exception("Image generation request failed")
            raise ImageGenerationError("图片生成服务暂时不可用，请稍后再试。") from exc

        try:
            items = data.get("data")
            if not isinstance(items, list) or not items:
                raise ImageGenerationError("图片生成服务没有返回图片数据。")

            first = items[0]
            if not isinstance(first, dict):
                raise ImageGenerationError("图片生成服务返回的数据格式异常。")

            url = first.get("url")
            if not url or not isinstance(url, str):
                raise ImageGenerationError("图片生成服务没有返回有效的图片 URL。")

            url = url.strip()
            if not url.startswith(("http://", "https://")):
                raise ImageGenerationError("图片生成服务返回的 URL 无效。")

            return url
        except ImageGenerationError:
            raise
        except Exception as exc:
            logger.exception("Failed to parse image generation response")
            raise ImageGenerationError("图片生成服务返回的数据无法解析。") from exc
