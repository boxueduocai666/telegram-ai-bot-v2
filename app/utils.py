from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
from zoneinfo import ZoneInfo

from telegram import Message

MDV2_SPECIAL = r"_ * [ ] ( ) ~ ` > # + - = | { } . ! \\"
MDV2_CHARS = "_[]()~`>#+-=|{}.!\\"


def truncate_text(text: str, max_chars: int, suffix: str = "…") -> str:
    text = text or ""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(suffix):
        return text[:max_chars]
    return text[: max_chars - len(suffix)].rstrip() + suffix


def normalize_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", (text or "").replace("\r\n", "\n")).strip()


def escape_markdown_v2(text: str) -> str:
    return re.sub(r"([_\[\]()~`>#+\-=|{}.!\\])", r"\\\1", text or "")


def markdown_to_markdown_v2(markdown: str) -> str:
    """A small, deterministic Markdown -> MarkdownV2 renderer.

    It intentionally supports the formats useful for bot replies instead of trying to
    implement all CommonMark. Code spans/blocks and links are handled before prose.
    """
    if not markdown:
        return ""

    placeholders: list[str] = []

    def stash(value: str) -> str:
        token = f"\u0000{len(placeholders)}\u0000"
        placeholders.append(value)
        return token

    text = markdown.replace("\r\n", "\n")

    def block_code(match: re.Match[str]) -> str:
        code = match.group(2).replace("`", "\\`")
        lang = match.group(1).strip() if match.group(1) else ""
        prefix = f"```{lang}\n" if lang else "```\n"
        return stash(prefix + code + "\n```")

    text = re.sub(r"```([^\n]*)\n([\s\S]*?)```", block_code, text)

    def inline_code(match: re.Match[str]) -> str:
        return stash(f"`{match.group(1).replace('`', '\\`')}`")

    text = re.sub(r"`([^`\n]+)`", inline_code, text)

    def link(match: re.Match[str]) -> str:
        label = escape_markdown_v2(match.group(1))
        url = match.group(2).replace("\\", "\\\\").replace(")", "\\)")
        return stash(f"[{label}]({url})")

    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, text)

    # Protect bold/italic/strike delimiters while escaping ordinary punctuation.
    markers = []
    marker_patterns = [r"\*\*(.+?)\*\*", r"__(.+?)__", r"~~(.+?)~~", r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"_([^_\n]+)_"]
    for pattern in marker_patterns:
        def repl(match: re.Match[str], pattern=pattern) -> str:
            inner = escape_markdown_v2(match.group(1))
            if pattern.startswith(r"\\*\\*") or pattern.startswith(r"\*\*"):
                rendered = f"*{inner}*"
            elif pattern.startswith("__"):
                rendered = f"*{inner}*"
            elif pattern.startswith("~~"):
                rendered = f"~{inner}~"
            elif pattern.startswith("(?<!"):
                rendered = f"_{inner}_"
            else:
                rendered = f"_{inner}_"
            return stash(rendered)
        text = re.sub(pattern, repl, text)

    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if re.match(r"^#{1,6}\s+", stripped):
            stripped = re.sub(r"^#{1,6}\s+", "", stripped)
            line = f"*{escape_markdown_v2(stripped)}*"
        elif re.match(r"^\s*[-*+]\s+", line):
            content = re.sub(r"^\s*[-*+]\s+", "", line)
            line = f"• {escape_markdown_v2(content)}"
        elif re.match(r"^\s*\d+[.)]\s+", line):
            match = re.match(r"^\s*(\d+)[.)]\s+(.*)$", line)
            assert match
            line = f"{match.group(1)}\\. {escape_markdown_v2(match.group(2))}"
        elif re.match(r"^\s*>\s?", line):
            content = re.sub(r"^\s*>\s?", "", line)
            line = f"> {escape_markdown_v2(content)}"
        else:
            line = escape_markdown_v2(line)
        lines.append(line)

    rendered = "\n".join(lines)
    for i, value in enumerate(placeholders):
        rendered = rendered.replace(f"\u0000{i}\u0000", value)
    return rendered


def safe_markdown_v2(markdown: str) -> str:
    """Compatibility alias for callers that want a clearly named safe formatter."""
    return markdown_to_markdown_v2(markdown)


def extract_bot_username(message: Message) -> str | None:
    if not message or not message.text:
        return None
    return None


def clean_command_args(args: Iterable[str]) -> str:
    return normalize_text(" ".join(args))


def parse_time(value: str) -> str | None:
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", value.strip())
    if not match:
        return None
    hour, minute = map(int, match.groups())
    if hour > 23 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def valid_timezone(value: str) -> bool:
    try:
        ZoneInfo(value)
        return True
    except Exception:
        return False


def now_in_timezone(timezone: str) -> datetime:
    return datetime.now(ZoneInfo(timezone))


@dataclass(frozen=True)
class ReplyContext:
    quoted_text: str | None = None
    replied_text: str | None = None
    replied_has_image: bool = False


def _message_text(message: Message | None) -> str:
    if not message:
        return ""
    return normalize_text(message.text or message.caption or "")


def build_reply_context(
    message: Message,
    max_length: int,
) -> ReplyContext:
    """Build one-level reply context: explicit quote > replied message.

    Telegram may expose a quoted fragment on the message object; this function checks
    it defensively because object shape differs by update type/version.
    """
    quote_text = getattr(getattr(message, "quote", None), "text", None)
    quote_text = normalize_text(quote_text or "") or None
    if quote_text:
        quote_text = truncate_text(quote_text, max_length)

    replied = message.reply_to_message
    replied_text = normalize_text(_message_text(replied)) or None
    if replied_text:
        replied_text = truncate_text(replied_text, max_length)
    replied_has_image = bool(replied and (replied.photo or replied.document))
    return ReplyContext(quoted_text=quote_text, replied_text=replied_text, replied_has_image=replied_has_image)


def split_text(text: str, max_length: int = 4000) -> list[str]:
    """Split text into Telegram-safe chunks without exceeding the message limit."""
    text = text or ""
    if not text:
        return [""]
    chunks = []
    remaining = text
    while len(remaining) > max_length:
        cut = remaining.rfind("\n", 0, max_length + 1)
        if cut < max_length // 2:
            cut = max_length
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip("\n")
    chunks.append(remaining)
    return chunks
