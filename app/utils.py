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


def _format_markdown_table(lines: list[str]) -> list[str]:
    """Convert GitHub-style Markdown tables into Telegram-friendly text.

    Telegram MarkdownV2 has no table syntax. A compact card-like layout is more
    readable on phones and remains valid Telegram MarkdownV2 after escaping.
    """
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            i + 1 < len(lines)
            and "|" in line
            and re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", lines[i + 1])
        ):
            def cells(value: str) -> list[str]:
                value = value.strip()
                if value.startswith("|"):
                    value = value[1:]
                if value.endswith("|") and not value.endswith("\\|"):
                    value = value[:-1]
                return [c.strip() for c in value.split("|")]

            headers = cells(line)
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                # A new Markdown block/list/header ends the table.
                if re.match(r"^\s*(#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s?)", lines[i]):
                    break
                rows.append(cells(lines[i]))
                i += 1

            if headers:
                for row in rows:
                    padded = row + [""] * max(0, len(headers) - len(row))
                    if padded:
                        title = padded[0]
                        result.append(f"**{title}**" if title else "")
                        for h, value in zip(headers[1:], padded[1:]):
                            if value:
                                result.append(f"• {h}: {value}")
                        result.append("")
                if not rows:
                    # A header-only table is still better than raw pipes.
                    result.append(f"**{headers[0]}**")
                    for h in headers[1:]:
                        result.append(f"• {h}")
                    result.append("")
            continue

        result.append(line)
        i += 1
    return result


def _normalize_math(text: str) -> str:
    """Turn common LaTeX math into readable plain text for Telegram.

    Telegram does not render LaTeX/MathJax, so keeping ``$``/``\\frac`` makes
    answers look broken. This intentionally handles common school-science/math
    notation without pretending to be a full LaTeX parser.
    """
    # Display and inline math delimiters.
    text = re.sub(r"\\\[([\s\S]*?)\\\]", r"\1", text)
    text = re.sub(r"\\\(([\s\S]*?)\\\)", r"\1", text)
    text = re.sub(r"\$\$([\s\S]*?)\$\$", r"\1", text)
    text = re.sub(r"(?<!\\)\$([^$\n]+)\$", r"\1", text)

    # Common LaTeX commands.
    replacements = [
        (r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)"),
        (r"\\dfrac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)"),
        (r"\\tfrac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)"),
        (r"\\sqrt\s*\{([^{}]*)\}", r"√(\1)"),
        (r"\\times", "×"),
        (r"\\cdot", "·"),
        (r"\\div", "÷"),
        (r"\\leq?", "≤"),
        (r"\\geq?", "≥"),
        (r"\\neq", "≠"),
        (r"\\approx", "≈"),
        (r"\\pm", "±"),
        (r"\\infty", "∞"),
        (r"\\pi", "π"),
        (r"\\rightarrow|\\to", "→"),
        (r"\\left|\\right", ""),
        (r"\\left|\\right.", ""),
        (r"\\left", ""),
        (r"\\right", ""),
        (r"\\text\s*\{([^{}]*)\}", r"\1"),
        (r"\\mathrm\s*\{([^{}]*)\}", r"\1"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    # Superscript/subscript braces: x^{2} -> x^2, a_{1} -> a_1.
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = re.sub(r"_\{([^{}]+)\}", r"_\1", text)
    text = text.replace(r"\%", "%").replace(r"\#", "#").replace(r"\&", "&")
    text = re.sub(r"\\([{}])", r"\1", text)
    return text


def markdown_to_markdown_v2(markdown: str) -> str:
    """Convert practical Markdown into Telegram MarkdownV2 safely.

    Telegram does not support Markdown tables or LaTeX. Tables are converted to
    compact mobile-friendly blocks, and common LaTeX math is normalized to readable
    Unicode/plain text. Formatting delimiters, links, code and quotes are protected
    before ordinary MarkdownV2 escaping.
    """
    if not markdown:
        return ""

    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    text = _normalize_math(text)
    text = "\n".join(_format_markdown_table(text.split("\n")))

    placeholders: list[str] = []

    def stash(value: str) -> str:
        token = f"\u0000{len(placeholders)}\u0000"
        placeholders.append(value)
        return token

    # Fenced code blocks first. Telegram code blocks should not be escaped like prose.
    def block_code(match: re.Match[str]) -> str:
        code = match.group(2).replace("`", "\\`")
        # Telegram accepts a language hint after the opening fence, but only use a
        # simple identifier so unusual punctuation cannot break MarkdownV2.
        lang = (match.group(1) or "").strip()
        lang = re.sub(r"[^A-Za-z0-9_+.-]", "", lang)
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

    # Protect emphasis before escaping ordinary punctuation.
    marker_patterns = [
        r"\*\*(.+?)\*\*",
        r"__(.+?)__",
        r"~~(.+?)~~",
        r"(?<!\*)\*([^*\n]+)\*(?!\*)",
        r"(?<!_)_([^_\n]+)_(?!_)",
    ]
    for pattern in marker_patterns:
        def repl(match: re.Match[str], pattern=pattern) -> str:
            inner = escape_markdown_v2(match.group(1))
            if pattern.startswith(r"\*\*") or pattern.startswith("__"):
                rendered = f"*{inner}*"
            elif pattern.startswith("~~"):
                rendered = f"~{inner}~"
            else:
                rendered = f"_{inner}_"
            return stash(rendered)
        text = re.sub(pattern, repl, text)

    lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if re.match(r"^#{1,6}\s+", stripped):
            stripped = re.sub(r"^#{1,6}\s+", "", stripped)
            line = f"*{escape_markdown_v2(stripped)}*"
        elif re.match(r"^\s*[-*+]\s+", line):
            content = re.sub(r"^\s*[-*+]\s+", "", line)
            line = f"• {escape_markdown_v2(content)}"
        elif re.match(r"^\s*•\s*", line):
            content = re.sub(r"^\s*•\s*", "", line)
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
