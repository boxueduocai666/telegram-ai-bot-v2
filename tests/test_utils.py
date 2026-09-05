from app.utils import markdown_to_markdown_v2, parse_time, truncate_text, valid_timezone, split_text


def test_truncate_text():
    assert truncate_text("abcdef", 4) == "abc…"


def test_parse_time():
    assert parse_time("8:05") == "08:05"
    assert parse_time("24:00") is None


def test_timezone():
    assert valid_timezone("Asia/Shanghai")
    assert not valid_timezone("Not/AZone")


def test_markdown_v2_escapes_plain_punctuation():
    result = markdown_to_markdown_v2("价格是 100.00 元!")
    assert "100\\.00" in result
    assert "\\!" in result


def test_markdown_v2_code_and_bold():
    result = markdown_to_markdown_v2("**标题** 和 `x=1`")
    assert "*标题*" in result
    assert "`x=1`" in result


def test_split_text():
    parts = split_text("a\n" * 20, 10)
    assert all(len(part) <= 10 for part in parts)
    assert "".join(parts).replace("\n", "") == "a" * 20
