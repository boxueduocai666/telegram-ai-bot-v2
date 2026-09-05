import pytest

from app.search import SearchService, SearchResult, format_search_results


def test_format_search_results():
    text = format_search_results([SearchResult("Title", "https://example.com", "Summary")])
    assert "Title" in text
    assert "https://example.com" in text


@pytest.mark.asyncio
async def test_search_empty_query():
    assert await SearchService().search("   ") == []
