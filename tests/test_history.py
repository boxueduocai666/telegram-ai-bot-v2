from app.main import render_history_events


def test_history_rendering():
    text = render_history_events(8, 8, None, [{"year": 1945, "text": "An event", "pages": [{"title": "Example"}]}])
    assert "8月8日" in text
    assert "1945" in text
    assert "来源" in text
