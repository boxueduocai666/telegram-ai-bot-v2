from app.database import Database


def test_model_persistence(tmp_path):
    db = Database(str(tmp_path / "bot.db"), "Asia/Shanghai", "08:00")
    db.initialize()
    assert db.set_user_model(123, "model-b")
    assert db.get_user_model(123, "model-a") == "model-b"


def test_group_settings_persistence(tmp_path):
    from app.database import GroupSettings
    db = Database(str(tmp_path / "bot.db"), "Asia/Shanghai", "08:00")
    db.initialize()
    original = GroupSettings(99, True, "07:30", "Asia/Taipei", "2026-09-05")
    assert db.save_group_settings(original)
    assert db.get_group_settings(99) == original


def test_group_message_persistence_and_bounded_clear(tmp_path):
    db = Database(str(tmp_path / "bot.db"), "Asia/Shanghai", "08:00")
    db.initialize()
    assert db.save_group_message(1, 10, "甲", "第一条", created_at="2026-09-05T10:00:00+00:00")
    assert db.save_group_message(1, 11, "乙", "第二条", created_at="2026-09-05T10:01:00+00:00")
    assert db.save_group_message(1, 12, "丙", "第三条", created_at="2026-09-05T10:02:00+00:00")
    assert db.count_group_messages(1) == 3
    rows = db.get_group_messages(1, limit=10)
    assert [row["message_id"] for row in rows] == [10, 11, 12]
    assert db.clear_group_messages(1, through_message_id=11)
    assert [row["message_id"] for row in db.get_group_messages(1, limit=10)] == [12]
