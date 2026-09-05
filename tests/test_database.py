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
