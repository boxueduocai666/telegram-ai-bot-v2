from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupSettings:
    chat_id: int
    auto_history_enabled: bool = False
    auto_history_time: str = "08:00"
    timezone: str = "Asia/Shanghai"
    last_history_sent: str | None = None


class Database:
    """Small SQLite persistence layer. Short-lived chat context stays in memory."""

    def __init__(self, path: str, default_timezone: str, default_auto_time: str) -> None:
        self.path = path
        self.default_timezone = default_timezone
        self.default_auto_time = default_auto_time
        self._lock = Lock()

    def initialize(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA busy_timeout=5000;
                CREATE TABLE IF NOT EXISTS user_models (
                    user_id INTEGER PRIMARY KEY,
                    model TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS group_settings (
                    chat_id INTEGER PRIMARY KEY,
                    auto_history_enabled INTEGER NOT NULL DEFAULT 0,
                    auto_history_time TEXT NOT NULL DEFAULT '08:00',
                    timezone TEXT NOT NULL DEFAULT 'Asia/Shanghai',
                    last_history_sent TEXT
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def get_user_model(self, user_id: int, default_model: str) -> str:
        try:
            with self._lock, closing(self._connect()) as conn:
                row = conn.execute("SELECT model FROM user_models WHERE user_id = ?", (user_id,)).fetchone()
            return str(row["model"]) if row else default_model
        except sqlite3.Error:
            logger.exception("Failed to read user model")
            return default_model

    def set_user_model(self, user_id: int, model: str) -> bool:
        try:
            with self._lock, closing(self._connect()) as conn:
                conn.execute(
                    "INSERT INTO user_models(user_id, model) VALUES(?, ?) "
                    "ON CONFLICT(user_id) DO UPDATE SET model=excluded.model, updated_at=CURRENT_TIMESTAMP",
                    (user_id, model),
                )
                conn.commit()
            return True
        except sqlite3.Error:
            logger.exception("Failed to save user model")
            return False

    def get_group_settings(self, chat_id: int) -> GroupSettings:
        try:
            with self._lock, closing(self._connect()) as conn:
                row = conn.execute("SELECT * FROM group_settings WHERE chat_id = ?", (chat_id,)).fetchone()
            if not row:
                return GroupSettings(chat_id, timezone=self.default_timezone, auto_history_time=self.default_auto_time)
            return GroupSettings(
                chat_id=chat_id,
                auto_history_enabled=bool(row["auto_history_enabled"]),
                auto_history_time=row["auto_history_time"],
                timezone=row["timezone"],
                last_history_sent=row["last_history_sent"],
            )
        except sqlite3.Error:
            logger.exception("Failed to read group settings")
            return GroupSettings(chat_id, timezone=self.default_timezone, auto_history_time=self.default_auto_time)

    def save_group_settings(self, settings: GroupSettings) -> bool:
        try:
            with self._lock, closing(self._connect()) as conn:
                conn.execute(
                    "INSERT INTO group_settings(chat_id, auto_history_enabled, auto_history_time, timezone, last_history_sent) "
                    "VALUES(?, ?, ?, ?, ?) "
                    "ON CONFLICT(chat_id) DO UPDATE SET auto_history_enabled=excluded.auto_history_enabled, "
                    "auto_history_time=excluded.auto_history_time, timezone=excluded.timezone, "
                    "last_history_sent=excluded.last_history_sent",
                    (
                        settings.chat_id,
                        int(settings.auto_history_enabled),
                        settings.auto_history_time,
                        settings.timezone,
                        settings.last_history_sent,
                    ),
                )
                conn.commit()
            return True
        except sqlite3.Error:
            logger.exception("Failed to save group settings")
            return False

    def list_auto_history_groups(self) -> list[GroupSettings]:
        try:
            with self._lock, closing(self._connect()) as conn:
                rows = conn.execute("SELECT * FROM group_settings WHERE auto_history_enabled = 1").fetchall()
            return [
                GroupSettings(
                    chat_id=row["chat_id"],
                    auto_history_enabled=True,
                    auto_history_time=row["auto_history_time"],
                    timezone=row["timezone"],
                    last_history_sent=row["last_history_sent"],
                )
                for row in rows
            ]
        except sqlite3.Error:
            logger.exception("Failed to list scheduled groups")
            return []

    def mark_history_sent(self, chat_id: int, date_key: str) -> bool:
        settings = self.get_group_settings(chat_id)
        return self.save_group_settings(
            GroupSettings(
                chat_id=settings.chat_id,
                auto_history_enabled=settings.auto_history_enabled,
                auto_history_time=settings.auto_history_time,
                timezone=settings.timezone,
                last_history_sent=date_key,
            )
        )
