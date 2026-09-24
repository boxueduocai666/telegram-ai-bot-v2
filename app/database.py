from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupSettings:
    chat_id: int
    auto_history_enabled: bool = False
    auto_history_time: str = "08:00"
    timezone: str = "Asia/Shanghai"
    last_history_sent: str | None = None
    auto_summary_enabled: bool = True
    auto_summary_time: str = "23:00"
    last_summary_sent: str | None = None


class Database:
    """Small SQLite persistence layer for user models and group chat history."""

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
                    last_history_sent TEXT,
                    auto_summary_enabled INTEGER NOT NULL DEFAULT 1,
                    auto_summary_time TEXT NOT NULL DEFAULT '23:00',
                    last_summary_sent TEXT
                );

                CREATE TABLE IF NOT EXISTS group_messages (
                    chat_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    author TEXT NOT NULL,
                    text TEXT NOT NULL,
                    chat_username TEXT,
                    created_at TEXT,
                    PRIMARY KEY (chat_id, message_id)
                );

                CREATE INDEX IF NOT EXISTS idx_group_messages_chat_time
                ON group_messages(chat_id, message_id);
                """
            )

            # Migrate an older SQLite V2 database in place. ALTER TABLE only adds
            # missing columns; existing messages/settings remain untouched.
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(group_settings)").fetchall()
            }
            migrations = (
                (
                    "auto_summary_enabled",
                    "ALTER TABLE group_settings ADD COLUMN auto_summary_enabled INTEGER NOT NULL DEFAULT 1",
                ),
                (
                    "auto_summary_time",
                    "ALTER TABLE group_settings ADD COLUMN auto_summary_time TEXT NOT NULL DEFAULT '23:00'",
                ),
                (
                    "last_summary_sent",
                    "ALTER TABLE group_settings ADD COLUMN last_summary_sent TEXT",
                ),
            )
            for name, statement in migrations:
                if name not in columns:
                    conn.execute(statement)

            # Existing groups with stored messages should become eligible for the
            # daily summary scheduler immediately after upgrading.
            conn.execute(
                """
                INSERT OR IGNORE INTO group_settings(
                    chat_id,
                    auto_history_enabled,
                    auto_history_time,
                    timezone,
                    last_history_sent,
                    auto_summary_enabled,
                    auto_summary_time,
                    last_summary_sent
                )
                SELECT DISTINCT
                    chat_id, 0, ?, ?, NULL, 1, '23:00', NULL
                FROM group_messages
                """,
                (self.default_auto_time, self.default_timezone),
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def get_user_model(self, user_id: int, default_model: str) -> str:
        try:
            with self._lock, closing(self._connect()) as conn:
                row = conn.execute(
                    "SELECT model FROM user_models WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
            return str(row["model"]) if row else default_model
        except sqlite3.Error:
            logger.exception("Failed to read user model")
            return default_model

    def set_user_model(self, user_id: int, model: str) -> bool:
        try:
            with self._lock, closing(self._connect()) as conn:
                conn.execute(
                    "INSERT INTO user_models(user_id, model) VALUES(?, ?) "
                    "ON CONFLICT(user_id) DO UPDATE SET "
                    "model=excluded.model, updated_at=CURRENT_TIMESTAMP",
                    (user_id, model),
                )
                conn.commit()
            return True
        except sqlite3.Error:
            logger.exception("Failed to save user model")
            return False

    def ensure_group_settings(self, chat_id: int) -> GroupSettings:
        current = self.get_group_settings(chat_id)
        try:
            with self._lock, closing(self._connect()) as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO group_settings(
                        chat_id,
                        auto_history_enabled,
                        auto_history_time,
                        timezone,
                        last_history_sent,
                        auto_summary_enabled,
                        auto_summary_time,
                        last_summary_sent
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chat_id,
                        int(current.auto_history_enabled),
                        current.auto_history_time,
                        current.timezone,
                        current.last_history_sent,
                        int(current.auto_summary_enabled),
                        current.auto_summary_time,
                        current.last_summary_sent,
                    ),
                )
                conn.commit()
        except sqlite3.Error:
            logger.exception("Failed to ensure group settings for %s", chat_id)
        return self.get_group_settings(chat_id)

    def save_group_message(
        self,
        chat_id: int,
        message_id: int,
        author: str,
        text: str,
        *,
        chat_username: str | None = None,
        created_at: str | None = None,
    ) -> bool:
        try:
            with self._lock, closing(self._connect()) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO group_messages "
                    "(chat_id, message_id, author, text, chat_username, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (chat_id, message_id, author, text, chat_username, created_at),
                )
                conn.commit()
            return True
        except sqlite3.Error:
            logger.exception("Failed to save group message")
            return False

    def get_group_messages(self, chat_id: int, limit: int = 120) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        try:
            with self._lock, closing(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT chat_id, message_id, author, text, chat_username, created_at "
                    "FROM group_messages WHERE chat_id = ? "
                    "ORDER BY message_id DESC LIMIT ?",
                    (chat_id, limit),
                ).fetchall()
            return [dict(row) for row in reversed(rows)]
        except sqlite3.Error:
            logger.exception("Failed to read group messages")
            return []

    def count_group_messages(self, chat_id: int) -> int:
        try:
            with self._lock, closing(self._connect()) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS count FROM group_messages WHERE chat_id = ?",
                    (chat_id,),
                ).fetchone()
            return int(row["count"]) if row else 0
        except sqlite3.Error:
            logger.exception("Failed to count group messages")
            return 0

    def clear_group_messages(
        self,
        chat_id: int,
        through_message_id: int | None = None,
    ) -> bool:
        try:
            with self._lock, closing(self._connect()) as conn:
                if through_message_id is None:
                    conn.execute(
                        "DELETE FROM group_messages WHERE chat_id = ?",
                        (chat_id,),
                    )
                else:
                    conn.execute(
                        "DELETE FROM group_messages "
                        "WHERE chat_id = ? AND message_id <= ?",
                        (chat_id, through_message_id),
                    )
                conn.commit()
            return True
        except sqlite3.Error:
            logger.exception("Failed to clear group messages")
            return False

    def get_group_settings(self, chat_id: int) -> GroupSettings:
        try:
            with self._lock, closing(self._connect()) as conn:
                row = conn.execute(
                    "SELECT * FROM group_settings WHERE chat_id = ?",
                    (chat_id,),
                ).fetchone()
            if not row:
                return GroupSettings(
                    chat_id=chat_id,
                    timezone=self.default_timezone,
                    auto_history_time=self.default_auto_time,
                )
            return self._row_to_group_settings(row)
        except (sqlite3.Error, KeyError):
            logger.exception("Failed to read group settings")
            return GroupSettings(
                chat_id=chat_id,
                timezone=self.default_timezone,
                auto_history_time=self.default_auto_time,
            )

    def save_group_settings(self, settings: GroupSettings) -> bool:
        try:
            with self._lock, closing(self._connect()) as conn:
                conn.execute(
                    """
                    INSERT INTO group_settings(
                        chat_id,
                        auto_history_enabled,
                        auto_history_time,
                        timezone,
                        last_history_sent,
                        auto_summary_enabled,
                        auto_summary_time,
                        last_summary_sent
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET
                        auto_history_enabled=excluded.auto_history_enabled,
                        auto_history_time=excluded.auto_history_time,
                        timezone=excluded.timezone,
                        last_history_sent=excluded.last_history_sent,
                        auto_summary_enabled=excluded.auto_summary_enabled,
                        auto_summary_time=excluded.auto_summary_time,
                        last_summary_sent=excluded.last_summary_sent
                    """,
                    (
                        settings.chat_id,
                        int(settings.auto_history_enabled),
                        settings.auto_history_time,
                        settings.timezone,
                        settings.last_history_sent,
                        int(settings.auto_summary_enabled),
                        settings.auto_summary_time,
                        settings.last_summary_sent,
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
                rows = conn.execute(
                    "SELECT * FROM group_settings WHERE auto_history_enabled = 1"
                ).fetchall()
            return [self._row_to_group_settings(row) for row in rows]
        except sqlite3.Error:
            logger.exception("Failed to list scheduled history groups")
            return []

    def list_auto_summary_groups(self) -> list[GroupSettings]:
        try:
            with self._lock, closing(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT * FROM group_settings WHERE auto_summary_enabled = 1"
                ).fetchall()
            return [self._row_to_group_settings(row) for row in rows]
        except sqlite3.Error:
            logger.exception("Failed to list scheduled summary groups")
            return []

    @staticmethod
    def _row_to_group_settings(row: sqlite3.Row) -> GroupSettings:
        return GroupSettings(
            chat_id=int(row["chat_id"]),
            auto_history_enabled=bool(row["auto_history_enabled"]),
            auto_history_time=str(row["auto_history_time"]),
            timezone=str(row["timezone"]),
            last_history_sent=row["last_history_sent"],
            auto_summary_enabled=bool(row["auto_summary_enabled"]),
            auto_summary_time=str(row["auto_summary_time"]),
            last_summary_sent=row["last_summary_sent"],
        )

    def mark_history_sent(self, chat_id: int, date_key: str) -> bool:
        settings = self.get_group_settings(chat_id)
        return self.save_group_settings(
            GroupSettings(
                chat_id=settings.chat_id,
                auto_history_enabled=settings.auto_history_enabled,
                auto_history_time=settings.auto_history_time,
                timezone=settings.timezone,
                last_history_sent=date_key,
                auto_summary_enabled=settings.auto_summary_enabled,
                auto_summary_time=settings.auto_summary_time,
                last_summary_sent=settings.last_summary_sent,
            )
        )

    def mark_summary_sent(self, chat_id: int, date_key: str) -> bool:
        settings = self.get_group_settings(chat_id)
        return self.save_group_settings(
            GroupSettings(
                chat_id=settings.chat_id,
                auto_history_enabled=settings.auto_history_enabled,
                auto_history_time=settings.auto_history_time,
                timezone=settings.timezone,
                last_history_sent=settings.last_history_sent,
                auto_summary_enabled=settings.auto_summary_enabled,
                auto_summary_time=settings.auto_summary_time,
                last_summary_sent=date_key,
            )
        )

    def finalize_summary(
        self,
        chat_id: int,
        message_ids: list[int],
        date_key: str,
    ) -> bool:
        """Atomically consume exactly the summarized messages and mark the day done."""
        ids_set: set[int] = set()
        for message_id in message_ids:
            try:
                value = int(message_id)
            except (TypeError, ValueError):
                continue
            if value > 0:
                ids_set.add(value)
        ids = sorted(ids_set)
        try:
            with self._lock, closing(self._connect()) as conn:
                if ids:
                    placeholders = ",".join("?" for _ in ids)
                    conn.execute(
                        "DELETE FROM group_messages "
                        f"WHERE chat_id = ? AND message_id IN ({placeholders})",
                        (chat_id, *ids),
                    )
                updated = conn.execute(
                    "UPDATE group_settings SET last_summary_sent = ? WHERE chat_id = ?",
                    (date_key, chat_id),
                ).rowcount
                if not updated:
                    conn.execute(
                        "INSERT INTO group_settings("
                        "chat_id, auto_history_enabled, auto_history_time, timezone, "
                        "last_history_sent, auto_summary_enabled, auto_summary_time, "
                        "last_summary_sent"
                        ") VALUES (?, 0, '08:00', ?, NULL, 1, '23:00', ?)",
                        (chat_id, self.default_timezone, date_key),
                    )
                conn.commit()
            return True
        except (sqlite3.Error, TypeError, ValueError):
            logger.exception("Failed to finalize scheduled summary for %s", chat_id)
            return False
