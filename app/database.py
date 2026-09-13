from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupSettings:
    chat_id: int
    auto_history_enabled: bool = False
    auto_history_time: str = "08:00"
    timezone: str = "Asia/Shanghai"
    last_history_sent: str | None = None


class Database:
    """Small synchronous PostgreSQL persistence layer.

    The public interface intentionally matches the old SQLite implementation so
    handlers.py does not need to know which database backend is being used.
    """

    def __init__(self, database_url: str, default_timezone: str, default_auto_time: str) -> None:
        self.database_url = database_url
        self.default_timezone = default_timezone
        self.default_auto_time = default_auto_time
        self._pool: ConnectionPool | None = None

    def _require_pool(self) -> ConnectionPool:
        if self._pool is None:
            raise RuntimeError("Database has not been initialized")
        return self._pool

    def initialize(self) -> None:
        if self._pool is None:
            self._pool = ConnectionPool(
                conninfo=self.database_url,
                min_size=1,
                max_size=5,
                open=True,
                kwargs={"row_factory": dict_row},
            )
        with self._require_pool().connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_models (
                    user_id BIGINT PRIMARY KEY,
                    model TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS group_settings (
                    chat_id BIGINT PRIMARY KEY,
                    auto_history_enabled BOOLEAN NOT NULL DEFAULT FALSE,
                    auto_history_time TEXT NOT NULL DEFAULT '08:00',
                    timezone TEXT NOT NULL DEFAULT 'Asia/Shanghai',
                    last_history_sent TEXT
                );

                CREATE TABLE IF NOT EXISTS group_messages (
                    chat_id BIGINT NOT NULL,
                    message_id BIGINT NOT NULL,
                    author TEXT NOT NULL,
                    text TEXT NOT NULL,
                    chat_username TEXT,
                    created_at TIMESTAMPTZ,
                    PRIMARY KEY (chat_id, message_id)
                );

                CREATE INDEX IF NOT EXISTS idx_group_messages_chat_time
                ON group_messages(chat_id, created_at, message_id);

                CREATE INDEX IF NOT EXISTS idx_group_messages_chat_message
                ON group_messages(chat_id, message_id);
                """
            )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def get_user_model(self, user_id: int, default_model: str) -> str:
        try:
            with self._require_pool().connection() as conn:
                row = conn.execute(
                    "SELECT model FROM user_models WHERE user_id = %s",
                    (user_id,),
                ).fetchone()
            return str(row["model"]) if row else default_model
        except Exception:
            logger.exception("Failed to read user model")
            return default_model

    def set_user_model(self, user_id: int, model: str) -> bool:
        try:
            with self._require_pool().connection() as conn:
                conn.execute(
                    """
                    INSERT INTO user_models(user_id, model)
                    VALUES (%s, %s)
                    ON CONFLICT(user_id) DO UPDATE SET
                        model = EXCLUDED.model,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (user_id, model),
                )
            return True
        except Exception:
            logger.exception("Failed to save user model")
            return False

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
            with self._require_pool().connection() as conn:
                conn.execute(
                    """
                    INSERT INTO group_messages(
                        chat_id,
                        message_id,
                        author,
                        text,
                        chat_username,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT(chat_id, message_id) DO NOTHING
                    """,
                    (
                        chat_id,
                        message_id,
                        author,
                        text,
                        chat_username,
                        created_at,
                    ),
                )
            return True
        except Exception:
            logger.exception("Failed to save group message")
            return False

    def get_group_messages(self, chat_id: int, limit: int = 80) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        try:
            with self._require_pool().connection() as conn:
                rows = conn.execute(
                    """
                    SELECT
                        chat_id,
                        message_id,
                        author,
                        text,
                        chat_username,
                        created_at::text AS created_at
                    FROM group_messages
                    WHERE chat_id = %s
                    ORDER BY message_id DESC
                    LIMIT %s
                    """,
                    (chat_id, limit),
                ).fetchall()
            return [dict(row) for row in reversed(rows)]
        except Exception:
            logger.exception("Failed to read group messages")
            return []

    def count_group_messages(self, chat_id: int) -> int:
        try:
            with self._require_pool().connection() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS count FROM group_messages WHERE chat_id = %s",
                    (chat_id,),
                ).fetchone()
            return int(row["count"]) if row else 0
        except Exception:
            logger.exception("Failed to count group messages")
            return 0

    def clear_group_messages(self, chat_id: int, through_message_id: int | None = None) -> bool:
        try:
            with self._require_pool().connection() as conn:
                if through_message_id is None:
                    conn.execute(
                        "DELETE FROM group_messages WHERE chat_id = %s",
                        (chat_id,),
                    )
                else:
                    conn.execute(
                        """
                        DELETE FROM group_messages
                        WHERE chat_id = %s AND message_id <= %s
                        """,
                        (chat_id, through_message_id),
                    )
            return True
        except Exception:
            logger.exception("Failed to clear group messages")
            return False

    def get_group_settings(self, chat_id: int) -> GroupSettings:
        try:
            with self._require_pool().connection() as conn:
                row = conn.execute(
                    "SELECT * FROM group_settings WHERE chat_id = %s",
                    (chat_id,),
                ).fetchone()
            if not row:
                return GroupSettings(
                    chat_id,
                    timezone=self.default_timezone,
                    auto_history_time=self.default_auto_time,
                )
            return GroupSettings(
                chat_id=chat_id,
                auto_history_enabled=bool(row["auto_history_enabled"]),
                auto_history_time=row["auto_history_time"],
                timezone=row["timezone"],
                last_history_sent=row["last_history_sent"],
            )
        except Exception:
            logger.exception("Failed to read group settings")
            return GroupSettings(
                chat_id,
                timezone=self.default_timezone,
                auto_history_time=self.default_auto_time,
            )

    def save_group_settings(self, settings: GroupSettings) -> bool:
        try:
            with self._require_pool().connection() as conn:
                conn.execute(
                    """
                    INSERT INTO group_settings(
                        chat_id,
                        auto_history_enabled,
                        auto_history_time,
                        timezone,
                        last_history_sent
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT(chat_id) DO UPDATE SET
                        auto_history_enabled = EXCLUDED.auto_history_enabled,
                        auto_history_time = EXCLUDED.auto_history_time,
                        timezone = EXCLUDED.timezone,
                        last_history_sent = EXCLUDED.last_history_sent
                    """,
                    (
                        settings.chat_id,
                        settings.auto_history_enabled,
                        settings.auto_history_time,
                        settings.timezone,
                        settings.last_history_sent,
                    ),
                )
            return True
        except Exception:
            logger.exception("Failed to save group settings")
            return False

    def list_auto_history_groups(self) -> list[GroupSettings]:
        try:
            with self._require_pool().connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM group_settings WHERE auto_history_enabled = TRUE"
                ).fetchall()
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
        except Exception:
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
