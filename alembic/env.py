from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.chat.repositories.pg_models import Base


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# LangGraph owns these tables through ``Async*Saver.setup()``.  They are not
# SQLAlchemy domain models, so Alembic autogenerate must neither propose nor
# apply their removal.
LANGGRAPH_CHECKPOINT_TABLES = frozenset(
    {"checkpoints", "checkpoint_writes", "checkpoint_blobs", "checkpoint_migrations"}
)


def include_name(name: str | None, type_: str, parent_names: dict[str, str | None]) -> bool:
    """Exclude exactly the LangGraph-managed checkpoint tables from Alembic."""

    return not (type_ == "table" and name in LANGGRAPH_CHECKPOINT_TABLES)


def _database_url() -> str:
    return (
        os.getenv("CHAT_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or config.get_main_option("sqlalchemy.url")
    )


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, include_name=include_name)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()
    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
