"""Database helpers.

The pipeline reads its connection from ``DATABASE_URL`` when it is set
(for example ``sqlite:///data/pipeline.db`` in tests). Otherwise it builds a
PostgreSQL URL from ``DB_USER``, ``DB_PASSWORD``, ``DB_HOST``, ``DB_PORT`` and
``DB_NAME``, which is what Docker Compose and Kubernetes provide.
"""

from __future__ import annotations

import logging
import os
import re

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

load_dotenv()
logger = logging.getLogger(__name__)

_SAFE_TABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REQUIRED_PG_VARS = ("DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME")


def database_url() -> str:
    """Return the SQLAlchemy URL for the pipeline database."""
    url = os.getenv("DATABASE_URL")
    if url:
        # Pin the driver: SQLAlchemy 2.1 maps a bare "postgresql://" to
        # psycopg 3, while this project installs psycopg2.
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg2://" + url[len("postgresql://"):]
        return url
    missing = [name for name in _REQUIRED_PG_VARS if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Database is not configured. Set DATABASE_URL or "
            f"{', '.join(missing)} (see .env.example)."
        )
    return (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )


def get_engine() -> Engine:
    """Create a SQLAlchemy engine with connection health checks."""
    return create_engine(database_url(), pool_pre_ping=True)


def _check_table_name(table_name: str) -> str:
    if not _SAFE_TABLE.match(table_name):
        raise ValueError(f"Invalid table name: {table_name!r}")
    return table_name


def table_exists(engine: Engine, table_name: str) -> bool:
    """Return True when ``table_name`` exists in the default schema."""
    return inspect(engine).has_table(_check_table_name(table_name))


def get_row_count(engine: Engine, table_name: str) -> int:
    """Count rows in a table whose name has been validated."""
    name = _check_table_name(table_name)
    with engine.connect() as conn:
        return int(conn.execute(text(f'SELECT COUNT(*) FROM "{name}"')).scalar_one())
