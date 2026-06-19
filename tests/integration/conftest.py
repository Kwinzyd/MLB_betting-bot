"""Shared pytest fixtures for integration tests.

The `db` fixture creates a real SQLite file in a pytest-managed temp dir,
patches `src.data.db.DB_PATH` to point at it, and yields the path.
All modules that call `get_db_connection()` resolve DB_PATH from the
`src.data.db` module global at call time, so patching there is sufficient
— no per-module patching required.
"""
import pytest
import src.data.db
from tests.fixtures.fixture_db import create_fixture_db_at_path


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Temp SQLite file seeded from fixture_db; DB_PATH patched for all modules."""
    db_path = str(tmp_path / 'fixture.db')
    create_fixture_db_at_path(db_path)
    monkeypatch.setattr(src.data.db, 'DB_PATH', db_path)
    yield db_path
