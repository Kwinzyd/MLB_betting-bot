import pytest
import sqlite3
from unittest.mock import patch, MagicMock


# --- Approach 1: Pure Mocking ---

@patch('src.pipelines.sync_events.get_db_connection')
def test_mock_integrity_error(mock_get_db):
    """Test how the app handles a mocked sqlite3.IntegrityError."""
    
    # Setup mock connection context manager
    mock_conn = MagicMock()
    mock_get_db.return_value.__enter__.return_value = mock_conn
    
    # Force the execute method to raise the specific SQLite error
    mock_conn.execute.side_effect = sqlite3.IntegrityError("UNIQUE constraint failed: games.game_id")
    
    # Here we assert that the error is raised. If your pipeline had a try/except block 
    # that caught this error and logged a warning instead, you would test for that behavior!
    with pytest.raises(sqlite3.IntegrityError) as exc_info:
        # Simulating a DB call that your pipeline might make
        mock_conn.execute("INSERT INTO games (game_id) VALUES ('mock_game_123')")
        
    assert "UNIQUE constraint failed" in str(exc_info.value)


# --- Approach 2: Real Error via In-Memory DB ---

def test_real_in_memory_integrity_error():
    """Test using an in-memory DB to trigger a real constraint failure."""
    conn = sqlite3.connect(':memory:')
    
    # Create a table with a UNIQUE constraint
    conn.execute("CREATE TABLE test_table (id INTEGER UNIQUE, name TEXT)")
    
    # Insert the first row successfully
    conn.execute("INSERT INTO test_table VALUES (1, 'Yankees')")
    
    # Attempting to insert a duplicate ID will raise the real sqlite3.IntegrityError
    with pytest.raises(sqlite3.IntegrityError) as exc_info:
        conn.execute("INSERT INTO test_table VALUES (1, 'Red Sox')")
        
    assert "UNIQUE constraint failed" in str(exc_info.value)