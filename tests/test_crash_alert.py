from unittest.mock import MagicMock, patch

import pytest


@patch('src.clients.telegram_bot.TelegramClient')
def test_notify_crash_sends_telegram(mock_telegram_cls):
    mock_bot = mock_telegram_cls.return_value
    mock_bot.send_message_sync = MagicMock()

    from main import _notify_crash
    _notify_crash("scan", ValueError("odds API returned 500"))

    mock_bot.send_message_sync.assert_called_once()
    msg = mock_bot.send_message_sync.call_args[0][0]
    assert "Pipeline Crash" in msg
    assert "scan" in msg
    assert "odds API returned 500" in msg


@patch('src.clients.telegram_bot.TelegramClient')
def test_notify_crash_swallows_telegram_failure(mock_telegram_cls):
    """If Telegram itself is down, the crash notifier must not mask the
    original exception by raising a secondary error."""
    mock_bot = mock_telegram_cls.return_value
    mock_bot.send_message_sync = MagicMock(side_effect=RuntimeError("telegram down"))

    from main import _notify_crash
    # Should not raise
    _notify_crash("scan", ValueError("boom"))


@patch('src.clients.telegram_bot.TelegramClient')
def test_notify_crash_truncates_long_error(mock_telegram_cls):
    mock_bot = mock_telegram_cls.return_value
    mock_bot.send_message_sync = MagicMock()

    from main import _notify_crash
    _notify_crash("run", ValueError("x" * 5000))

    msg = mock_bot.send_message_sync.call_args[0][0]
    # Long error gets clipped (traceback tail capped at 800 chars + envelope)
    assert len(msg) < 1100
    assert "x" * 5000 not in msg
