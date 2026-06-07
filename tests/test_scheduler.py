from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.scheduler import (
    _notify_job_crash,
    _tick,
    morning_run_minutes,
    settle_minutes,
    should_run_at,
)


def _dt(h: int, m: int) -> datetime:
    return datetime(2026, 4, 23, h, m, 0)


class TestSchedule:
    def test_run_fires_at_morning_slot(self):
        assert should_run_at("run", _dt(8, 0))

    def test_run_fires_every_half_hour_in_game_window(self):
        assert should_run_at("run", _dt(11, 0))
        assert should_run_at("run", _dt(11, 30))
        assert should_run_at("run", _dt(15, 0))
        assert should_run_at("run", _dt(22, 30))

    def test_run_does_not_fire_off_half_hour(self):
        assert not should_run_at("run", _dt(14, 15))
        assert not should_run_at("run", _dt(14, 1))

    def test_run_does_not_fire_overnight(self):
        assert not should_run_at("run", _dt(3, 0))
        assert not should_run_at("run", _dt(9, 0))   # 09:00 gap between morning+window
        assert not should_run_at("run", _dt(23, 0))  # after window closes

    def test_settle_fires_only_at_2am(self):
        assert should_run_at("settle", _dt(2, 0))
        assert not should_run_at("settle", _dt(2, 1))
        assert not should_run_at("settle", _dt(1, 0))
        assert not should_run_at("settle", _dt(14, 0))

    def test_morning_run_slots_are_complete(self):
        slots = morning_run_minutes()
        # 8:00 + (11:00, 11:30, ..., 22:00, 22:30) = 1 + 24 = 25
        assert len(slots) == 25
        assert (8, 0) in slots
        assert (22, 30) in slots

    def test_settle_has_one_slot(self):
        assert settle_minutes() == [(2, 0)]


class TestTick:
    async def test_tick_dispatches_matching_command(self):
        dispatch = AsyncMock()
        last_fire: dict = {}

        await _tick(_dt(8, 0), last_fire, dispatch)

        dispatch.assert_called_once_with("run")

    async def test_tick_skips_when_no_command_matches(self):
        dispatch = AsyncMock()
        last_fire: dict = {}

        # 14:07 hits no slot: run/trigger fire on :00/:15/:30/:45, nightly jobs
        # are pre-noon. (14:15 is now a real `trigger` slot.)
        await _tick(_dt(14, 7), last_fire, dispatch)

        dispatch.assert_not_called()

    async def test_tick_deduplicates_within_same_minute(self):
        """If the scheduler ticks twice inside the same wall-clock minute
        (e.g., the first tick ran long), the job fires only once."""
        dispatch = AsyncMock()
        last_fire: dict = {}

        await _tick(_dt(11, 0), last_fire, dispatch)
        await _tick(_dt(11, 0), last_fire, dispatch)

        dispatch.assert_called_once_with("run")

    async def test_tick_fires_again_on_next_scheduled_slot(self):
        dispatch = AsyncMock()
        last_fire: dict = {}

        await _tick(_dt(11, 0), last_fire, dispatch)
        await _tick(_dt(11, 30), last_fire, dispatch)

        assert dispatch.call_count == 2

    async def test_tick_swallows_job_exception_and_continues(self):
        """A failing job must not propagate — the scheduler stays alive."""
        dispatch = AsyncMock(side_effect=RuntimeError("boom"))
        last_fire: dict = {}

        with patch('src.scheduler._notify_job_crash') as mock_notify:
            await _tick(_dt(8, 0), last_fire, dispatch)

        dispatch.assert_called_once()
        mock_notify.assert_called_once()
        args = mock_notify.call_args[0]
        assert args[0] == "run"
        assert isinstance(args[1], RuntimeError)

    async def test_tick_dispatches_settle_at_2am(self):
        dispatch = AsyncMock()
        await _tick(_dt(2, 0), {}, dispatch)
        dispatch.assert_called_once_with("settle")


class TestNotify:
    @patch('src.clients.telegram_bot.TelegramClient')
    def test_crash_notify_sends_telegram(self, mock_cls):
        mock_bot = mock_cls.return_value
        mock_bot.send_message_sync = MagicMock()

        _notify_job_crash("run", ValueError("odds API 503"))

        mock_bot.send_message_sync.assert_called_once()
        msg = mock_bot.send_message_sync.call_args[0][0]
        assert "Scheduled Job Failed" in msg
        assert "run" in msg
        assert "odds API 503" in msg

    @patch('src.clients.telegram_bot.TelegramClient')
    def test_crash_notify_swallows_telegram_failure(self, mock_cls):
        mock_bot = mock_cls.return_value
        mock_bot.send_message_sync = MagicMock(side_effect=RuntimeError("down"))

        _notify_job_crash("run", ValueError("boom"))  # must not raise
