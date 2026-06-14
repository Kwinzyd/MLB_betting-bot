from unittest.mock import patch, AsyncMock

import httpx

from src.clients.llm import OpenRouterClient, _extract_json


class TestExtractJson:
    def test_raw_object(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_prose_wrapped(self):
        assert _extract_json('Sure! {"a": 1} hope that helps') == {"a": 1}

    def test_array(self):
        assert _extract_json('[1, 2, 3]') == [1, 2, 3]

    def test_garbage_is_none(self):
        assert _extract_json('no json here at all') is None

    def test_empty_is_none(self):
        assert _extract_json('') is None


def _client():
    c = OpenRouterClient()
    c.api_key = 'test-key'
    return c


@patch('src.clients.llm.LLM_ENABLED', False)
async def test_unavailable_returns_none_everywhere():
    c = _client()
    assert c.available is False
    assert await c.chat([{'role': 'user', 'content': 'x'}]) is None
    assert await c.complete_json('s', 'u') is None
    assert await c.complete_text('s', 'u') is None


@patch('src.clients.llm.LLM_ENABLED', True)
async def test_complete_json_parses_response():
    c = _client()
    resp = {"choices": [{"message": {"content": '{"questionable": true}'}}], "usage": {}}
    with patch.object(c, '_post', new=AsyncMock(return_value=resp)):
        out = await c.complete_json('sys', 'unique-user-1', use_cache=False)
    assert out == {"questionable": True}


@patch('src.clients.llm.LLM_ENABLED', True)
async def test_complete_text_returns_content():
    c = _client()
    resp = {"choices": [{"message": {"content": "hello"}}], "usage": {}}
    with patch.object(c, '_post', new=AsyncMock(return_value=resp)):
        out = await c.complete_text('sys', 'unique-user-2', use_cache=False)
    assert out == "hello"


@patch('src.clients.llm.LLM_ENABLED', True)
async def test_chat_failsafe_on_http_error():
    c = _client()
    with patch.object(c, '_post', new=AsyncMock(side_effect=httpx.HTTPError('boom'))):
        assert await c.chat([{'role': 'user', 'content': 'x'}]) is None


@patch('src.clients.llm.LLM_ENABLED', True)
async def test_complete_json_failsafe_on_unparseable():
    c = _client()
    resp = {"choices": [{"message": {"content": "I cannot answer that."}}], "usage": {}}
    with patch.object(c, '_post', new=AsyncMock(return_value=resp)):
        out = await c.complete_json('sys', 'unique-user-3', use_cache=False)
    assert out is None
