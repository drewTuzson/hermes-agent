import asyncio

import pytest

from tools.send_message_tool import (
    _send_telegram_message_with_retry,
    _telegram_retry_delay,
)


class ConnectTimeout(Exception):
    pass


class PoolTimeout(Exception):
    pass


class GenericTimedOut(Exception):
    pass


class FakeBot:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    async def send_message(self, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return {"ok": True, "kwargs": kwargs}


def _with_cause(exc, cause):
    exc.__cause__ = cause
    return exc


def test_generic_timeout_is_not_retried_to_avoid_duplicates():
    assert _telegram_retry_delay(GenericTimedOut("Timed out"), 0) is None


def test_wrapped_connect_timeout_is_retried():
    exc = _with_cause(GenericTimedOut("Timed out"), ConnectTimeout("connect timed out"))
    assert _telegram_retry_delay(exc, 0) == 1.0
    assert _telegram_retry_delay(exc, 1) == 2.0


def test_pool_timeout_not_sent_message_is_retried():
    exc = _with_cause(
        GenericTimedOut("Timed out"),
        PoolTimeout("Pool timeout: All connections in the connection pool are occupied. Request was *not* sent to Telegram."),
    )
    assert _telegram_retry_delay(exc, 0) == 1.0


def test_send_telegram_message_retries_safe_timeout(monkeypatch):
    waits = []

    async def fake_sleep(delay):
        waits.append(delay)

    monkeypatch.setattr("tools.send_message_tool.asyncio.sleep", fake_sleep)
    failure = _with_cause(GenericTimedOut("Timed out"), ConnectTimeout("connect timed out"))
    bot = FakeBot([failure])

    result = asyncio.run(_send_telegram_message_with_retry(bot, chat_id=1, text="hello"))

    assert result["ok"] is True
    assert bot.calls == 2
    assert waits == [1.0]


def test_send_telegram_message_does_not_retry_generic_timeout(monkeypatch):
    async def fake_sleep(delay):
        raise AssertionError("generic timeout should not sleep/retry")

    monkeypatch.setattr("tools.send_message_tool.asyncio.sleep", fake_sleep)
    bot = FakeBot([GenericTimedOut("Timed out")])

    with pytest.raises(GenericTimedOut):
        asyncio.run(_send_telegram_message_with_retry(bot, chat_id=1, text="hello"))

    assert bot.calls == 1
