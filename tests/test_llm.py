"""LLM client helpers and the evaluation cache (no real API calls)."""

import json
from types import SimpleNamespace as NS

import httpx
import openai
import pytest

from conftest import raw_reply

from jobagent import llm_cache
from jobagent.config import Settings
from jobagent.llm import (DailyLimitError, EmptyReplyError, LLMClient, LLMError,
                          ToolUseFailedError, _check_not_empty, _is_daily_limit)


def _settings(key=""):
    return Settings(provider="groq", base_url="https://example.invalid", model="m",
                    compare_model="m2", api_key_var="GROQ_API_KEY", reasoning_effort="low",
                    db_path=":memory:", history_budget=2500, job_source="local_csv",
                    live_offline=False, api_key=key)


def test_missing_key_gives_clear_error_and_key_is_never_in_repr():
    with pytest.raises(LLMError, match="No API key found"):
        LLMClient(_settings(""))
    assert "secret-test-value" not in repr(_settings("secret-test-value"))


def test_empty_reply_is_detected():
    with pytest.raises(EmptyReplyError, match="thinking"):
        _check_not_empty(NS(choices=[NS(finish_reason="length",
                                        message=NS(content="", tool_calls=None))]), 40)
    _check_not_empty(raw_reply("Hello"), 40)


def _err429(message, retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after else {}
    resp = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://x"))
    return openai.RateLimitError(message, response=resp, body={"message": message})


def test_daily_limit_detection():
    assert not _is_daily_limit(_err429("... tokens per minute (TPM): Limit 8000 ...", 10))
    assert _is_daily_limit(_err429("... tokens per day (TPD): Limit 200000 ..."))
    assert _is_daily_limit(_err429("Too many requests", 400))


class _Inner:
    settings = NS(model="m", reasoning_effort="low")

    def __init__(self, items):
        self.items, self.calls = list(items), 0

    def chat(self, *a, **k):
        self.calls += 1
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_cache, "CACHE_DIR", tmp_path)
    return tmp_path


def test_cache_second_call_is_free(cache_dir):
    inner = _Inner([raw_reply('{"message": "hi"}')])
    llm = llm_cache.CachingLLM(llm=inner, model="m", tokens_per_minute=10**9)
    msgs = [{"role": "user", "content": "hello"}]
    a = llm.chat(msgs)
    b = llm.chat(msgs)
    assert inner.calls == 1 and (llm.hits, llm.misses) == (1, 1)
    assert a.choices[0].message.content == b.choices[0].message.content


def test_cache_replays_fake_tool_error_but_not_daily_limit(cache_dir):
    inner = _Inner([ToolUseFailedError("bad", '{"name": "json"}'), DailyLimitError("TPD")])
    llm = llm_cache.CachingLLM(llm=inner, model="m", tokens_per_minute=10**9)
    for _ in range(2):                                     # 2nd time comes from the cache
        with pytest.raises(ToolUseFailedError):
            llm.chat([{"role": "user", "content": "a"}])
    with pytest.raises(DailyLimitError):
        llm.chat([{"role": "user", "content": "b"}])
    with pytest.raises(DailyLimitError):                   # stopped: no new call made
        llm.chat([{"role": "user", "content": "c"}])
    assert inner.calls == 2 and llm.stop_reason == "TPD"
    assert len(list(cache_dir.glob("*.json"))) == 1        # the daily limit is not cached
    assert json.loads(next(cache_dir.glob("*.json")).read_text())["error"] == "tool_use_failed"
