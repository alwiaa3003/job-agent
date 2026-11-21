"""LLM client wrapper that caches every response on disk, so evaluation re-runs are free.

Model errors (empty reply, fake tool call) are cached; quota and network errors are not."""

import hashlib
import json
import time
from collections import deque
from types import SimpleNamespace as NS

from jobagent.config import PROJECT_ROOT
from jobagent.llm import DailyLimitError, EmptyReplyError, LLMClient, ToolUseFailedError

CACHE_DIR = PROJECT_ROOT / "eval" / "cache" / "llm"


def _to_dict(response) -> dict:
    choice = response.choices[0]
    msg = choice.message
    usage = response.usage
    return {
        "content": msg.content,
        "finish_reason": choice.finish_reason,
        "tool_calls": [{"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
                       for c in (msg.tool_calls or [])],
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _from_dict(d: dict):
    calls = [NS(id=c["id"], type="function", function=NS(name=c["name"], arguments=c["arguments"]))
             for c in d["tool_calls"]] or None
    return NS(choices=[NS(finish_reason=d["finish_reason"],
                          message=NS(content=d["content"], tool_calls=calls))],
              usage=NS(prompt_tokens=d["prompt_tokens"], total_tokens=d["total_tokens"]))


class CachingLLM:
    def __init__(self, llm: LLMClient | None = None, model: str | None = None,
                 tokens_per_minute: int = 7000):
        self.llm = llm or LLMClient()
        self.settings = self.llm.settings
        self.model = model or self.settings.model
        self.tpm = tokens_per_minute
        self._window = deque()            # (time, tokens) of recent uncached calls
        self.hits = self.misses = 0
        self.stop_reason = ""             # set on a daily limit; no further calls
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def chat(self, messages, tools=None, model=None, temperature=0.2, max_tokens=1024):
        model = model or self.model
        key_data = {"model": model, "messages": messages, "tools": tools,
                    "temperature": temperature, "max_tokens": max_tokens,
                    "reasoning_effort": self.settings.reasoning_effort}
        key = hashlib.sha256(json.dumps(key_data, sort_keys=True, ensure_ascii=False)
                             .encode("utf-8")).hexdigest()
        path = CACHE_DIR / f"{key}.json"

        if path.exists():
            self.hits += 1
            return self._replay(json.loads(path.read_text(encoding="utf-8")))

        if self.stop_reason:
            raise DailyLimitError(self.stop_reason)
        self.misses += 1
        self._pace()
        try:
            response = self.llm.chat(messages, tools=tools, model=model,
                                     temperature=temperature, max_tokens=max_tokens)
        except ToolUseFailedError as err:
            self._save(path, {"error": "tool_use_failed", "message": str(err),
                              "failed_generation": err.failed_generation})
            self._used(1500)
            raise
        except EmptyReplyError as err:
            self._save(path, {"error": "empty_reply", "message": str(err)})
            self._used(max_tokens)
            raise
        except DailyLimitError as err:
            self.misses -= 1
            self.stop_reason = str(err)
            raise
        data = _to_dict(response)
        self._save(path, data)
        self._used(data["total_tokens"] or 1500)
        return response

    @staticmethod
    def _replay(d: dict):
        if d.get("error") == "tool_use_failed":
            raise ToolUseFailedError(d["message"], d["failed_generation"])
        if d.get("error") == "empty_reply":
            raise EmptyReplyError(d["message"])
        return _from_dict(d)

    @staticmethod
    def _save(path, data: dict) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _used(self, tokens: int) -> None:
        self._window.append((time.time(), tokens))

    def _pace(self) -> None:
        # Stay under the free tier's tokens-per-minute limit.
        expected = 2500                   # typical request + answer
        while True:
            now = time.time()
            while self._window and now - self._window[0][0] > 60:
                self._window.popleft()
            if sum(t for _, t in self._window) + expected <= self.tpm:
                return
            time.sleep(max(1.0, 60 - (now - self._window[0][0])))
