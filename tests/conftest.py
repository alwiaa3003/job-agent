"""Shared fixtures: fake LLM, sockets blocked, Hugging Face offline, local CSV source."""

import json
import os
import socket
from types import SimpleNamespace as NS

import pytest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _no_network(*args, **kwargs):
    raise OSError("Network access is blocked in tests")


socket.socket.connect = _no_network
socket.create_connection = _no_network

# The India data is optional and not in the repo; tests that need it use @needs_india.
from jobagent.config import PROJECT_ROOT  # noqa: E402

HAS_INDIA_DATA = (PROJECT_ROOT / "data" / "jobs_india.csv").exists()
needs_india = pytest.mark.skipif(not HAS_INDIA_DATA, reason="optional India data not "
                                 "installed (see README: python scripts/prepare_india.py)")


def pytest_sessionstart(session):
    # Downloads are blocked, so fail early with one clear message if the model isn't cached.
    from huggingface_hub import try_to_load_from_cache
    from jobagent.embed import EMBED_MODEL
    if not isinstance(try_to_load_from_cache(f"sentence-transformers/{EMBED_MODEL}",
                                             "config.json"), str):
        pytest.exit("The embedding model is not downloaded yet (tests never use the "
                    "internet). Run once:  python scripts/try_tools.py   (downloads "
                    "about 90 MB), then run pytest again.", returncode=1)


@pytest.fixture(autouse=True)
def local_source():
    from jobagent.sources import LocalCSVSource, set_source
    set_source(LocalCSVSource())
    yield


@pytest.fixture
def store():
    from jobagent.store import Store
    s = Store(":memory:")
    yield s
    s.close()


def tool_reply(name, arguments, call_id="call_1"):
    """Fake response with one tool call."""
    call = NS(id=call_id, function=NS(name=name, arguments=arguments))
    return NS(choices=[NS(finish_reason="tool_calls", message=NS(content=None, tool_calls=[call]))],
              usage=NS(prompt_tokens=100, total_tokens=150))


def raw_reply(content):
    """Fake response with this exact text."""
    return NS(choices=[NS(finish_reason="stop", message=NS(content=content, tool_calls=None))],
              usage=NS(prompt_tokens=100, total_tokens=150))


def answer_reply(message, job_ids=(), kind="info"):
    """Fake response with a valid JSON final answer."""
    return raw_reply(json.dumps({"message": message, "job_ids": list(job_ids), "kind": kind}))


class FakeLLM:
    """Returns or raises scripted items in order; records the messages and whether tools were sent."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []
        self.tools_sent = []

    def chat(self, messages, tools=None, **kwargs):
        self.seen.append([dict(m) for m in messages])
        self.tools_sent.append(bool(tools))
        if not self.script:
            raise AssertionError("FakeLLM was called more times than scripted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
