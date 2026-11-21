"""The agent loop with a scripted FAKE LLM: every error path, no API calls."""

import json

from conftest import FakeLLM, answer_reply, raw_reply, tool_reply

from jobagent.agent import Agent
from jobagent.llm import EmptyReplyError, LLMError, ToolUseFailedError


def run(script, store, max_steps=5, known=None):
    llm = FakeLLM(script)
    messages = [{"role": "system", "content": "test"},
                {"role": "user", "content": "Find entry-level Python jobs"}]
    result = Agent(llm, store, max_steps=max_steps).run(messages, known)
    return result, llm, messages


def test_tool_then_answer(store):
    r, llm, msgs = run([tool_reply("search_jobs", '{"query": "python engineer", "location": "Belcamp"}'),
                        answer_reply("- Python Engineer - ESO - Belcamp, MD (3904982937)",
                                     ["3904982937"], "jobs")], store)
    assert r.stopped_reason == "answer" and r.steps == 2 and r.structured == "first_try"
    assert r.answer.job_ids == ["3904982937"] and not r.guard.changed
    assert [m["role"] for m in msgs[2:]] == ["assistant", "tool", "assistant"]


def test_tool_error_is_sent_back_so_the_model_can_fix_it(store):
    r, llm, _ = run([tool_reply("search_jobs", '{"query": "python", "experience_level": "Junior"}'),
                     tool_reply("search_jobs", '{"query": "python", "experience_level": "Entry level"}',
                                "call_2"),
                     answer_reply("Done.")], store)
    tool_msg = [m for m in llm.seen[1] if m["role"] == "tool"][0]
    assert '"ok": false' in tool_msg["content"] and "hint" in tool_msg["content"]
    assert [c.ok for c in r.tool_calls] == [False, True]


def test_empty_reply_is_retried_once(store):
    r, _, _ = run([EmptyReplyError("no answer"), answer_reply("Which city?", kind="clarify")], store)
    assert r.stopped_reason == "answer" and r.answer.kind == "clarify"


def test_api_error_gives_a_friendly_answer_not_a_crash(store):
    r, _, _ = run([LLMError("Rate limit reached (HTTP 429)")], store)
    assert r.stopped_reason == "llm_error" and "429" in r.reply and r.answer.kind == "error"


def test_step_limit(store):
    script = [tool_reply("search_jobs", json.dumps({"query": f"python {i}"}), f"c{i}")
              for i in range(10)]
    r, llm, _ = run(script, store, max_steps=3)
    assert r.stopped_reason == "max_steps" and r.steps == 3 and len(llm.seen) == 3


def test_identical_repeat_call_is_refused(store):
    r, _, _ = run([tool_reply("search_jobs", '{"query": "python", "remote_only": true}'),
                   tool_reply("search_jobs", '{"remote_only": true, "query": "python"}', "c2"),
                   answer_reply("None found.", kind="no_results")], store)
    assert "already made this exact call" in r.tool_calls[1].error


def test_invalid_json_answer_retried_once_without_tools(store):
    r, llm, msgs = run([raw_reply('{"message": "x", "job_ids": ["abc"], "kind": "list"}'),
                        answer_reply("Fixed.")], store)
    assert r.structured == "after_retry" and r.reply == "Fixed."
    assert llm.tools_sent == [True, False]          # the retry has no tools
    assert "invalid" in llm.seen[1][-1]["content"]
    assert json.loads(msgs[-1]["content"])["message"] == "Fixed."   # only clean JSON kept


def test_invalid_twice_falls_back_safely(store):
    r, _, _ = run([raw_reply('{"message": "broken'), raw_reply('{"message": "broken')], store)
    assert r.structured == "fallback" and "{" not in r.reply and r.answer.kind == "error"


def test_answer_sent_as_fake_tool_is_recovered_without_extra_call(store):
    fg = '{"name": "json", "arguments": {"message": "What role?", "job_ids": [], "kind": "clarify"}}'
    r, llm, _ = run([ToolUseFailedError("bad tool", fg)], store)
    assert r.recovered_fake_tool and r.answer.kind == "clarify" and len(llm.seen) == 1


def test_invented_job_is_removed_by_the_guard(store):
    r, _, msgs = run([tool_reply("search_jobs", '{"query": "python engineer", "location": "Belcamp", "limit": 1}'),
                      answer_reply("- Python Engineer - ESO - Belcamp, MD (3904982937)\n"
                                   "- Senior Python Engineer - Google - Remote (3999999999)",
                                   ["3904982937", "3999999999"], "jobs")], store)
    assert r.guard.removed_ids == ["3999999999"] and "3999999999" not in r.reply
    assert "3999999999" not in msgs[-1]["content"]   # not kept in history either


def test_saved_jobs_count_as_known(store):
    store.save_job("3904982937", "Python Engineer", "ESO")
    r, _, _ = run([answer_reply("Saved: Python Engineer - ESO (3904982937)", ["3904982937"])], store)
    assert r.answer.job_ids == ["3904982937"] and not r.guard.changed


def test_fake_tool_twice_is_a_model_failure_not_an_outage(store):
    """Regression (eval V2/D4): two fake tool calls must end in a fallback, not llm_error."""
    useless = '{"name": "browser.search", "arguments": {}}'
    r, llm, _ = run([ToolUseFailedError("bad tool", useless),
                     ToolUseFailedError("bad tool again", useless),
                     answer_reply("Which role are you looking for?", kind="clarify")], store)
    assert r.stopped_reason == "answer" and r.structured == "after_retry"
    assert r.answer.kind == "clarify" and len(llm.seen) == 3


def test_retry_that_hits_an_outage_is_reported_as_llm_error(store):
    r, _, _ = run([raw_reply("not json"), LLMError("Could not reach the API")], store)
    assert r.stopped_reason == "llm_error"           # the runner won't count it


def test_retry_answered_through_a_fake_tool_is_used(store):
    fg = '{"name": "json", "arguments": {"message": "Fixed", "job_ids": [], "kind": "info"}}'
    r, _, _ = run([raw_reply("not json"), ToolUseFailedError("bad", fg)], store)
    assert r.structured == "after_retry" and r.reply == "Fixed"
