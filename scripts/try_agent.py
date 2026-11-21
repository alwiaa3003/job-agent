"""Walk through the agent loop's error paths with a scripted fake model (no API, no key).

Usage: python scripts/try_agent.py"""

import json
from types import SimpleNamespace as NS

from jobagent.agent import Agent
from jobagent.llm import EmptyReplyError, LLMError, ToolUseFailedError
from jobagent.store import MemoryStore


def tool_reply(name, arguments, call_id="call_1"):
    call = NS(id=call_id, function=NS(name=name, arguments=arguments))
    return NS(choices=[NS(message=NS(content=None, tool_calls=[call]))])


def raw_reply(content):
    return NS(choices=[NS(message=NS(content=content, tool_calls=None))])


def text_reply(text, job_ids=(), kind="info"):
    return raw_reply(json.dumps({"message": text, "job_ids": list(job_ids), "kind": kind}))


class FakeLLM:
    """Returns or raises the scripted items in order and records the messages it received."""
    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def chat(self, messages, tools=None, **kwargs):
        self.seen.append(list(messages))
        item = self.script.pop(0) if self.script else text_reply("(script ended)")
        if isinstance(item, Exception):
            raise item
        return item


def run(title, script):
    print(f"\n=== {title}")
    llm = FakeLLM(script)
    messages = [{"role": "system", "content": "test"},
                {"role": "user", "content": "Find entry-level Python jobs"}]
    result = Agent(llm, MemoryStore()).run(messages)
    for c in result.tool_calls:
        print(f"  [tool] {c.name}({c.arguments}) -> {'ok' if c.ok else 'ERROR: ' + c.error}")
    print(f"  steps={result.steps}  stopped_reason={result.stopped_reason}  "
          f"json={result.structured}")
    if result.format_error:
        print(f"  format error: {result.format_error}")
    print(f"  answer: kind={result.answer.kind} job_ids={result.answer.job_ids}"
          f"  recovered_fake_tool={result.recovered_fake_tool}")
    if result.guard.changed:
        g = result.guard
        print(f"  guard: removed_ids={g.removed_ids} removed_lines={g.removed_lines} "
              f"replaced={g.replaced}")
    print(f"  reply: {result.reply}")
    return llm


def main():
    llm = run("1. Bad argument, then the model fixes it", [
        tool_reply("search_jobs", '{"query": "python", "experience_level": "Junior"}'),
        tool_reply("search_jobs", '{"query": "python", "experience_level": "Entry level"}',
                   "call_2"),
        text_reply("- Python Engineer - ESO - Belcamp, MD (3904982937)",
                   ["3904982937"], "jobs"),
    ])
    tool_msg = [m for m in llm.seen[1] if m["role"] == "tool"][0]
    print(f"  what the model saw after its bad call: {tool_msg['content']}")

    run("2. Empty reply once, then a real answer (one automatic retry)", [
        EmptyReplyError("The model used all 1024 tokens on thinking and wrote no answer."),
        text_reply("Which city or state are you interested in?"),
    ])

    run("3. Rate limit (429) still failing after the client's retries", [
        LLMError("Rate limit reached (HTTP 429) and retries ran out. Wait a minute."),
    ])

    run("4. Model keeps calling tools forever -> step limit", [
        tool_reply("search_jobs", f'{{"query": "python {i}"}}', f"call_{i}") for i in range(10)
    ])

    run("5. Exact same call twice -> second one is refused", [
        tool_reply("search_jobs", '{"query": "python", "remote_only": true}'),
        tool_reply("search_jobs", '{"remote_only": true,  "query": "python"}', "call_2"),
        text_reply("No remote Python jobs found.", kind="no_results"),
    ])

    llm = run("6. Invalid final answer (bad kind + bad job_id), fixed after ONE retry", [
        raw_reply('```json\n{"message": "Here you go", "job_ids": ["abc"], "kind": "list"}\n```'),
        text_reply("Here you go", [], "info"),
    ])
    print(f"  what the model was told: {llm.seen[1][-1]['content']}")

    run("7. Plain text instead of JSON, twice -> safe fallback (text kept)", [
        raw_reply("Which city are you interested in?"),
        raw_reply("Which city are you interested in?"),
    ])

    run("8. Broken JSON twice -> fallback message (broken JSON never shown)", [
        raw_reply('{"message": "Jobs: ...", "job_ids": [3904982937'),
        raw_reply('{"message": "Jobs: ...", "job_ids": [3904982937'),
    ])

    # gpt-oss on Groq really does this (HTTP 400 tool_use_failed).
    run("9. Answer sent as a fake tool named 'json' -> recovered, no extra call", [
        ToolUseFailedError("invalid tool call", '{"name": "json", "arguments": '
                           '{"message": "What kind of role?", "job_ids": [], "kind": "clarify"}}'),
    ])

    run("11. Real search, then the model adds an INVENTED job -> guard removes it", [
        tool_reply("search_jobs", '{"query": "python engineer", "location": "Belcamp", "limit": 1}'),
        text_reply("- Python Engineer - ESO - Belcamp, MD (3904982937)\n"
                   "- Senior Python Engineer - Google - Remote (3999999999)",
                   ["3904982937", "3999999999"], "jobs"),
    ])

    run("10. Fake tool with no usable answer -> one call without tools", [
        ToolUseFailedError("invalid tool call", '{"name": "browser.search", "arguments": {}}'),
        text_reply("Sorry, I can't browse the web.", kind="refuse"),
    ])


if __name__ == "__main__":
    main()
