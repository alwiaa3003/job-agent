"""Run evaluation scenarios (local CSV source, in-memory DB per scenario) and compute metrics."""

import json
import re

from jobagent.agent import Agent
from jobagent.guard import JOB_ID_IN_TEXT
from jobagent.memory import Conversation
from jobagent.prompts import system_prompt
from jobagent.sources import LocalCSVSource, get_source, set_source
from jobagent.store import Store
from jobagent.tools import REGISTRY

ARG_ERRORS = ("Invalid arguments", "Arguments are not valid JSON", "Unknown tool",
              "Arguments must be")


def _turn_log(user: str, result, known: set[str]) -> dict:
    a, g = result.answer, result.guard
    user_ids = set(JOB_ID_IN_TEXT.findall(user))
    shown = set(a.job_ids) | set(JOB_ID_IN_TEXT.findall(a.message))
    return {
        "user": user,
        "tool_calls": [{"name": c.name, "arguments": c.arguments, "ok": c.ok, "error": c.error}
                       for c in result.tool_calls],
        "answer": a.model_dump(),
        "structured": result.structured,
        "format_error": result.format_error,
        "fake_tool": result.recovered_fake_tool,
        "stopped": result.stopped_reason,
        "steps": result.steps,
        "prompt_tokens": result.prompt_tokens,
        "guard": {"removed_ids": g.removed_ids, "removed_lines": g.removed_lines,
                  "fixed_lines": g.fixed_lines, "replaced": g.replaced},
        "shown_ungrounded": sorted(shown - known - user_ids),
    }


def run_scenario(scn: dict, llm, prompt_version: str) -> dict:
    set_source(LocalCSVSource())
    store = Store(":memory:")
    for job_id in scn.get("setup", {}).get("saved", []):
        job = get_source().get_job(job_id)
        store.save_job(job["job_id"], job["title"], job["company"])
    agent = Agent(llm, store)

    turns = []
    for session in scn["sessions"]:                 # new history, same database
        conv = Conversation(system_prompt("local_csv", prompt_version), store)
        for user in session:
            messages = conv.start_turn(user)
            result = agent.run(messages, conv.known_ids)
            conv.end_turn(messages, keep=result.stopped_reason != "llm_error")
            known = conv.known_ids | {j["job_id"] for j in store.list_saved()}
            turns.append(_turn_log(user, result, known))

    saved = [j["job_id"] for j in store.list_saved()]
    return {"id": scn["id"], "category": scn["category"], "memory": scn.get("memory", False),
            "turns": turns, "saved_at_end": saved,
            "checks": check(scn["expect"], turns[-1], saved)}


def _arg_matches(value, accepted: list) -> bool:
    if isinstance(value, bool) or any(isinstance(x, bool) for x in accepted):
        return value in accepted
    return any(str(x).lower() in str(value).lower() for x in accepted)


def _call_args(call: dict) -> dict | None:
    # Compare validated arguments, so "india" counts as "India" and defaults are filled in.
    model = REGISTRY.get(call["name"], (None,))[0]
    try:
        return model.model_validate(json.loads(call["arguments"] or "{}")).model_dump()
    except Exception:
        return None


def check(expect: dict, turn: dict, saved: list[str]) -> dict:
    """Pass/fail per check for the last turn; checks that don't apply are left out."""
    called = [c["name"] for c in turn["tool_calls"]]
    msg = turn["answer"]["message"].lower()
    answer_ids = set(turn["answer"]["job_ids"]) | set(JOB_ID_IN_TEXT.findall(msg))
    out = {"answered": turn["stopped"] == "answer"}
    if "tools" in expect:
        out["tool"] = all(t in called for t in expect["tools"])
    if expect.get("no_tools"):
        out["tool"] = not called
    if "args" in expect:
        ok = True
        for tool, wanted in expect["args"].items():
            calls = [_call_args(c) for c in turn["tool_calls"] if c["name"] == tool]
            ok &= any(a is not None and all(_arg_matches(a.get(k), v) for k, v in wanted.items())
                      for a in calls)
        out["args"] = ok
    if "kinds" in expect:
        out["kind"] = turn["answer"]["kind"] in expect["kinds"]
    if expect.get("no_jobs"):
        out["no_jobs"] = not turn["answer"]["job_ids"]
    if "mention" in expect:
        out["mention"] = any(m.lower() in msg for m in expect["mention"])
    if expect.get("mention_saved"):
        out["mention_saved"] = bool(saved) and bool(answer_ids & set(saved))
    if "not_regex" in expect:
        out["not_regex"] = not re.search(expect["not_regex"], msg)
    return out


def _rate(good: int, total: int) -> dict:
    return {"good": good, "total": total, "pct": round(100 * good / total, 1) if total else None}


def summarize(results: list[dict]) -> dict:
    """Aggregate metrics over scenario results."""
    turns = [t for r in results for t in r["turns"]]
    calls = [c for t in turns for c in t["tool_calls"]]
    with_check = lambda key: [r for r in results if key in r["checks"]]
    passed = lambda r: all(r["checks"].values())

    categories = {}
    for r in results:
        cat = categories.setdefault(r["category"], [0, 0])
        cat[0] += passed(r)
        cat[1] += 1

    return {
        "scenarios_passed": _rate(sum(passed(r) for r in results), len(results)),
        "correct_tool": _rate(sum(r["checks"]["tool"] for r in with_check("tool")),
                              len(with_check("tool"))),
        "valid_arguments": _rate(sum(not c["error"].startswith(ARG_ERRORS) for c in calls),
                                 len(calls)),
        "correct_arguments": _rate(sum(r["checks"]["args"] for r in with_check("args")),
                                   len(with_check("args"))),
        "json_valid_first_try": _rate(sum(t["structured"] == "first_try" for t in turns),
                                      sum(t["structured"] != "none" for t in turns)),
        "json_valid_after_retry": _rate(sum(t["structured"] in ("first_try", "after_retry")
                                            for t in turns),
                                        sum(t["structured"] != "none" for t in turns)),
        "no_invented_jobs_model": _rate(sum(not (t["guard"]["removed_ids"] or t["guard"]["replaced"])
                                            for t in turns), len(turns)),
        "shown_answers_grounded": _rate(sum(not t["shown_ungrounded"] for t in turns), len(turns)),
        "answer_kind_correct": _rate(sum(r["checks"]["kind"] for r in with_check("kind")),
                                     len(with_check("kind"))),
        "memory_recall": _rate(sum(passed(r) for r in results if r["memory"]),
                               sum(r["memory"] for r in results)),
        "by_category": {k: _rate(*v) for k, v in categories.items()},
        "fake_tool_recoveries": sum(t["fake_tool"] for t in turns),
        "guard_fixed_lines": sum(t["guard"]["fixed_lines"] for t in turns),
        "llm_errors": sum(t["stopped"] == "llm_error" for t in turns),
        "model_calls": sum(t["steps"] for t in turns),
        "prompt_tokens": sum(sum(t["prompt_tokens"]) for t in turns),
        # Groq reports no usage for tool_use_failed responses.
        "calls_without_token_count": sum(t["steps"] - len(t["prompt_tokens"]) for t in turns),
        "turns": len(turns),
    }
