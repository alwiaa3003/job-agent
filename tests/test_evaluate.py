"""Evaluation checks and metrics, on hand-made turn logs (no LLM)."""

from jobagent.eval_scenarios import SCENARIOS
from jobagent.evaluate import check, summarize


def _turn(calls=(), kind="jobs", message="", job_ids=(), structured="first_try", guard=None):
    return {"tool_calls": [{"name": n, "arguments": a, "ok": True, "error": e} for n, a, e in calls],
            "answer": {"message": message, "job_ids": list(job_ids), "kind": kind},
            "structured": structured, "fake_tool": False, "stopped": "answer", "steps": 1,
            "prompt_tokens": [100], "shown_ungrounded": [],
            "guard": guard or {"removed_ids": [], "removed_lines": 0, "fixed_lines": 0,
                               "replaced": False}}


def test_scenario_ids_unique_and_about_thirty():
    ids = [s["id"] for s in SCENARIOS]
    assert len(ids) == len(set(ids)) and 28 <= len(ids) <= 35


def test_args_check_uses_validated_values():
    expect = {"tools": ["search_jobs"],
              "args": {"search_jobs": {"experience_level": ["Entry level"], "country": ["India"]}}}
    good = _turn([("search_jobs", '{"query": "py", "experience_level": "entry level", '
                                  '"country": "india"}', "")])
    bad = _turn([("search_jobs", '{"query": "py"}', "")])
    assert check(expect, good, []) == {"answered": True, "tool": True, "args": True}
    assert check(expect, bad, [])["args"] is False


def test_no_tools_mention_and_regex_checks():
    t = _turn(kind="clarify", message="Which role do you want?")
    assert all(check({"no_tools": True, "kinds": ["clarify"], "no_jobs": True}, t, []).values())
    salary = _turn(message="It pays $120,000 a year.")
    assert check({"not_regex": r"[$]\s?\d"}, salary, [])["not_regex"] is False
    assert check({"mention_saved": True}, _turn(message="(3904982937)"), ["3904982937"])["mention_saved"]


def test_summary_counts():
    results = [
        {"category": "a", "memory": False, "checks": {"answered": True, "tool": True},
         "turns": [_turn([("search_jobs", "{}", "")])]},
        {"category": "a", "memory": True, "checks": {"answered": True, "tool": False},
         "turns": [_turn([("search_jobs", "{}", "Invalid arguments - x")], structured="after_retry",
                         guard={"removed_ids": ["1"], "removed_lines": 1, "fixed_lines": 0,
                                "replaced": False})]},
    ]
    s = summarize(results)
    assert s["scenarios_passed"] == {"good": 1, "total": 2, "pct": 50.0}
    assert s["valid_arguments"]["good"] == 1 and s["json_valid_first_try"]["good"] == 1
    assert s["no_invented_jobs_model"]["good"] == 1 and s["memory_recall"]["good"] == 0
