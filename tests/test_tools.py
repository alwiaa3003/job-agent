"""Tools: argument validation, results, and errors (no LLM involved)."""

import json

from conftest import needs_india

from jobagent.tools import run_tool, tool_schemas

REAL_JOB = "3904982937"   # Python Engineer - ESO (in data/jobs.csv)


def test_schemas_are_generated_for_all_five_tools():
    names = [t["function"]["name"] for t in tool_schemas()]
    assert names == ["search_jobs", "get_job_details", "save_job", "list_saved",
                     "remember_preference"]


def test_search_returns_at_most_limit_with_source(store):
    res = run_tool("search_jobs", '{"query": "python developer", "limit": 3}', store)
    assert res["ok"] and 1 <= res["count"] <= 3
    assert all(j["source"] == "local_csv" for j in res["jobs"])


@needs_india
def test_search_normalises_level_and_country_spelling(store):
    res = run_tool("search_jobs", json.dumps({"query": "python developer",
                                              "experience_level": "entry level",
                                              "country": "india"}), store)
    assert res["ok"] and res["count"] > 0                 # else the all() below passes vacuously
    assert all(j["level"] == "Entry level" and j["country"] == "India" for j in res["jobs"])


def test_unknown_argument_is_rejected(store):
    res = run_tool("search_jobs", '{"query": "nurse", "remote": true}', store)
    assert not res["ok"] and "remote: Extra inputs are not permitted" in res["error"]


def test_bad_level_lists_allowed_values(store):
    res = run_tool("search_jobs", '{"query": "nurse", "experience_level": "Senior"}', store)
    assert not res["ok"] and "Entry level" in res["error"]


def test_limit_above_five_is_rejected(store):
    res = run_tool("search_jobs", '{"query": "nurse", "limit": 50}', store)
    assert not res["ok"] and "limit" in res["error"]


def test_broken_json_and_unknown_tool(store):
    assert "not valid JSON" in run_tool("search_jobs", '{"query": "nurse"', store)["error"]
    assert "Unknown tool" in run_tool("apply_to_job", "{}", store)["error"]


def test_no_results_gives_a_note(store):
    res = run_tool("search_jobs", '{"query": "nurse", "location": "Atlantis"}', store)
    assert res["ok"] and res["count"] == 0 and "note" in res


def test_details_of_real_and_unknown_job(store):
    res = run_tool("get_job_details", json.dumps({"job_id": REAL_JOB}), store)
    assert res["ok"] and res["job"]["title"] == "Python Engineer"
    assert len(res["job"]["description"]) <= 600
    assert not run_tool("get_job_details", '{"job_id": "123456789"}', store)["ok"]


def test_save_job_only_real_jobs_and_no_duplicates(store):
    first = run_tool("save_job", json.dumps({"job_id": REAL_JOB}), store)
    again = run_tool("save_job", json.dumps({"job_id": REAL_JOB}), store)
    assert first["ok"] and not first["already_saved"] and again["already_saved"]
    assert not run_tool("save_job", '{"job_id": "123456789"}', store)["ok"]
    assert not run_tool("save_job", '{"job_id": "abc"}', store)["ok"]
    assert run_tool("list_saved", "", store)["count"] == 1


def test_remember_preference_allowed_keys_only(store):
    ok = run_tool("remember_preference", '{"key": "Excluded_Locations", "value": "Gurugram"}', store)
    assert ok["ok"] and store.get_preferences() == {"excluded_locations": "Gurugram"}
    bad = run_tool("remember_preference", '{"key": "favourite_food", "value": "pizza"}', store)
    assert not bad["ok"] and "Allowed: remote, location, level, excluded_locations" in bad["error"]
