"""Run every tool directly with good and bad arguments (no LLM, no key).

Usage: python scripts/try_tools.py"""

import json

from jobagent.store import MemoryStore
from jobagent.tools import run_tool, tool_schemas


MAX_SHOW = 900   # display only; the tool result itself is not cut


def show(name, args, store):
    result = run_tool(name, args, store)
    print(f"\n>>> {name}({args})")
    text = json.dumps(result, indent=1)
    print(text[:MAX_SHOW] + (" ...[display cut]" if len(text) > MAX_SHOW else ""))
    return result


def main():
    schemas = json.dumps(tool_schemas(), separators=(",", ":"))
    print(f"Tool schemas: {len(tool_schemas())} tools, {len(schemas)} chars "
          f"(~{len(schemas) // 4} tokens)")

    store = MemoryStore()
    # Arguments are JSON strings, as the model sends them.
    res = show("search_jobs", '{"query": "python developer", "experience_level": "entry level",'
                              ' "exclude_location": "Gurugram", "limit": 3}', store)
    first_id = res["jobs"][0]["job_id"] if res.get("jobs") else "0"
    show("search_jobs", '{"query": "python developer", "remote_only": true, "limit": 2}', store)
    # Needs the optional India data (scripts/prepare_india.py).
    india = show("search_jobs", '{"query": "python developer", "experience_level": "Entry level",'
                                ' "country": "india", "exclude_location": "Gurugram"}', store)
    show("search_jobs", '{"query": "software engineer", "location": "Bangalore", "limit": 3}',
         store)
    if india.get("jobs"):
        show("get_job_details", json.dumps({"job_id": india["jobs"][0]["job_id"]}), store)
    show("get_job_details", json.dumps({"job_id": first_id}), store)
    show("save_job", json.dumps({"job_id": first_id}), store)
    show("save_job", json.dumps({"job_id": first_id}), store)        # already saved
    show("list_saved", "", store)
    show("remember_preference", '{"key": "remote", "value": "prefer remote"}', store)
    show("remember_preference", '{"key": "excluded_locations", "value": "Gurugram"}', store)

    # Each should return {"ok": false, "error": ...}.
    print("\n---------- bad calls ----------")
    show("search_jobs", '{"query": "x"}', store)                         # too short
    show("search_jobs", '{"query": "nurse", "remote": true}', store)     # wrong arg name
    show("search_jobs", '{"query": "nurse", "experience_level": "Senior"}', store)
    show("search_jobs", '{"query": "nurse", "location": "Atlantis"}', store)   # no matches
    show("get_job_details", '{"job_id": "123456789"}', store)            # unknown id
    show("save_job", '{"job_id": "abc"}', store)                         # not digits
    show("save_job", '{"job_id": "123456789"}', store)                   # valid format, not in data
    show("remember_preference", '{"key": "favourite_food", "value": "pizza"}', store)  # bad key
    show("search_jobs", '{"query": "nurse"', store)                      # broken JSON
    show("apply_to_job", '{}', store)                                    # unknown tool


if __name__ == "__main__":
    main()
