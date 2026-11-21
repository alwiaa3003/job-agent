"""Run the tools against the live Himalayas source (no LLM).

Usage: python scripts/try_live.py [--offline]   (--offline: cache only, sockets blocked)"""

import json
import socket
import sys

from jobagent.live import HimalayasSource, make_job_id
from jobagent.sources import set_source
from jobagent.store import MemoryStore
from jobagent.tools import run_tool

SEARCHES = [
    '{"query": "python developer", "country": "US", "limit": 3}',
    '{"query": "python developer", "country": "India", "experience_level": "Entry level", "limit": 3}',
    '{"query": "data analyst", "location": "Germany", "limit": 3}',
    '{"query": "data analyst", "location": "Pune", "limit": 3}',
]


def block_network():
    def refuse(*args, **kwargs):
        raise OSError("network blocked by try_live.py --offline")
    socket.socket.connect = refuse
    socket.create_connection = refuse


def show(name, args, store):
    result = run_tool(name, args, store)
    print(f"\n>>> {name}({args})")
    if result.get("ok") and "jobs" in result:
        print(f"  ok, {result['count']} job(s)")
        for j in result["jobs"]:
            print(f"  - {j['title']} | {j['company']} | {j['location']} | level: {j['level']} "
                  f"| id {j['job_id']} | source={j['source']} | posted {j['posted']}")
    else:
        text = json.dumps(result, ensure_ascii=False)
        print("  " + text[:700] + (" ...[display cut]" if len(text) > 700 else ""))
    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    offline = "--offline" in sys.argv
    if offline:
        block_network()
    set_source(HimalayasSource(offline=offline))
    print(f"Mode: {'OFFLINE (cache only, network blocked)' if offline else 'online'}")

    guid = "https://himalayas.app/companies/example/jobs/python-developer"
    print("job_id is stable:", make_job_id(guid), "==", make_job_id(guid))

    store = MemoryStore()
    first = None
    for args in SEARCHES:
        res = show("search_jobs", args, store)
        if first is None and res.get("jobs"):
            first = res["jobs"][0]["job_id"]
    if first:
        show("get_job_details", json.dumps({"job_id": first}), store)
        show("save_job", json.dumps({"job_id": first}), store)
    if offline:
        print("\n--- a search that was never cached:")
        show("search_jobs", '{"query": "underwater basket weaving"}', store)


if __name__ == "__main__":
    main()
