"""Show SQLite memory surviving a restart and the conversation token budget (no LLM, no key).

Usage: python scripts/try_memory.py"""

import json
import tempfile
from pathlib import Path

from jobagent.memory import Conversation, estimate_tokens
from jobagent.store import Store


def part1_persistence():
    print("=== Part 1: memory survives closing the program")
    path = Path(tempfile.mkdtemp()) / "demo.db"

    store = Store(path)
    store.set_preference("remote", "prefer remote")
    store.set_preference("excluded_locations", "Gurugram")
    store.set_preference("remote", "remote only")      # replaces the earlier value
    print("save_job first time :", store.save_job("3904982937", "Python Engineer", "ESO"))
    print("save_job second time:", store.save_job("3904982937", "Python Engineer", "ESO"))
    store.set_preference("location", "Pune'); DROP TABLE preferences; --")  # stored as text
    store.close()

    store = Store(path)                                # new connection = next session
    print("preferences after reopening:", store.get_preferences())
    print("saved jobs after reopening :", store.list_saved())
    store.close()


def fake_turn(i: int, n_jobs: int = 5) -> list[dict]:
    jobs = [{"job_id": f"39000000{i}{k}", "title": f"Python Developer {i}.{k}",
             "company": "Acme", "location": "Austin, TX", "snippet": "x" * 120}
            for k in range(n_jobs)]
    ids = [j["job_id"] for j in jobs]
    return [
        {"role": "user", "content": f"Find python jobs, variant {i}"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function",
         "function": {"name": "search_jobs", "arguments": json.dumps({"query": f"python {i}"})}}]},
        {"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps({"ok": True, "jobs": jobs})},
        {"role": "assistant", "content": json.dumps(
            {"message": f"Here are {n_jobs} Python jobs (variant {i}).", "job_ids": ids,
             "kind": "jobs"})},
    ]


def part2_budget():
    print("\n=== Part 2: token budget (budget = 1200 tokens, last 2 turns kept whole)")
    store = Store(":memory:")
    store.set_preference("remote", "prefer remote")
    conv = Conversation("SYSTEM PROMPT", store, budget=1200)
    for i in range(1, 9):
        messages = conv.start_turn(f"Find python jobs, variant {i}")
        messages += fake_turn(i)[1:]
        conv.end_turn(messages)
        sizes = [len(t) for t in conv.turns]
        print(f"after turn {i}: history ~{conv.history_tokens():4d} tokens | "
              f"messages per kept turn {sizes} | summary lines {len(conv.summary)}")

    print("\nOne uncompacted turn alone is ~", estimate_tokens(fake_turn(1)), "tokens")
    sent = conv.start_turn("next question")
    ok = all(m["role"] != "tool" or any(c["id"] == m["tool_call_id"]
             for p in sent[:i] for c in p.get("tool_calls", []))
             for i, m in enumerate(sent))
    print("every tool result has its tool call:", ok)
    print("\nSystem message sent with the next request:\n" + sent[0]["content"])


if __name__ == "__main__":
    part1_persistence()
    part2_budget()
