"""SQLite memory across sessions and the token-budgeted conversation."""

import json

from jobagent.memory import MAX_USER_CHARS, Conversation, memory_block
from jobagent.store import Store


def test_memory_survives_closing_the_database(tmp_path):
    path = tmp_path / "test.db"
    s = Store(path)
    s.set_preference("remote", "prefer remote")
    s.set_preference("remote", "remote only")
    assert s.save_job("3904982937", "Python Engineer", "ESO")
    assert not s.save_job("3904982937", "Python Engineer", "ESO")
    s.close()
    s2 = Store(path)                                      # new session
    assert s2.get_preferences() == {"remote": "remote only"}
    assert [j["job_id"] for j in s2.list_saved()] == ["3904982937"]
    s2.close()


def test_sql_injection_text_is_stored_as_plain_text(store):
    store.set_preference("location", "Pune'); DROP TABLE preferences; --")
    assert store.get_preferences()["location"].startswith("Pune')")


def test_memory_block_lists_preferences_and_saved_jobs(store):
    store.set_preference("excluded_locations", "Gurugram")
    store.save_job("3904982937", "Python Engineer", "ESO")
    block = memory_block(store)
    assert "excluded_locations=Gurugram" in block and "(3904982937)" in block


def _fake_turn(conv, i):
    messages = conv.start_turn(f"find python jobs {i}")
    jobs = [{"job_id": f"39000000{i}{k}", "title": "Dev", "snippet": "x" * 120} for k in range(5)]
    messages += [
        {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function",
         "function": {"name": "search_jobs", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps({"ok": True, "jobs": jobs})},
        {"role": "assistant", "content": json.dumps(
            {"message": f"answer {i}", "job_ids": [j["job_id"] for j in jobs], "kind": "jobs"})},
    ]
    conv.end_turn(messages)


def test_budget_compacts_then_summarises_and_keeps_pairs(store):
    conv = Conversation("SYSTEM", store, budget=700)
    for i in range(1, 9):
        _fake_turn(conv, i)
    assert conv.history_tokens() <= 700
    assert len(conv.turns[-1]) == 4                            # newest turn kept whole
    assert all(len(t) == 2 for t in conv.turns[:-2])           # older ones compacted
    assert conv.summary and "job_ids: 3900000010" in conv.summary[0]
    sent = conv.start_turn("next")
    call_ids = {c["id"] for m in sent for c in m.get("tool_calls", [])}
    assert all(m["tool_call_id"] in call_ids for m in sent if m["role"] == "tool")
    assert "3900000010" in conv.known_ids                     # survives compaction


def test_long_user_message_is_cut(store):
    msgs = Conversation("S", store).start_turn("x" * 5000)
    assert len(msgs[-1]["content"]) < MAX_USER_CHARS + 30
    assert msgs[-1]["content"].endswith("[message cut: too long]")


def test_failed_turn_is_not_kept(store):
    conv = Conversation("S", store)
    msgs = conv.start_turn("hello")
    conv.end_turn(msgs, keep=False)
    assert conv.turns == []
