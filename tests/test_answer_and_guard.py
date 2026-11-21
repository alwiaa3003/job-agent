"""Structured final answers (Pydantic) and the grounding guard."""

from jobagent.answer import FinalAnswer, parse_answer
from jobagent.guard import check_answer, ids_from_tool_messages

KNOWN = {"3904982937", "3889738868"}


def test_parse_valid_and_fenced_json():
    ans, err = parse_answer('```json\n{"message": "Hi", "job_ids": [], "kind": "clarify"}\n```')
    assert ans and not err and ans.kind == "clarify"


def test_parse_rejects_bad_kind_and_bad_id():
    ans, err = parse_answer('{"message": "x", "job_ids": ["abc"], "kind": "list"}')
    assert ans is None and "job_ids" in err and "kind" in err


def test_parse_rejects_plain_text_and_extra_fields():
    assert parse_answer("Which city?")[0] is None
    assert parse_answer('{"message": "x", "extra": 1}')[0] is None


def test_guard_keeps_grounded_answer():
    a = FinalAnswer(message="- Python Engineer - ESO - Belcamp, MD (3904982937)",
                    job_ids=["3904982937"], kind="jobs")
    out, rep = check_answer(a, KNOWN)
    assert out == a and not rep.changed


def test_guard_drops_invented_id_only_in_job_ids():
    a = FinalAnswer(message="Saved: Online Data Analyst - TELUS International AI Data Solutions (3889738868)",
                    job_ids=["3889738860", "3889738868"], kind="info")
    out, rep = check_answer(a, KNOWN)
    assert out.job_ids == ["3889738868"] and rep.removed_ids == ["3889738860"]


def test_guard_removes_invented_line_and_adds_note():
    a = FinalAnswer(message="- Python Engineer - ESO (3904982937)\n- AI Lead - Google (3999999999)",
                    job_ids=["3904982937", "3999999999"], kind="jobs")
    out, rep = check_answer(a, KNOWN)
    assert "3999999999" not in out.message and "could not verify" in out.message
    assert out.job_ids == ["3904982937"] and rep.removed_lines == 1


def test_guard_rewrites_mixed_up_line_from_data():
    a = FinalAnswer(message="- Staff Engineer - Microsoft (3904982937)",
                    job_ids=["3904982937"], kind="jobs")
    out, rep = check_answer(a, KNOWN)
    assert "Python Engineer - ESO" in out.message and rep.fixed_lines == 1


def test_guard_replaces_jobs_answer_without_any_verified_id():
    a = FinalAnswer(message="- AI Engineer at OpenAI\n- SWE at Apple", kind="jobs")
    out, rep = check_answer(a, KNOWN)
    assert rep.replaced and out.kind == "error" and not out.job_ids


def test_guard_allows_repeating_users_own_missing_id_but_never_verifies_it():
    a = FinalAnswer(message="There is no job with job_id 3999999999.", kind="info")
    out, rep = check_answer(a, KNOWN, {"3999999999"})
    assert out.message == a.message and not out.job_ids
    b = FinalAnswer(message="- ML Engineer (3999999999)", job_ids=["3999999999"], kind="jobs")
    assert check_answer(b, KNOWN, {"3999999999"})[1].replaced


def test_salary_and_year_are_not_job_ids():
    a = FinalAnswer(message="ESO (3904982937) pays 120000 USD (2024).",
                    job_ids=["3904982937"], kind="jobs")
    assert not check_answer(a, KNOWN)[1].changed


def test_ids_from_tool_messages_ignores_errors():
    msgs = [{"role": "tool", "content": '{"ok": true, "jobs": [{"job_id": "111111111"}]}'},
            {"role": "tool", "content": '{"ok": true, "job": {"job_id": "222222222"}}'},
            {"role": "tool", "content": '{"ok": true, "saved": "333333333"}'},
            {"role": "tool", "content": '{"ok": false, "error": "No job with job_id 444444444."}'}]
    assert ids_from_tool_messages(msgs) == {"111111111", "222222222", "333333333"}


def test_guard_keeps_a_confirmation_sentence_and_adds_the_facts():
    """Regression: a confirmation sentence must not be replaced by a bare job line."""
    a = FinalAnswer(message="Job 3904982937 has been saved to your list.",
                    job_ids=["3904982937"], kind="info")
    out, rep = check_answer(a, KNOWN)
    assert out.message == "Job 3904982937 (Python Engineer - ESO) has been saved to your list."
    assert rep.fixed_lines == 1
