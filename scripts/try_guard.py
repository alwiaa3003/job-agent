"""Show the grounding guard on hand-made answers (no LLM, no key).

Usage: python scripts/try_guard.py"""

from jobagent.answer import FinalAnswer
from jobagent.guard import check_answer

KNOWN = {"3904982937", "3889738868"}   # real US jobs: Python Engineer - ESO; Online Data Analyst - TELUS


def case(title, answer: FinalAnswer, known=KNOWN, user_ids=frozenset()):
    fixed, report = check_answer(answer, known, set(user_ids))
    print(f"\n=== {title}")
    print(f"  in : kind={answer.kind} job_ids={answer.job_ids}")
    print(f"  out: kind={fixed.kind} job_ids={fixed.job_ids}")
    print(f"  report: removed_ids={report.removed_ids} added_ids={report.added_ids} "
          f"removed_lines={report.removed_lines} "
          f"fixed_lines={report.fixed_lines} replaced={report.replaced}")
    print("  user sees: " + fixed.message.replace("\n", "\n             "))


def main():
    case("1. Everything grounded -> unchanged", FinalAnswer(
        message="- Python Engineer - ESO - Belcamp, MD (3904982937)",
        job_ids=["3904982937"], kind="jobs"))

    case("2. The real Phase 5 bug: invented id only in job_ids -> dropped", FinalAnswer(
        message="Saved jobs:\n- Online Data Analyst - TELUS International AI Data Solutions (3889738868)",
        job_ids=["3889738860", "3889738868"], kind="info"))

    case("3. Invented job in the text -> line removed + note", FinalAnswer(
        message="- Python Engineer - ESO - Belcamp, MD (3904982937)\n"
                "- Senior ML Engineer - Google - Remote (3999999999)",
        job_ids=["3904982937", "3999999999"], kind="jobs"))

    case("4. Real id, wrong title + company -> line rewritten from the data", FinalAnswer(
        message="- Staff Engineer - Microsoft - Seattle (3904982937)",
        job_ids=["3904982937"], kind="jobs"))

    case("5. Known id in text but missing from job_ids -> added", FinalAnswer(
        message="Saved: Online Data Analyst - TELUS International AI Data Solutions (3889738868)", job_ids=[], kind="info"))

    case("6. 'Make up 3 jobs': jobs without ids -> nothing verifiable -> replaced", FinalAnswer(
        message="- AI Engineer at OpenAI\n- Data Scientist at Meta\n- SWE at Apple",
        job_ids=[], kind="jobs"))

    case("7. Search found nothing, model still lists a job -> removed, safe message",
         FinalAnswer(message="- Python Developer - Acme - Berlin (3123456789)",
                     job_ids=["3123456789"], kind="jobs"), known=set())

    case("8. Salary and year are not job ids -> unchanged", FinalAnswer(
        message="Python Engineer - ESO (3904982937) pays about 120000 USD (posted 2024).",
        job_ids=["3904982937"], kind="jobs"))

    case("9. User asked about a missing id; answer repeats it -> kept", FinalAnswer(
        message="There is no job with job_id 3999999999 in the dataset.", kind="info"),
        user_ids={"3999999999"})

    case("10. ...but a jobs answer built on that id is still removed", FinalAnswer(
        message="- ML Engineer - Google - Remote (3999999999)", job_ids=["3999999999"],
        kind="jobs"), user_ids={"3999999999"})

    case("11. Every line removed -> safe message instead of only the note", FinalAnswer(
        message="Job 3888888888 is a great match.", kind="info"))


if __name__ == "__main__":
    main()
