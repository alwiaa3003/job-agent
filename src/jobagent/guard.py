"""Grounding guard: an answer may only mention jobs a tool returned or that are saved.

Unknown ids and invented lines are removed; mixed-up lines are corrected from the data."""

import json
import re
from dataclasses import dataclass, field

from jobagent.answer import FinalAnswer
from jobagent.sources import SourceError, get_source

# Job ids are 10 digits (local) or 12 (live); salaries and years are shorter.
JOB_ID_IN_TEXT = re.compile(r"(?<!\d)\d{9,12}(?!\d)")
BULLET = re.compile(r"\s*([-*•]|\d+[.)])\s")


@dataclass
class GuardReport:
    removed_ids: list[str] = field(default_factory=list)
    added_ids: list[str] = field(default_factory=list)      # known ids found only in the text
    removed_lines: int = 0
    fixed_lines: int = 0
    replaced: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.removed_ids or self.added_ids or self.removed_lines
                    or self.fixed_lines or self.replaced)


def ids_from_tool_messages(messages: list[dict]) -> set[str]:
    """Every job_id contained in successful tool results."""
    ids = set()
    for m in messages:
        if m.get("role") != "tool":
            continue
        try:
            data = json.loads(m["content"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if not isinstance(data, dict) or not data.get("ok"):
            continue
        ids |= {str(j["job_id"]) for j in data.get("jobs", []) if "job_id" in j}
        if isinstance(data.get("job"), dict) and "job_id" in data["job"]:
            ids.add(str(data["job"]["job_id"]))
        if "saved" in data:
            ids.add(str(data["saved"]))
    return ids


def _line_matches(line: str, job: dict) -> bool:
    low = line.lower()
    title_words = re.findall(r"\w+", job["title"].lower())[:2]
    company = job["company"].lower()
    return (bool(title_words) and all(w in low for w in title_words)) or \
           (company != "unknown" and company in low)


def _canonical_line(job: dict) -> str:
    return f"- {job['title']} - {job['company']} - {job['location']} ({job['job_id']})"


def check_answer(answer: FinalAnswer, known_ids: set[str],
                 user_ids: set[str] = frozenset()) -> tuple[FinalAnswer, GuardReport]:
    """Grounded copy of the answer plus a report of what changed.

    user_ids (typed by the user) may be repeated in non-jobs answers but are never verified ids."""
    report = GuardReport()
    if answer.kind == "error":                     # the agent's own messages
        return answer, report

    lines_out, seen_in_text = [], []
    for line in answer.message.split("\n"):
        ids = JOB_ID_IN_TEXT.findall(line)
        unknown = [i for i in ids if i not in known_ids]
        if unknown and answer.kind != "jobs" and all(i in user_ids for i in unknown):
            lines_out.append(line)                 # e.g. "job X does not exist"
            continue
        if unknown:
            report.removed_lines += 1
            report.removed_ids += [i for i in unknown if i not in report.removed_ids]
            continue
        if len(ids) == 1:
            try:
                job = get_source().get_job(ids[0])
            except SourceError:
                job = None
            if job and not _line_matches(line, job):
                if BULLET.match(line):
                    line = _canonical_line(job)
                else:                              # keep the sentence, add the real facts
                    line = line.replace(ids[0], f"{ids[0]} ({job['title']} - {job['company']})", 1)
                report.fixed_lines += 1
        seen_in_text += [i for i in ids if i not in seen_in_text]
        lines_out.append(line)

    job_ids = []
    for i in answer.job_ids + seen_in_text:
        if i not in known_ids:
            if i not in report.removed_ids:
                report.removed_ids.append(i)
        elif i not in job_ids:
            job_ids.append(i)
    report.added_ids = [i for i in job_ids if i not in answer.job_ids]

    message = "\n".join(lines_out).strip()

    if (report.removed_lines and not message) or (answer.kind == "jobs" and not job_ids):
        report.replaced = True
        return FinalAnswer(message="I could not verify any jobs for this answer, so I "
                           "did not show them. Please try the search again.",
                           job_ids=[], kind="error"), report

    if not report.changed:
        return answer, report
    if report.removed_lines:
        message += (f"\n\n(I removed {report.removed_lines} job(s) that I could not "
                    "verify in the search results.)")
    return FinalAnswer(message=message[:3000] or "(no answer)", job_ids=job_ids[:10],
                       kind=answer.kind), report
