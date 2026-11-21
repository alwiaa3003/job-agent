"""The structured final answer and its validation.

Groq rejects JSON mode when tools are enabled, so the prompt asks for JSON and it is validated here."""

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Kind = Literal["jobs", "no_results", "clarify", "info", "refuse", "error"]


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True,
                              coerce_numbers_to_str=True)

    message: str = Field(min_length=1, max_length=3000,
                         description="The text shown to the user")
    job_ids: list[str] = Field(default_factory=list, max_length=10,
                               description="job_id of every job mentioned in message")
    kind: Kind = "info"

    @field_validator("job_ids")
    @classmethod
    def check_ids(cls, ids):
        for job_id in ids:
            if not re.fullmatch(r"\d{5,15}", job_id):
                raise ValueError(f"'{job_id}' is not a job_id (digits only)")
        return ids


def _extract_json(text: str) -> str:
    # Models sometimes wrap the JSON in ```json fences or add a sentence around it.
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


def parse_answer(text: str) -> tuple[FinalAnswer | None, str]:
    """(answer, "") if text is a valid final answer, else (None, error message)."""
    raw = _extract_json(text or "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as err:
        return None, f"not valid JSON ({err.msg})"
    if not isinstance(data, dict):
        return None, "the JSON must be an object {...}"
    try:
        return FinalAnswer.model_validate(data), ""
    except ValidationError as err:
        return None, _short_error(err)


def _short_error(err: ValidationError) -> str:
    # One short line, so the retry prompt stays small.
    parts = []
    for e in err.errors():
        field = ".".join(str(p) for p in e["loc"]) or "answer"
        parts.append(f"{field}: {e['msg'].removeprefix('Value error, ')}")
    return "; ".join(parts)
