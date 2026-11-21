"""Agent tools: Pydantic argument models, implementations and generated JSON schemas.

The schemas sent to the model are generated from the same models used for validation.
"""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from jobagent.sources import SourceError, get_source
from jobagent.store import Store

LEVELS = ["Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
Level = Literal["any", "Internship", "Entry level", "Associate", "Mid-Senior level",
                "Director", "Executive"]
PREF_KEYS = ["remote", "location", "level", "excluded_locations"]
PrefKey = Literal["remote", "location", "level", "excluded_locations"]


class StrictArgs(BaseModel):
    # Unknown argument names are errors (so the model learns), and numeric job_ids are accepted.
    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True,
                              str_strip_whitespace=True)


class SearchJobsArgs(StrictArgs):
    query: str = Field(min_length=2, max_length=100,
                       description="Role or skills, e.g. 'python developer'")
    remote_only: bool = Field(False, description="Only remote jobs")
    location: str = Field("", max_length=50,
                          description="Only this place: city, country or 2-letter US state code")
    exclude_location: str = Field("", max_length=50, description="Leave out this place")
    experience_level: Level = "any"
    country: Literal["any", "US", "India"] = "any"
    limit: int = Field(5, ge=1, le=5, description="Max results")

    @field_validator("country", mode="before")
    @classmethod
    def fix_country(cls, v):
        if isinstance(v, str):
            low = v.strip().lower()
            if low in {"us", "usa", "united states", "united states of america"}:
                return "US"
            if low in {"india", "any"}:
                return low.capitalize() if low == "india" else "any"
        return v

    @field_validator("experience_level", mode="before")
    @classmethod
    def fix_level_case(cls, v):
        if isinstance(v, str):
            for level in ["any", *LEVELS]:
                if v.strip().lower() == level.lower():
                    return level
        return v


class JobIdArgs(StrictArgs):
    job_id: str = Field(pattern=r"^\d{5,15}$", description="job_id from search results")


class NoArgs(StrictArgs):
    pass


class RememberPreferenceArgs(StrictArgs):
    key: PrefKey = Field(description="One of: " + ", ".join(PREF_KEYS))
    value: str = Field(min_length=1, max_length=100)

    @field_validator("key", mode="before")
    @classmethod
    def check_key(cls, v):
        if isinstance(v, str) and v.strip().lower() in PREF_KEYS:
            return v.strip().lower()
        raise ValueError(f"'{v}' is not an allowed preference key. "
                         f"Allowed: {', '.join(PREF_KEYS)}")


def search_jobs(args: SearchJobsArgs, store: Store) -> dict:
    jobs = get_source().search(args.query, k=args.limit, remote_only=args.remote_only,
                         location=args.location, exclude_location=args.exclude_location,
                         experience_level=args.experience_level, country=args.country)
    result = {"ok": True, "count": len(jobs), "jobs": jobs}
    if not jobs:
        result["note"] = "No jobs matched. Suggest removing a filter."
    return result


def get_job_details(args: JobIdArgs, store: Store) -> dict:
    job = get_source().get_job(args.job_id)
    if job is None:
        return {"ok": False, "error": f"No job with job_id {args.job_id}."}
    return {"ok": True, "job": job}


def save_job(args: JobIdArgs, store: Store) -> dict:
    job = get_source().get_job(args.job_id)
    if job is None:
        return {"ok": False, "error": f"No job with job_id {args.job_id}."}
    added = store.save_job(job["job_id"], job["title"], job["company"])
    return {"ok": True, "saved": job["job_id"], "already_saved": not added}


def list_saved(args: NoArgs, store: Store) -> dict:
    jobs = store.list_saved()
    return {"ok": True, "count": len(jobs), "jobs": jobs}


def remember_preference(args: RememberPreferenceArgs, store: Store) -> dict:
    store.set_preference(args.key, args.value)
    return {"ok": True, "remembered": {args.key: args.value}}


# name -> (argument model, function, description for the model)
REGISTRY = {
    "search_jobs": (SearchJobsArgs, search_jobs,
                    "Search jobs by meaning, with optional filters. Max 5 results."),
    "get_job_details": (JobIdArgs, get_job_details,
                        "Get details (description, salary, url) of one job by job_id."),
    "save_job": (JobIdArgs, save_job, "Save a job to the user's list by job_id."),
    "list_saved": (NoArgs, list_saved, "List the user's saved jobs."),
    "remember_preference": (RememberPreferenceArgs, remember_preference,
                            "Remember a lasting user preference for future sessions."),
}


def _strip_titles(schema):
    # Pydantic's per-field "title" adds tokens and tells the model nothing.
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


def tool_schemas() -> list[dict]:
    """Tool definitions in OpenAI/Groq format."""
    return [
        {"type": "function",
         "function": {"name": name, "description": desc,
                      "parameters": _strip_titles(model.model_json_schema())}}
        for name, (model, _, desc) in REGISTRY.items()
    ]


def _format_validation_error(err: ValidationError) -> str:
    parts = []
    for e in err.errors():
        field = ".".join(str(p) for p in e["loc"]) or "arguments"
        msg = e["msg"].removeprefix("Value error, ")
        parts.append(f"{field}: {msg}")
    return "Invalid arguments - " + "; ".join(parts)


def run_tool(name: str, raw_args, store: Store) -> dict:
    """Validate and run one tool call; never raises (errors are returned as {"ok": False})."""
    if name not in REGISTRY:
        return {"ok": False, "error": f"Unknown tool '{name}'. Available: {', '.join(REGISTRY)}."}
    model, func, _ = REGISTRY[name]

    # Arguments arrive as a JSON string, sometimes empty for no-arg tools.
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError as err:
            return {"ok": False, "error": f"Arguments are not valid JSON: {err.msg}."}
    if raw_args is None:
        raw_args = {}
    if not isinstance(raw_args, dict):
        return {"ok": False, "error": "Arguments must be a JSON object."}

    try:
        args = model.model_validate(raw_args)
    except ValidationError as err:
        return {"ok": False, "error": _format_validation_error(err)}

    try:
        return func(args, store)
    except SourceError as err:
        return {"ok": False, "error": str(err)}
    except Exception as err:   # a tool bug must not end the chat session
        return {"ok": False, "error": f"Tool '{name}' failed: {type(err).__name__}: {err}"}
