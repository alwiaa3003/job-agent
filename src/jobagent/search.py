"""Semantic search over the local job CSVs: filter, then rank by cosine similarity.

US data is required; India data is loaded only if scripts/prepare_india.py was run.
"""

import re

import numpy as np
import pandas as pd

from jobagent.config import PROJECT_ROOT
from jobagent import embed

DATA = PROJECT_ROOT / "data"
# (country, jobs file, vectors file, required?)
SOURCES = [
    ("US", DATA / "jobs.csv", DATA / "job_vecs.npy", True),
    ("India", DATA / "jobs_india.csv", DATA / "job_vecs_india.npy", False),
]
COUNTRIES = [c for c, *_ in SOURCES]

PLACE_ALIASES = {"bangalore": "bengaluru", "gurgaon": "gurugram", "bombay": "mumbai",
                 "madras": "chennai", "calcutta": "kolkata"}

_jobs: pd.DataFrame | None = None
_job_vecs: np.ndarray | None = None


def _load_one(country, jobs_file, vecs_file, required):
    if not jobs_file.exists() or not vecs_file.exists():
        if required:
            raise FileNotFoundError(f"{jobs_file.name} or {vecs_file.name} not found in data/.")
        return None
    jobs = pd.read_csv(jobs_file, dtype={"job_id": str})
    vecs = np.load(vecs_file)
    if len(jobs) != len(vecs):
        raise ValueError(f"{jobs_file.name} has {len(jobs)} rows but "
                         f"{vecs_file.name} has {len(vecs)}. Re-run the prepare script.")
    jobs["country"] = country
    return jobs, vecs


def load_index() -> tuple[pd.DataFrame, np.ndarray]:
    """All available jobs and their vectors (loaded once)."""
    global _jobs, _job_vecs
    if _jobs is None:
        loaded = [part for src in SOURCES if (part := _load_one(*src)) is not None]
        jobs = pd.concat([j for j, _ in loaded], ignore_index=True)
        vecs = np.vstack([v for _, v in loaded])
        if jobs["job_id"].duplicated().any():
            raise ValueError("The same job_id appears in two datasets.")
        _jobs, _job_vecs = jobs, vecs
    return _jobs, _job_vecs


def _location_matches(locations: pd.Series, place: str) -> np.ndarray:
    # A 2-letter code is a US state suffix (", TX"); anything else must match whole
    # words, so "India" does not match "Indiana".
    place = place.strip().lower()
    place = PLACE_ALIASES.get(place, place)
    if len(place) == 2 and place.isalpha():
        return locations.str.endswith(", " + place).to_numpy()
    return locations.str.contains(r"\b" + re.escape(place) + r"\b", regex=True).to_numpy()


def filter_mask(jobs: pd.DataFrame, remote_only: bool = False, location: str = "",
                exclude_location: str = "", experience_level: str = "any",
                country: str = "any") -> np.ndarray:
    """Boolean mask of jobs that pass every filter."""
    mask = np.ones(len(jobs), dtype=bool)
    locations = jobs["location"].fillna("").str.lower()

    if country and country != "any":
        mask &= (jobs["country"] == country).to_numpy()
    if remote_only:
        # remote_allowed is 1.0 or blank
        mask &= (jobs["remote_allowed"] == 1).to_numpy()
    if location.strip():
        mask &= _location_matches(locations, location)
    if exclude_location.strip():
        mask &= ~_location_matches(locations, exclude_location)
    if experience_level and experience_level != "any":
        levels = jobs["formatted_experience_level"].fillna("").str.lower()
        mask &= (levels == experience_level.lower()).to_numpy()
    return mask


def _clean(value, default=""):
    return default if pd.isna(value) else value


EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Separators are required so 10-digit job ids are never matched. Adjacent digits or
# hyphens block a match (requisition codes like "2024-30909 2024", "163-818-4549-001"),
# but adjacent letters do not, since descriptions often glue words on ("703-955-1234Email").
PHONE = re.compile(
    r"(?<![\d-])(?:"
    r"(?:\+?1[\s.-]?)?\(?[2-9]\d{2}(?:\)[\s.-]?|[\s.-])\d{3}[\s.-]\d{4}"  # US / Canada
    r"|\+\d{1,3}[\s.-]?\(?\d{1,5}\)?[\s.-]\d{3,5}[\s.-]?\d{3,4}"         # international, needs +
    r"|(?:\+91[\s-]?)?[6-9]\d{4}[\s-]\d{5}|\+91[\s-]?[6-9]\d{9}"         # India mobile
    r")(?!\d|-\w)"
    r"|(?<=\d)1[-.][2-9]\d{2}[-.]\d{3}[-.]\d{4}(?!\d|-\w)")   # "1-800-..." glued to a zip code
# Without separators a number only counts as a phone after a label ("Contact :", "//").
LABELED_PHONE = re.compile(
    # Case-sensitive lookbehind: skips "graph 123..." but still finds "COSMOSContact : 925...".
    r"((?<![a-z])(?i:phone|ph|tel|mobile|mob|cell|contact|call|direct)"
    r"(?i:\s*(?:number|num|no\.?|#))?\s*[:.\-]*\s*|//\s*)(\+?\d{10,12})(?!\d)")


def scrub_contacts(text: str) -> str:
    """Blank e-mail addresses and phone numbers before job text is sent to the model."""
    text = EMAIL.sub("[email removed]", text)
    text = PHONE.sub("[phone removed]", text)
    return LABELED_PHONE.sub(r"\1[phone removed]", text)


def _short_job(row: pd.Series, score: float) -> dict:
    snippet = scrub_contacts(" ".join(str(row["description"]).split()))[:120]
    return {
        "job_id": row["job_id"],
        "title": row["title"],
        "company": _clean(row["company_name"], "unknown"),
        "location": _clean(row["location"], "unknown"),
        "country": row["country"],
        "level": _clean(row["formatted_experience_level"], "not stated"),
        "remote": bool(row["remote_allowed"] == 1),   # numpy bools are not JSON-serialisable
        "score": round(float(score), 2),
        "snippet": snippet,
    }


def _dedupe_key(row: pd.Series) -> tuple:
    # Same title, company and location look identical to the user: show only the best one.
    return tuple(str(_clean(row[c])).strip().lower()
                 for c in ("title", "company_name", "location"))


def search(query: str, k: int = 5, **filters) -> list[dict]:
    """Up to k jobs closest in meaning to query, best first, without near-duplicates."""
    jobs, vecs = load_index()
    rows = np.flatnonzero(filter_mask(jobs, **filters))
    if len(rows) == 0:
        return []
    scores = vecs[rows] @ embed.embed_text(query)   # vectors are normalised: dot = cosine
    results, seen = [], set()
    for i in np.argsort(-scores):
        row = jobs.iloc[rows[i]]
        key = _dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)
        results.append(_short_job(row, scores[i]))
        if len(results) == k:
            break
    return results


def get_job(job_id: str, desc_chars: int = 600) -> dict | None:
    """Details of one job (description trimmed), or None if the id is unknown."""
    jobs, _ = load_index()
    match = jobs[jobs["job_id"] == str(job_id)]
    if match.empty:
        return None
    row = match.iloc[0]
    salary = _clean(row["normalized_salary"], None)
    return {
        "job_id": row["job_id"],
        "title": row["title"],
        "company": _clean(row["company_name"], "unknown"),
        "location": _clean(row["location"], "unknown"),
        "country": row["country"],
        "level": _clean(row["formatted_experience_level"], "not stated"),
        "work_type": _clean(row["formatted_work_type"], "not stated"),
        "remote": bool(row["remote_allowed"] == 1),
        "salary_per_year_usd": round(float(salary)) if salary is not None else None,
        "url": _clean(row["job_posting_url"], None),
        "description": scrub_contacts(" ".join(str(row["description"]).split()))[:desc_chars],
    }
