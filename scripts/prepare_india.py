"""Convert data/india__jobs/linkdin_Job_data.csv into data/jobs_india.csv and its vectors.

Usage: python scripts/prepare_india.py  (prints counts only, never job or personal content)"""

import re

import numpy as np
import pandas as pd

from jobagent.config import PROJECT_ROOT
from jobagent.embed import embed_texts, job_text

SOURCE = PROJECT_ROOT / "data" / "india__jobs" / "linkdin_Job_data.csv"
OUT_CSV = PROJECT_ROOT / "data" / "jobs_india.csv"
OUT_VECS = PROJECT_ROOT / "data" / "job_vecs_india.npy"

# Recruiter name and profile-link columns are never read into the job table.
KEEP = ["job_ID", "job", "company_name", "location", "work_type",
        "full_time_remote", "job_details"]

LEVELS = {"Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"}
WORK_TYPES = {"Full-time", "Part-time", "Contract", "Temporary", "Internship", "Volunteer"}

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{4}[\s-]?\d{5}\b")


def split_level(text) -> tuple:
    # "Full-time · Entry level" -> ("Full-time", "Entry level")
    parts = [p.strip() for p in str(text).split("·")] if pd.notna(text) else []
    work = parts[0] if parts and parts[0] in WORK_TYPES else None
    level = parts[1] if len(parts) > 1 and parts[1] in LEVELS else None
    return work, level


def names_pattern(company_names: set[str]) -> re.Pattern | None:
    # Recruiter names are read only to blank them in descriptions; never printed or saved.
    # Single words and company names in that column are not personal data, so they're skipped.
    people = pd.read_csv(SOURCE, usecols=["Hiring_person"], dtype=str,
                         encoding="utf-8-sig")["Hiring_person"].dropna().str.strip()
    names = {n for n in people if len(n.split()) >= 2 and n not in company_names}
    if not names:
        return None
    # Longest first, so a full name is removed before its prefix.
    alternatives = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    # Not \b: names may end in a dot ("S. Kumar Jr.").
    return re.compile(r"(?<!\w)(?:" + alternatives + r")(?!\w)")


def scrub(text: str, names: re.Pattern | None) -> tuple[str, int]:
    text, n_email = EMAIL.subn("[email removed]", text)
    text, n_phone = PHONE.subn("[phone removed]", text)
    n_name = 0
    if names is not None:
        text, n_name = names.subn("[name removed]", text)
    return text, n_email + n_phone + n_name


def main():
    if not SOURCE.exists():
        raise SystemExit(f"{SOURCE} not found. Download the India LinkedIn dataset first.")

    raw = pd.read_csv(SOURCE, usecols=KEEP, dtype=str, encoding="utf-8-sig")
    print(f"Read {len(raw):,} rows ({len(KEEP)} columns kept, personal columns never loaded)")

    raw["job"] = raw["job"].str.strip()
    raw["job_details"] = raw["job_details"].fillna("").str.strip()
    jobs = raw[raw["job"].notna() & (raw["job"] != "") & (raw["job_details"] != "")]
    print(f"After dropping empty title/description: {len(jobs):,}")

    jobs = jobs.drop_duplicates("job_ID")
    jobs = jobs.drop_duplicates(["job", "company_name", "location", "job_details"])
    print(f"After removing duplicates: {len(jobs):,}")

    names = names_pattern(set(jobs["company_name"].dropna().str.strip()))
    cleaned = jobs["job_details"].map(lambda t: scrub(t, names))
    titles = jobs["job"].map(lambda t: scrub(t, names))
    removed = int(sum(n for _, n in cleaned) + sum(n for _, n in titles))
    print(f"E-mails/phones/names blanked in titles + descriptions: {removed}")

    split = jobs["full_time_remote"].map(split_level)
    out = pd.DataFrame({
        "job_id": jobs["job_ID"].str.strip(),
        "title": [text for text, _ in titles],
        "company_name": jobs["company_name"],
        "location": jobs["location"],
        "description": [text for text, _ in cleaned],
        "formatted_work_type": [w for w, _ in split],
        "formatted_experience_level": [lvl for _, lvl in split],
        # Same coding as the US file: 1.0 or blank.
        "remote_allowed": np.where(jobs["work_type"] == "Remote", 1.0, np.nan),
        "normalized_salary": np.nan,   # not in the India file
        # Every US URL follows this pattern.
        "job_posting_url": "https://www.linkedin.com/jobs/view/" + jobs["job_ID"].str.strip() + "/",
    })
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(out):,} jobs to {OUT_CSV.name}")

    texts = [job_text(t, d) for t, d in zip(out["title"], out["description"])]
    vecs = embed_texts(texts)
    np.save(OUT_VECS, vecs)
    print(f"Saved vectors {vecs.shape} to {OUT_VECS.name}")


if __name__ == "__main__":
    main()
