"""Live remote jobs from the Himalayas public API (https://himalayas.app/api).

Responses are cached in cache/himalayas/, so LIVE_OFFLINE=1 can replay them without the network."""

import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from jobagent import embed
from jobagent.config import PROJECT_ROOT
from jobagent.search import scrub_contacts
from jobagent.sources import SourceError

API_URL = "https://himalayas.app/jobs/api/search"
CACHE_DIR = PROJECT_ROOT / "cache" / "himalayas"
INDEX_FILE = CACHE_DIR / "jobs_index.json"   # job_id -> job, so details and saving work later
USER_AGENT = "JobAgent-portfolio-demo/0.1"
TIMEOUT_S = 10
MIN_GAP_S = 1.0
CACHE_TTL_S = 24 * 3600  # Himalayas refreshes its data daily

# API terms: visible credit, applicationLink unchanged, cache, stay within limits,
# drop expired jobs. An empty locationRestrictions list means worldwide.
LEVEL_FROM_SENIORITY = {"Entry-level": "Entry level", "Mid-level": "Mid-Senior level",
                        "Senior": "Mid-Senior level", "Manager": "Director",
                        "Director": "Director", "Executive": "Executive"}
# Levels the API can filter on; the rest are filtered locally.
SENIORITY_PARAM = {"Entry level": "Entry-level", "Director": "Director",
                   "Executive": "Executive"}
COUNTRY_PARAM = {"US": "US", "India": "India"}
COUNTRY_NAME = {"US": "United States", "India": "India"}
WORK_TYPES = {"Full Time": "Full-time", "Part Time": "Part-time", "Contractor": "Contract",
              "Intern": "Internship", "Temporary": "Temporary"}
PLACE_ALIASES = {"us": "united states", "usa": "united states", "uk": "united kingdom"}


class LiveSourceError(SourceError):
    """Live jobs could not be fetched."""


def make_job_id(guid: str) -> str:
    """Stable 12-digit id from the guid; can't clash with the 10-digit LinkedIn ids."""
    digest = int(hashlib.sha256(guid.encode("utf-8")).hexdigest(), 16)
    return "9" + str(digest % 10**11).zfill(11)


def _html_to_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"[​-‍﻿]", "", text)   # invisible zero-width characters
    return " ".join(html.unescape(text).split())


def _salary_per_year_usd(job: dict):
    # USD only: no currency conversion.
    lo, hi = job.get("minSalary"), job.get("maxSalary")
    if job.get("currency") != "USD" or not (lo or hi):
        return None
    mid = ((lo or hi) + (hi or lo)) / 2
    return round(mid * 12) if job.get("salaryPeriod") == "monthly" else round(mid)


def to_our_job(raw: dict) -> dict:
    """Map one Himalayas job to the local job format."""
    places = raw.get("locationRestrictions") or []
    levels = sorted({LEVEL_FROM_SENIORITY.get(s, "not stated") for s in raw.get("seniority") or []})
    return {
        "job_id": make_job_id(raw["guid"]),
        "title": raw.get("title") or "unknown",
        "company": raw.get("companyName") or "unknown",
        "location": "Remote (" + (", ".join(places[:4]) + (" ..." if len(places) > 4 else "")
                                  if places else "worldwide") + ")",
        "country": ", ".join(places) if places else "worldwide",
        "places": places,                               # [] means worldwide
        "levels": levels or ["not stated"],
        "level": ", ".join(levels) if levels else "not stated",
        "remote": True,
        "work_type": WORK_TYPES.get(raw.get("employmentType"), raw.get("employmentType")
                                    or "not stated"),
        "salary_per_year_usd": _salary_per_year_usd(raw),
        "url": raw.get("applicationLink") or raw.get("guid"),   # API terms: link unchanged
        "snippet": scrub_contacts(" ".join((raw.get("excerpt") or "").split()))[:120],
        "description": scrub_contacts(_html_to_text(raw.get("description"))),
        "posted": time.strftime("%Y-%m-%d", time.gmtime(raw["pubDate"])) if raw.get("pubDate") else None,
        "expires_at": raw.get("expiryDate"),
        "source": "live",
    }


def _place_match(places: list[str], place: str) -> bool:
    place = PLACE_ALIASES.get(place.strip().lower(), place.strip().lower())
    pattern = r"\b" + re.escape(place) + r"\b"
    return any(re.search(pattern, p.lower()) for p in places)


def _passes(job: dict, location: str, exclude_location: str, experience_level: str,
            country: str) -> bool:
    worldwide = not job["places"]
    if country != "any" and not (worldwide or COUNTRY_NAME[country] in job["places"]):
        return False
    if location.strip() and not (worldwide or _place_match(job["places"], location)):
        return False
    # Only drop jobs open solely in the excluded place; worldwide jobs stay.
    if exclude_location.strip() and not worldwide and \
            all(_place_match([p], exclude_location) for p in job["places"]):
        return False
    if experience_level != "any" and experience_level not in job["levels"]:
        return False
    return True


class HimalayasSource:
    name = "live"

    def __init__(self, offline: bool = False):
        self.offline = offline
        self._last_request = 0.0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def search(self, query: str, k: int = 5, remote_only: bool = False, location: str = "",
               exclude_location: str = "", experience_level: str = "any",
               country: str = "any") -> list[dict]:
        params = {"q": query}
        if country != "any":
            params["country"] = COUNTRY_PARAM[country]
        if experience_level in SENIORITY_PARAM:
            params["seniority"] = SENIORITY_PARAM[experience_level]
        response = None
        if location.strip() and country == "any":
            # The country parameter accepts country names but returns 400 for cities.
            response = self._fetch({**params, "country": location.strip()})
            if response.get("_bad_request"):
                response = None
        if response is None:
            response = self._fetch(params)
        if response.get("_bad_request"):
            raise LiveSourceError("The live job source rejected this search.")
        raw_jobs = response.get("jobs") or []

        now = time.time()
        jobs = [to_our_job(r) for r in raw_jobs if r.get("guid")]
        jobs = [j for j in jobs if not j["expires_at"] or j["expires_at"] > now]
        self._remember(jobs)
        jobs = [j for j in jobs if _passes(j, location, exclude_location, experience_level, country)]
        if not jobs:
            return []

        texts = [embed.job_text(j["title"], j["snippet"]) for j in jobs]
        scores = embed.embed_texts(texts) @ embed.embed_text(query)
        results, seen = [], set()
        for i in np.argsort(-scores):
            j = jobs[i]
            key = (j["title"].lower(), j["company"].lower(), j["location"].lower())
            if key in seen:
                continue
            seen.add(key)
            results.append(_short(j, float(scores[i])))
            if len(results) == k:
                break
        return results

    def get_job(self, job_id: str, desc_chars: int = 600) -> dict | None:
        job = self._load_index().get(str(job_id))
        if job is None or (job.get("expires_at") and job["expires_at"] <= time.time()):
            return None
        keys = ["job_id", "title", "company", "location", "country", "level", "work_type",
                "remote", "salary_per_year_usd", "url", "posted", "source"]
        out = {key: job[key] for key in keys}
        out["description"] = job["description"][:desc_chars]
        return out

    def _cache_file(self, params: dict):
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        return CACHE_DIR / f"search_{key}.json"

    def _fetch(self, params: dict) -> dict:
        path = self._cache_file(params)
        cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if cached and (self.offline or time.time() - cached["fetched_at"] < CACHE_TTL_S):
            return cached["response"]
        if self.offline:
            raise LiveSourceError("Offline mode: no saved live results for this search. "
                                  "Run it once online first (LIVE_OFFLINE=0).")
        response = self._get(params)
        path.write_text(json.dumps({"fetched_at": time.time(), "params": params,
                                    "response": response}, ensure_ascii=False), encoding="utf-8")
        return response

    def _get(self, params: dict) -> dict:
        url = API_URL + "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                       "Accept": "application/json"})
        for attempt in range(2):
            wait = self._last_request + MIN_GAP_S - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.time()
            try:
                with urllib.request.urlopen(request, timeout=TIMEOUT_S) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as err:
                if err.code == 400:
                    return {"_bad_request": True}           # cached, so it isn't retried
                if err.code in (429, 500, 502, 503, 504) and attempt == 0:
                    retry_after = err.headers.get("Retry-After", "")
                    time.sleep(min(float(retry_after), 30) if retry_after.isdigit() else 5)
                    continue
                raise LiveSourceError(f"Live job source returned HTTP {err.code}. "
                                      "Try again later.") from err
            except (urllib.error.URLError, TimeoutError) as err:
                raise LiveSourceError("Could not reach the live job source "
                                      "(network problem or timeout).") from err
            except json.JSONDecodeError as err:
                raise LiveSourceError("The live job source sent an unreadable reply.") from err
        raise LiveSourceError("Live job source is busy (rate limited). Try again later.")

    def _load_index(self) -> dict:
        if not INDEX_FILE.exists():
            return {}
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))

    def _remember(self, jobs: list[dict]) -> None:
        # Expired jobs are removed, as the API terms ask.
        now = time.time()
        index = {i: j for i, j in self._load_index().items()
                 if not j.get("expires_at") or j["expires_at"] > now}
        index.update({j["job_id"]: j for j in jobs})
        INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")


def _short(job: dict, score: float) -> dict:
    return {"job_id": job["job_id"], "title": job["title"], "company": job["company"],
            "location": job["location"], "country": job["country"], "level": job["level"],
            "remote": True, "score": round(score, 2), "snippet": job["snippet"],
            "posted": job["posted"], "source": "live"}
