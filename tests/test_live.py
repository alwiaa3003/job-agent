"""Live Himalayas source, tested OFFLINE with a fake cached response (no network)."""

import json
import time

import pytest

from jobagent import live

FAKE_RAW = [   # made-up jobs in the real Himalayas field format
    {"guid": "https://himalayas.app/companies/acme/jobs/python-dev", "title": "Python Developer",
     "companyName": "Acme", "excerpt": "Build Python APIs.", "description": "<p>Build &amp; ship</p>",
     "employmentType": "Full Time", "seniority": ["Entry-level"], "locationRestrictions": ["India"],
     "minSalary": 5000, "maxSalary": 7000, "currency": "USD", "salaryPeriod": "monthly",
     "pubDate": 1795000000, "expiryDate": time.time() + 86400,
     "applicationLink": "https://himalayas.app/companies/acme/jobs/python-dev?utm_source=api"},
    {"guid": "https://himalayas.app/companies/globex/jobs/backend", "title": "Backend Engineer",
     "companyName": "Globex", "excerpt": "Python backend.", "description": "Backend",
     "employmentType": "Contractor", "seniority": ["Senior"], "locationRestrictions": [],
     "pubDate": 1795000000, "expiryDate": time.time() + 86400,
     "applicationLink": "https://himalayas.app/companies/globex/jobs/backend?utm_source=api"},
    {"guid": "https://himalayas.app/companies/old/jobs/expired", "title": "Python Old Job",
     "companyName": "Old", "excerpt": "", "description": "", "seniority": [],
     "locationRestrictions": ["Germany"], "pubDate": 1700000000, "expiryDate": time.time() - 60,
     "applicationLink": "https://himalayas.app/companies/old/jobs/expired?utm_source=api"},
]


def test_job_id_is_stable_12_digits_starting_with_9():
    a = live.make_job_id("https://himalayas.app/x")
    assert a == live.make_job_id("https://himalayas.app/x") and len(a) == 12 and a[0] == "9"
    assert a.isdigit() and a != live.make_job_id("https://himalayas.app/y")


def test_mapping_to_our_format():
    j = live.to_our_job(FAKE_RAW[0])
    assert j["level"] == "Entry level" and j["work_type"] == "Full-time" and j["remote"]
    assert j["salary_per_year_usd"] == 72000                # (5000+7000)/2 * 12
    assert j["description"] == "Build & ship" and j["source"] == "live"
    assert j["url"] == FAKE_RAW[0]["applicationLink"]       # API terms: link unchanged
    assert live.to_our_job(FAKE_RAW[1])["location"] == "Remote (worldwide)"


def test_worldwide_matches_any_country_and_location():
    ww = live.to_our_job(FAKE_RAW[1])
    india = live.to_our_job(FAKE_RAW[0])
    assert live._passes(ww, "Germany", "", "any", "US")
    assert live._passes(india, "", "", "any", "India") and not live._passes(india, "", "", "any", "US")
    assert not live._passes(india, "", "India", "any", "any")   # open ONLY in India
    assert live._passes(ww, "", "India", "any", "any")          # worldwide never excluded


@pytest.fixture
def offline_source(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(live, "INDEX_FILE", tmp_path / "jobs_index.json")
    src = live.HimalayasSource(offline=True)
    path = src._cache_file({"q": "python"})
    path.write_text(json.dumps({"fetched_at": time.time(), "params": {"q": "python"},
                                "response": {"jobs": FAKE_RAW}}))
    return src


def test_offline_replay_drops_expired_and_details_work(offline_source):
    jobs = offline_source.search("python", k=5)
    assert {j["title"] for j in jobs} == {"Python Developer", "Backend Engineer"}
    assert all(j["source"] == "live" for j in jobs)
    details = offline_source.get_job(jobs[0]["job_id"])
    assert details and details["source"] == "live"


def test_offline_uncached_search_fails_cleanly(offline_source):
    with pytest.raises(live.LiveSourceError, match="Offline mode"):
        offline_source.search("underwater basket weaving")
