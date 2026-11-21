"""Local CSV search: filters and near-duplicate hiding."""

from conftest import needs_india

from jobagent import search


def test_both_countries_loaded_without_id_clashes():
    jobs, vecs = search.load_index()
    assert len(jobs) == len(vecs)
    assert not jobs["job_id"].duplicated().any()
    assert set(jobs["country"]) <= {"US", "India"}


def test_india_does_not_match_indiana():
    for j in search.search("nurse", k=5, location="India", country="US"):
        assert "indiana" not in j["location"].lower()


def test_state_code_matches_end_of_location():
    jobs = search.search("software engineer", k=5, location="TX")
    assert jobs and all(j["location"].endswith(", TX") for j in jobs)


@needs_india
def test_exclude_location_and_alias():
    jobs = search.search("python developer", k=5, country="India", exclude_location="Gurgaon")
    assert jobs and not any("gurugram" in j["location"].lower() for j in jobs)
    blr = search.search("software engineer", k=3, location="Bangalore")
    assert blr and all("bengaluru" in j["location"].lower() for j in blr)


def test_results_hide_same_title_company_location():
    jobs = search.search("senior automation tester", k=5)
    keys = [(j["title"].lower(), j["company"].lower(), j["location"].lower()) for j in jobs]
    assert len(keys) == len(set(keys))


def test_contact_details_never_reach_the_model():
    from jobagent.search import scrub_contacts
    text = "Mail jane.doe@example.com or call (555) 123-4567 / +91 98765 43210. Job 3904982937."
    out = scrub_contacts(text)
    assert "@" not in out and "123-4567" not in out and "98765" not in out
    assert "3904982937" in out                          # job ids are untouched
    # glued-on words and labelled numbers without separators are still caught
    glued = scrub_contacts("HR at555-123-4567. Direct: 555-123-4567Email. Contact : 5551234567 "
                           "COSMOSContact : 5551234567")
    assert glued.count("[phone removed]") == 4
    # requisition / position codes and years are NOT phone numbers
    for code in ["2024-30909 2024", "0372-2024-2025", "163-818-4549-001", "2023-2024"]:
        assert scrub_contacts(code) == code


def test_no_email_in_any_snippet_or_details_sent_to_the_model():
    import re
    jobs, _ = search.load_index()
    email = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
    for i in range(len(jobs)):
        row = jobs.iloc[i]
        assert not email.search(search._short_job(row, 0.0)["snippet"])
    for job_id in jobs["job_id"].head(2000):
        assert not email.search(search.get_job(job_id)["description"])
