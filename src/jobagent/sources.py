"""Job sources behind one interface (search, get_job), selected by JOB_SOURCE."""

from jobagent import search as csv_search
from jobagent.config import load_settings


class SourceError(Exception):
    """A source could not answer; the message is safe to show to the user and the model."""


class LocalCSVSource:
    """The bundled LinkedIn dataset (US, plus India if installed)."""
    name = "local_csv"

    def search(self, query: str, k: int = 5, **filters) -> list[dict]:
        return [{**job, "source": self.name} for job in csv_search.search(query, k=k, **filters)]

    def get_job(self, job_id: str) -> dict | None:
        job = csv_search.get_job(job_id)
        return {**job, "source": self.name} if job else None


_source = None


def get_source():
    """The source configured in .env, created once."""
    global _source
    if _source is None:
        settings = load_settings()
        if settings.job_source == "live":
            from jobagent.live import HimalayasSource
            _source = HimalayasSource(offline=settings.live_offline)
        elif settings.job_source == "local_csv":
            _source = LocalCSVSource()
        else:
            raise ValueError(f"JOB_SOURCE must be local_csv or live, not "
                             f"'{settings.job_source}'.")
    return _source


def set_source(source) -> None:
    """Override the source (evaluation and tests)."""
    global _source
    _source = source
