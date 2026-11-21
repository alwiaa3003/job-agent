"""Settings loaded from .env and the environment. The API key is never printed."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The value in .env.example; treated as "no key set".
PLACEHOLDER_KEY = "your-groq-key-here"


@dataclass
class Settings:
    provider: str
    base_url: str
    model: str
    compare_model: str
    api_key_var: str
    reasoning_effort: str   # gpt-oss only: low | medium | high
    db_path: str
    history_budget: int     # estimated tokens for summary + recent turns
    job_source: str         # local_csv | live
    live_offline: bool      # live source reads only its cache
    api_key: str = field(default="", repr=False)   # keep the key out of logs

    @property
    def has_key(self) -> bool:
        return bool(self.api_key) and self.api_key != PLACEHOLDER_KEY


def _project_path(path: str) -> str:
    # Relative paths are resolved against the project, not the current directory.
    p = Path(path)
    return str(p if p.is_absolute() or path == ":memory:" else PROJECT_ROOT / p)


def load_settings(env_file: Path | None = None) -> Settings:
    """Read .env (if present) and environment variables into Settings."""
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

    key_var = os.getenv("LLM_API_KEY_VAR", "GROQ_API_KEY")
    return Settings(
        provider=os.getenv("LLM_PROVIDER", "groq"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
        compare_model=os.getenv("LLM_COMPARE_MODEL", "openai/gpt-oss-20b"),
        api_key_var=key_var,
        reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "low"),
        db_path=_project_path(os.getenv("JOBAGENT_DB", "data/jobagent.db")),
        history_budget=int(os.getenv("HISTORY_TOKEN_BUDGET", "2500")),
        job_source=os.getenv("JOB_SOURCE", "local_csv").strip().lower(),
        live_offline=os.getenv("LIVE_OFFLINE", "0").strip() in {"1", "true", "yes"},
        api_key=os.getenv(key_var, "").strip(),
    )
