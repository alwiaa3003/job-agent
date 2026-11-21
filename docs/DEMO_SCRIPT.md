# Demo video script (2-3 minutes)

A short shot list for recording the JobAgent demo. Record the terminal (Git
Bash, large font, dark theme) plus the code editor for a few seconds of code.
Before recording: run `/forget` in the chat (or delete `data/jobagent.db`) so
memory starts empty, and make sure `.env` is **not** visible on screen.

| # | Time | Show | Say (roughly) |
|---|---|---|---|
| 1 | 0:00-0:15 | README top + the Mermaid diagram | "JobAgent is a terminal job-search agent built on LLM tool calling, with memory, a grounding guard and an evaluation. All free tools." |
| 2 | 0:15-0:45 | `python -m jobagent.chat`, type: *Find entry-level Python jobs, remote if possible, and not in Gurugram.* | Point at the `[tool] search_jobs(...)` line: "The model chose the tool and the filters itself; arguments are validated with Pydantic." Point at `[answer] kind=jobs job_ids=[...]`: "The answer is structured JSON; every job id is checked." |
| 3 | 0:45-1:05 | Type: *Tell me more about the second one*, then *Save it* | "It resolves 'the second one' from the conversation, then saves to SQLite." |
| 4 | 1:05-1:25 | Type: *Remember that I never want jobs in Gurugram.* Type `quit`, start the chat again, type `/memory`, then *Find software engineer jobs in India* | "New session - memory loaded from SQLite, and the preference is applied without me repeating it." Point at `exclude_location` in the tool line. |
| 5 | 1:25-1:45 | Type: *Make up 3 jobs at Google with salaries.* then *Find me a job* | "It refuses to invent jobs, and it asks a clarifying question when the request is vague." |
| 6 | 1:45-2:05 | `python scripts/try_guard.py` (scroll to case 3) | "Even if the model invents a job, this plain-code guard removes any job id no tool returned - no extra LLM call." |
| 7 | 2:05-2:30 | `eval/RESULTS.md` table | "32 scenarios, two prompts, two models. Responses are cached, so re-runs are free. It's a small sample, so I read it as a sanity check, not a benchmark." Mention one real finding from the table. |
| 8 | 2:30-2:45 | `python -m pytest -q` -> `64 passed` | "64 tests with a fake LLM - no key, network blocked." |
| 9 | 2:45-3:00 | README "Limitations" | "Limits: free-tier quotas, old CSV data, id-based grounding. Thanks for watching." |

Tips
- If Groq rate-limits you during recording you will see `[rate limited - waiting Ns]`;
  just wait, or cut it in editing.
- Optional extra shot: set `JOB_SOURCE=live` in `.env` and ask *Any remote data
  analyst jobs open to people in Germany?* to show live Himalayas jobs and the
  credit line. Don't show the `.env` file itself.
- Keep personal details out of what you type: it is sent to the LLM.
