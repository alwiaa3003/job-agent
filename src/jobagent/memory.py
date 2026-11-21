"""Conversation history kept inside a token budget.

Requests are system prompt + memory block + summary of old turns + recent (whole) turns."""

import json

from jobagent.guard import ids_from_tool_messages
from jobagent.store import Store

KEEP_FULL_TURNS = 2       # older turns are compacted to user text + final answer
MAX_USER_CHARS = 1000
MAX_SUMMARY_LINES = 12
MAX_MEMORY_JOBS = 10


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token count (~4 characters per token); slightly overestimates Groq's count."""
    return sum(len(json.dumps(m, ensure_ascii=False)) // 4 + 4 for m in messages)


def memory_block(store: Store) -> str:
    """Preferences and saved jobs from earlier sessions, as prompt text."""
    prefs = store.get_preferences()
    saved = store.list_saved()
    lines = ["Memory from earlier sessions (apply preferences as default filters "
             "unless the user says otherwise):"]
    lines.append("Preferences: " + ("; ".join(f"{k}={v}" for k, v in prefs.items())
                                    if prefs else "none"))
    if saved:
        shown = saved[-MAX_MEMORY_JOBS:]
        lines.append(f"Saved jobs ({len(saved)}): " + "; ".join(
            f"{j['title']} - {j['company']} ({j['job_id']})" for j in shown))
    else:
        lines.append("Saved jobs: none")
    return "\n".join(lines)


def _final_answer(turn: list[dict]) -> dict:
    return turn[-1]


def _summary_line(turn: list[dict]) -> str:
    # Extractive (no LLM call), so it can't invent anything and keeps job_ids exact.
    user = " ".join(turn[0]["content"].split())[:100]
    try:
        answer = json.loads(_final_answer(turn)["content"])
        kind, text, ids = answer["kind"], answer["message"], answer.get("job_ids", [])
    except (json.JSONDecodeError, KeyError, TypeError):
        kind, text, ids = "info", str(_final_answer(turn).get("content", "")), []
    text = " ".join(text.split())[:120]
    id_part = f" [job_ids: {', '.join(ids)}]" if ids else ""
    return f'- User: "{user}" -> {kind}: {text}{id_part}'


class Conversation:
    def __init__(self, system_prompt: str, store: Store, budget: int = 2500,
                 keep_full_turns: int = KEEP_FULL_TURNS):
        self.system_prompt = system_prompt
        self.store = store
        self.budget = budget
        self.keep_full_turns = keep_full_turns
        self.turns: list[list[dict]] = []
        self.summary: list[str] = []
        # Compaction drops tool results, but the guard still needs their job_ids.
        self.known_ids: set[str] = set()
        self._prefix_len = 0

    def system_message(self) -> dict:
        parts = [self.system_prompt, memory_block(self.store)]
        if self.summary:
            parts.append("Summary of earlier turns in this session:\n" + "\n".join(self.summary))
        return {"role": "system", "content": "\n\n".join(parts)}

    def start_turn(self, user_text: str) -> list[dict]:
        """Messages for a new user message; agent.run appends to them."""
        base = [self.system_message()] + [m for turn in self.turns for m in turn]
        self._prefix_len = len(base)
        if len(user_text) > MAX_USER_CHARS:
            user_text = user_text[:MAX_USER_CHARS] + " [message cut: too long]"
        return base + [{"role": "user", "content": user_text}]

    def end_turn(self, messages: list[dict], keep: bool = True) -> None:
        """Store the finished turn (keep=False drops it), then fit the budget."""
        if keep:
            turn = messages[self._prefix_len:]
            self.known_ids |= ids_from_tool_messages(turn)
            self.turns.append(turn)
        self._fit_budget()

    def history_tokens(self) -> int:
        summary = [{"role": "system", "content": "\n".join(self.summary)}] if self.summary else []
        return estimate_tokens(summary + [m for turn in self.turns for m in turn])

    def _fit_budget(self) -> None:
        for i in range(max(0, len(self.turns) - self.keep_full_turns)):
            turn = self.turns[i]
            if len(turn) > 2:
                self.turns[i] = [turn[0], _final_answer(turn)]
        while self.history_tokens() > self.budget and len(self.turns) > 1:
            self.summary.append(_summary_line(self.turns.pop(0)))
        self.summary = self.summary[-MAX_SUMMARY_LINES:]
