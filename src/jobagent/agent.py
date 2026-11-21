"""Agent loop: model, tool calls, tool results, validated and grounded final answer.

Agent.run never raises; API errors and bad tool calls end in a user-facing message.
"""

import json
from dataclasses import dataclass, field

from jobagent.answer import FinalAnswer, parse_answer
from jobagent.guard import (JOB_ID_IN_TEXT, GuardReport, check_answer,
                            ids_from_tool_messages)
from jobagent.llm import EmptyReplyError, LLMClient, LLMError, ToolUseFailedError
from jobagent.sources import get_source
from jobagent.store import Store
from jobagent.tools import run_tool, tool_schemas

LIVE_CREDIT = "Jobs from Himalayas: https://himalayas.app"


def _credit_live_source(answer: FinalAnswer) -> FinalAnswer:
    # Himalayas' API terms require a visible credit and link.
    if (get_source().name == "live" and answer.job_ids
            and "himalayas" not in answer.message.lower()):
        message = (answer.message + "\n\n" + LIVE_CREDIT)[-3000:]
        return answer.model_copy(update={"message": message})
    return answer

MAX_STEPS = 5   # model calls per user message


@dataclass
class ToolCallRecord:
    name: str
    arguments: str   # raw JSON as sent by the model
    ok: bool
    error: str = ""


@dataclass
class AgentResult:
    answer: FinalAnswer
    steps: int = 0                               # model calls made
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    stopped_reason: str = "answer"               # answer | max_steps | llm_error
    structured: str = "none"                     # first_try | after_retry | fallback | none
    format_error: str = ""
    recovered_fake_tool: bool = False
    prompt_tokens: list[int] = field(default_factory=list)   # as reported by the API
    guard: GuardReport = field(default_factory=GuardReport)

    @property
    def reply(self) -> str:
        return self.answer.message


class Agent:
    def __init__(self, llm: LLMClient, store: Store, max_steps: int = MAX_STEPS):
        self.llm = llm
        self.store = store
        self.max_steps = max_steps
        self.tools = tool_schemas()

    def run(self, messages: list[dict], known_ids: set[str] | None = None) -> AgentResult:
        """Answer the last user message, appending tool calls, results and the answer to messages.

        known_ids are job_ids returned by tools earlier in the session."""
        self._known_before = set(known_ids or ())
        result = AgentResult(answer=FinalAnswer(message="(no answer yet)"))
        empty_retry_used = False
        seen_calls = set()

        while result.steps < self.max_steps:
            result.steps += 1
            try:
                response = self.llm.chat(messages, tools=self.tools)
                _record_usage(result, response)
            except ToolUseFailedError as err:
                result.recovered_fake_tool = True
                text = _answer_from_fake_tool(err.failed_generation)
                if text is None:
                    # Ask once more with tools disabled.
                    result.steps += 1
                    try:
                        text = self.llm.chat(messages).choices[0].message.content
                    except (ToolUseFailedError, EmptyReplyError) as err2:
                        # Model misbehaviour, not an outage: let the format retry/fallback handle it.
                        text = _answer_from_fake_tool(getattr(err2, "failed_generation", "")) or ""
                    except LLMError as err2:
                        return self._fail(result, f"Sorry, I could not reach the model. {err2}")
                return self._finish(messages, text, result)
            except EmptyReplyError as err:
                if not empty_retry_used:          # empty replies are usually one-offs
                    empty_retry_used = True
                    continue
                return self._fail(result, f"Sorry, the model gave no answer. ({err})")
            except LLMError as err:
                return self._fail(result, f"Sorry, I could not reach the model. {err}")

            message = response.choices[0].message

            if not message.tool_calls:
                return self._finish(messages, message.content, result)

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name,
                                  "arguments": c.function.arguments or "{}"}}
                    for c in message.tool_calls
                ],
            })
            for call in message.tool_calls:
                key = (call.function.name, _normalise(call.function.arguments))
                if key in seen_calls:
                    output = {"ok": False, "error": "You already made this exact call. "
                              "Use its earlier result or answer the user."}
                else:
                    seen_calls.add(key)
                    output = run_tool(call.function.name, call.function.arguments, self.store)
                result.tool_calls.append(ToolCallRecord(
                    name=call.function.name,
                    arguments=call.function.arguments or "{}",
                    ok=output.get("ok", False),
                    error=output.get("error", ""),
                ))
                if not output.get("ok", False):
                    output["hint"] = "Fix the arguments and call the tool again, or tell the user."
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(output, ensure_ascii=False),
                })

        result.stopped_reason = "max_steps"
        result.answer = FinalAnswer(
            message=f"Sorry, I could not finish this in {self.max_steps} steps. "
                    "Please try a simpler or more specific request.", kind="error")
        messages.append({"role": "assistant", "content": result.answer.model_dump_json()})
        return result

    def _finish(self, messages: list[dict], text: str, result: AgentResult) -> AgentResult:
        answer = self._final_answer(messages, text, result)
        known = (self._known_before | ids_from_tool_messages(messages)
                 | {j["job_id"] for j in self.store.list_saved()})
        user_text = next((m["content"] for m in reversed(messages)
                          if m["role"] == "user"), "")
        user_ids = set(JOB_ID_IN_TEXT.findall(user_text))
        result.answer, result.guard = check_answer(answer, known, user_ids)
        result.answer = _credit_live_source(result.answer)
        # History keeps the grounded answer, never the raw model output.
        messages.append({"role": "assistant", "content": result.answer.model_dump_json()})
        return result

    def _final_answer(self, messages: list[dict], text: str,
                      result: AgentResult) -> FinalAnswer:
        answer, error = parse_answer(text)
        if answer:
            result.structured = "first_try"
            return answer

        result.format_error = error
        # One retry showing the bad reply and the error; not kept in the history.
        retry_messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": f"Your final answer was invalid: {error}. Reply "
             "again with ONLY the JSON object: "
             '{"message": "...", "job_ids": [...], "kind": "..."}'},
        ]
        result.steps += 1
        try:
            response = self.llm.chat(retry_messages)
            _record_usage(result, response)
            answer, error = parse_answer(response.choices[0].message.content)
        except ToolUseFailedError as err:
            answer, error = parse_answer(_answer_from_fake_tool(err.failed_generation) or "")
        except EmptyReplyError as err:
            answer, error = None, str(err)
        except LLMError as err:                # outage, not a format failure
            result.stopped_reason = "llm_error"
            answer, error = None, str(err)
        if answer:
            result.structured = "after_retry"
            return answer

        # Plain text is shown as is; broken JSON is never shown.
        result.structured = "fallback"
        result.format_error += f" | retry: {error}"
        plain = (text or "").strip()
        if not plain or "{" in plain:
            plain = "Sorry, I could not format my answer. Please ask again."
        return FinalAnswer(message=plain[:3000], kind="error")

    @staticmethod
    def _fail(result: AgentResult, text: str) -> AgentResult:
        result.stopped_reason = "llm_error"
        result.answer = FinalAnswer(message=text, kind="error")
        return result


def _record_usage(result: AgentResult, response) -> None:
    usage = getattr(response, "usage", None)
    if usage is not None and getattr(usage, "prompt_tokens", None):
        result.prompt_tokens.append(usage.prompt_tokens)


def _answer_from_fake_tool(failed_generation: str) -> str | None:
    # failed_generation looks like {"name": "json", "arguments": {"message": ...}}.
    try:
        data = json.loads(failed_generation)
    except (json.JSONDecodeError, TypeError):
        return None
    args = data.get("arguments") if isinstance(data, dict) else None
    if isinstance(args, str):            # arguments may be a string or an object
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if isinstance(args, dict) and "message" in args:
        return json.dumps(args)
    return None


def _normalise(arguments: str | None) -> str:
    # Key order and spacing don't make a call different.
    try:
        return json.dumps(json.loads(arguments or "{}"), sort_keys=True)
    except json.JSONDecodeError:
        return arguments or ""
