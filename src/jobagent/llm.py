"""OpenAI-compatible client wrapper (Groq) with 429 backoff and user-facing errors."""

import time

import openai

from jobagent.config import Settings, load_settings


class LLMError(Exception):
    """An error whose message is safe to show to the user."""


class EmptyReplyError(LLMError):
    """The model returned neither text nor a tool call."""


class DailyLimitError(LLMError):
    """HTTP 429 for a per-day limit; retrying within minutes is pointless."""


class ToolUseFailedError(LLMError):
    """HTTP 400 tool_use_failed; failed_generation holds the model's raw output."""
    def __init__(self, message: str, failed_generation: str = ""):
        super().__init__(message)
        self.failed_generation = failed_generation


class LLMClient:
    def __init__(self, settings: Settings | None = None, max_retries: int = 4,
                 timeout: float = 30.0):
        self.settings = settings or load_settings()
        if not self.settings.has_key:
            raise LLMError(
                f"No API key found. Copy .env.example to .env and set "
                f"{self.settings.api_key_var}=<your key>. "
                f"Free key: https://console.groq.com/keys"
            )
        self.max_retries = max_retries
        # Retries are done in _with_retries so 429s get a clear message and daily limits stop early.
        self._client = openai.OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout=timeout,
            max_retries=0,
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             model: str | None = None, temperature: float = 0.2,
             max_tokens: int = 1024):
        """Send a chat request; raises LLMError if nothing usable comes back.

        For gpt-oss, max_tokens includes hidden reasoning, so small values give empty replies."""
        model = model or self.settings.model
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if is_reasoning_model(model):
            kwargs["reasoning_effort"] = self.settings.reasoning_effort
        if tools:
            kwargs["tools"] = tools
        response = self._with_retries(lambda: self._client.chat.completions.create(**kwargs))
        _check_not_empty(response, max_tokens)
        return response

    def list_models(self) -> list[str]:
        """Model ids available to this key."""
        page = self._with_retries(lambda: self._client.models.list())
        return sorted(m.id for m in page.data)

    def _with_retries(self, call):
        # Exponential backoff (2, 4, 8, 16 s) unless the server sends retry-after.
        for attempt in range(self.max_retries + 1):
            try:
                return call()
            except openai.RateLimitError as err:
                if _is_daily_limit(err):
                    raise DailyLimitError(
                        f"Daily limit reached for this model (HTTP 429): {_short(err, 300)}"
                    ) from err
                if attempt == self.max_retries:
                    raise LLMError(
                        "Rate limit reached (HTTP 429) and retries ran out. "
                        "The free tier allows only a few requests per minute - "
                        "wait a minute and try again."
                    ) from err
                wait = _retry_after(err) or 2 ** (attempt + 1)
                print(f"[rate limited - waiting {wait:.0f}s, "
                      f"retry {attempt + 1}/{self.max_retries}]")
                time.sleep(wait)
            except openai.AuthenticationError as err:
                raise LLMError(
                    f"The API key was rejected (HTTP 401). Check "
                    f"{self.settings.api_key_var} in your .env file."
                ) from err
            except openai.NotFoundError as err:
                raise LLMError(
                    "Model not found (HTTP 404). Check LLM_MODEL in .env."
                ) from err
            except openai.APIConnectionError as err:
                raise LLMError(
                    f"Could not reach {self.settings.base_url}. "
                    "Check your internet connection."
                ) from err
            except openai.BadRequestError as err:
                body = err.body if isinstance(err.body, dict) else {}
                if body.get("code") == "tool_use_failed":
                    # gpt-oss sometimes sends its JSON answer as a call to a tool named "json";
                    # Groq returns the attempted output, so the agent can recover it.
                    raise ToolUseFailedError("The model made an invalid tool call.",
                                             body.get("failed_generation") or "") from err
                raise LLMError(f"The API rejected the request (HTTP 400): "
                               f"{_short(err)}") from err
            except openai.APIStatusError as err:
                raise LLMError(f"The API returned an error (HTTP {err.status_code}): "
                               f"{_short(err)}") from err


def is_reasoning_model(model: str) -> bool:
    """True for models that accept reasoning_effort (gpt-oss)."""
    return "gpt-oss" in model


def _check_not_empty(response, max_tokens: int) -> None:
    if not response.choices:
        raise EmptyReplyError("The model returned no reply at all. Please try again.")
    choice = response.choices[0]
    if (choice.message.content or "").strip() or choice.message.tool_calls:
        return
    if choice.finish_reason == "length":
        raise EmptyReplyError(
            f"The model used all {max_tokens} tokens on thinking and wrote no answer. "
            "Try again, ask a simpler question, or raise max_tokens."
        )
    raise EmptyReplyError(f"The model returned an empty reply "
                   f"(finish_reason={choice.finish_reason}). Please try again.")


def _is_daily_limit(err: openai.RateLimitError) -> bool:
    # Per-minute limits clear within a minute, so a long retry-after also means a daily limit.
    body = err.body if isinstance(err.body, dict) else {}
    text = f"{body.get('message', '')} {err}".lower()
    if "per day" in text or "(tpd)" in text or "(rpd)" in text:
        return True
    try:
        return float(err.response.headers.get("retry-after", "0")) > 120
    except (ValueError, AttributeError):
        return False


def _retry_after(err: openai.RateLimitError) -> float | None:
    try:
        value = float(err.response.headers.get("retry-after", ""))
        return min(value, 60.0)
    except (ValueError, AttributeError):
        return None


def _short(err: Exception, limit: int = 200) -> str:
    text = str(err)
    return text if len(text) <= limit else text[:limit] + "..."
