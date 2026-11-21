"""Check the API key, list available models and test tool calling on each chat model.

Usage: python scripts/hello.py  (1 model list + 1 hello call + 1 tool call per chat model)"""

import json

from jobagent.llm import LLMClient, LLMError

# Speech and prompt-guard models can't chat.
NON_CHAT_WORDS = ["whisper", "orpheus", "prompt-guard"]

TEST_TOOL = {
    "type": "function",
    "function": {
        "name": "search_jobs",
        "description": "Search job postings.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Role or skills"},
                "remote": {"type": "boolean"},
            },
            "required": ["query"],
        },
    },
}


def is_chat_model(model: str) -> bool:
    return not any(word in model for word in NON_CHAT_WORDS)


def check_tool_call(llm: LLMClient, model: str) -> str:
    messages = [{"role": "user", "content": "Find remote Python developer jobs."}]
    try:
        resp = llm.chat(messages, tools=[TEST_TOOL], model=model, max_tokens=512)
    except LLMError as err:      # keep testing the other models
        return f"ERROR: {err}"
    calls = resp.choices[0].message.tool_calls
    tokens = resp.usage.total_tokens if resp.usage else "?"
    if not calls:
        return f"NO tool call (answered in text instead) [{tokens} tokens]"
    call = calls[0]
    try:
        args = json.loads(call.function.arguments)
    except json.JSONDecodeError:
        return f"tool call with INVALID JSON args: {call.function.arguments!r}"
    return f"OK -> {call.function.name}({args}) [{tokens} tokens]"


def main() -> None:
    try:
        llm = LLMClient()
        print(f"Provider: {llm.settings.provider}  ({llm.settings.base_url})")
        print(f"Default model: {llm.settings.model}   "
              f"Compare model: {llm.settings.compare_model}")

        models = llm.list_models()
        print(f"\nModels available to your key ({len(models)}):")
        for m in models:
            kind = "chat/text" if is_chat_model(m) else "skipped (not a chat model)"
            print(f"  {m:40s} {kind}")
        for label, m in [("LLM_MODEL", llm.settings.model),
                         ("LLM_COMPARE_MODEL", llm.settings.compare_model)]:
            if m not in models:
                print(f"\nWARNING: {label}={m} is NOT available to your key. "
                      f"Change it in .env (suggested: openai/gpt-oss-20b).")

        resp = llm.chat([{"role": "user", "content": "Say hello in one short sentence."}])
        print(f"\nHello reply ({llm.settings.model}): {resp.choices[0].message.content}")
        if resp.usage:
            details = resp.usage.completion_tokens_details
            thinking = getattr(details, "reasoning_tokens", None) if details else None
            print(f"Tokens used: {resp.usage.total_tokens} (thinking: {thinking})")

        print("\nTool-call test:")
        for m in models:
            if is_chat_model(m):
                print(f"  {m:30s} {check_tool_call(llm, m)}")
    except LLMError as err:
        print(f"Error: {err}")


if __name__ == "__main__":
    main()
