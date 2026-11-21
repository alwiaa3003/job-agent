"""Terminal chat: python -m jobagent.chat  (commands: quit, /memory, /forget)."""

import sys

from jobagent.agent import Agent
from jobagent.llm import LLMClient, LLMError
from jobagent.memory import Conversation, memory_block
from jobagent.prompts import system_prompt
from jobagent.store import Store

QUIT_WORDS = {"quit", "exit", "q"}


def main() -> None:
    # Windows consoles default to cp1252, which can't print some characters models use.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        llm = LLMClient()
    except LLMError as err:
        print(f"Error: {err}")
        return

    store = Store(llm.settings.db_path)
    agent = Agent(llm, store)
    source = llm.settings.job_source
    conv = Conversation(system_prompt(source), store, budget=llm.settings.history_budget)

    prefs, saved = store.get_preferences(), store.list_saved()
    print(f"JobAgent (model: {llm.settings.model}). Type 'quit' to exit, "
          f"'/memory' or '/forget'.")
    if source == "live":
        mode = "cache only, no network" if llm.settings.live_offline else "online"
        print(f"Job source: LIVE remote jobs from Himalayas - https://himalayas.app "
              f"({mode}; jobs can change or close).")
    else:
        print("Job source: local LinkedIn dataset (US + India, 2023-2024).")
    print(f"Memory loaded: {len(prefs)} preference(s), {len(saved)} saved job(s).\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_text:
            continue
        if user_text.lower() in QUIT_WORDS:
            print("Bye!")
            break
        if user_text.lower() == "/memory":
            print(memory_block(store) + "\n")
            continue
        if user_text.lower() == "/forget":
            if input("Delete all saved jobs and preferences? (yes/no): ").strip() == "yes":
                store.clear()
                print("Memory deleted.\n")
            continue

        messages = conv.start_turn(user_text)
        result = agent.run(messages, conv.known_ids)
        conv.end_turn(messages, keep=result.stopped_reason != "llm_error")

        for call in result.tool_calls:
            status = "ok" if call.ok else f"ERROR: {call.error}"
            print(f"  [tool] {call.name}({call.arguments}) -> {status}")
        a = result.answer
        fixed = " (recovered from a fake tool call)" if result.recovered_fake_tool else ""
        print(f"  [answer] kind={a.kind} job_ids={a.job_ids} json={result.structured}{fixed}")
        g = result.guard
        if g.changed:
            print(f"  [guard] removed ids {g.removed_ids}, added ids {g.added_ids}, "
                  f"removed lines {g.removed_lines}, "
                  f"fixed lines {g.fixed_lines}, replaced answer {g.replaced}")
        real = f", biggest request {max(result.prompt_tokens)} tokens (real)" \
            if result.prompt_tokens else ""
        print(f"  [context] history ~{conv.history_tokens()} tokens, "
              f"{len(conv.turns)} turn(s), {len(conv.summary)} summary line(s){real}")
        print(f"\nAgent: {result.reply}\n")

    store.close()


if __name__ == "__main__":
    main()
