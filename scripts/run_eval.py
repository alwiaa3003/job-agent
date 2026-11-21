"""Run the evaluation for one prompt version and model, then rebuild eval/RESULTS.md.

Usage: python scripts/run_eval.py --prompt v1|v2 --model <id> [--only A1,F3] [--report]"""

import argparse
import json
import sys
import time

from jobagent.config import PROJECT_ROOT
from jobagent.eval_scenarios import SCENARIOS
from jobagent.evaluate import run_scenario, summarize
from jobagent.llm import LLMError
from jobagent.llm_cache import CachingLLM

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"                 # metrics + checks only
FULL_DIR = PROJECT_ROOT / "eval" / "cache" / "results_full"     # full transcripts, kept local
REPORT = PROJECT_ROOT / "eval" / "RESULTS.md"

ROWS = [("scenarios_passed", "Scenarios passed (all checks)"),
        ("correct_tool", "Correct tool chosen"),
        ("valid_arguments", "Valid tool arguments"),
        ("correct_arguments", "Correct filter values"),
        ("json_valid_first_try", "Valid JSON answer, first try"),
        ("json_valid_after_retry", "Valid JSON answer, after 1 retry"),
        ("no_invented_jobs_model", "No invented jobs (model, before guard)"),
        ("shown_answers_grounded", "Shown answers grounded (after guard)"),
        ("answer_kind_correct", "Answer kind correct"),
        ("memory_recall", "Memory recall (new session)")]


def run(prompt: str, model: str, only: set[str] | None) -> None:
    scenarios = [s for s in SCENARIOS if not only or s["id"] in only]
    skipped = []
    if not (PROJECT_ROOT / "data" / "jobs_india.csv").exists():
        skipped = [s["id"] for s in scenarios if s.get("needs_india")]
        scenarios = [s for s in scenarios if not s.get("needs_india")]
        print(f"NOTE: India data not installed, so {len(skipped)} scenario(s) are skipped: "
              f"{', '.join(skipped)}. Results are NOT comparable with the full 32-scenario "
              "table in the README.")
    try:
        llm = CachingLLM(model=model)
    except LLMError as err:                         # e.g. no API key
        print(f"Error: {err}")
        print("Tip: 'python scripts/run_eval.py --report' rebuilds the table from the "
              "saved results without any API key.")
        return
    results, start = [], time.time()
    stopped = ""
    for scn in scenarios:
        try:
            r = run_scenario(scn, llm, prompt)
        except LLMError as err:
            stopped = str(err)
        else:
            lost = [t for t in r["turns"] if t["stopped"] == "llm_error"]
            if llm.stop_reason or lost:
                # Quota or API errors aren't the model's fault: don't score it, resume later.
                stopped = llm.stop_reason or lost[0]["answer"]["message"]
        if stopped:
            print(f"  {scn['id']}: STOPPED (not counted) - {stopped[:300]}")
            print(f"  -> {len(results)} scenario(s) kept. Everything is cached: run the "
                  "same command again later to resume (cached calls are free).")
            break
        results.append(r)
        failed = [k for k, v in r["checks"].items() if not v]
        print(f"  {scn['id']:3s} {'PASS' if not failed else 'FAIL ' + ','.join(failed):30s}"
              f" kind={r['turns'][-1]['answer']['kind']:10s} cache hits/new: "
              f"{llm.hits}/{llm.misses}  ({time.time() - start:.0f}s)", flush=True)

    if only:                                        # partial runs don't overwrite results
        return
    name = f"{prompt}__{model.replace('/', '_')}"
    out = {"prompt": prompt, "model": model, "complete": len(results) == len(scenarios),
           "stopped": stopped, "skipped_no_india_data": skipped,
           "summary": summarize(results), "results": results}
    # Transcripts can contain jobs from the unlicensed India data, so they stay local.
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    (FULL_DIR / f"{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(public_view(out), indent=1),
                                              encoding="utf-8")
    print(f"Saved eval/results/{name}.json (metrics + checks) and "
          f"eval/cache/results_full/{name}.json (full, not committed)  "
          f"(new API calls this run: {llm.misses})")


def public_view(out: dict) -> dict:
    """Results without job text: metrics plus each scenario's id, category and checks."""
    slim = [{k: r[k] for k in ("id", "category", "memory", "checks")} for r in out["results"]]
    return {**out, "results": slim}


def fmt(rate: dict) -> str:
    return "-" if rate["pct"] is None else f"{rate['pct']:.0f}% ({rate['good']}/{rate['total']})"


def report() -> None:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS_DIR.glob("*.json"))]
    if not runs:
        print("No results yet.")
        return
    def label(r):
        tag = "" if r["complete"] else " (partial)"
        if r.get("skipped_no_india_data"):
            tag += f" (without India data: {len(r['results'])} scenarios)"
        return f"{r['prompt']} + {r['model'].split('/')[-1]}{tag}"
    head = "| Metric | " + " | ".join(label(r) for r in runs) + " |"
    lines = ["# Evaluation results", "",
             f"{len(SCENARIOS)} scenarios, local CSV data, LLM responses cached "
             "(re-runs are free and identical). Generated by `python scripts/run_eval.py`.", "",
             head, "|" + "---|" * (len(runs) + 1)]
    for key, label in ROWS:
        lines.append(f"| {label} | " + " | ".join(fmt(r["summary"][key]) for r in runs) + " |")
    for key, label in [("fake_tool_recoveries", "Fake tool calls recovered"),
                       ("guard_fixed_lines", "Lines corrected or annotated by guard"),
                       ("llm_errors", "Turns lost to API errors"),
                       ("model_calls", "Model calls"),
                       ("prompt_tokens", "Prompt tokens (counted calls only)"),
                       ("calls_without_token_count", "Calls with no token count")]:
        lines.append(f"| {label} | " + " | ".join(str(r["summary"][key]) for r in runs) + " |")

    lines += ["", "Notes: percentages are rounded; counts are exact. 32 scenarios is a small "
              "sample, so a difference of one or two scenarios is not meaningful. "
              "\"Prompt tokens\" leaves out calls that ended in a fake tool call (Groq "
              "returns no token count for them), so token totals are NOT comparable "
              "between columns with different numbers of such calls."]
    lines += ["", "## Pass rate by category", "",
              "| Category | " + " | ".join(f"{r['prompt']} + {r['model'].split('/')[-1]}"
                                          for r in runs) + " |",
              "|" + "---|" * (len(runs) + 1)]
    for cat in runs[0]["summary"]["by_category"]:
        lines.append(f"| {cat} | " + " | ".join(fmt(r["summary"]["by_category"].get(
            cat, {"pct": None})) for r in runs) + " |")

    lines += ["", "## Failed checks", ""]
    for r in runs:
        fails = [f"{x['id']} ({', '.join(k for k, v in x['checks'].items() if not v)})"
                 for x in r["results"] if not all(x["checks"].values())]
        lines.append(f"- **{r['prompt']} + {r['model'].split('/')[-1]}:** "
                     + (", ".join(fails) or "none"))
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="v1", choices=["v1", "v2"])
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--only", default="", help="comma-separated scenario ids")
    p.add_argument("--report", action="store_true", help="only rebuild eval/RESULTS.md")
    args = p.parse_args()
    if not args.report:
        print(f"Evaluating prompt {args.prompt} on {args.model}")
        run(args.prompt, args.model, set(filter(None, args.only.split(","))) or None)
    report()


if __name__ == "__main__":
    main()
