"""Hand-authored agentic-workflow prompts for distillation seed dataset.

Covers the core competencies any capable agent needs:
  - tool selection & composition
  - multi-step planning with revision
  - reflection and self-correction
  - error recovery from tool failures
  - long-horizon task decomposition
  - context-window awareness
  - user-intent clarification vs. autonomous action
  - structured output (JSON/function-call) discipline
"""
from __future__ import annotations

from typing import List, TypedDict


class AgenticPrompt(TypedDict):
    id: str
    category: str
    instruction: str


AGENTIC_PROMPTS: List[AgenticPrompt] = [
    {
        "id": "agent-tool-select-01",
        "category": "tool_selection",
        "instruction": (
            "You are an agent with access to three tools: web_search(query), "
            "read_file(path), and run_shell(command). A user asks: 'What's the "
            "latest Python version installed on this machine, and is it newer "
            "than the one mentioned in my requirements.txt?' Explain — in order — "
            "which tools you would call, with exact arguments, and why. Do NOT "
            "call tools you don't need, and don't guess where a single tool call "
            "could answer the question."
        ),
    },
    {
        "id": "agent-plan-revise-01",
        "category": "planning",
        "instruction": (
            "You are planning a task: 'Migrate our PostgreSQL database from "
            "version 13 to 16 with zero downtime.' Produce a numbered plan with "
            "at least 6 steps. Then identify the single riskiest step and "
            "propose one concrete fallback if that step fails mid-migration. "
            "Do not pad the plan with obvious prep steps (like 'read the docs')."
        ),
    },
    {
        "id": "agent-reflection-01",
        "category": "reflection",
        "instruction": (
            "You just ran `grep -r 'TODO' src/` and got 847 matches. Your "
            "original goal was to 'find TODOs that block the v2 release'. "
            "Reflect: was grep the right tool? If not, what would you do "
            "differently now, and what specifically went wrong with the first "
            "approach? Then write the corrected next action as a single tool "
            "call with exact arguments."
        ),
    },
    {
        "id": "agent-error-recovery-01",
        "category": "error_recovery",
        "instruction": (
            "A tool call `run_shell('npm install')` failed with: "
            "'ENOSPC: no space left on device'. You need to install dependencies "
            "to continue the task. Produce a 3-step recovery plan and the exact "
            "shell command for step 1. Do not just retry the failing command."
        ),
    },
    {
        "id": "agent-decomp-01",
        "category": "decomposition",
        "instruction": (
            "Decompose this request into sub-tasks suitable for parallel "
            "execution by separate agent workers: 'Audit our codebase for "
            "deprecated API usage, missing type hints, and security "
            "vulnerabilities, then prioritize findings by severity.' Output a "
            "DAG: list the sub-tasks, which can run in parallel, and which must "
            "be sequential (with explicit reasons)."
        ),
    },
    {
        "id": "agent-clarify-vs-act-01",
        "category": "intent_clarification",
        "instruction": (
            "A user says: 'Clean up the database.' You are an autonomous agent "
            "with write access to production. Decide: do you ask for "
            "clarification, or act? State your decision in one sentence, then "
            "either (a) produce exactly the clarifying question you'd ask "
            "(one question, not a menu), or (b) produce the exact plan you'd "
            "execute with explicit assumptions you're committing to."
        ),
    },
    {
        "id": "agent-structured-output-01",
        "category": "structured_output",
        "instruction": (
            "Given this user request — 'Book me a window seat on the cheapest "
            "direct flight from SFO to JFK next Friday under $400' — produce "
            "exactly one JSON object suitable for a flight_search() tool call "
            "with fields origin, destination, date, seat_preference, "
            "max_price_usd, direct_only. If any field is ambiguous, use null "
            "and explain in a separate `assumptions` array. Output only the "
            "JSON — no prose."
        ),
    },
    {
        "id": "agent-context-budget-01",
        "category": "context_management",
        "instruction": (
            "You have 8,000 tokens of remaining context budget and 14 source "
            "files (average ~2,500 tokens each) that may be relevant to a bug "
            "report. Describe your exact triage strategy: which files you read "
            "first, how you decide when to stop reading, and what you do if the "
            "first 3 reads don't localize the bug. Produce a concrete numbered "
            "procedure, not general advice."
        ),
    },
    {
        "id": "agent-tool-compose-01",
        "category": "tool_selection",
        "instruction": (
            "You have: git_log(path), git_blame(path, line), and "
            "gh_pr_view(number). A user asks: 'Why was this line added to "
            "src/auth.py:42?' Produce the exact sequence of tool calls (with "
            "real argument values — assume this is a real repo where auth.py "
            "exists) that leads to the most specific answer in the fewest "
            "calls. Explain the chain of inference between each call."
        ),
    },
    {
        "id": "agent-reflection-02",
        "category": "reflection",
        "instruction": (
            "Your last 3 attempts to write unit tests for a function kept "
            "producing tests that pass trivially (e.g., asserting the result "
            "is not None). Reflect on what you're missing. Then produce a "
            "specific checklist of 5 concrete test qualities — each phrased "
            "as something testable — you'll apply to the next attempt. The "
            "checklist should be specific to this failure mode, not generic "
            "'good test' advice."
        ),
    },
    {
        "id": "agent-safety-gate-01",
        "category": "safety_gates",
        "instruction": (
            "You are executing this task: 'Update all users in the production "
            "database who haven't logged in for 90 days to set status=inactive.' "
            "Before running the UPDATE statement, produce the exact pre-flight "
            "sequence you'd run (read-only commands only), the specific "
            "numeric sanity checks you'd confirm, and the rollback plan if "
            "the UPDATE affects more rows than expected. Do NOT include the "
            "UPDATE itself in your answer."
        ),
    },
    {
        "id": "agent-long-horizon-01",
        "category": "long_horizon",
        "instruction": (
            "A user gives you a 2-week project: 'Replace our internal "
            "monitoring stack with OpenTelemetry.' You can only see the first "
            "day's context at a time. Design a written handoff format your "
            "future-self can read on day 2 to resume without re-reading "
            "everything. Show an example handoff note after day 1, with "
            "enough structure that any future agent (not just you) could "
            "continue effectively."
        ),
    },
    {
        "id": "agent-permission-scope-01",
        "category": "safety_gates",
        "instruction": (
            "The user has approved you to 'run the test suite'. During the "
            "run you notice a failing test appears to be caused by an obvious "
            "typo in a source file. Should you fix the typo? Answer in one "
            "sentence, then justify briefly with reference to scope of "
            "authorization."
        ),
    },
    {
        "id": "agent-tool-abuse-detect-01",
        "category": "tool_selection",
        "instruction": (
            "An agent has a `run_shell` tool and a task: 'Count the number of "
            "files in the project.' It runs: "
            "`find . -type f | wc -l` (returns 1,247) and then "
            "`ls -la | wc -l` (returns 28) and then `tree | tail -1`. "
            "Critique this tool usage: which calls were redundant, which was "
            "wrong, and what was the minimum-call correct solution?"
        ),
    },
    {
        "id": "agent-multi-agent-coord-01",
        "category": "decomposition",
        "instruction": (
            "You are a coordinator agent dispatching work to three specialist "
            "sub-agents: backend, frontend, and devops. A user wants a new "
            "feature: 'Add rate limiting to our public API, with a dashboard "
            "showing usage per user.' Produce the three delegation prompts "
            "(one per sub-agent), each self-contained (no references to the "
            "others), with explicit success criteria and explicit scope limits "
            "so they don't step on each other's work."
        ),
    },
]


def get_prompts() -> List[AgenticPrompt]:
    return list(AGENTIC_PROMPTS)


if __name__ == "__main__":
    for p in AGENTIC_PROMPTS:
        print(f"[{p['category']}] {p['id']}: {p['instruction'][:100]}...")
    print(f"\nTotal: {len(AGENTIC_PROMPTS)} prompts across "
          f"{len({p['category'] for p in AGENTIC_PROMPTS})} categories")
