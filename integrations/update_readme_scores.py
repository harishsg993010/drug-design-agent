#!/usr/bin/env python3
"""
Refresh the benchmark results block in README.md

Reads ``local_runs/summary.json`` (written by ``run_all_local.py``) and rewrites
the region between the BENCH markers, so the published numbers always match the
run on disk instead of drifting from hand edits.

Usage::

    python integrations/run_all_local.py --summary-only   # refresh summary.json
    python integrations/update_readme_scores.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

START = "<!-- BENCH:START -->"
END = "<!-- BENCH:END -->"


def build_block(summary: dict, top_n: int) -> str:
    results = summary.get("results", {})
    graded = summary.get("n_graded", len(results))
    total = summary.get("n_tasks", 82)
    mean = summary.get("mean_score")

    scored = [
        (tid, r.get("score"))
        for tid, r in results.items()
        if isinstance(r.get("score"), (int, float))
    ]
    solved = sum(1 for _, s in scored if s >= 0.999)
    partial = sum(1 for _, s in scored if 0 < s < 0.999)
    zero = sum(1 for _, s in scored if s <= 0)
    no_answer = sum(1 for _, r in results.items() if not r.get("answer_file_present"))
    judge_failed = sum(1 for _, r in results.items() if r.get("judge_failed"))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status = "complete" if graded >= total else f"in progress ({graded}/{total})"

    lines = [
        START,
        "",
        f"Agent **Mistral Vibe** (`mistral-vibe-cli-latest`) with this project's MCP",
        f"server registered; judge **`mistral-large-latest`**. Run {status}, "
        f"last updated {stamp}.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| tasks graded | {graded} / {total} |",
        f"| **mean score** (outcome) | **{mean:.3f}** |" if isinstance(mean, (int, float))
        else "| **mean score** (outcome) | n/a |",
        f"| solved (1.0) | {solved} |",
        f"| partial (0 < s < 1) | {partial} |",
        f"| zero | {zero} |",
        f"| no answer produced | {no_answer} |",
        f"| judge failures | {judge_failed} |",
        "",
    ]

    if scored:
        lines += [
            f"Top {min(top_n, len(scored))} tasks:",
            "",
            "| score | task |",
            "|---|---|",
        ]
        for tid, score in sorted(scored, key=lambda kv: -kv[1])[:top_n]:
            lines.append(f"| {score:.3f} | `{tid}` |")
        lines.append("")

    lines += [
        "Scored by each task's own `judge.py` against the gated expert rubrics.",
        "",
        END,
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh README benchmark results")
    ap.add_argument("--summary", type=Path, default=Path("local_runs/summary.json"))
    ap.add_argument("--readme", type=Path, default=Path("README.md"))
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    if not args.summary.is_file():
        print(f"No summary at {args.summary}; run run_all_local.py --summary-only first.",
              file=sys.stderr)
        return 1

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    block = build_block(summary, args.top)

    readme = args.readme.read_text(encoding="utf-8")
    if START not in readme or END not in readme:
        print(f"README is missing the {START} / {END} markers.", file=sys.stderr)
        return 1

    head, _, rest = readme.partition(START)
    _, _, tail = rest.partition(END)
    updated = head + block + tail

    # Keep the headline row in step with the table; a hand-written score there
    # goes stale the moment another task lands.
    results = summary.get("results", {})
    scored = [r.get("score") for r in results.values()
              if isinstance(r.get("score"), (int, float))]
    if scored:
        mean_pct = 100 * sum(scored) / len(scored)
        solved = sum(1 for s in scored if s >= 0.999)
        above = sum(1 for s in scored if s > 0)
        updated = re.sub(
            r"\| \*\*Benchmark score\*\* \|[^|]*\|",
            f"| **Benchmark score** | **{mean_pct:.1f}** mean outcome on "
            f"DrugDiscoveryBench ({len(scored)} tasks) |",
            updated,
        )
        updated = re.sub(
            r"\| \*\*Tasks solved outright\*\* \|[^|]*\|",
            f"| **Tasks solved outright** | **{solved}** · {above} scoring above zero |",
            updated,
        )

    args.readme.write_text(updated, encoding="utf-8")

    print(f"README updated: {summary.get('n_graded')}/{summary.get('n_tasks')} graded, "
          f"mean {summary.get('mean_score')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
