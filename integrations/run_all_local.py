#!/usr/bin/env python3
"""
Run the whole DrugDiscoveryBench suite locally

Drives ``run_task_local.py`` over every task, with limited concurrency and
resume, then summarises the scores.

Note that ``judge.py`` computes ``score = outcome_pct / 100`` -- the process
rubric is graded and recorded but does not contribute to the headline number.
This summary reports both.

As with the single-task runner, these scores are NOT comparable to official
DrugDiscoveryBench results: the agent has a different toolset and the prompt
carries an added environment note.

Usage::

    export MISTRAL_API_KEY=...
    python integrations/run_all_local.py --workers 3
    python integrations/run_all_local.py --summary-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
DEFAULT_BENCH = HERE.parents[1] / "DrugDiscoveryBench"
RUNNER = HERE / "run_task_local.py"


def task_ids(bench_dir: Path) -> list[str]:
    tasks = bench_dir / "benchmark" / "tasks"
    return sorted(p.name for p in tasks.iterdir() if (p / "task.toml").is_file())


def load_reward(out_dir: Path, task_id: str) -> dict | None:
    path = out_dir / task_id / "reward.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None


def run_one(task_id: str, args, index: int, total: int) -> tuple[str, dict | None]:
    cmd = [
        sys.executable, str(RUNNER), task_id,
        "--bench-dir", str(args.bench_dir),
        "--out-dir", str(args.out_dir),
        "--model", args.model,
        "--judge-model", args.judge_model,
        "--max-turns", str(args.max_turns),
        "--timeout", str(args.timeout),
    ]
    if not args.process:
        cmd.append("--no-process")

    started = time.time()
    log_dir = args.out_dir / task_id
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "runner.log", "w", encoding="utf-8") as log:
        subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            # A stuck agent must not stall the sweep: allow the agent budget
            # plus headroom for the judge.
            timeout=args.timeout + 900,
        )

    reward = load_reward(args.out_dir, task_id)
    score = reward.get("score") if reward else None
    elapsed = time.time() - started
    label = f"{score:.3f}" if isinstance(score, (int, float)) else "FAILED"
    print(f"[{index}/{total}] {task_id}  score={label}  ({elapsed:.0f}s)", flush=True)
    return task_id, reward


def summarise(out_dir: Path, ids: list[str]) -> None:
    rows = []
    for task_id in ids:
        reward = load_reward(out_dir, task_id)
        if reward:
            rows.append((task_id, reward))

    if not rows:
        print("No results yet.")
        return

    scores = [r["score"] for _, r in rows if isinstance(r.get("score"), (int, float))]
    proc = [r["process_pct"] for _, r in rows
            if isinstance(r.get("process_pct"), (int, float)) and r.get("process_present")]
    judge_failed = sum(1 for _, r in rows if r.get("judge_failed"))
    no_answer = sum(1 for _, r in rows if not r.get("answer_file_present"))

    print()
    print("=" * 62)
    print(f"tasks graded          : {len(rows)} / {len(ids)}")
    print(f"mean score (outcome)  : {sum(scores)/len(scores):.4f}" if scores else "no scores")
    print(f"solved (score = 1.0)  : {sum(1 for s in scores if s >= 0.999)}")
    print(f"partial (0 < s < 1)   : {sum(1 for s in scores if 0 < s < 0.999)}")
    print(f"zero (score = 0)      : {sum(1 for s in scores if s <= 0)}")
    if proc:
        print(f"mean process pct      : {sum(proc)/len(proc):.1f}%")
    print(f"judge failures        : {judge_failed}")
    print(f"no answer produced    : {no_answer}")
    print("=" * 62)

    print("\ntop scores:")
    for task_id, r in sorted(rows, key=lambda kv: -(kv[1].get("score") or 0))[:10]:
        print(f"  {r.get('score', 0):.3f}  {task_id}  (process {r.get('process_pct', 0):.0f}%)")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "n_graded": len(rows),
                "n_tasks": len(ids),
                "mean_score": (sum(scores) / len(scores)) if scores else None,
                "mean_process_pct": (sum(proc) / len(proc)) if proc else None,
                "results": {tid: r for tid, r in rows},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {summary_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all DrugDiscoveryBench tasks locally")
    ap.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--out-dir", type=Path, default=Path("local_runs"))
    ap.add_argument("--model", default=os.environ.get("VIBE_MODEL", "mistral-vibe-cli-latest"))
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "mistral-large-latest"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=120)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--process", action="store_true",
                    help="Also grade the process rubric (slower; does not change score)")
    ap.add_argument("--limit", type=int, help="Only run the first N pending tasks")
    ap.add_argument("--redo", action="store_true", help="Re-run tasks that already have a reward")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("MISTRAL_API_KEY"):
        print("Set MISTRAL_API_KEY.", file=sys.stderr)
        return 1

    ids = task_ids(args.bench_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        summarise(args.out_dir, ids)
        return 0

    pending = ids if args.redo else [t for t in ids if load_reward(args.out_dir, t) is None]
    if args.limit:
        pending = pending[: args.limit]

    print(f"tasks: {len(ids)} total, {len(pending)} to run, {args.workers} workers")
    print(f"agent: {args.model}   judge: {args.judge_model}")
    print()

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, t, args, i, len(pending)): t
            for i, t in enumerate(pending, 1)
        }
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"  {futures[fut]} raised {type(e).__name__}: {e}", flush=True)

    print(f"\nswept {len(pending)} task(s) in {(time.time()-started)/60:.1f} min")
    summarise(args.out_dir, ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
