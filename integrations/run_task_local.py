#!/usr/bin/env python3
"""
Run one DrugDiscoveryBench task locally, without Docker

The official harness runs each task inside a ~23 GB image that carries the
BiOMNI toolchain and a biomedical data lake, orchestrated by Harbor. This
script does the same two steps on the host instead:

    1. Mistral Vibe answers the task, with this project's MCP server registered
       so it has real database tools (UniProt, ChEMBL, PDB, PubChem, KEGG,
       NCBI, OpenTargets).
    2. The task's own ``tests/judge.py`` -- which is a standalone rubric judge
       -- grades the answer and writes ``reward.json``.

IMPORTANT: scores from this are NOT comparable to official DrugDiscoveryBench
results. The agent has a different toolset (no BiOMNI, no local data lake) and
no egress hardening, and the prompt carries an added note about that. Treat it
as a local development loop for the tools, not as a benchmark submission.

Usage::

    export MISTRAL_API_KEY=...
    python integrations/run_task_local.py 69b025e20c10fe76b7aaf812
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Answers and transcripts contain non-ASCII; Windows would otherwise give this
# process's stdout the ANSI codepage and raise on the first arrow or dash.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_BENCH = Path(__file__).resolve().parents[2] / "DrugDiscoveryBench"

# Appended to the task prompt. The instruction assumes the BiOMNI container, so
# the agent is told plainly what it actually has instead of being left to
# discover that /biomni does not exist.
LOCAL_ENV_NOTE = """

---

## Environment note for this run

You are NOT running inside the benchmark container. Ignore any instruction
above that refers to `/biomni/...`, `/workspace/data/biomni_data/...`, the
conda env `biomni_e1`, or importing `biomni.tool.*` -- none of that exists here.

### What you have instead

The `drug-discovery` MCP server, with live tools for:

- Databases: `query_uniprot`, `query_chembl`, `query_pdb`, `query_pubchem`,
  `query_kegg`, `query_ncbi` (genes and PubMed), `query_opentargets`,
  `search_compounds`, `search_proteins`
- Cheminformatics: `calculate_descriptors`, `molecular_similarity`,
  `calculate_fingerprint`, `check_drug_likeness`, `predict_admet`,
  `smiles_to_inchi`, `generate_conformers`
- Structures: `parse_pdb`, `download_pdb`, `superimpose_structures`,
  `calculate_rmsd`, `analyze_binding_site`, `find_interactions`

You also have a shell with network access. You may `pip install` Python
packages, `curl` public APIs and data files, and run R if a task needs a dataset
that ships with an R package.

### Ground every number in retrieved data

This is the part that decides whether your answer is right.

- Do NOT answer from memory. Recalled statistics, dataset contents and paper
  figures are frequently wrong in detail, and a plausible-looking number that
  you did not verify will be marked incorrect.
- For any quantity the task asks about, fetch the underlying data and compute
  it. Load the actual dataset, query the actual database, read the actual
  record -- then derive the value.
- If a task names a paper, dataset, gene, compound or structure, look it up
  through the tools above or the web before relying on anything about it.
- If you genuinely cannot retrieve a needed source, say so explicitly in your
  working and state the assumption you fell back on, rather than presenting a
  remembered figure as if it were measured.
- Sanity-check your result before writing it: does it follow from the data you
  actually pulled, and does it answer the exact question asked?

### Final answer

Write your final answer -- and only the answer, no prose explanation -- to a
file named exactly `answer.md` in your current working directory:

    answer.md

Write it with a relative path. Do not nest it in a subdirectory, and do not
repeat the answer as prose in your reply -- the file is what gets graded.

Write your best current answer to `answer.md` as soon as you have one, then
overwrite it as you learn more. Do not save it for the end: if you run out of
turns before writing the file, the run scores zero regardless of your work.

Keep it terse, and if the task specifies an output format (YES/NO, a number to
a stated precision, JSON, a SMILES string, a list), produce exactly that format.
"""


def run_agent(instruction: str, workdir: Path, model: str, max_turns: int, timeout: int) -> Path:
    """Have Vibe answer the task; return the path it was told to write."""
    answer_path = workdir / "answer.md"
    prompt = instruction + LOCAL_ENV_NOTE

    (workdir / "instruction_used.md").write_text(prompt, encoding="utf-8")

    env = dict(os.environ)
    env["VIBE_ACTIVE_MODEL"] = model
    # Vibe emits non-ASCII (e.g. U+2713) and dies with a charmap error when
    # Windows gives a piped stdout the ANSI codepage instead of UTF-8.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    argv = [
        "vibe", "-p", prompt,
        "--output", "json",
        "--trust",
        "--auto-approve",
        "--max-turns", str(max_turns),
    ]

    print(f"[agent] vibe ({model}), max-turns={max_turns}, timeout={timeout}s")
    started = time.time()
    with open(workdir / "vibe.json", "w", encoding="utf-8") as out, \
         open(workdir / "vibe.stderr", "w", encoding="utf-8") as err:
        try:
            proc = subprocess.run(
                argv, cwd=workdir, env=env, stdout=out, stderr=err, timeout=timeout
            )
            code = proc.returncode
        except subprocess.TimeoutExpired:
            print(f"[agent] timed out after {timeout}s")
            code = -1

    print(f"[agent] finished in {time.time() - started:.0f}s (exit {code})")
    return answer_path


def extract_answer_from_transcript(workdir: Path, answer_path: Path) -> bool:
    """
    Fall back to the assistant's last message if no answer file was written

    Agents do not always honour "write it to this path", and an empty answer
    scores 0 for the wrong reason.
    """
    if answer_path.is_file() and answer_path.read_text(encoding="utf-8", errors="replace").strip():
        return True

    # Agents sometimes resolve the path against their own cwd and nest it.
    # Prefer the shallowest match so a stray deep copy cannot win.
    nested = sorted(
        (p for p in workdir.rglob("answer.md")
         if p != answer_path and p.read_text(encoding="utf-8", errors="replace").strip()),
        key=lambda p: len(p.relative_to(workdir).parts),
    )
    if nested:
        found = nested[0]
        print(f"[agent] found answer at {found.relative_to(workdir)}")
        answer_path.write_text(found.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        return True

    transcript = workdir / "vibe.json"
    if not transcript.is_file():
        return False

    raw = transcript.read_text(encoding="utf-8", errors="replace")
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        # A session stopped at the turn/price ceiling prints a stop event
        # instead of the closing array, so fall back to per-line parsing.
        entries = []
        for line in raw.splitlines():
            line = line.strip().rstrip(",")
            if line.startswith("{"):
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not entries:
            stop = next((l for l in raw.splitlines() if "vibe_stop_event" in l), "")
            print(f"[agent] unusable transcript{(': ' + stop.strip()) if stop else ''}")
            return False

    texts = [
        "".join(p.get("text", "") for p in (e.get("content") or []) if p.get("type") == "text")
        for e in entries
        if e.get("type") == "message" and e.get("role") == "assistant"
    ]
    final = next((t.strip() for t in reversed(texts) if t.strip()), "")
    if not final:
        return False

    print("[agent] no answer.md written; using the final assistant message")
    answer_path.write_text(final, encoding="utf-8")
    return True


def run_judge(task_dir: Path, workdir: Path, answer_path: Path, python: str,
              grade_process: bool = True) -> dict | None:
    """
    Grade the answer with the task's own standalone judge

    ``judge.py`` computes ``score = outcome_pct / 100``, so process grading
    costs a chunked pass over the whole trajectory (hundreds of KB) without
    moving the score. ``grade_process=False`` skips it, which is roughly 4x
    faster per task.
    """
    reward_path = workdir / "reward.json"
    trajectory = workdir / "vibe.json"
    use_trajectory = grade_process and trajectory.is_file()

    cmd = [
        python, str(task_dir / "tests" / "judge.py"),
        "--answer-file", str(answer_path),
        "--rubrics-file", str(task_dir / "tests" / "rubrics.json"),
        "--trajectory-file", str(trajectory) if use_trajectory else "",
        "--output", str(reward_path),
    ]

    print(f"[judge] {os.environ.get('JUDGE_MODEL')} via {os.environ.get('JUDGE_BASE_URL')}")
    env = dict(os.environ)
    env["JUDGE_RAW_DUMP_DIR"] = str(workdir)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if proc.stderr.strip():
        print("[judge] " + proc.stderr.strip()[:600])

    if not reward_path.is_file():
        print("[judge] no reward.json produced")
        return None

    return json.loads(reward_path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a DrugDiscoveryBench task locally, without Docker")
    ap.add_argument("task_id")
    ap.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--out-dir", type=Path, default=Path("local_runs"))
    ap.add_argument("--model", default=os.environ.get("VIBE_MODEL", "mistral-vibe-cli-latest"))
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "mistral-large-latest"))
    ap.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL", "https://api.mistral.ai/v1"))
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=3600, help="Agent wall-clock limit in seconds")
    ap.add_argument("--python", default=sys.executable, help="Interpreter for judge.py (needs `openai`)")
    ap.add_argument("--skip-agent", action="store_true", help="Re-grade an existing answer.md")
    ap.add_argument("--no-process", action="store_true",
                    help="Skip process-rubric grading (does not affect score; ~4x faster judge)")
    args = ap.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY") or os.environ.get("JUDGE_API_KEY")
    if not api_key:
        print("Set MISTRAL_API_KEY (or JUDGE_API_KEY).", file=sys.stderr)
        return 1

    task_dir = args.bench_dir / "benchmark" / "tasks" / args.task_id
    if not task_dir.is_dir():
        print(f"No such task: {task_dir}", file=sys.stderr)
        return 1

    rubrics = json.loads((task_dir / "tests" / "rubrics.json").read_text(encoding="utf-8"))
    if not rubrics.get("outcome_rubrics") and not rubrics.get("ground_truth"):
        print("Rubrics are empty -- run scripts/populate_rubrics.py first.", file=sys.stderr)
        return 1

    os.environ.setdefault("JUDGE_API_KEY", api_key)
    os.environ["JUDGE_BASE_URL"] = args.judge_base_url
    os.environ["JUDGE_MODEL"] = args.judge_model

    workdir = args.out_dir / args.task_id
    workdir.mkdir(parents=True, exist_ok=True)
    answer_path = workdir / "answer.md"

    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
    print(f"task    : {args.task_id}")
    print(f"workdir : {workdir.resolve()}")
    print(f"rubrics : {len(rubrics.get('outcome_rubrics', []))} outcome, "
          f"{len(rubrics.get('process_rubrics', []))} process")
    print()

    if not args.skip_agent:
        run_agent(instruction, workdir, args.model, args.max_turns, args.timeout)

    if not extract_answer_from_transcript(workdir, answer_path):
        print("[agent] produced no answer at all", file=sys.stderr)
        (workdir / "reward.json").write_text(
            json.dumps({"score": 0.0, "answer_file_present": 0}), encoding="utf-8"
        )
        return 1

    answer = answer_path.read_text(encoding="utf-8", errors="replace").strip()
    print(f"\n[answer] {len(answer)} chars")
    print("  " + answer[:300].replace("\n", "\n  "))
    print()

    reward = run_judge(task_dir, workdir, answer_path, args.python,
                       grade_process=not args.no_process)
    if reward is None:
        return 1

    print("\n=== reward.json ===")
    print(json.dumps(reward, indent=2)[:800])

    detail = workdir / "grades_detail.json"
    if detail.is_file():
        print(f"\nPer-criterion detail: {detail}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
