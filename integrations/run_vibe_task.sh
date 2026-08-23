#!/usr/bin/env bash
# Run one DrugDiscoveryBench task with Mistral Vibe as the agent.
#
#   MISTRAL_API_KEY=...  ./integrations/run_vibe_task.sh <task_id> [bench_dir]
#
# Both roles run on Mistral: Vibe drives the trajectory, and the judge uses
# Mistral's OpenAI-compatible endpoint. Set VIBE_MODEL / JUDGE_MODEL to override.
set -euo pipefail

TASK_ID="${1:?usage: run_vibe_task.sh <task_id> [bench_dir]}"
BENCH_DIR="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../DrugDiscoveryBench" && pwd)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${MISTRAL_API_KEY:?set MISTRAL_API_KEY}"

# The model Vibe drives. Vibe has no --model flag; it reads config, which
# VIBE_ACTIVE_MODEL overrides.
VIBE_MODEL="${VIBE_MODEL:-mistral-vibe-cli-latest}"

# The judge grades against the rubrics. Any OpenAI-compatible endpoint works.
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.mistral.ai/v1}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-$MISTRAL_API_KEY}"
export JUDGE_MODEL="${JUDGE_MODEL:-mistral-large-latest}"

TASK_PATH="$BENCH_DIR/benchmark/tasks/$TASK_ID"
[ -d "$TASK_PATH" ] || { echo "No such task: $TASK_PATH" >&2; exit 1; }

echo "task   : $TASK_ID"
echo "agent  : vibe ($VIBE_MODEL)"
echo "judge  : $JUDGE_MODEL via $JUDGE_BASE_URL"
echo

cd "$BENCH_DIR"
PYTHONPATH="$HERE" harbor run \
    -p "$TASK_PATH" \
    -m "$VIBE_MODEL" \
    -a vibe \
    --agent-import-path vibe_agent:VibeAgent \
    -e docker \
    --ae "MISTRAL_API_KEY=$MISTRAL_API_KEY" \
    --ae "VIBE_ACTIVE_MODEL=$VIBE_MODEL" \
    --environment-build-timeout-multiplier 3.0 \
    -k 1 -n 1 --yes

echo
echo "Results under $BENCH_DIR/jobs/<timestamp>/$TASK_ID/"
echo "  verifier/reward.json          score 0-100"
echo "  verifier/grades_detail.json   per-criterion breakdown"
echo "  agent/                        Vibe transcript + trajectory.json"
