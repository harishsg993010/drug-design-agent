#!/usr/bin/env python3
"""
Render the DrugDiscoveryBench leaderboard comparison chart for the README

Reads our score from ``local_runs/summary.json`` so the chart cannot drift from
the run on disk, and writes ``assets/leaderboard.png``.

Usage::

    python integrations/plot_leaderboard.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# Published DrugDiscoveryBench results: (label, score, stderr)
LEADERBOARD = [
    ("GPT-5.5 (mini-SWE-agent) xhigh", 51.60, 4.30),
    ("Claude Sonnet 5 (mini-SWE-agent) max", 50.04, 0.70),
    ("Gemini 3.5 Flash (Gemini CLI) high", 50.00, 2.40),
    ("Gemini 3.5 Flash (mini-SWE-agent) high", 48.80, 2.10),
    ("Claude Opus 4.8 (mini-SWE-agent) max", 46.80, 1.40),
    ("Claude Opus 4.8 (Claude Code) max", 45.10, 4.38),
    ("GPT-5.5 (Codex) xhigh", 45.10, 4.90),
    ("Claude Sonnet 5.0 (Claude Code) max", 44.70, 1.90),
    ("Gemini 3.1 Pro (Gemini CLI) high", 41.90, 4.60),
    ("GLM 5.2 (mini-SWE-agent) xhigh", 36.20, 3.70),
    ("Kimi K2.7 Code (mini-SWE-agent) xhigh", 35.30, 2.10),
    ("DeepSeek V4 Pro (mini-SWE-agent) xhigh", 31.70, 3.20),
    ("Claude Sonnet 4.6 (mini-SWE-agent) max", 31.30, 0.70),
    ("GPT-5.2 (Codex) xhigh", 29.30, 3.20),
    ("Qwen 3.7 Max (mini-SWE-agent) xhigh", 29.30, 2.50),
    ("Claude Opus 4.6 (Claude Code) max", 27.70, 6.70),
    ("Claude Sonnet 4.6 (Claude Code) max", 24.00, 0.70),
    ("MiniMax M3 (mini-SWE-agent) xhigh", 22.80, 4.60),
]

OURS_LABEL = "Mistral Vibe + Drug Discovery MCP"

# Validated against the light chart surface: CVD dE 15.9, normal-vision 17.8,
# both fills >= 3:1 on surface. The rest-of-field fill is deliberately neutral
# so the highlighted bar carries the only hue.
SURFACE = "#fcfcfb"
ACCENT = "#2a78d6"
REST = "#898781"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
AXIS = "#c3c2b7"


def rounded_bar(ax, y, width, height, color, radius=0.34):
    """Draw a bar with rounded data-end, anchored flat at the baseline."""
    # A tiny bar cannot host the corner radius; clamp so it stays anchored.
    r = min(radius, max(width, 0.01) / 2)
    patch = FancyBboxPatch(
        (0, y - height / 2),
        max(width - r, 0.01),
        height,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=0,
        facecolor=color,
        mutation_aspect=0.42,
    )
    ax.add_patch(patch)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot the leaderboard comparison")
    ap.add_argument("--summary", type=Path, default=Path("local_runs/summary.json"))
    ap.add_argument("--out", type=Path, default=Path("assets/leaderboard.png"))
    ap.add_argument("--score", type=float, help="Override our score (percent)")
    args = ap.parse_args()

    if args.score is not None:
        ours, n_tasks = args.score, None
    else:
        if not args.summary.is_file():
            print(f"No summary at {args.summary}", file=sys.stderr)
            return 1
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        scores = [
            r["score"] for r in summary.get("results", {}).values()
            if isinstance(r.get("score"), (int, float))
        ]
        if not scores:
            print("No scores in summary", file=sys.stderr)
            return 1
        ours = 100 * sum(scores) / len(scores)
        n_tasks = len(scores)

    rows = sorted(
        LEADERBOARD + [(OURS_LABEL, ours, None)],
        key=lambda r: r[1],
    )

    fig, ax = plt.subplots(figsize=(9.6, 7.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, (label, score, err) in enumerate(rows):
        is_ours = label == OURS_LABEL
        rounded_bar(ax, i, score, 0.62, ACCENT if is_ours else REST)

        if err:
            ax.errorbar(
                score, i, xerr=err, fmt="none",
                ecolor=AXIS, elinewidth=1.4, capsize=3, capthick=1.4, zorder=3,
            )

        ax.text(
            score + (err or 0) + 1.1, i, f"{score:.1f}",
            va="center", ha="left", fontsize=8.5,
            color=INK if is_ours else INK_SECONDARY,
            fontweight="bold" if is_ours else "normal",
        )

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [r[0] for r in rows],
        fontsize=8.5,
    )
    for tick, (label, _, _) in zip(ax.get_yticklabels(), rows):
        if label == OURS_LABEL:
            tick.set_color(INK)
            tick.set_fontweight("bold")
        else:
            tick.set_color(INK_SECONDARY)

    ax.set_xlim(0, 62)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_xlabel("Mean outcome score", fontsize=9, color=MUTED, labelpad=8)

    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["left"].set_linewidth(1)

    subtitle = "DrugDiscoveryBench · 82 tasks, LLM judge against expert rubrics"
    if n_tasks:
        subtitle += f"\nOurs: {n_tasks} tasks graded, single run (no error bar)"

    # The subtitle occupies two 8.5pt lines above the axes, so the title has to
    # clear roughly 30pt or it lands on top of it.
    ax.set_title(
        "Mistral Vibe with the Drug Discovery MCP server",
        fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=44,
    )
    ax.text(
        0, 1.015, subtitle,
        transform=ax.transAxes, fontsize=8.5, color=MUTED,
        va="bottom", ha="left", linespacing=1.6,
    )

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out}  (ours = {ours:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
