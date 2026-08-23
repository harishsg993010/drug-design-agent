#!/usr/bin/env python3
"""
Register this project's MCP server with DrugDiscoveryBench tasks

Harbor reads MCP servers from a task's ``[[environment.mcp_servers]]`` and hands
them to the agent adapter, which registers them with the agent CLI before the
trajectory starts. This script stamps that block into task.toml files so the
benchmark's agents can call the drug discovery tools alongside the image's
BiOMNI suite.

Two transports, with different trade-offs:

``http`` (default)
    The MCP server runs on the *host*; the trial container reaches it at
    ``host.docker.internal``. Nothing to rebuild. The benchmark's egress proxy
    is a denylist of Scale hosts, so this is not blocked.

``stdio``
    The agent launches the server *inside* the container, which means this
    project must be installed in the trial image -- build a derived image
    ``FROM ghcr.io/scaleapi/drugdiscoverybench:1.0.0-lightweight`` that pip
    installs it, then point tasks at your tag.

Usage::

    python integrations/add_mcp_to_tasks.py /path/to/DrugDiscoveryBench
    python integrations/add_mcp_to_tasks.py /path/to/DrugDiscoveryBench --transport stdio
    python integrations/add_mcp_to_tasks.py /path/to/DrugDiscoveryBench --revert
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MARKER = "# --- drug-discovery MCP server (added by add_mcp_to_tasks.py) ---"
END_MARKER = "# --- end drug-discovery MCP server ---"

BLOCK_TEMPLATE = """
{marker}
[[environment.mcp_servers]]
name = "drug-discovery"
{body}
{end_marker}
"""

# Matches the block above, so --revert and re-runs are clean
BLOCK_RE = re.compile(
    re.escape(MARKER) + r".*?" + re.escape(END_MARKER) + r"\n?",
    re.DOTALL,
)


def build_block(transport: str, url: str, command: str) -> str:
    if transport == "stdio":
        body = f'transport = "stdio"\ncommand = "{command}"'
    else:
        body = f'transport = "{transport}"\nurl = "{url}"'

    return BLOCK_TEMPLATE.format(marker=MARKER, body=body, end_marker=END_MARKER)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("bench_dir", type=Path, help="Path to the DrugDiscoveryBench clone")
    parser.add_argument(
        "--transport",
        choices=["http", "streamable-http", "stdio"],
        default="http",
        help="MCP transport (default: http, served from the host)",
    )
    parser.add_argument(
        "--url",
        default="http://host.docker.internal:8080/mcp",
        help="Server URL for the http transports",
    )
    parser.add_argument(
        "--command",
        default="drug-discovery-mcp",
        help="Executable for the stdio transport (must exist in the trial image)",
    )
    parser.add_argument("--task", help="Only patch this task id (default: all tasks)")
    parser.add_argument("--revert", action="store_true", help="Remove a previously added block")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = parser.parse_args()

    tasks_dir = args.bench_dir / "benchmark" / "tasks"
    if not tasks_dir.is_dir():
        print(f"No tasks directory at {tasks_dir}", file=sys.stderr)
        return 1

    pattern = f"{args.task}/task.toml" if args.task else "*/task.toml"
    task_files = sorted(tasks_dir.glob(pattern))
    if not task_files:
        print(f"No task.toml matched {pattern}", file=sys.stderr)
        return 1

    block = build_block(args.transport, args.url, args.command)
    changed = 0

    for path in task_files:
        original = path.read_text(encoding="utf-8")
        # Strip any previous block first, so re-running switches transport
        # cleanly instead of stacking duplicates.
        # Normalising the trailing newline on both paths keeps --revert
        # byte-identical to the original file.
        stripped = BLOCK_RE.sub("", original).rstrip("\n") + "\n"
        updated = stripped if args.revert else stripped + block

        if updated != original:
            changed += 1
            if not args.dry_run:
                path.write_text(updated, encoding="utf-8")

    verb = "Would update" if args.dry_run else ("Reverted" if args.revert else "Updated")
    print(f"{verb} {changed} of {len(task_files)} task(s)")

    if not args.revert and not args.dry_run and args.transport != "stdio":
        print(f"\nStart the server on the host so trials can reach {args.url}:")
        print("  drug-discovery-mcp --transport streamable-http --host 0.0.0.0 --port 8080")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
