# Benchmark integrations

Running this project's tools inside
[DrugDiscoveryBench](https://github.com/scaleapi/DrugDiscoveryBench), and running
[Mistral Vibe](https://github.com/mistralai/mistral-vibe) as the agent under
evaluation.

| File | Purpose |
|---|---|
| `vibe_agent.py` | Harbor agent adapter for Mistral Vibe |
| `add_mcp_to_tasks.py` | Stamps this project's MCP server into task configs |
| `test_vibe_agent.py` | Tests for the adapter's trajectory conversion |

## Why these live outside the package

Harbor requires **Python 3.12+** and loads the agent adapter inside *its own*
interpreter, where `drug_discovery_mcp` and its dependencies are not installed.
`vibe_agent.py` therefore imports only Harbor — never this project. The MCP
server reaches the agent through task *configuration*, not through an import.

The tests sit here rather than in `tests/` because `tests/__init__.py`
star-imports the whole package, which Harbor's interpreter cannot satisfy.

## Registering the MCP server with a benchmark task

Harbor has first-class MCP support: it reads `[[environment.mcp_servers]]` from
a task's `task.toml` and each agent adapter registers those servers with its CLI
before the trajectory starts.

```bash
# HTTP transport — server runs on the host, container reaches it via
# host.docker.internal. Nothing to rebuild.
python integrations/add_mcp_to_tasks.py /path/to/DrugDiscoveryBench

# then, on the host:
drug-discovery-mcp --transport streamable-http --host 0.0.0.0 --port 8080
```

The benchmark's egress proxy is a *denylist* of Scale hosts, so host-bound
traffic is tunnelled rather than blocked, and the task healthcheck (which only
asserts `scale.com` → 403) is unaffected.

Use `--transport stdio` instead if you have built a derived trial image that
installs this project:

```dockerfile
FROM ghcr.io/scaleapi/drugdiscoverybench:1.0.0-lightweight
RUN pip install drug-discovery-mcp
```

`--revert` removes the block byte-for-byte; `--dry-run` previews. Re-running
with a different transport replaces the block rather than stacking duplicates.

## Running Mistral Vibe as the agent

Harbor ships adapters for claude-code, codex and gemini-cli, but not Vibe.
`vibe_agent.py` supplies one, wired in through `--agent-import-path` (which
bypasses Harbor's `AgentName` enum):

```bash
PYTHONPATH=/path/to/drug-design-agent/integrations \
harbor run \
    -p benchmark/tasks/<task_id> \
    -m mistral-large-latest \
    -a vibe \
    --agent-import-path vibe_agent:VibeAgent \
    -e docker \
    --ae "MISTRAL_API_KEY=$MISTRAL_API_KEY" \
    -k 1 -n 1 --yes
```

The adapter:

- installs Vibe with `uv tool install mistral-vibe` inside the container, and
  reuses an existing CLI if the image already has one. Vibe is a Python package,
  so this avoids the npm/nvm install paths that the benchmark's own
  `scripts/harbor_agents.py` documents as being SIGKILLed inside the image;
- registers every MCP server Harbor passes it via `vibe mcp add`;
- runs `vibe -p <instruction> --output json` and converts the session into
  Harbor's ATIF trajectory format, so the judge and the trajectory viewer both
  work.

**Credentials.** Vibe's interactive login lives under `$VIBE_HOME`, which does
not exist in a fresh container, so a non-interactive run needs a key in the
environment. The adapter forwards `MISTRAL_API_KEY`, `VIBE_API_KEY`,
`VIBE_BASE_URL` and `VIBE_ACTIVE_MODEL` from the host.

**Turn budget.** The adapter defaults to `max_turns=60`. Vibe's own default is
low enough that a tool round trip can be cut off before the result returns,
which surfaces as an empty answer rather than an error.

## Running the adapter tests

They need Harbor, so run them in Harbor's interpreter:

```bash
uv tool run --with pytest --from harbor==0.13.1 pytest integrations/
```

They are excluded from the main suite (`testpaths = ["tests"]`), which runs on
Python 3.10+ where Harbor cannot be installed.

## Notes on the benchmark clone

`scripts/populate_rubrics.py` writes answers into 82 **tracked** `rubrics.json`
files. They are gated content — do not commit that clone.

On Windows the script fails with `UnicodeDecodeError` because it calls
`read_text()` without an encoding; prefix the command with `PYTHONUTF8=1`.
