"""
Harbor agent adapter for Mistral Vibe

Lets `DrugDiscoveryBench <https://github.com/scaleapi/DrugDiscoveryBench>`_ (and
any other Harbor benchmark) run `Mistral Vibe
<https://github.com/mistralai/mistral-vibe>`_ as the agent under evaluation,
with this project's MCP tools registered inside the trial container.

Harbor ships adapters for claude-code, codex and gemini-cli but not Vibe. Rather
than fork Harbor, this subclasses ``BaseInstalledAgent`` and is wired in through
``--agent-import-path``, which bypasses the ``AgentName`` enum::

    PYTHONPATH=/path/to/drug-design-agent/integrations harbor run \\
        -p benchmark/tasks/<task_id> \\
        -m mistral-large-latest \\
        -a vibe \\
        --agent-import-path vibe_agent:VibeAgent \\
        -e docker

This module is deliberately standalone -- it imports only Harbor, never
``drug_discovery_mcp``. Harbor loads it inside its own interpreter (which
requires Python 3.12+), where this project's dependencies are absent. The MCP
server reaches the agent through task configuration, not through an import.

Vibe is a Python package, so installing it in the trial container is a plain
``uv tool install`` -- it avoids the npm/nvm install paths that the benchmark's
own ``scripts/harbor_agents.py`` documents as being SIGKILLed inside the image.

Any MCP servers Harbor passes in (``[[environment.mcp_servers]]`` in a task's
``task.toml``) are registered with ``vibe mcp add`` before the run, so the agent
can call this project's drug discovery tools alongside the image's BiOMNI suite.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
from typing import Any, Dict, List, Optional

from harbor.agents.installed.base import (
    BaseInstalledAgent,
    NonZeroAgentExitCodeError,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.utils.trajectory_utils import format_trajectory_json

logger = logging.getLogger(__name__)

# Credentials forwarded from the host into the trial container. Vibe stores an
# interactive login under $VIBE_HOME, which does not exist in a fresh
# container, so a non-interactive run needs a key in the environment.
_CREDENTIAL_ENV_VARS = (
    "MISTRAL_API_KEY",
    "VIBE_API_KEY",
    "VIBE_BASE_URL",
    "VIBE_ACTIVE_MODEL",
)


class VibeAgent(BaseInstalledAgent):
    """
    Runs Mistral Vibe in programmatic mode and converts its transcript to ATIF

    Vibe's ``-p/--prompt`` mode emits a JSON array of session entries. Three
    entry types matter here:

    ``message``    a user or assistant turn, with ``content`` text parts
    ``reasoning``  the model's thinking for the turn
    ``effect``     a tool call -- ``detail`` holds the tool name and input,
                   ``state`` holds the result
    """

    SUPPORTS_ATIF: bool = True

    _OUTPUT_FILENAME = "vibe.json"
    _CONTAINER_LOG = "/logs/agent/vibe.json"

    def __init__(self, *args, max_turns: int = 60, vibe_agent: str = "auto-approve", **kwargs):
        """
        Args:
            max_turns: Turn ceiling for the session. Vibe's default is low
                enough that a tool round trip can be cut off before the result
                comes back, which shows up as an empty answer.
            vibe_agent: Which Vibe agent profile to run under.
        """
        super().__init__(*args, **kwargs)
        self._max_turns = max_turns
        self._vibe_agent = vibe_agent
        self._instruction: Optional[str] = None

    @staticmethod
    def name() -> str:
        return "vibe"

    def get_version_command(self) -> str | None:
        return "vibe --version"

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    async def install(self, environment: BaseEnvironment) -> None:
        """Install the Vibe CLI, reusing it if the image already ships one"""
        probe = await environment.exec(command="command -v vibe")
        if probe.return_code == 0 and (probe.stdout or "").strip():
            logger.info("Reusing the Vibe CLI already present in the image")
            return

        version_spec = f"=={self._version}" if self._version else ""
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                # uv is the documented installer for Vibe and needs no Node
                "command -v uv >/dev/null 2>&1 || "
                "  curl -LsSf https://astral.sh/uv/install.sh | sh; "
                'export PATH="$HOME/.local/bin:$PATH"; '
                f"uv tool install 'mistral-vibe{version_spec}'; "
                "vibe --version"
            ),
        )

    def _build_register_mcp_commands(self) -> List[str]:
        """
        Return `vibe mcp add` commands for every MCP server Harbor passed in

        Harbor hands us ``MCPServerConfig`` objects from the task's
        ``[[environment.mcp_servers]]``; Vibe takes the same information through
        its own CLI rather than a config file we would have to hand-write.
        """
        commands: List[str] = []

        for server in self.mcp_servers:
            argv = ["vibe", "mcp", "add", server.name, "--transport", server.transport]

            if server.transport == "stdio":
                argv += ["--command", server.command or ""]
                for arg in server.args:
                    argv += ["--arg", arg]
            else:
                argv += ["--url", server.url or ""]

            commands.append(shlex.join(argv))

        return commands

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._instruction = instruction

        env = {k: os.environ[k] for k in _CREDENTIAL_ENV_VARS if k in os.environ}
        if self.model_name:
            env["VIBE_ACTIVE_MODEL"] = self.model_name

        for command in self._build_register_mcp_commands():
            await self.exec_as_agent(environment, command=command, env=env)

        argv = [
            "vibe",
            "-p",
            instruction,
            "--output",
            "json",
            "--agent",
            self._vibe_agent,
            "--auto-approve",
            "--trust",
            "--max-turns",
            str(self._max_turns),
        ]
        # Vibe has no --model flag: the model is selected through config, which
        # VIBE_ACTIVE_MODEL overrides (set from self.model_name in `env` above).

        cli_flags = self.build_cli_flags()
        flags = f" {cli_flags}" if cli_flags else ""

        result = await self.exec_as_agent(
            environment,
            command=(
                'export PATH="$HOME/.local/bin:$PATH"; '
                f"{shlex.join(argv)}{flags} "
                f"</dev/null > {self._CONTAINER_LOG} 2>/logs/agent/vibe.stderr"
            ),
            env=env,
        )

        if result is not None and getattr(result, "return_code", 0) not in (0, None):
            raise NonZeroAgentExitCodeError(
                f"vibe exited {result.return_code}: "
                f"{self._truncate_output(getattr(result, 'stderr', None))}"
            )

        self._write_trajectory()

    # ------------------------------------------------------------------
    # trajectory
    # ------------------------------------------------------------------

    def _read_entries(self) -> List[Dict[str, Any]]:
        """Load the session entries Vibe wrote, tolerating a truncated file"""
        path = self.logs_dir / self._OUTPUT_FILENAME
        if not path.exists():
            logger.warning("Vibe produced no output at %s", path)
            return []

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            # A session killed at the turn/price ceiling can leave a partial array
            logger.warning("Could not parse Vibe output: %s", e)
            return []

        return data if isinstance(data, list) else []

    @staticmethod
    def _text_of(entry: Dict[str, Any]) -> str:
        """Join the text parts of a message entry"""
        return "".join(
            part.get("text", "")
            for part in entry.get("content") or []
            if part.get("type") == "text"
        ).strip()

    def _convert_entries_to_trajectory(self, entries: List[Dict[str, Any]]) -> Trajectory:
        """
        Fold Vibe's session entries into ATIF steps

        Vibe emits reasoning, tool calls and the assistant message as separate
        entries within one turn, so they are accumulated and flushed together
        whenever an assistant message closes the turn.
        """
        steps: List[Step] = []
        reasoning: List[str] = []
        tool_calls: List[ToolCall] = []
        results: List[ObservationResult] = []

        def flush(message: str, timestamp: Optional[str]) -> None:
            steps.append(
                Step(
                    step_id=str(len(steps) + 1),
                    source="agent",
                    message=message,
                    timestamp=timestamp,
                    model_name=self.model_name,
                    reasoning_content="\n".join(reasoning) or None,
                    tool_calls=list(tool_calls) or None,
                    observation=Observation(results=list(results)) if results else None,
                )
            )
            reasoning.clear()
            tool_calls.clear()
            results.clear()

        for entry in entries:
            kind = entry.get("type")
            timestamp = self._iso(entry.get("createdAt"))

            if kind == "message" and entry.get("role") == "user":
                steps.append(
                    Step(
                        step_id=str(len(steps) + 1),
                        source="user",
                        message=self._text_of(entry),
                        timestamp=timestamp,
                    )
                )

            elif kind == "reasoning":
                text = (entry.get("text") or entry.get("summary") or "").strip()
                if text:
                    reasoning.append(text)

            elif kind == "effect":
                call_id = str(entry.get("id", len(tool_calls)))
                detail = entry.get("detail") or {}
                state = entry.get("state") or {}

                tool_calls.append(
                    ToolCall(
                        tool_call_id=call_id,
                        function_name=detail.get("toolName") or entry.get("title") or "tool",
                        arguments=detail.get("input") or {},
                    )
                )
                results.append(
                    ObservationResult(
                        source_call_id=call_id,
                        content=self._effect_output(state),
                    )
                )

            elif kind == "message" and entry.get("role") == "assistant":
                flush(self._text_of(entry), timestamp)

        # A run cut off mid-turn still has reasoning/tool calls worth keeping
        if reasoning or tool_calls:
            flush("", None)

        # Harbor rejects a trajectory with no steps, so a session that produced
        # nothing (killed at startup, missing credentials) is recorded as such
        # rather than raising here and masking the real failure.
        if not steps:
            steps.append(
                Step(
                    step_id="1",
                    source="agent",
                    message="Vibe produced no session output.",
                )
            )

        return Trajectory(
            agent=Agent(
                # version() is a method on BaseInstalledAgent, not a property
                name=self.name(),
                version=self.version() or "unknown",
                model_name=self.model_name,
            ),
            steps=steps,
            final_metrics=FinalMetrics(total_steps=len(steps)),
        )

    @staticmethod
    def _effect_output(state: Dict[str, Any]) -> str:
        """Pull the readable result out of a tool effect's state"""
        if text := (state.get("outputText") or "").strip():
            return text

        output = state.get("output")
        if isinstance(output, dict):
            joined = "".join(
                str(output.get(key) or "") for key in ("stdout", "stderr", "output")
            ).strip()
            if joined:
                return joined
        elif isinstance(output, str):
            return output

        return str(state.get("status", ""))

    @staticmethod
    def _iso(millis: Any) -> Optional[str]:
        """Convert Vibe's epoch-millisecond timestamps to ISO-8601"""
        if not isinstance(millis, (int, float)):
            return None
        from datetime import datetime, timezone

        return datetime.fromtimestamp(millis / 1000, tz=timezone.utc).isoformat()

    def _write_trajectory(self) -> None:
        """Write trajectory.json next to the raw Vibe output"""
        trajectory = self._convert_entries_to_trajectory(self._read_entries())
        path = self.logs_dir / "trajectory.json"
        path.write_text(
            format_trajectory_json(trajectory.to_json_dict()),
            encoding="utf-8",
        )
