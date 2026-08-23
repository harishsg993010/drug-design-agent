"""
Tests for the Mistral Vibe Harbor agent adapter

Harbor requires Python 3.12+, so these skip on older interpreters (this
project targets 3.10+). Run them under a 3.12+ environment with Harbor
installed:

    uv tool run --with pytest --from harbor==0.13.1 pytest integrations/

It lives beside the adapter rather than in `tests/`, because `tests/__init__.py`
star-imports the whole `drug_discovery_mcp` package, whose dependencies are not
installed in Harbor's interpreter.

The Vibe entry fixtures below are trimmed from real `vibe -p --output json`
output, not invented.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("harbor", reason="Harbor requires Python 3.12+")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vibe_agent import VibeAgent  # noqa: E402


# A complete Vibe session: prompt -> reasoning -> bash tool call -> answer.
VIBE_SESSION = [
    {
        "type": "message",
        "role": "user",
        "createdAt": 1787429000000,
        "content": [{"type": "text", "text": "Run: echo benchmark-probe"}],
    },
    {
        "type": "reasoning",
        "createdAt": 1787429001000,
        "text": "The user wants me to run a shell command.",
        "summary": "",
    },
    {
        "type": "effect",
        "createdAt": 1787429002000,
        "id": "ldVoM8ptu",
        "title": "bash",
        "detail": {
            "toolName": "bash",
            "kind": "shell",
            "input": {"command": "echo benchmark-probe"},
        },
        "state": {
            "status": "completed",
            "output": {"stdout": "benchmark-probe\n", "stderr": "", "output": ""},
            "outputText": "benchmark-probe\n",
            "durationMs": 526.7,
        },
    },
    {
        "type": "message",
        "role": "assistant",
        "createdAt": 1787429003000,
        "content": [{"type": "text", "text": "benchmark-probe"}],
    },
]


def make_agent(**attrs):
    """Build an adapter without running Harbor's __init__ machinery."""
    agent = VibeAgent.__new__(VibeAgent)
    agent.model_name = attrs.get("model_name", "mistral-large-latest")
    agent._version = attrs.get("version", "2.24.3")
    agent.mcp_servers = attrs.get("mcp_servers", [])
    return agent


class TestIdentity:
    def test_name_is_not_a_harbor_builtin(self):
        """The adapter is loaded by import path, so it needs its own name"""
        assert VibeAgent.name() == "vibe"

    def test_declares_atif_support(self):
        assert VibeAgent.SUPPORTS_ATIF is True


class TestTrajectoryConversion:
    def test_user_turn_becomes_first_step(self):
        traj = make_agent()._convert_entries_to_trajectory(VIBE_SESSION)

        assert traj.steps[0].source == "user"
        assert traj.steps[0].message == "Run: echo benchmark-probe"

    def test_turn_is_folded_into_one_agent_step(self):
        """Reasoning, tool call and answer are separate Vibe entries, one ATIF step"""
        traj = make_agent()._convert_entries_to_trajectory(VIBE_SESSION)

        assert len(traj.steps) == 2
        step = traj.steps[1]
        assert step.source == "agent"
        assert step.message == "benchmark-probe"
        assert "shell command" in step.reasoning_content

    def test_tool_call_and_result_are_linked(self):
        traj = make_agent()._convert_entries_to_trajectory(VIBE_SESSION)
        step = traj.steps[1]

        assert len(step.tool_calls) == 1
        call = step.tool_calls[0]
        assert call.function_name == "bash"
        assert call.arguments == {"command": "echo benchmark-probe"}

        result = step.observation.results[0]
        assert result.source_call_id == call.tool_call_id
        assert result.content == "benchmark-probe"

    def test_agent_metadata_is_recorded(self):
        traj = make_agent()._convert_entries_to_trajectory(VIBE_SESSION)

        assert traj.agent.name == "vibe"
        assert traj.agent.version == "2.24.3"
        assert traj.agent.model_name == "mistral-large-latest"
        assert traj.final_metrics.total_steps == 2

    def test_serialises_to_atif(self):
        traj = make_agent()._convert_entries_to_trajectory(VIBE_SESSION)

        payload = traj.to_json_dict()
        assert payload["agent"]["name"] == "vibe"
        assert len(payload["steps"]) == 2

    def test_empty_session_does_not_crash(self):
        """
        A run killed before producing output still yields a valid trajectory

        Harbor's Trajectory requires at least one step, so the adapter records
        the empty run rather than raising and hiding the underlying failure.
        """
        traj = make_agent()._convert_entries_to_trajectory([])

        assert len(traj.steps) == 1
        assert "no session output" in traj.steps[0].message
        assert traj.to_json_dict()["agent"]["name"] == "vibe"

    def test_trailing_tool_call_without_answer_is_kept(self):
        """A session cut off at the turn limit still records its work"""
        truncated = VIBE_SESSION[:-1]  # drop the assistant message

        traj = make_agent()._convert_entries_to_trajectory(truncated)

        assert len(traj.steps) == 2
        assert traj.steps[1].tool_calls[0].function_name == "bash"

    def test_falls_back_to_stdout_when_output_text_missing(self):
        entries = [dict(e) for e in VIBE_SESSION]
        entries[2] = dict(entries[2])
        entries[2]["state"] = {
            "status": "completed",
            "output": {"stdout": "from-stdout", "stderr": "", "output": ""},
        }

        traj = make_agent()._convert_entries_to_trajectory(entries)

        assert traj.steps[1].observation.results[0].content == "from-stdout"


class TestMCPRegistration:
    """The MCP servers Harbor passes in must reach Vibe's own CLI"""

    def test_stdio_server_becomes_vibe_mcp_add(self):
        from harbor.models.task.config import MCPServerConfig

        agent = make_agent(mcp_servers=[
            MCPServerConfig(
                name="drug-discovery",
                transport="stdio",
                command="drug-discovery-mcp",
                args=["--transport", "stdio"],
            )
        ])

        commands = agent._build_register_mcp_commands()

        assert len(commands) == 1
        assert "vibe mcp add drug-discovery" in commands[0]
        assert "--transport stdio" in commands[0]
        assert "--command drug-discovery-mcp" in commands[0]
        assert commands[0].count("--arg") == 2

    def test_http_server_uses_url(self):
        from harbor.models.task.config import MCPServerConfig

        agent = make_agent(mcp_servers=[
            MCPServerConfig(
                name="drug-discovery",
                transport="streamable-http",
                url="http://host.docker.internal:8080/mcp",
            )
        ])

        command = agent._build_register_mcp_commands()[0]

        assert "--url http://host.docker.internal:8080/mcp" in command
        assert "--command" not in command

    def test_no_servers_means_no_commands(self):
        assert make_agent()._build_register_mcp_commands() == []
