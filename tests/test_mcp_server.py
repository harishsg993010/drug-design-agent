"""
Tests for the MCP server

These exercise the server through the official SDK's own surface -- tool
discovery, schema generation and dispatch -- rather than calling the underlying
tool functions directly.
"""

import json

import pytest
from unittest.mock import patch

from drug_discovery_mcp.mcp_server import build_server


EXPECTED_TOOLS = {
    # databases
    "query_uniprot", "query_chembl", "query_pdb", "query_opentargets",
    "query_kegg", "query_pubchem", "query_ncbi",
    "search_compounds", "search_proteins", "search_patents",
    # cheminformatics
    "calculate_descriptors", "smiles_to_inchi", "inchi_to_smiles",
    "molecular_similarity", "calculate_fingerprint", "predict_admet",
    "check_drug_likeness", "generate_conformers", "optimize_geometry",
    "calculate_charge",
    # structural biology
    "superimpose_structures", "analyze_binding_site", "download_pdb",
    "parse_pdb", "calculate_rmsd", "find_interactions",
    "analyze_conformation", "compare_structures", "extract_ligand",
    "analyze_solvent_accessibility",
}


@pytest.fixture(scope="module")
def server():
    return build_server()


class TestServerMetadata:
    """Tests for server identity"""

    def test_name_and_version(self, server):
        """The server identifies itself to clients"""
        assert server.name == "drug-discovery"
        assert server.version

    def test_instructions_are_provided(self, server):
        """Clients get guidance on what the toolset covers"""
        assert server.instructions
        assert "drug discovery" in server.instructions.lower()


class TestToolDiscovery:
    """Tests for tool registration and schema generation"""

    async def test_all_tools_registered(self, server):
        """Every tool in the catalogue is exposed over MCP"""
        names = {tool.name for tool in await server.list_tools()}

        assert names == EXPECTED_TOOLS

    async def test_every_tool_is_described(self, server):
        """A tool with no description is unusable by a model"""
        for tool in await server.list_tools():
            assert tool.description, f"{tool.name} has no description"

    async def test_schema_marks_required_arguments(self, server):
        """Required parameters are those without defaults"""
        tools = {tool.name: tool for tool in await server.list_tools()}

        schema = tools["molecular_similarity"].input_schema
        assert set(schema["required"]) == {"smiles1", "smiles2"}
        # optional parameters still appear, so a client can set them
        assert "metric" in schema["properties"]
        assert "fingerprint_type" in schema["properties"]

    async def test_schema_types_come_from_annotations(self, server):
        """Parameter types are derived from the function signature"""
        tools = {tool.name: tool for tool in await server.list_tools()}

        schema = tools["calculate_charge"].input_schema
        assert schema["properties"]["smiles"]["type"] == "string"
        assert schema["properties"]["ph"]["type"] == "number"

    async def test_parameters_are_documented(self, server):
        """The docstring reaches the client, so arguments are explained"""
        tools = {tool.name: tool for tool in await server.list_tools()}

        description = tools["query_uniprot"].description
        assert "accession" in description
        assert "P04637" in description


class TestToolDispatch:
    """Tests for calling tools through the MCP layer"""

    async def test_call_tool_returns_result(self, server):
        """A successful call comes back as tool content"""
        with patch.object(
            server._tool_manager.get_tool("query_uniprot"), "fn",
            return_value={"accession": "P04637", "entry_name": "P53_HUMAN"},
        ):
            result = await server.call_tool("query_uniprot", {"accession": "P04637"})

        assert result.is_error is False
        assert json.loads(result.content[0].text)["entry_name"] == "P53_HUMAN"

    async def test_arguments_are_validated(self, server):
        """A missing required argument is rejected before the tool runs"""
        from mcp.server.mcpserver.exceptions import ToolError

        with pytest.raises(ToolError, match="accession"):
            await server.call_tool("query_uniprot", {})

    async def test_unknown_tool_is_rejected(self, server):
        """Calling a tool that was never registered fails"""
        from mcp.server.mcpserver.exceptions import ToolError

        with pytest.raises(ToolError):
            await server.call_tool("no_such_tool", {})

    async def test_defaults_are_applied(self, server):
        """Omitted optional arguments fall back to the declared default"""
        captured = {}

        def fake(smiles, ph=7.0):
            captured["ph"] = ph
            return {"smiles": smiles, "total_charge": 0}

        with patch.object(server._tool_manager.get_tool("calculate_charge"), "fn", fake):
            await server.call_tool("calculate_charge", {"smiles": "CCO"})

        assert captured["ph"] == 7.0


class TestToolsAreSynchronous:
    """
    The SDK runs sync tools in a worker thread

    The tool functions block on network and RDKit calls, so they must stay
    synchronous rather than being declared async and stalling the event loop.
    """

    async def test_tools_are_not_coroutines(self, server):
        for tool in await server.list_tools():
            registered = server._tool_manager.get_tool(tool.name)
            assert registered.is_async is False, f"{tool.name} is async"
