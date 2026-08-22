"""
Tests for Server Module
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from drug_discovery_mcp.server import DrugDiscoveryMCPServer, MCPRequest, MCPResponse


class TestDrugDiscoveryMCPServer:
    """Tests for DrugDiscoveryMCPServer"""
    
    @pytest.fixture
    def server(self):
        # Create server without starting it
        return DrugDiscoveryMCPServer(host="0.0.0.0", port=8080)
    
    def test_initialization(self, server):
        """Test server initialization"""
        assert server is not None
        assert server.host == "0.0.0.0"
        assert server.port == 8080
        assert server.app is not None
    
    def test_register_all_tools(self, server):
        """Test that all tools are registered"""
        assert len(server.tools) > 0
        assert len(server.tool_info) > 0
        
        # Check for some expected tools
        expected_tools = [
            "query_uniprot",
            "query_chembl",
            "query_pdb",
            "calculate_descriptors",
            "smiles_to_inchi",
            "molecular_similarity",
            "superimpose_structures",
            "analyze_binding_site"
        ]
        
        for tool in expected_tools:
            assert tool in server.tools, f"Tool {tool} not found"
    
    def test_tool_info(self, server):
        """Test that tool info is available"""
        # Check a specific tool
        if "query_uniprot" in server.tool_info:
            info = server.tool_info["query_uniprot"]
            
            assert info.name == "query_uniprot"
            assert info.category is not None
            assert info.description is not None
            assert info.parameters is not None


class TestServerRoutes:
    """Tests for server API routes"""
    
    @pytest.fixture
    def client(self):
        # Create a test client
        server = DrugDiscoveryMCPServer(host="0.0.0.0", port=8080)
        return TestClient(server.app)
    
    def test_root_route(self, client):
        """Test root route"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "description" in data
    
    def test_health_route(self, client):
        """Test health check route"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_list_tools_route(self, client):
        """Test list tools route"""
        response = client.get("/tools")
        
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "count" in data
        assert len(data["tools"]) > 0
    
    def test_get_tool_info_route(self, client):
        """Test get tool info route"""
        response = client.get("/tools/query_uniprot")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "query_uniprot"
    
    def test_get_tool_info_not_found(self, client):
        """Test get tool info for non-existent tool"""
        response = client.get("/tools/nonexistent_tool")
        
        assert response.status_code == 404
    
    def test_list_categories_route(self, client):
        """Test list categories route"""
        response = client.get("/categories")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        # Should have some categories
        assert len(data) > 0


class TestMCPProtocol:
    """Tests for MCP protocol handling"""
    
    @pytest.fixture
    def client(self):
        server = DrugDiscoveryMCPServer(host="0.0.0.0", port=8080)
        return TestClient(server.app)
    
    def test_mcp_endpoint(self, client):
        """Test MCP protocol endpoint"""
        # Test with a simple request
        request_data = {
            "method": "query_uniprot",
            "params": {"accession": "P12345"},
            "id": "test_request_001"
        }
        
        response = client.post("/mcp", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "success" in data
    
    def test_call_tool_endpoint(self, client):
        """Test call tool endpoint"""
        response = client.post("/call/query_uniprot", json={"params": {"accession": "P12345"}})
        
        assert response.status_code == 200
        data = response.json()
        assert "result" in data or "error" in data
        assert "tool" in data
    
    def test_batch_endpoint(self, client):
        """Test batch call endpoint"""
        calls = [
            {"tool": "query_uniprot", "params": {"accession": "P12345"}},
            {"tool": "query_chembl", "params": {"compound_id": "CHEMBL123"}}
        ]
        
        response = client.post("/batch", json={"calls": calls})
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2


class TestMCPRequestResponse:
    """Tests for MCP request/response models"""
    
    def test_mcp_request(self):
        """Test MCPRequest model"""
        request = MCPRequest(
            method="query_uniprot",
            params={"accession": "P12345"},
            id="test_001"
        )
        
        assert request.method == "query_uniprot"
        assert request.params == {"accession": "P12345"}
        assert request.id == "test_001"
    
    def test_mcp_response(self):
        """Test MCPResponse model"""
        response = MCPResponse(
            id="test_001",
            result={"accession": "P12345"},
            success=True
        )
        
        assert response.id == "test_001"
        assert response.success is True
        assert response.result == {"accession": "P12345"}


class TestServerInitialization:
    """Tests for server initialization"""
    
    def test_default_initialization(self):
        """Test server with default parameters"""
        server = DrugDiscoveryMCPServer()
        
        assert server is not None
        assert server.host == "0.0.0.0"
        assert server.port == 8080
    
    def test_custom_initialization(self):
        """Test server with custom parameters"""
        server = DrugDiscoveryMCPServer(host="localhost", port=9000)
        
        assert server.host == "localhost"
        assert server.port == 9000


class TestServerComponents:
    """Tests for server component initialization"""
    
    @pytest.fixture
    def server(self):
        return DrugDiscoveryMCPServer()
    
    def test_database_tools_initialized(self, server):
        """Test that database tools are initialized"""
        assert server.db_tools is not None
    
    def test_cheminformatics_tools_initialized(self, server):
        """Test that cheminformatics tools are initialized"""
        assert server.chem_tools is not None
    
    def test_structural_biology_tools_initialized(self, server):
        """Test that structural biology tools are initialized"""
        assert server.struct_tools is not None
