"""
Tests for Database Modules
"""

import pytest
from unittest.mock import patch, MagicMock
from drug_discovery_mcp.databases import (
    UniProtClient,
    ChEMBLClient,
    PDBClient,
    OpenTargetsClient,
    DatabaseTools,
    DatabaseError
)


class TestUniProtClient:
    """Tests for UniProtClient"""
    
    @pytest.fixture
    def client(self):
        return UniProtClient()
    
    def test_initialization(self, client):
        """Test client initialization"""
        assert client is not None
        assert client.get_name() == "UniProt"
    
    def test_query(self, client):
        """Test querying UniProt"""
        # This test would normally make a real API call
        # For testing, we'll mock the request
        with patch.object(client, '_make_request') as mock_request:
            mock_data = {
                "accession": "P12345",
                "entryName": "TEST_HUMAN",
                "protein": {
                    "recommendedName": {
                        "fullName": {"value": "Test protein"}
                    }
                },
                "organism": {
                    "scientificName": "Homo sapiens",
                    "taxonId": 9606
                },
                "sequence": {"sequence": "MATEST"}
            }
            mock_request.return_value = mock_data
            
            result = client.query("P12345")
            
            assert result is not None
            assert result.accession == "P12345"
    
    def test_search(self, client):
        """Test searching UniProt"""
        with patch.object(client, '_make_request') as mock_request:
            mock_data = {
                "results": [
                    {
                        "primaryAccession": "P12345",
                        "entryName": "TEST_HUMAN",
                        "protein": {"recommendedName": {"fullName": {"value": "Test protein"}}},
                        "organism": {"scientificName": "Homo sapiens"},
                        "score": 100.0
                    }
                ],
                "total": 1
            }
            mock_request.return_value = mock_data
            
            result = client.search("test", limit=10)
            
            assert result is not None
            assert "results" in result
            assert len(result["results"]) == 1


class TestChEMBLClient:
    """Tests for ChEMBLClient"""
    
    @pytest.fixture
    def client(self):
        return ChEMBLClient()
    
    def test_initialization(self, client):
        """Test client initialization"""
        assert client is not None
        assert client.get_name() == "ChEMBL"
    
    def test_query_compound(self, client):
        """Test querying ChEMBL compound"""
        with patch.object(client, '_make_request') as mock_request:
            mock_data = {
                "molecule": {
                    "chembl_id": "CHEMBL123",
                    "smiles": "CCO",
                    "pref_name": "Ethanol",
                    "molecular_weight": 46.07
                }
            }
            mock_request.return_value = mock_data
            
            result = client.query_compound("CHEMBL123")
            
            assert result is not None
            assert result.compound_id == "CHEMBL123"
    
    def test_search_compounds(self, client):
        """Test searching ChEMBL compounds"""
        with patch.object(client, '_make_request') as mock_request:
            mock_data = {
                "molecules": [
                    {
                        "chembl_id": "CHEMBL123",
                        "smiles": "CCO",
                        "pref_name": "Ethanol"
                    }
                ],
                "total": 1
            }
            mock_request.return_value = mock_data
            
            result = client.search_compounds("ethanol", limit=10)
            
            assert result is not None
            assert "results" in result
            assert len(result["results"]) == 1


class TestPDBClient:
    """Tests for PDBClient"""
    
    @pytest.fixture
    def client(self):
        return PDBClient()
    
    def test_initialization(self, client):
        """Test client initialization"""
        assert client is not None
        assert client.get_name() == "PDB"
    
    def test_query(self, client):
        """Test querying PDB"""
        with patch.object(client, '_make_request') as mock_request:
            mock_data = {
                "pdb_id": "1ABC",
                "title": "Test structure",
                "resolution": 1.8,
                "method": "X-RAY DIFFRACTION"
            }
            mock_request.return_value = mock_data
            
            result = client.query("1ABC")
            
            assert result is not None
            assert result.pdb_id == "1ABC"


class TestOpenTargetsClient:
    """Tests for OpenTargetsClient"""
    
    @pytest.fixture
    def client(self):
        return OpenTargetsClient()
    
    def test_initialization(self, client):
        """Test client initialization"""
        assert client is not None
        assert client.get_name() == "OpenTargets"
    
    def test_query_target(self, client):
        """Test querying OpenTargets target"""
        with patch.object(client, '_make_request') as mock_request:
            mock_data = {
                "target": {
                    "id": "ENSG00000123456",
                    "name": "Test target",
                    "gene_symbol": "TEST",
                    "target_type": "SINGLE PROTEIN"
                }
            }
            mock_request.return_value = mock_data
            
            result = client.query_target("ENSG00000123456")
            
            assert result is not None
            assert result.id == "ENSG00000123456"


class TestDatabaseTools:
    """Tests for DatabaseTools"""
    
    @pytest.fixture
    def tools(self):
        return DatabaseTools()
    
    def test_initialization(self, tools):
        """Test tools initialization"""
        assert tools is not None
        assert hasattr(tools, 'uniprot')
        assert hasattr(tools, 'chembl')
        assert hasattr(tools, 'pdb')
    
    def test_query_uniprot(self, tools):
        """Test querying UniProt through tools"""
        with patch.object(tools.uniprot, 'query') as mock_query:
            mock_query.return_value = {"accession": "P12345"}
            
            result = tools.query_uniprot("P12345")
            
            assert result is not None
            assert "accession" in result
    
    def test_query_chembl(self, tools):
        """Test querying ChEMBL through tools"""
        with patch.object(tools.chembl, 'query_compound') as mock_query:
            mock_query.return_value = {"compound_id": "CHEMBL123"}
            
            result = tools.query_chembl("CHEMBL123")
            
            assert result is not None
            assert "compound_id" in result


class TestDatabaseError:
    """Tests for DatabaseError"""
    
    def test_error_creation(self):
        """Test creating a DatabaseError"""
        error = DatabaseError("Test error", status_code=404)
        
        assert error.message == "Test error"
        assert error.status_code == 404
        assert "404" in str(error)
    
    def test_error_without_status(self):
        """Test creating a DatabaseError without status code"""
        error = DatabaseError("Test error")
        
        assert error.message == "Test error"
        assert error.status_code is None
        assert "Test error" in str(error)
