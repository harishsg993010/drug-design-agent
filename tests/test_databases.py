"""
Tests for Database Modules
"""

import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from drug_discovery_mcp.databases import (
    UniProtClient,
    ChEMBLClient,
    PDBClient,
    OpenTargetsClient,
    KEGGClient,
    PubChemClient,
    NCBIClient,
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
        # Mocked with the shape rest.uniprot.org actually returns
        with patch.object(client, '_make_request') as mock_request:
            mock_data = {
                "primaryAccession": "P12345",
                "uniProtkbId": "TEST_HUMAN",
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "Test protein"}
                    }
                },
                "genes": [{"geneName": {"value": "TEST"}, "synonyms": [{"value": "T1"}]}],
                "organism": {
                    "scientificName": "Homo sapiens",
                    "taxonId": 9606,
                    "lineage": ["Eukaryota", "Metazoa"]
                },
                "sequence": {"value": "MATEST", "length": 6, "molWeight": 654},
                "comments": [
                    {
                        "commentType": "FUNCTION",
                        "texts": [{"value": "Does a test thing."}]
                    }
                ],
                "keywords": [{"name": "Reference proteome"}],
                "uniProtKBCrossReferences": [
                    {
                        "database": "GO",
                        "id": "GO:0005634",
                        "properties": [
                            {"key": "GoTerm", "value": "C:nucleus"},
                            {"key": "GoEvidenceType", "value": "IDA:UniProtKB"},
                        ],
                    }
                ],
            }
            mock_request.return_value = mock_data

            result = client.query("P12345")

            assert result is not None
            assert result.accession == "P12345"
            assert result.entry_name == "TEST_HUMAN"
            assert result.protein_name == "Test protein"
            assert result.gene_names == ["TEST", "T1"]
            assert result.organism == "Homo sapiens"
            assert result.organism_id == 9606
            assert result.sequence == "MATEST"
            assert result.molecular_weight == 654
            assert result.function == "Does a test thing."
            assert result.keywords == ["Reference proteome"]
            assert result.go_annotations[0]["term"] == "nucleus"
    
    def test_search(self, client):
        """Test searching UniProt"""
        # search() needs the response headers for the result total, so it goes
        # through _request_raw rather than _make_request
        with patch.object(client, '_request_raw') as mock_request:
            response = MagicMock()
            response.json.return_value = {
                "results": [
                    {
                        "primaryAccession": "P12345",
                        "uniProtkbId": "TEST_HUMAN",
                        "proteinDescription": {
                            "recommendedName": {"fullName": {"value": "Test protein"}}
                        },
                        "genes": [{"geneName": {"value": "TEST"}}],
                        "organism": {"scientificName": "Homo sapiens"},
                        "sequence": {"length": 6},
                    }
                ]
            }
            response.headers = {"X-Total-Results": "42"}
            mock_request.return_value = response

            result = client.search("test", limit=10)

            assert result is not None
            assert "results" in result
            assert len(result["results"]) == 1
            assert result["total"] == 42
            assert result["results"][0]["accession"] == "P12345"
            assert result["results"][0]["gene_names"] == ["TEST"]


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
        # ChEMBL nests structures/properties and returns numbers as strings
        with patch.object(client, '_make_request') as mock_request:
            mock_data = {
                "molecule_chembl_id": "CHEMBL123",
                "pref_name": "Ethanol",
                "molecule_type": "Small molecule",
                "molecule_structures": {
                    "canonical_smiles": "CCO",
                    "standard_inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                },
                "molecule_properties": {
                    "full_mwt": "46.07",
                    "alogp": "-0.14",
                    "hba": 1,
                    "hbd": 1,
                    "psa": "20.23",
                    "full_molformula": "C2H6O",
                },
                "molecule_synonyms": [{"molecule_synonym": "Alcohol"}],
            }
            mock_request.return_value = mock_data

            result = client.query_compound("CHEMBL123")

            assert result is not None
            assert result.compound_id == "CHEMBL123"
            assert result.smiles == "CCO"
            assert result.molecular_formula == "C2H6O"
            # numeric strings are coerced to numbers
            assert result.molecular_weight == 46.07
            assert result.logp == -0.14
            assert result.tpsa == 20.23
            assert result.synonyms == ["Alcohol"]
    
    def test_search_compounds(self, client):
        """Test searching ChEMBL compounds"""
        with patch.object(client, '_make_request') as mock_request:
            mock_data = {
                "molecules": [
                    {
                        "molecule_chembl_id": "CHEMBL123",
                        "pref_name": "Ethanol",
                        "molecule_structures": {"canonical_smiles": "CCO"},
                        "molecule_properties": {"full_mwt": "46.07"},
                    }
                ],
                "page_meta": {"total_count": 1},
            }
            mock_request.return_value = mock_data

            result = client.search_compounds("ethanol", limit=10)

            assert result is not None
            assert "results" in result
            assert len(result["results"]) == 1
            assert result["total"] == 1
            assert result["results"][0]["compound_id"] == "CHEMBL123"
            assert result["results"][0]["smiles"] == "CCO"


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
        # query() resolves the entry, its chains and its ligands in one
        # GraphQL call, so the payload arrives in a "data" envelope
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = {
                "data": {
                    "entry": {
                        "rcsb_id": "1ABC",
                        "struct": {"title": "Test structure"},
                        "exptl": [{"method": "X-RAY DIFFRACTION"}],
                        "refine": [{"ls_d_res_high": 1.8, "ls_R_factor_R_work": 0.2}],
                        "rcsb_accession_info": {"initial_release_date": "1995-07-11T00:00:00Z"},
                        "struct_keywords": {"pdbx_keywords": "HYDROLASE", "text": "enzyme, test"},
                        "audit_author": [{"name": "Doe, J."}],
                        "polymer_entities": [
                            {"rcsb_polymer_entity_container_identifiers": {
                                "auth_asym_ids": ["A", "B"]}}
                        ],
                        "nonpolymer_entities": [
                            {
                                "nonpolymer_comp": {"chem_comp": {
                                    "id": "ZN", "name": "ZINC ION", "formula": "Zn"}},
                                "rcsb_nonpolymer_entity_container_identifiers": {
                                    "auth_asym_ids": ["A"]},
                            }
                        ],
                    }
                }
            }

            result = client.query("1ABC")

            assert result is not None
            assert result.pdb_id == "1ABC"
            assert result.title == "Test structure"
            assert result.resolution == 1.8
            assert result.method == "X-RAY DIFFRACTION"
            assert result.release_date == "1995-07-11"
            assert result.classification == "HYDROLASE"
            assert result.keywords == ["enzyme", "test"]
            assert result.authors == ["Doe, J."]
            # chains are real chain names, not entity numbers
            assert result.chains == ["A", "B"]
            assert result.ligands == [
                {"ligand_id": "ZN", "name": "ZINC ION", "formula": "Zn", "chains": ["A"]}
            ]
            assert result.experimental_data["r_factor_work"] == 0.2

    def test_query_reports_graphql_errors(self, client):
        """GraphQL errors arrive with HTTP 200 and must still raise"""
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = {"errors": [{"message": "boom"}]}

            with pytest.raises(DatabaseError, match="boom"):
                client.query("1ABC")


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
        # OpenTargets is GraphQL: payloads arrive wrapped in a "data" envelope
        with patch.object(client, '_make_request') as mock_request:
            mock_data = {
                "data": {
                    "target": {
                        "id": "ENSG00000123456",
                        "approvedSymbol": "TEST",
                        "approvedName": "Test target",
                        "biotype": "protein_coding",
                        "functionDescriptions": ["Tests things."],
                        "synonyms": [{"label": "TST", "source": "HGNC"}],
                        "proteinIds": [{"id": "P12345", "source": "uniprot_swissprot"}],
                    }
                }
            }
            mock_request.return_value = mock_data

            result = client.query_target("ENSG00000123456")

            assert result is not None
            assert result.id == "ENSG00000123456"
            assert result.gene_symbol == "TEST"
            assert result.name == "Test target"
            assert result.protein_id == "P12345"
            assert result.description == "Tests things."
            assert result.synonyms == ["TST"]

    def test_query_target_reports_graphql_errors(self, client):
        """GraphQL errors arrive with HTTP 200 and must still raise"""
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = {"errors": [{"message": "boom"}]}

            with pytest.raises(DatabaseError, match="boom"):
                client.query_target("ENSG00000123456")


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


# KEGG serves fixed-width flat text: the section name occupies the first 12
# columns and continuation lines are blank across that same width.
KEGG_PATHWAY_RECORD = """\
ENTRY       hsa04110                    Pathway
NAME        Cell cycle - Homo sapiens (human)
DESCRIPTION Mitotic cell cycle progression is accomplished through
            a reproducible sequence of events.
CLASS       Cellular Processes; Cell growth and death
PATHWAY_MAP hsa04110  Cell cycle
DRUG        D05988  Tacedinaline (USAN/INN)
ORGANISM    Homo sapiens (human) [GN:hsa]
GENE        1017  CDK2; cyclin-dependent kinase 2 isoform 1 [KO:K02206] [EC:2.7.11.22]
            1019  CDK4; cyclin-dependent kinase 4 [KO:K02089]
COMPOUND    C00575  3',5'-Cyclic AMP
REFERENCE   PMID:7908906
///
"""


class TestKEGGClient:
    """Tests for KEGGClient"""

    @pytest.fixture
    def client(self):
        return KEGGClient()

    def test_initialization(self, client):
        """Test client initialization"""
        assert client is not None
        assert client.get_name() == "KEGG"

    def test_query_pathway(self, client):
        """Test parsing a KEGG flat-file pathway record"""
        with patch.object(client, '_get_text') as mock_get:
            mock_get.return_value = KEGG_PATHWAY_RECORD

            result = client.query_pathway("hsa04110")

            assert result.pathway_id == "hsa04110"
            assert result.name == "Cell cycle - Homo sapiens (human)"
            assert result.pathway_class == "Cellular Processes; Cell growth and death"
            assert result.organism == "Homo sapiens (human) [GN:hsa]"
            # continuation lines are folded into the section above them
            assert result.description.endswith("a reproducible sequence of events.")
            assert len(result.genes) == 2
            assert len(result.compounds) == 1
            assert len(result.drugs) == 1
            assert result.references == ["PMID:7908906"]

    def test_gene_entry_is_split_into_parts(self, client):
        """Gene lines carry an ID, symbol, description and bracketed annotations"""
        with patch.object(client, '_get_text') as mock_get:
            mock_get.return_value = KEGG_PATHWAY_RECORD

            gene = client.query_pathway("hsa04110").genes[0]

            assert gene["id"] == "1017"
            assert gene["symbol"] == "CDK2"
            assert gene["description"] == "cyclin-dependent kinase 2 isoform 1"
            assert gene["annotations"] == ["KO:K02206", "EC:2.7.11.22"]

    def test_search_parses_tab_separated_results(self, client):
        """KEGG find returns tab separated id/name pairs"""
        with patch.object(client, '_get_text') as mock_get:
            mock_get.return_value = "map04110\tCell cycle\nmap04111\tCell cycle - yeast\n"

            results = client.search("pathway", "cell cycle")

            assert results == [
                {"id": "map04110", "name": "Cell cycle"},
                {"id": "map04111", "name": "Cell cycle - yeast"},
            ]

    def test_search_returns_empty_on_no_match(self, client):
        """KEGG answers an empty result set with a blank body"""
        with patch.object(client, '_get_text') as mock_get:
            mock_get.side_effect = DatabaseError("KEGG returned no data")

            assert client.search("pathway", "nonexistent") == []


class TestPubChemClient:
    """Tests for PubChemClient"""

    @pytest.fixture
    def client(self):
        return PubChemClient()

    def test_initialization(self, client):
        """Test client initialization"""
        assert client is not None
        assert client.get_name() == "PubChem"

    def test_query_compound(self, client):
        """Test querying a PubChem compound by CID"""
        with patch.object(client, '_make_request') as mock_request, \
             patch.object(client, 'get_synonyms') as mock_synonyms:
            mock_request.return_value = {
                "PropertyTable": {
                    "Properties": [
                        {
                            "CID": 2244,
                            "MolecularFormula": "C9H8O4",
                            "MolecularWeight": "180.16",
                            "ExactMass": "180.04225873",
                            "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
                            "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                            "IUPACName": "2-acetyloxybenzoic acid",
                            "XLogP": 1.2,
                            "TPSA": 63.6,
                            "Charge": 0,
                            "HBondDonorCount": 1,
                            "HBondAcceptorCount": 4,
                        }
                    ]
                }
            }
            mock_synonyms.return_value = ["aspirin", "ACETYLSALICYLIC ACID"]

            result = client.query_compound(2244)

            assert result.compound_id == "2244"
            assert result.formula == "C9H8O4"
            # PubChem returns masses as strings; they must be coerced
            assert result.molecular_weight == 180.16
            assert result.exact_mass == 180.04225873
            assert result.logp == 1.2
            assert result.charge == 0
            assert result.name == "aspirin"

    def test_resolve_cid_returns_empty_on_404(self, client):
        """An unknown identifier is a 404, not an error condition"""
        with patch.object(client, '_make_request') as mock_request:
            mock_request.side_effect = DatabaseError("not found", status_code=404)

            assert client.resolve_cid("nonexistent-compound") == []

    def test_search_by_structure_rejects_unknown_type(self, client):
        """Only the three documented structure searches are accepted"""
        with pytest.raises(DatabaseError, match="Unknown search type"):
            client.search_by_structure("CCO", search_type="telepathy")


class TestNCBIClient:
    """Tests for NCBIClient"""

    @pytest.fixture
    def client(self):
        return NCBIClient()

    def test_initialization(self, client):
        """Test client initialization"""
        assert client is not None
        assert client.get_name() == "NCBI"

    def test_query_gene(self, client):
        """Test parsing an Entrez gene summary"""
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = {
                "result": {
                    "7157": {
                        "uid": "7157",
                        "name": "TP53",
                        "description": "tumor protein p53",
                        "chromosome": "17",
                        "maplocation": "17p13.1",
                        "otheraliases": "BCC7, LFS1, P53",
                        "summary": "This gene encodes a tumor suppressor protein.",
                        "organism": {"scientificname": "Homo sapiens", "taxid": 9606},
                        "genomicinfo": [{"chrloc": "17", "chraccver": "NC_000017.11"}],
                        "mim": ["191170"],
                    }
                }
            }

            result = client.query_gene(7157)

            assert result.gene_id == "7157"
            assert result.symbol == "TP53"
            assert result.organism == "Homo sapiens"
            assert result.taxonomy_id == 9606
            assert result.map_location == "17p13.1"
            assert result.aliases == ["BCC7", "LFS1", "P53"]
            assert result.mim_ids == ["191170"]

    def test_search_genes_prefers_symbol_match(self, client):
        """
        A bare term must be matched against gene symbols first

        An unqualified Entrez search matches any record that merely mentions
        the term, which puts other genes ahead of the one asked for.
        """
        with patch.object(client, 'search') as mock_search, \
             patch.object(client, 'query_gene') as mock_gene:
            mock_search.return_value = {"ids": ["672"], "total": 1}
            mock_gene.return_value = SimpleNamespace(gene_id="672")

            client.search_genes("BRCA1")

            term = mock_search.call_args[0][1]
            assert "BRCA1[sym]" in term
            assert "Homo sapiens[orgn]" in term

    def test_search_genes_falls_back_to_free_text(self, client):
        """A phrase that is not a symbol still searches"""
        with patch.object(client, 'search') as mock_search, \
             patch.object(client, 'query_gene') as mock_gene:
            mock_search.side_effect = [
                {"ids": [], "total": 0},           # no symbol match
                {"ids": ["1956"], "total": 12},    # free-text fallback
            ]
            mock_gene.return_value = SimpleNamespace(gene_id="1956")

            result = client.search_genes("tumor suppressor")

            assert mock_search.call_count == 2
            assert "[sym]" not in mock_search.call_args[0][1]
            assert result["total"] == 12

    def test_search_reports_entrez_errors(self, client):
        """Entrez reports failures inside a 200 response body"""
        with patch.object(client, '_make_request') as mock_request:
            mock_request.return_value = {"esearchresult": {"ERROR": "Invalid db name"}}

            with pytest.raises(DatabaseError, match="Invalid db name"):
                client.search("nosuchdb", "test")
