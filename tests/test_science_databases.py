"""
Tests for the additional scientific database clients

Fixtures are trimmed from real API responses, so the parsers are checked
against the shapes these services actually return.
"""

import pytest
from unittest.mock import patch

from drug_discovery_mcp.databases.science import (
    AlphaFoldClient,
    ClinicalTrialsClient,
    ClinVarClient,
    EuropePMCClient,
    GTExClient,
    ReactomeClient,
)
from drug_discovery_mcp.databases.base import DatabaseError


class TestAlphaFoldClient:
    @pytest.fixture
    def client(self):
        return AlphaFoldClient()

    def test_name(self, client):
        assert client.get_name() == "AlphaFold"

    def test_query_parses_prediction(self, client):
        """AlphaFold returns a bare list of models, newest first"""
        with patch.object(client, "_make_request") as request:
            request.return_value = [{
                "entryId": "AF-P04637-F1",
                "uniprotAccession": "P04637",
                "uniprotDescription": "Cellular tumor antigen p53",
                "gene": "TP53",
                "organismScientificName": "Homo sapiens",
                "uniprotSequence": "MEEPQSDPSV",
                "globalMetricValue": 75.06,
                "fractionPlddtConfident": 0.2,
                "latestVersion": 6,
                "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P04637-F1-model_v6.pdb",
            }]

            result = client.query("P04637")

            assert result.entry_id == "AF-P04637-F1"
            assert result.gene == "TP53"
            assert result.mean_plddt == 75.06
            assert result.length == 10
            assert result.version == 6

    def test_missing_model_raises(self, client):
        with patch.object(client, "_make_request") as request:
            request.return_value = []

            with pytest.raises(DatabaseError, match="No AlphaFold model"):
                client.query("NOTREAL")


class TestEuropePMCClient:
    @pytest.fixture
    def client(self):
        return EuropePMCClient()

    def test_name(self, client):
        assert client.get_name() == "EuropePMC"

    def test_journal_read_from_core_shape(self, client):
        """
        'core' results nest the journal under journalInfo

        Only 'lite' results carry a top-level journalTitle, so a parser that
        reads just that field reports every core result as having no journal.
        """
        with patch.object(client, "_make_request") as request:
            request.return_value = {
                "hitCount": 139163,
                "resultList": {"result": [{
                    "id": "42607633",
                    "pmid": "42607633",
                    "doi": "10.1016/j.canep.2026.103204",
                    "title": "TP53 mutations predict poor survival",
                    "authorString": "Ragnarsdottir S, Frick EA",
                    "journalInfo": {"journal": {"title": "Cancer epidemiology"}},
                    "pubYear": "2026",
                    "citedByCount": 3,
                    "isOpenAccess": "N",
                }]},
            }

            result = client.search("TP53 AND cancer", limit=5)

            assert result["total"] == 139163
            paper = result["results"][0]
            assert paper["journal"] == "Cancer epidemiology"
            assert paper["cited_by"] == 3
            assert paper["is_open_access"] is False

    def test_journal_read_from_lite_shape(self, client):
        with patch.object(client, "_make_request") as request:
            request.return_value = {
                "hitCount": 1,
                "resultList": {"result": [{"id": "1", "journalTitle": "Nature"}]},
            }

            assert client.search("x")["results"][0]["journal"] == "Nature"

    def test_open_access_filter_is_applied(self, client):
        with patch.object(client, "_make_request") as request:
            request.return_value = {"hitCount": 0, "resultList": {"result": []}}

            client.search("TP53", open_access_only=True)

            sent = request.call_args.kwargs["params"]["query"]
            assert "OPEN_ACCESS:y" in sent


class TestClinicalTrialsClient:
    @pytest.fixture
    def client(self):
        return ClinicalTrialsClient()

    def test_name(self, client):
        assert client.get_name() == "ClinicalTrials"

    def test_parses_nested_protocol_section(self, client):
        """Study fields are spread across protocolSection modules"""
        with patch.object(client, "_make_request") as request:
            request.return_value = {
                "totalCount": 397,
                "studies": [{
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT03228667",
                            "briefTitle": "A Study of Combination Immunotherapies",
                        },
                        "statusModule": {
                            "overallStatus": "ACTIVE_NOT_RECRUITING",
                            "startDateStruct": {"date": "2017-09-12"},
                        },
                        "designModule": {
                            "phases": ["PHASE2"],
                            "enrollmentInfo": {"count": 40},
                        },
                        "conditionsModule": {"conditions": ["Melanoma"]},
                        "armsInterventionsModule": {
                            "interventions": [{"type": "DRUG", "name": "pembrolizumab"}]
                        },
                        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "NantKwest"}},
                    }
                }],
            }

            result = client.search(condition="melanoma", limit=5)

            assert result["total"] == 397
            trial = result["results"][0]
            assert trial["nct_id"] == "NCT03228667"
            assert trial["phases"] == ["PHASE2"]
            assert trial["enrollment"] == 40
            assert trial["conditions"] == ["Melanoma"]
            assert trial["interventions"][0]["name"] == "pembrolizumab"
            assert trial["sponsor"] == "NantKwest"
            assert trial["url"] == "https://clinicaltrials.gov/study/NCT03228667"

    def test_requests_total_count(self, client):
        """totalCount is omitted by the API unless explicitly requested"""
        with patch.object(client, "_make_request") as request:
            request.return_value = {"studies": []}

            client.search(condition="melanoma")

            assert request.call_args.kwargs["params"]["countTotal"] == "true"

    def test_search_needs_a_criterion(self, client):
        with pytest.raises(DatabaseError, match="at least one"):
            client.search()


class TestReactomeClient:
    @pytest.fixture
    def client(self):
        return ReactomeClient()

    def test_name(self, client):
        assert client.get_name() == "Reactome"

    def test_search_strips_highlight_markup(self, client):
        """Reactome wraps matched terms in <span class="highlighting">"""
        with patch.object(client, "_make_request") as request:
            request.return_value = {"results": [{
                "typeName": "Protein",
                "entries": [{
                    "stId": "R-HSA-69488",
                    "name": '<span class="highlighting" >TP53</span>',
                    "exactType": "ReferenceGeneProduct",
                    "species": ["Homo sapiens"],
                }],
            }]}

            hit = client.search("TP53")["results"][0]

            assert hit["name"] == "TP53"
            assert hit["id"] == "R-HSA-69488"

    def test_search_respects_limit_across_groups(self, client):
        with patch.object(client, "_make_request") as request:
            request.return_value = {"results": [
                {"typeName": "Protein", "entries": [{"stId": f"R-{i}", "name": f"n{i}"}
                                                    for i in range(5)]},
                {"typeName": "Pathway", "entries": [{"stId": f"P-{i}", "name": f"p{i}"}
                                                    for i in range(5)]},
            ]}

            assert len(client.search("TP53", limit=3)["results"]) == 3

    def test_query_pathway_joins_summation(self, client):
        with patch.object(client, "_make_request") as request:
            request.return_value = {
                "stId": "R-HSA-3700989",
                "displayName": "Transcriptional Regulation by TP53",
                "schemaClass": "Pathway",
                "speciesName": "Homo sapiens",
                "summation": [{"text": "The tumor suppressor TP53 is a transcription factor."}],
            }

            pathway = client.query_pathway("R-HSA-3700989")

            assert pathway["type"] == "Pathway"
            assert pathway["name"] == "Transcriptional Regulation by TP53"
            assert "tumor suppressor" in pathway["summary"]


class TestClinVarClient:
    @pytest.fixture
    def client(self):
        return ClinVarClient()

    def test_name(self, client):
        assert client.get_name() == "ClinVar"

    def test_builds_entrez_query_from_filters(self, client):
        with patch.object(client, "_make_request") as request:
            request.return_value = {"esearchresult": {"count": "0", "idlist": []}}

            client.search_variants(gene="BRCA1", classification="pathogenic")

            term = request.call_args.kwargs["params"]["term"]
            assert "BRCA1[gene]" in term
            assert '"pathogenic"[clinsig]' in term

    def test_search_needs_a_filter(self, client):
        with pytest.raises(DatabaseError, match="at least one"):
            client.search_variants()

    def test_parses_germline_classification(self, client):
        """Newer ClinVar records nest the call under germline_classification"""
        with patch.object(client, "_make_request") as request:
            request.side_effect = [
                {"esearchresult": {"count": "14306", "idlist": ["4884209"]}},
                {"result": {"4884209": {
                    "uid": "4884209",
                    "accession": "VCV004884209",
                    "title": "NM_007294.4(BRCA1):c.241_256del (p.Gln81fs)",
                    "obj_type": "Deletion",
                    "gene_sort": "BRCA1",
                    "protein_change": "Q81fs",
                    "germline_classification": {
                        "description": "Pathogenic",
                        "review_status": "no assertion criteria provided",
                        "last_evaluated": "2025/05/01 00:00",
                    },
                    "variation_set": [{"cdna_change": "c.241_256del"}],
                }}},
            ]

            result = client.search_variants(gene="BRCA1")

            assert result["total"] == 14306
            v = result["results"][0]
            assert v["classification"] == "Pathogenic"
            assert v["gene"] == "BRCA1"
            assert v["cdna_change"] == "c.241_256del"

    def test_falls_back_to_clinical_significance(self, client):
        """Older records use clinical_significance instead"""
        with patch.object(client, "_make_request") as request:
            request.side_effect = [
                {"esearchresult": {"count": "1", "idlist": ["1"]}},
                {"result": {"1": {"uid": "1", "clinical_significance": {"description": "Benign"}}}},
            ]

            assert client.search_variants(gene="X")["results"][0]["classification"] == "Benign"

    def test_error_records_are_skipped(self, client):
        with patch.object(client, "_make_request") as request:
            request.side_effect = [
                {"esearchresult": {"count": "1", "idlist": ["999"]}},
                {"result": {"999": {"uid": "999", "error": "cannot get document summary"}}},
            ]

            assert client.search_variants(gene="X")["results"] == []


class TestGTExClient:
    @pytest.fixture
    def client(self):
        return GTExClient()

    def test_name(self, client):
        assert client.get_name() == "GTEx"

    def test_expression_resolves_versioned_gencode_id(self, client):
        """
        GTEx matches the versioned GENCODE id exactly

        An unversioned or stale id returns an empty list rather than an error,
        so the symbol must be resolved before querying expression.
        """
        with patch.object(client, "_make_request") as request:
            request.side_effect = [
                {"data": [{"geneSymbol": "TP53", "gencodeId": "ENSG00000141510.16"}]},
                {"data": [
                    {"tissueSiteDetailId": "Ovary", "median": 32.4, "unit": "TPM"},
                    {"tissueSiteDetailId": "Cells_EBV-transformed_lymphocytes",
                     "median": 72.9, "unit": "TPM"},
                ]},
            ]

            result = client.median_expression("TP53")

            # the versioned id from the lookup is what gets queried
            assert request.call_args.kwargs["params"]["gencodeId"] == "ENSG00000141510.16"
            assert result["n_tissues"] == 2
            # highest expressing tissue first
            assert result["tissues"][0]["tissue"] == "Cells_EBV-transformed_lymphocytes"

    def test_top_limits_tissues(self, client):
        with patch.object(client, "_make_request") as request:
            request.side_effect = [
                {"data": [{"geneSymbol": "TP53", "gencodeId": "ENSG00000141510.16"}]},
                {"data": [{"tissueSiteDetailId": f"T{i}", "median": float(i), "unit": "TPM"}
                          for i in range(10)]},
            ]

            assert len(client.median_expression("TP53", top=3)["tissues"]) == 3

    def test_unknown_gene_raises(self, client):
        with patch.object(client, "_make_request") as request:
            request.return_value = {"data": []}

            with pytest.raises(DatabaseError, match="not found in GTEx"):
                client.resolve_gene("NOTAGENE")
