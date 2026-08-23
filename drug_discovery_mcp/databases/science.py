"""
Additional scientific database clients

Covers sources the core clients do not: predicted structures (AlphaFold),
full-text literature (Europe PMC), clinical trials (ClinicalTrials.gov) and
curated pathways/reactions (Reactome).

The endpoint map was taken from Google DeepMind's `science-skills
<https://github.com/google-deepmind/science-skills>`_ (Apache-2.0), which
documents verified request shapes for each of these services. The clients here
are written against this project's ``DatabaseClient`` base so they inherit its
rate limiting, retry and error handling.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import DatabaseClient, DatabaseConfig, DatabaseError

logger = logging.getLogger(__name__)

# Reactome wraps search matches in markup, e.g. '<span class="highlighting">TP53</span>'
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_markup(value: Optional[str]) -> str:
    """Remove the highlight markup Reactome embeds in search results"""
    return _TAG_RE.sub("", value or "").strip()


# --------------------------------------------------------------------------
# AlphaFold
# --------------------------------------------------------------------------


@dataclass
class AlphaFoldPrediction:
    """A predicted structure from the AlphaFold DB"""
    entry_id: str
    uniprot_accession: str
    description: Optional[str] = None
    gene: Optional[str] = None
    organism: Optional[str] = None
    sequence: str = ""
    length: int = 0
    # Mean pLDDT across the model: >90 very high, 70-90 confident, 50-70 low
    mean_plddt: Optional[float] = None
    fraction_confident: Optional[float] = None
    fraction_very_high: Optional[float] = None
    version: Optional[int] = None
    pdb_url: Optional[str] = None
    cif_url: Optional[str] = None


class AlphaFoldClient(DatabaseClient):
    """
    Client for the AlphaFold Protein Structure Database

    Complements the PDB client: where PDB has experimentally determined
    structures, AlphaFold has predicted ones for proteins with no solved
    structure, along with per-model confidence (pLDDT).
    """

    @classmethod
    def get_default_config(cls) -> DatabaseConfig:
        return DatabaseConfig(
            endpoint="https://alphafold.ebi.ac.uk/api",
            rate_limit=10,
            timeout=30,
            retries=3,
            cache_enabled=True,
            cache_ttl=86400,
            headers={"Accept": "application/json", "User-Agent": "DrugDiscoveryMCP/0.1.0"},
        )

    def get_name(self) -> str:
        return "AlphaFold"

    def query(self, accession: str) -> AlphaFoldPrediction:
        """
        Fetch the predicted structure for a UniProt accession

        Args:
            accession: UniProt accession, e.g. "P04637"

        Returns:
            AlphaFoldPrediction with confidence metrics and download URLs
        """
        url = f"{self.config.endpoint}/prediction/{accession}"

        try:
            data = self._make_request("GET", url)
            # The API returns a bare list of models, newest first
            entries: List[Dict[str, Any]] = (
                data if isinstance(data, list) else (data.get("results") or [])
            )
            if not entries:
                raise DatabaseError(f"No AlphaFold model for {accession}")

            return self._parse(entries[0])
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"AlphaFold query failed for {accession}: {e}")
            raise DatabaseError(f"Failed to query AlphaFold for {accession}: {e}")

    @staticmethod
    def _parse(data: Dict[str, Any]) -> AlphaFoldPrediction:
        sequence = data.get("uniprotSequence") or ""
        return AlphaFoldPrediction(
            entry_id=data.get("entryId", ""),
            uniprot_accession=data.get("uniprotAccession", ""),
            description=data.get("uniprotDescription"),
            gene=data.get("gene"),
            organism=data.get("organismScientificName"),
            sequence=sequence,
            length=len(sequence),
            mean_plddt=data.get("globalMetricValue"),
            fraction_confident=data.get("fractionPlddtConfident"),
            fraction_very_high=data.get("fractionPlddtVeryHigh"),
            version=data.get("latestVersion"),
            pdb_url=data.get("pdbUrl"),
            cif_url=data.get("cifUrl"),
        )


# --------------------------------------------------------------------------
# Europe PMC
# --------------------------------------------------------------------------


@dataclass
class Publication:
    """A literature record from Europe PMC"""
    id: str
    source: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[str] = None
    cited_by: Optional[int] = None
    is_open_access: bool = False
    abstract: Optional[str] = None


class EuropePMCClient(DatabaseClient):
    """
    Client for Europe PMC

    Broader than PubMed alone: indexes preprints, patents and full text, and
    exposes citation counts and open-access status directly on each hit.
    """

    @classmethod
    def get_default_config(cls) -> DatabaseConfig:
        return DatabaseConfig(
            endpoint="https://www.ebi.ac.uk/europepmc/webservices/rest",
            rate_limit=10,
            timeout=30,
            retries=3,
            cache_enabled=True,
            cache_ttl=3600,
            headers={"Accept": "application/json", "User-Agent": "DrugDiscoveryMCP/0.1.0"},
        )

    def get_name(self) -> str:
        return "EuropePMC"

    def search(
        self,
        query: str,
        limit: int = 10,
        open_access_only: bool = False,
        with_abstract: bool = True,
    ) -> Dict[str, Any]:
        """
        Search the literature

        Args:
            query: Europe PMC query, e.g. 'TP53 AND cancer' or 'AUTH:"Cho Y"'
            limit: Maximum number of results
            open_access_only: Restrict to open-access records
            with_abstract: Request abstracts (``resultType=core``)

        Returns:
            Dictionary with the matching publications and the total hit count
        """
        search_query = f"({query}) AND OPEN_ACCESS:y" if open_access_only else query
        params = {
            "query": search_query,
            "format": "json",
            "pageSize": min(limit, 100),
            "resultType": "core" if with_abstract else "lite",
        }

        try:
            data = self._make_request("GET", f"{self.config.endpoint}/search", params=params)
            results = ((data.get("resultList") or {}).get("result")) or []

            return {
                "query": query,
                "total": data.get("hitCount", 0),
                "results": [self._parse(r).__dict__ for r in results[:limit]],
            }
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Europe PMC search failed: {e}")
            raise DatabaseError(f"Failed to search Europe PMC: {e}")

    def get_citations(self, pmid: str, limit: int = 25) -> List[Dict[str, Any]]:
        """
        List the papers citing a given article

        Args:
            pmid: PubMed ID of the cited article
            limit: Maximum number of citing papers

        Returns:
            List of citing records
        """
        url = f"{self.config.endpoint}/MED/{pmid}/citations"
        params = {"format": "json", "pageSize": min(limit, 100)}

        try:
            data = self._make_request("GET", url, params=params)
            return ((data.get("citationList") or {}).get("citation")) or []
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Europe PMC citations failed for {pmid}: {e}")
            raise DatabaseError(f"Failed to fetch citations for {pmid}: {e}")

    @staticmethod
    def _parse(data: Dict[str, Any]) -> Publication:
        # 'lite' results carry journalTitle at the top level; 'core' results
        # nest it under journalInfo.journal.title instead.
        journal = data.get("journalTitle") or (
            ((data.get("journalInfo") or {}).get("journal") or {}).get("title")
        )

        return Publication(
            id=str(data.get("id", "")),
            source=data.get("source"),
            pmid=data.get("pmid"),
            pmcid=data.get("pmcid"),
            doi=data.get("doi"),
            title=data.get("title"),
            authors=data.get("authorString"),
            journal=journal,
            year=data.get("pubYear"),
            cited_by=data.get("citedByCount"),
            is_open_access=data.get("isOpenAccess") == "Y",
            abstract=data.get("abstractText"),
        )


# --------------------------------------------------------------------------
# ClinicalTrials.gov
# --------------------------------------------------------------------------


@dataclass
class ClinicalTrial:
    """A study record from ClinicalTrials.gov"""
    nct_id: str
    title: Optional[str] = None
    status: Optional[str] = None
    phases: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    interventions: List[Dict[str, Any]] = field(default_factory=list)
    enrollment: Optional[int] = None
    start_date: Optional[str] = None
    completion_date: Optional[str] = None
    sponsor: Optional[str] = None
    summary: Optional[str] = None
    url: Optional[str] = None


class ClinicalTrialsClient(DatabaseClient):
    """
    Client for the ClinicalTrials.gov v2 API

    Useful for checking whether a target or compound has reached the clinic,
    and at what phase.
    """

    @classmethod
    def get_default_config(cls) -> DatabaseConfig:
        return DatabaseConfig(
            endpoint="https://clinicaltrials.gov/api/v2",
            rate_limit=10,
            timeout=30,
            retries=3,
            cache_enabled=True,
            cache_ttl=3600,
            headers={"Accept": "application/json", "User-Agent": "DrugDiscoveryMCP/0.1.0"},
        )

    def get_name(self) -> str:
        return "ClinicalTrials"

    def search(
        self,
        condition: Optional[str] = None,
        intervention: Optional[str] = None,
        query: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Search studies

        Args:
            condition: Disease or condition, e.g. "melanoma"
            intervention: Drug or intervention, e.g. "pembrolizumab"
            query: Free-text search across the record
            status: Filter by recruitment status, e.g. "RECRUITING"
            limit: Maximum number of studies

        Returns:
            Dictionary with the matching studies and the total count
        """
        if not any((condition, intervention, query)):
            raise DatabaseError("Provide at least one of condition, intervention or query")

        params: Dict[str, Any] = {
            "pageSize": min(limit, 100),
            # totalCount is omitted unless explicitly requested
            "countTotal": "true",
        }
        if condition:
            params["query.cond"] = condition
        if intervention:
            params["query.intr"] = intervention
        if query:
            params["query.term"] = query
        if status:
            params["filter.overallStatus"] = status

        try:
            data = self._make_request("GET", f"{self.config.endpoint}/studies", params=params)
            studies = data.get("studies", []) or []

            return {
                "total": data.get("totalCount", len(studies)),
                "results": [self._parse(s).__dict__ for s in studies[:limit]],
            }
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"ClinicalTrials search failed: {e}")
            raise DatabaseError(f"Failed to search ClinicalTrials.gov: {e}")

    def query_study(self, nct_id: str) -> ClinicalTrial:
        """
        Fetch one study by its NCT identifier

        Args:
            nct_id: e.g. "NCT03228667"

        Returns:
            ClinicalTrial record
        """
        try:
            data = self._make_request("GET", f"{self.config.endpoint}/studies/{nct_id}")
            return self._parse(data)
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"ClinicalTrials query failed for {nct_id}: {e}")
            raise DatabaseError(f"Failed to query study {nct_id}: {e}")

    @staticmethod
    def _parse(study: Dict[str, Any]) -> ClinicalTrial:
        protocol = study.get("protocolSection", study) or {}
        ident = protocol.get("identificationModule", {}) or {}
        status_mod = protocol.get("statusModule", {}) or {}
        design = protocol.get("designModule", {}) or {}
        conditions = protocol.get("conditionsModule", {}) or {}
        arms = protocol.get("armsInterventionsModule", {}) or {}
        sponsor = protocol.get("sponsorCollaboratorsModule", {}) or {}
        description = protocol.get("descriptionModule", {}) or {}

        nct_id = ident.get("nctId", "")
        return ClinicalTrial(
            nct_id=nct_id,
            title=ident.get("briefTitle"),
            status=status_mod.get("overallStatus"),
            phases=design.get("phases", []) or [],
            conditions=conditions.get("conditions", []) or [],
            interventions=[
                {"type": i.get("type"), "name": i.get("name")}
                for i in arms.get("interventions", []) or []
            ],
            enrollment=(design.get("enrollmentInfo") or {}).get("count"),
            start_date=(status_mod.get("startDateStruct") or {}).get("date"),
            completion_date=(status_mod.get("completionDateStruct") or {}).get("date"),
            sponsor=((sponsor.get("leadSponsor") or {}).get("name")),
            summary=description.get("briefSummary"),
            url=f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else None,
        )


# --------------------------------------------------------------------------
# Reactome
# --------------------------------------------------------------------------


class ReactomeClient(DatabaseClient):
    """
    Client for Reactome

    Curated human pathways and reactions. Complements KEGG: Reactome models
    reactions and participants in more detail, and is human-centric.
    """

    @classmethod
    def get_default_config(cls) -> DatabaseConfig:
        return DatabaseConfig(
            endpoint="https://reactome.org/ContentService",
            rate_limit=10,
            timeout=30,
            retries=3,
            cache_enabled=True,
            cache_ttl=86400,
            headers={"Accept": "application/json", "User-Agent": "DrugDiscoveryMCP/0.1.0"},
        )

    def get_name(self) -> str:
        return "Reactome"

    def search(self, query: str, species: str = "Homo sapiens", limit: int = 10) -> Dict[str, Any]:
        """
        Search Reactome entities and pathways

        Args:
            query: Gene, protein or pathway name, e.g. "TP53"
            species: Species filter
            limit: Maximum number of results

        Returns:
            Dictionary with flattened search hits
        """
        params = {"query": query, "species": species, "cluster": "true"}

        try:
            data = self._make_request("GET", f"{self.config.endpoint}/search/query", params=params)
            hits = []
            for group in data.get("results", []) or []:
                for entry in group.get("entries", []) or []:
                    hits.append({
                        "id": entry.get("stId") or entry.get("id"),
                        "name": _strip_markup(entry.get("name")),
                        "type": entry.get("exactType") or group.get("typeName"),
                        "species": entry.get("species"),
                        "is_disease": entry.get("isDisease", False),
                    })
                    if len(hits) >= limit:
                        break
                if len(hits) >= limit:
                    break

            return {"query": query, "total": len(hits), "results": hits}
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Reactome search failed: {e}")
            raise DatabaseError(f"Failed to search Reactome: {e}")

    def query_pathway(self, stable_id: str) -> Dict[str, Any]:
        """
        Fetch a pathway or event by its stable identifier

        Args:
            stable_id: Reactome stable ID, e.g. "R-HSA-69488"

        Returns:
            Dictionary describing the pathway
        """
        try:
            data = self._make_request("GET", f"{self.config.endpoint}/data/query/{stable_id}")
            species = data.get("speciesName") or [
                s.get("displayName") for s in data.get("species", []) or []
            ]

            return {
                "id": data.get("stId", stable_id),
                "name": _strip_markup(data.get("displayName")),
                "type": data.get("schemaClass"),
                "species": species,
                "is_disease": data.get("isInDisease", False),
                "summary": " ".join(
                    _strip_markup(s.get("text"))
                    for s in data.get("summation", []) or []
                    if s.get("text")
                ) or None,
            }
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Reactome pathway query failed for {stable_id}: {e}")
            raise DatabaseError(f"Failed to query Reactome pathway {stable_id}: {e}")

    def get_pathways_for_gene(self, gene: str, species: str = "Homo sapiens") -> List[Dict[str, Any]]:
        """
        List pathways a gene participates in

        Args:
            gene: Gene symbol, e.g. "TP53"
            species: Species filter

        Returns:
            List of pathway records
        """
        results = self.search(gene, species=species, limit=100)["results"]
        return [r for r in results if "Pathway" in str(r.get("type", ""))]


# Singleton instances
alphafold_client = AlphaFoldClient()
europepmc_client = EuropePMCClient()
clinicaltrials_client = ClinicalTrialsClient()
reactome_client = ReactomeClient()


# Convenience functions
def query_alphafold(accession: str) -> Dict[str, Any]:
    """Fetch the AlphaFold predicted structure for a UniProt accession"""
    try:
        return alphafold_client.query(accession).__dict__
    except DatabaseError as e:
        return {"error": str(e), "accession": accession}


def search_literature(query: str, **kwargs) -> Dict[str, Any]:
    """Search Europe PMC"""
    try:
        return europepmc_client.search(query, **kwargs)
    except DatabaseError as e:
        return {"error": str(e), "query": query}


def search_clinical_trials(**kwargs) -> Dict[str, Any]:
    """Search ClinicalTrials.gov"""
    try:
        return clinicaltrials_client.search(**kwargs)
    except DatabaseError as e:
        return {"error": str(e)}


def search_reactome(query: str, **kwargs) -> Dict[str, Any]:
    """Search Reactome"""
    try:
        return reactome_client.search(query, **kwargs)
    except DatabaseError as e:
        return {"error": str(e), "query": query}


# --------------------------------------------------------------------------
# ClinVar
# --------------------------------------------------------------------------


@dataclass
class ClinVarVariant:
    """A variant record from ClinVar"""
    uid: str
    accession: Optional[str] = None
    title: Optional[str] = None
    gene: Optional[str] = None
    variant_type: Optional[str] = None
    classification: Optional[str] = None
    review_status: Optional[str] = None
    last_evaluated: Optional[str] = None
    protein_change: Optional[str] = None
    cdna_change: Optional[str] = None


class ClinVarClient(DatabaseClient):
    """
    Client for ClinVar, via NCBI Entrez

    The consensus record for clinical interpretation of human variants:
    Pathogenic / Likely pathogenic / VUS / Benign, together with the review
    status (the star rating) behind each call.
    """

    @classmethod
    def get_default_config(cls) -> DatabaseConfig:
        return DatabaseConfig(
            endpoint="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            rate_limit=3,  # 10/s with an NCBI_API_KEY
            timeout=30,
            retries=3,
            cache_enabled=True,
            cache_ttl=86400,
            headers={"Accept": "application/json", "User-Agent": "DrugDiscoveryMCP/0.1.0"},
        )

    def __init__(self, config: Optional[DatabaseConfig] = None, api_key: Optional[str] = None):
        import os

        super().__init__(config or self.get_default_config())
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        if self.api_key:
            self.config.rate_limit = 10

    def get_name(self) -> str:
        return "ClinVar"

    def _params(self, **kwargs) -> Dict[str, Any]:
        params = {k: v for k, v in kwargs.items() if v is not None}
        params.setdefault("tool", "DrugDiscoveryMCP")
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def search_variants(
        self,
        gene: Optional[str] = None,
        term: Optional[str] = None,
        classification: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Search ClinVar

        Args:
            gene: Gene symbol, e.g. "BRCA1"
            term: Free-text Entrez query, combined with the other filters
            classification: e.g. "pathogenic", "benign", "likely pathogenic"
            limit: Maximum number of variants

        Returns:
            Dictionary with the matching variants and the total hit count
        """
        clauses = []
        if gene:
            clauses.append(f"{gene}[gene]")
        if classification:
            clauses.append(f'"{classification}"[clinsig]')
        if term:
            clauses.append(term)
        if not clauses:
            raise DatabaseError("Provide at least one of gene, term or classification")

        query = " AND ".join(clauses)

        try:
            found = self._make_request(
                "GET",
                f"{self.config.endpoint}/esearch.fcgi",
                params=self._params(db="clinvar", term=query, retmode="json", retmax=limit),
            )
            result = found.get("esearchresult") or {}
            if result.get("ERROR"):
                raise DatabaseError(f"Entrez error: {result['ERROR']}")

            ids = result.get("idlist", []) or []
            variants = self._summaries(ids) if ids else []

            return {
                "query": query,
                "total": int(result.get("count", 0) or 0),
                "results": [v.__dict__ for v in variants],
            }
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"ClinVar search failed: {e}")
            raise DatabaseError(f"Failed to search ClinVar: {e}")

    def _summaries(self, ids: List[str]) -> List[ClinVarVariant]:
        """Fetch summaries for a batch of ClinVar UIDs in one call"""
        data = self._make_request(
            "GET",
            f"{self.config.endpoint}/esummary.fcgi",
            params=self._params(db="clinvar", id=",".join(ids), retmode="json"),
        )
        result = data.get("result") or {}
        return [
            self._parse(uid, result[uid])
            for uid in ids
            if uid in result and not result[uid].get("error")
        ]

    @staticmethod
    def _parse(uid: str, data: Dict[str, Any]) -> ClinVarVariant:
        # Newer records use germline_classification, older ones
        # clinical_significance; both carry the same sub-fields.
        classification = (
            data.get("germline_classification") or data.get("clinical_significance") or {}
        )
        variation = (data.get("variation_set") or [{}])[0]

        return ClinVarVariant(
            uid=str(data.get("uid", uid)),
            accession=data.get("accession"),
            title=data.get("title"),
            gene=data.get("gene_sort") or None,
            variant_type=data.get("obj_type"),
            classification=classification.get("description"),
            review_status=classification.get("review_status"),
            last_evaluated=classification.get("last_evaluated"),
            protein_change=data.get("protein_change") or None,
            cdna_change=variation.get("cdna_change"),
        )


# --------------------------------------------------------------------------
# GTEx
# --------------------------------------------------------------------------


class GTExClient(DatabaseClient):
    """
    Client for the GTEx portal

    Bulk tissue expression across ~54 human tissues, useful for judging where a
    target is expressed and where on-target toxicity might land.
    """

    @classmethod
    def get_default_config(cls) -> DatabaseConfig:
        return DatabaseConfig(
            endpoint="https://gtexportal.org/api/v2",
            rate_limit=5,
            timeout=30,
            retries=3,
            cache_enabled=True,
            cache_ttl=86400,
            headers={"Accept": "application/json", "User-Agent": "DrugDiscoveryMCP/0.1.0"},
        )

    def get_name(self) -> str:
        return "GTEx"

    def resolve_gene(self, gene: str) -> Dict[str, Any]:
        """
        Look up a gene's versioned GENCODE identifier

        The expression endpoint matches the *versioned* id exactly and silently
        returns an empty list for an unversioned or stale one, so the symbol has
        to be resolved first.

        Args:
            gene: Gene symbol or Ensembl ID, e.g. "TP53"

        Returns:
            Dictionary describing the gene, including ``gencode_id``
        """
        try:
            data = self._make_request(
                "GET", f"{self.config.endpoint}/reference/gene", params={"geneId": gene}
            )
            rows = data.get("data") or []
            if not rows:
                raise DatabaseError(f"Gene not found in GTEx: {gene}")

            row = rows[0]
            return {
                "gene_symbol": row.get("geneSymbol"),
                "gencode_id": row.get("gencodeId"),
                "entrez_gene_id": row.get("entrezGeneId"),
                "chromosome": row.get("chromosome"),
                "gene_type": row.get("geneType"),
                "description": row.get("description"),
            }
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"GTEx gene lookup failed for {gene}: {e}")
            raise DatabaseError(f"Failed to resolve gene {gene} in GTEx: {e}")

    def median_expression(
        self,
        gene: str,
        dataset_id: str = "gtex_v8",
        top: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Median expression of a gene across tissues

        Args:
            gene: Gene symbol, e.g. "TP53" (resolved to a versioned GENCODE id)
            dataset_id: GTEx release, e.g. "gtex_v8"
            top: Return only the N highest-expressing tissues

        Returns:
            Dictionary with per-tissue medians, highest first
        """
        try:
            resolved = self.resolve_gene(gene)
            gencode_id = resolved["gencode_id"]

            data = self._make_request(
                "GET",
                f"{self.config.endpoint}/expression/medianGeneExpression",
                params={"gencodeId": gencode_id, "datasetId": dataset_id},
            )
            rows = data.get("data") or []
            tissues = sorted(
                (
                    {
                        "tissue": r.get("tissueSiteDetailId"),
                        "median": r.get("median"),
                        "unit": r.get("unit"),
                    }
                    for r in rows
                ),
                key=lambda r: r["median"] if isinstance(r["median"], (int, float)) else -1,
                reverse=True,
            )

            return {
                "gene_symbol": resolved["gene_symbol"],
                "gencode_id": gencode_id,
                "dataset_id": dataset_id,
                "n_tissues": len(tissues),
                "tissues": tissues[:top] if top else tissues,
            }
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"GTEx expression query failed for {gene}: {e}")
            raise DatabaseError(f"Failed to query GTEx expression for {gene}: {e}")


clinvar_client = ClinVarClient()
gtex_client = GTExClient()


def search_clinvar(**kwargs) -> Dict[str, Any]:
    """Search ClinVar for variant classifications"""
    try:
        return clinvar_client.search_variants(**kwargs)
    except DatabaseError as e:
        return {"error": str(e)}


def query_gtex_expression(gene: str, **kwargs) -> Dict[str, Any]:
    """Median tissue expression for a gene from GTEx"""
    try:
        return gtex_client.median_expression(gene, **kwargs)
    except DatabaseError as e:
        return {"error": str(e), "gene": gene}
