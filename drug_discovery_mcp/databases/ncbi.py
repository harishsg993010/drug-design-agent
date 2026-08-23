"""
NCBI Database Client

Provides access to NCBI Entrez (E-utilities) for genetic, genomic and
literature data.

An API key is optional but raises the rate limit from 3 to 10 requests per
second; set ``NCBI_API_KEY`` in the environment or pass one via the config.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from .base import DatabaseClient, DatabaseConfig, DatabaseError

logger = logging.getLogger(__name__)


@dataclass
class NCBIGene:
    """Represents an NCBI gene record"""
    gene_id: str
    symbol: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    organism: Optional[str] = None
    taxonomy_id: Optional[int] = None
    chromosome: Optional[str] = None
    map_location: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    genomic_info: List[Dict[str, Any]] = field(default_factory=list)
    mim_ids: List[str] = field(default_factory=list)


@dataclass
class NCBIPublication:
    """Represents a PubMed record"""
    pmid: str
    title: Optional[str] = None
    journal: Optional[str] = None
    pub_date: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    doi: Optional[str] = None


class NCBIClient(DatabaseClient):
    """
    Client for querying NCBI Entrez

    NCBI hosts the primary databases for genes, sequences and biomedical
    literature. This client covers:
    - Gene records (location, aliases, RefSeq summary)
    - Free-text search across any Entrez database
    - Sequence retrieval in FASTA format
    - PubMed literature lookup
    """

    def __init__(self, config: Optional[DatabaseConfig] = None, api_key: Optional[str] = None):
        """
        Initialize NCBI client

        Args:
            config: Custom configuration
            api_key: NCBI API key (falls back to the NCBI_API_KEY environment variable)
        """
        super().__init__(config or self.get_default_config())
        self.api_key = api_key or self.config.api_key or os.environ.get("NCBI_API_KEY")

        # An API key lifts the per-second cap from 3 to 10
        if self.api_key:
            self.config.rate_limit = 10

    @classmethod
    def get_default_config(cls) -> DatabaseConfig:
        """Get default NCBI configuration"""
        return DatabaseConfig(
            endpoint="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            rate_limit=3,  # NCBI's unauthenticated limit
            timeout=30,
            retries=3,
            cache_enabled=True,
            cache_ttl=86400,
            headers={
                "Accept": "application/json",
                "User-Agent": "DrugDiscoveryMCP/0.1.0"
            }
        )

    def get_name(self) -> str:
        """Get the name of this database"""
        return "NCBI"

    # --- transport ------------------------------------------------------------

    def _params(self, **kwargs) -> Dict[str, Any]:
        """Build Entrez query parameters, attaching the API key when present"""
        params = {k: v for k, v in kwargs.items() if v is not None}
        params.setdefault("tool", "DrugDiscoveryMCP")
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    # --- queries --------------------------------------------------------------

    def query_gene(self, gene_id: Union[str, int]) -> NCBIGene:
        """
        Query NCBI Gene for a gene record

        Args:
            gene_id: NCBI Gene ID, e.g. "7157"

        Returns:
            NCBIGene with the parsed record
        """
        gene_id = str(gene_id)
        url = f"{self.config.endpoint}/esummary.fcgi"

        try:
            data = self._make_request("GET", url, params=self._params(
                db="gene", id=gene_id, retmode="json"
            ))

            result = data.get("result") or {}
            record = result.get(gene_id)
            if not record:
                raise DatabaseError(f"Gene not found: {gene_id}")

            # Entrez reports an unknown UID inside the record, with HTTP 200
            if record.get("error"):
                raise DatabaseError(f"Gene {gene_id}: {record['error']}")

            return self._parse_gene(gene_id, record)
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"NCBI gene query failed for {gene_id}: {e}")
            raise DatabaseError(f"Failed to query NCBI gene {gene_id}: {e}")

    def search_genes(
        self,
        query: str,
        organism: Optional[str] = "Homo sapiens",
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Search NCBI Gene

        A bare term is matched against gene symbols first. An unqualified
        Entrez search matches any record that merely *mentions* the term, so
        searching "BRCA1" that way returns EGFR and TP53 ahead of BRCA1 itself.

        Args:
            query: Search terms, e.g. a gene symbol
            organism: Restrict to an organism (None searches all)
            limit: Maximum number of results

        Returns:
            Dictionary with the matching gene records
        """
        organism_filter = f" AND {organism}[orgn]" if organism else ""

        try:
            # Symbol-qualified first, then fall back to free text
            found = self.search("gene", f"{query}[sym]{organism_filter}", limit=limit)
            if not found["ids"]:
                found = self.search("gene", f"{query}{organism_filter}", limit=limit)

            ids = found["ids"]
            return {
                "query": query,
                "total": found["total"],
                "results": [self.query_gene(gene_id).__dict__ for gene_id in ids],
            }
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"NCBI gene search failed: {e}")
            raise DatabaseError(f"Failed to search NCBI genes: {e}")

    def search(self, database: str, term: str, limit: int = 20) -> Dict[str, Any]:
        """
        Search any Entrez database

        Args:
            database: Entrez database name ("gene", "pubmed", "protein", "nuccore", ...)
            term: Entrez query expression
            limit: Maximum number of IDs to return

        Returns:
            Dictionary with the matching UIDs and the total hit count
        """
        url = f"{self.config.endpoint}/esearch.fcgi"

        try:
            data = self._make_request("GET", url, params=self._params(
                db=database, term=term, retmode="json", retmax=limit
            ))

            result = data.get("esearchresult") or {}
            if result.get("ERROR"):
                raise DatabaseError(f"Entrez error: {result['ERROR']}")

            return {
                "database": database,
                "term": term,
                "ids": result.get("idlist", []) or [],
                "total": int(result.get("count", 0) or 0),
            }
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"NCBI search failed for {database}: {e}")
            raise DatabaseError(f"Failed to search NCBI {database}: {e}")

    def get_sequence(
        self,
        accession: str,
        database: str = "protein",
        rettype: str = "fasta",
    ) -> str:
        """
        Fetch a sequence record

        Args:
            accession: Sequence accession, e.g. "NP_000537.3"
            database: "protein" or "nuccore"
            rettype: Record type, "fasta" by default

        Returns:
            The record as text
        """
        url = f"{self.config.endpoint}/efetch.fcgi"

        try:
            response = self._request_raw("GET", url, params=self._params(
                db=database, id=accession, rettype=rettype, retmode="text"
            ))
            text = response.text

            if not text.strip():
                raise DatabaseError(f"No sequence returned for {accession}")
            return text
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"NCBI sequence fetch failed for {accession}: {e}")
            raise DatabaseError(f"Failed to fetch sequence {accession}: {e}")

    def get_protein_sequence(self, accession: str) -> str:
        """
        Fetch a protein sequence as plain residues

        Args:
            accession: Protein accession, e.g. "NP_000537.3"

        Returns:
            The amino acid sequence, with the FASTA header removed
        """
        fasta = self.get_sequence(accession, database="protein")
        return "".join(
            line.strip() for line in fasta.splitlines() if not line.startswith(">")
        )

    def query_publication(self, pmid: Union[str, int]) -> NCBIPublication:
        """
        Query PubMed for a publication

        Args:
            pmid: PubMed ID

        Returns:
            NCBIPublication with the parsed record
        """
        pmid = str(pmid)
        url = f"{self.config.endpoint}/esummary.fcgi"

        try:
            data = self._make_request("GET", url, params=self._params(
                db="pubmed", id=pmid, retmode="json"
            ))

            record = (data.get("result") or {}).get(pmid)
            if not record:
                raise DatabaseError(f"Publication not found: {pmid}")

            if record.get("error"):
                raise DatabaseError(f"Publication {pmid}: {record['error']}")

            return self._parse_publication(pmid, record)
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"NCBI publication query failed for {pmid}: {e}")
            raise DatabaseError(f"Failed to query PubMed record {pmid}: {e}")

    def search_publications(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search PubMed

        Args:
            query: Search terms
            limit: Maximum number of results

        Returns:
            Dictionary with the matching publications
        """
        try:
            found = self.search("pubmed", query, limit=limit)
            pmids = found["ids"]

            if not pmids:
                return {"query": query, "total": found["total"], "results": []}

            # One esummary call covers the whole result set
            url = f"{self.config.endpoint}/esummary.fcgi"
            data = self._make_request("GET", url, params=self._params(
                db="pubmed", id=",".join(pmids), retmode="json"
            ))
            result = data.get("result") or {}

            return {
                "query": query,
                "total": found["total"],
                "results": [
                    self._parse_publication(pmid, result[pmid]).__dict__
                    for pmid in pmids
                    if pmid in result and not result[pmid].get("error")
                ],
            }
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            raise DatabaseError(f"Failed to search PubMed: {e}")

    # --- parsing --------------------------------------------------------------

    def _parse_gene(self, gene_id: str, data: Dict[str, Any]) -> NCBIGene:
        """Parse an Entrez gene summary into an NCBIGene"""
        organism = data.get("organism") or {}
        aliases = [
            alias.strip()
            for alias in (data.get("otheraliases") or "").split(",")
            if alias.strip()
        ]

        return NCBIGene(
            gene_id=str(data.get("uid", gene_id)),
            symbol=data.get("name"),
            name=data.get("nomenclaturename") or data.get("description"),
            description=data.get("description"),
            organism=organism.get("scientificname"),
            taxonomy_id=organism.get("taxid"),
            chromosome=data.get("chromosome"),
            map_location=data.get("maplocation"),
            aliases=aliases,
            summary=data.get("summary") or None,
            genomic_info=data.get("genomicinfo", []) or [],
            mim_ids=[str(m) for m in (data.get("mim") or [])],
        )

    @staticmethod
    def _parse_publication(pmid: str, data: Dict[str, Any]) -> NCBIPublication:
        """Parse an Entrez PubMed summary into an NCBIPublication"""
        doi = None
        for identifier in data.get("articleids", []) or []:
            if identifier.get("idtype") == "doi":
                doi = identifier.get("value")
                break

        return NCBIPublication(
            pmid=str(data.get("uid", pmid)),
            title=data.get("title"),
            journal=data.get("source"),
            pub_date=data.get("pubdate"),
            authors=[
                author.get("name", "")
                for author in data.get("authors", []) or []
                if author.get("name")
            ],
            doi=doi,
        )


# Singleton instance
ncbi_client = NCBIClient()


# Convenience functions for direct use
def query_ncbi(gene_id: str, **kwargs) -> Dict[str, Any]:
    """Query NCBI database for gene information"""
    try:
        gene = ncbi_client.query_gene(gene_id, **kwargs)
        return gene.__dict__
    except DatabaseError as e:
        return {"error": str(e), "gene_id": gene_id}


def search_ncbi(query: str, **kwargs) -> Dict[str, Any]:
    """Search NCBI Gene"""
    try:
        return ncbi_client.search_genes(query, **kwargs)
    except DatabaseError as e:
        return {"error": str(e), "query": query}
