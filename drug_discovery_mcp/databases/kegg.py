"""
KEGG Database Client

Provides access to the KEGG REST API for pathway, gene, compound and drug data.

KEGG serves flat text records rather than JSON: each record is a sequence of
sections whose name occupies the first 12 columns, with continuation lines
left-padded to that same width and the record terminated by ``///``.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .base import DatabaseClient, DatabaseConfig, DatabaseError

logger = logging.getLogger(__name__)

# KEGG pads section names into a fixed-width leading column
_FIELD_WIDTH = 12


@dataclass
class KEGGPathway:
    """Represents a KEGG pathway"""
    pathway_id: str
    name: str
    description: Optional[str] = None
    pathway_class: Optional[str] = None
    organism: Optional[str] = None
    genes: List[Dict[str, Any]] = field(default_factory=list)
    compounds: List[Dict[str, Any]] = field(default_factory=list)
    drugs: List[Dict[str, Any]] = field(default_factory=list)
    modules: List[Dict[str, Any]] = field(default_factory=list)
    related_pathways: List[Dict[str, Any]] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


@dataclass
class KEGGCompound:
    """Represents a KEGG compound"""
    compound_id: str
    names: List[str] = field(default_factory=list)
    formula: Optional[str] = None
    exact_mass: Optional[float] = None
    molecular_weight: Optional[float] = None
    pathways: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class KEGGGene:
    """Represents a KEGG gene"""
    gene_id: str
    symbol: Optional[str] = None
    description: Optional[str] = None
    organism: Optional[str] = None
    pathways: List[Dict[str, Any]] = field(default_factory=list)
    orthology: List[Dict[str, Any]] = field(default_factory=list)


class KEGGClient(DatabaseClient):
    """
    Client for querying KEGG

    KEGG provides pathway maps, genes, compounds, drugs and the relationships
    between them. This client covers:
    - Pathway records (genes, compounds, drugs, cross-referenced maps)
    - Compound records (formula, mass, pathway membership)
    - Gene records
    - Free-text search across any KEGG database
    - ID conversion against outside namespaces (NCBI, UniProt, PubChem, ChEBI)
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize KEGG client

        Args:
            config: Custom configuration
        """
        super().__init__(config or self.get_default_config())

    @classmethod
    def get_default_config(cls) -> DatabaseConfig:
        """Get default KEGG configuration"""
        return DatabaseConfig(
            endpoint="https://rest.kegg.jp",
            rate_limit=3,  # KEGG asks for modest request rates
            timeout=30,
            retries=3,
            cache_enabled=True,
            cache_ttl=86400,
            headers={
                "Accept": "text/plain",
                "User-Agent": "DrugDiscoveryMCP/0.1.0"
            }
        )

    def get_name(self) -> str:
        """Get the name of this database"""
        return "KEGG"

    # --- transport ------------------------------------------------------------

    def _get_text(self, path: str) -> str:
        """
        Fetch a KEGG endpoint as text

        Args:
            path: Path below the API root, e.g. "get/hsa04110"

        Returns:
            Response body

        Raises:
            DatabaseError: If the request fails or the entry is empty
        """
        response = self._request_raw("GET", f"{self.config.endpoint}/{path}")
        text = response.text

        if not text.strip():
            raise DatabaseError(f"KEGG returned no data for {path}")
        return text

    # --- flat-file parsing ----------------------------------------------------

    @staticmethod
    def _parse_record(text: str) -> Dict[str, List[str]]:
        """
        Split a KEGG flat record into ``section name -> content lines``

        Continuation lines are blank in the leading fixed-width column and are
        appended to the section that opened above them.
        """
        sections: Dict[str, List[str]] = {}
        current = None

        for line in text.splitlines():
            if not line or line.startswith("///"):
                continue

            head, content = line[:_FIELD_WIDTH].strip(), line[_FIELD_WIDTH:].strip()

            if head:
                current = head
                sections.setdefault(current, [])
                if content:
                    sections[current].append(content)
            elif current and content:
                sections[current].append(content)

        return sections

    @staticmethod
    def _split_entry(line: str) -> Dict[str, Any]:
        """
        Split a "<id>  <label>" list line into its two halves

        KEGG separates the identifier from the human-readable label with
        whitespace, and appends bracketed annotations such as [KO:K02206].
        """
        parts = line.split(None, 1)
        entry_id = parts[0] if parts else ""
        label = parts[1].strip() if len(parts) > 1 else ""

        annotations = []
        while label.endswith("]") and "[" in label:
            start = label.rindex("[")
            annotations.insert(0, label[start + 1:-1])
            label = label[:start].strip()

        # Gene lines read "SYMBOL; description"
        symbol, _, description = label.partition(";")

        return {
            "id": entry_id,
            "name": label,
            "symbol": symbol.strip() if description else None,
            "description": description.strip() or None,
            "annotations": annotations,
        }

    def _entries(self, sections: Dict[str, List[str]], name: str) -> List[Dict[str, Any]]:
        """Parse one section of the record into a list of entries"""
        return [self._split_entry(line) for line in sections.get(name, []) if line]

    # --- queries --------------------------------------------------------------

    def query_pathway(self, pathway_id: str) -> KEGGPathway:
        """
        Query KEGG for a pathway

        Args:
            pathway_id: KEGG pathway ID, e.g. "hsa04110" or "map04110"

        Returns:
            KEGGPathway with the parsed record
        """
        try:
            sections = self._parse_record(self._get_text(f"get/{pathway_id}"))

            return KEGGPathway(
                pathway_id=self._first(sections, "ENTRY").split()[0] if sections.get("ENTRY") else pathway_id,
                name=self._first(sections, "NAME"),
                description=" ".join(sections.get("DESCRIPTION", [])) or None,
                pathway_class=self._first(sections, "CLASS") or None,
                organism=self._first(sections, "ORGANISM") or None,
                genes=self._entries(sections, "GENE"),
                compounds=self._entries(sections, "COMPOUND"),
                drugs=self._entries(sections, "DRUG"),
                modules=self._entries(sections, "MODULE"),
                related_pathways=self._entries(sections, "REL_PATHWAY"),
                references=[
                    line for line in sections.get("REFERENCE", []) if line.startswith("PMID")
                ],
            )
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"KEGG pathway query failed for {pathway_id}: {e}")
            raise DatabaseError(f"Failed to query KEGG pathway {pathway_id}: {e}")

    def query_compound(self, compound_id: str) -> KEGGCompound:
        """
        Query KEGG for a compound

        Args:
            compound_id: KEGG compound ID, e.g. "C00002"

        Returns:
            KEGGCompound with the parsed record
        """
        try:
            sections = self._parse_record(self._get_text(f"get/{compound_id}"))
            names = [n.rstrip(";") for n in sections.get("NAME", [])]

            return KEGGCompound(
                compound_id=self._first(sections, "ENTRY").split()[0] if sections.get("ENTRY") else compound_id,
                names=names,
                formula=self._first(sections, "FORMULA") or None,
                exact_mass=self._number(self._first(sections, "EXACT_MASS")),
                molecular_weight=self._number(self._first(sections, "MOL_WEIGHT")),
                pathways=self._entries(sections, "PATHWAY"),
            )
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"KEGG compound query failed for {compound_id}: {e}")
            raise DatabaseError(f"Failed to query KEGG compound {compound_id}: {e}")

    def query_gene(self, gene_id: str) -> KEGGGene:
        """
        Query KEGG for a gene

        Args:
            gene_id: KEGG gene ID, e.g. "hsa:7157"

        Returns:
            KEGGGene with the parsed record
        """
        try:
            sections = self._parse_record(self._get_text(f"get/{gene_id}"))
            symbol = self._first(sections, "SYMBOL") or None

            return KEGGGene(
                gene_id=self._first(sections, "ENTRY").split()[0] if sections.get("ENTRY") else gene_id,
                symbol=symbol,
                description=self._first(sections, "NAME") or self._first(sections, "DEFINITION") or None,
                organism=self._first(sections, "ORGANISM") or None,
                pathways=self._entries(sections, "PATHWAY"),
                orthology=self._entries(sections, "ORTHOLOGY"),
            )
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"KEGG gene query failed for {gene_id}: {e}")
            raise DatabaseError(f"Failed to query KEGG gene {gene_id}: {e}")

    def search(self, database: str, query: str, limit: int = 20) -> List[Dict[str, str]]:
        """
        Search a KEGG database by keyword

        Args:
            database: KEGG database name ("pathway", "compound", "genes", "drug", ...)
            query: Search keywords
            limit: Maximum number of results

        Returns:
            List of ``{"id": ..., "name": ...}`` records
        """
        from urllib.parse import quote

        try:
            text = self._get_text(f"find/{database}/{quote(query)}")
        except DatabaseError:
            # KEGG answers an empty result set with a blank body
            return []

        results = []
        for line in text.splitlines()[:limit]:
            if not line.strip():
                continue
            entry_id, _, name = line.partition("\t")
            results.append({"id": entry_id.strip(), "name": name.strip()})
        return results

    def convert(self, target: str, source_id: str) -> List[str]:
        """
        Convert an identifier between KEGG and an outside namespace

        Args:
            target: Target namespace, e.g. "hsa", "ncbi-geneid", "uniprot", "pubchem"
            source_id: Identifier to convert, e.g. "hsa:7157" or "ncbi-geneid:7157"

        Returns:
            List of identifiers in the target namespace
        """
        try:
            text = self._get_text(f"conv/{target}/{source_id}")
        except DatabaseError:
            return []

        converted = []
        for line in text.splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                converted.append(parts[1].strip())
        return converted

    def get_pathways_for_gene(self, gene_id: str) -> List[Dict[str, Any]]:
        """
        List the pathways a gene participates in

        Args:
            gene_id: KEGG gene ID, e.g. "hsa:7157"

        Returns:
            List of pathway entries
        """
        return self.query_gene(gene_id).pathways

    # --- helpers --------------------------------------------------------------

    @staticmethod
    def _first(sections: Dict[str, List[str]], name: str) -> str:
        """Read the first line of a section, or an empty string"""
        lines = sections.get(name) or []
        return lines[0] if lines else ""

    @staticmethod
    def _number(value: str) -> Optional[float]:
        """Parse a numeric field, tolerating absent or non-numeric values"""
        if not value:
            return None
        try:
            return float(value.split()[0])
        except (ValueError, IndexError):
            return None


# Singleton instance
kegg_client = KEGGClient()


# Convenience functions for direct use
def query_kegg(pathway_id: str, **kwargs) -> Dict[str, Any]:
    """Query KEGG database for pathway information"""
    try:
        pathway = kegg_client.query_pathway(pathway_id, **kwargs)
        return pathway.__dict__
    except DatabaseError as e:
        return {"error": str(e), "pathway_id": pathway_id}


def search_kegg(database: str, query: str, **kwargs) -> Dict[str, Any]:
    """Search a KEGG database"""
    try:
        return {
            "database": database,
            "query": query,
            "results": kegg_client.search(database, query, **kwargs),
        }
    except DatabaseError as e:
        return {"error": str(e), "database": database, "query": query}
