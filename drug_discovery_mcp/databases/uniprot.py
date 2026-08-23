"""
UniProt Database Client

Provides access to UniProt protein database for sequence, function, and annotation data.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from .base import DatabaseClient, DatabaseConfig, DatabaseError

logger = logging.getLogger(__name__)


@dataclass
class UniProtEntry:
    """Represents a UniProt protein entry"""
    accession: str
    entry_name: str
    protein_name: str
    gene_names: List[str]
    organism: str
    organism_id: int
    taxonomy: List[str]
    sequence: str
    length: int
    molecular_weight: Optional[float] = None
    pi: Optional[float] = None  # Isoelectric point
    function: Optional[str] = None
    pathways: List[Dict[str, Any]] = None
    go_annotations: List[Dict[str, Any]] = None
    keywords: List[str] = None
    features: List[Dict[str, Any]] = None
    diseases: List[Dict[str, Any]] = None
    interactions: List[Dict[str, Any]] = None
    references: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.pathways is None:
            self.pathways = []
        if self.go_annotations is None:
            self.go_annotations = []
        if self.keywords is None:
            self.keywords = []
        if self.features is None:
            self.features = []
        if self.diseases is None:
            self.diseases = []
        if self.interactions is None:
            self.interactions = []
        if self.references is None:
            self.references = []


class UniProtClient(DatabaseClient):
    """
    Client for querying UniProt database
    
    UniProt is a comprehensive resource for protein sequence and functional information.
    It provides:
    - Protein sequences
    - Functional annotations
    - Gene ontology terms
    - Pathway information
    - Disease associations
    - Protein-protein interactions
    - Post-translational modifications
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize UniProt client
        
        Args:
            config: Custom configuration
        """
        super().__init__(config or self.get_default_config())
    
    @classmethod
    def get_default_config(cls) -> DatabaseConfig:
        """Get default UniProt configuration"""
        return DatabaseConfig(
            endpoint="https://rest.uniprot.org/uniprotkb",
            rate_limit=10,
            timeout=30,
            retries=3,
            cache_enabled=True,
            cache_ttl=3600,
            headers={
                "Accept": "application/json",
                "User-Agent": "DrugDiscoveryMCP/0.1.0"
            }
        )
    
    def get_name(self) -> str:
        """Get the name of this database"""
        return "UniProt"
    
    def query(
        self,
        accession: str,
        fields: Optional[List[str]] = None,
        format: str = "json"
    ) -> UniProtEntry:
        """
        Query UniProt for a specific protein entry
        
        Args:
            accession: UniProt accession number (e.g., "P12345")
            fields: List of fields to include (None for all)
            format: Response format ("json" or "xml")
            
        Returns:
            UniProtEntry object with protein information
            
        Raises:
            DatabaseError: If the query fails
        """
        url = f"{self.config.endpoint}/{accession}"
        
        if fields:
            url += f"?fields={','.join(fields)}"
        
        try:
            data = self._make_request("GET", url)
            return self._parse_entry(data)
        except Exception as e:
            logger.error(f"UniProt query failed for {accession}: {e}")
            raise DatabaseError(f"Failed to query UniProt for {accession}: {e}")
    
    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        fields: Optional[List[str]] = None,
        sort: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search UniProt database
        
        Args:
            query: Search query (e.g., "insulin AND human")
            limit: Maximum number of results
            offset: Pagination offset
            fields: Fields to include in results
            sort: Sort by field ("score", "accession", etc.)
            
        Returns:
            Dictionary with search results
        """
        url = f"{self.config.endpoint}/search"

        # UniProt paginates by cursor, not offset, so fetch through the offset
        # and slice locally.
        params = {
            "query": query,
            "size": min(500, offset + limit),
        }

        # The API rejects an unknown sort field; "score" means relevance, which
        # is already the default ordering, so it is simply omitted.
        if sort and sort != "score":
            params["sort"] = sort

        if fields:
            params["fields"] = ",".join(fields)

        try:
            response = self._request_raw("GET", url, params=params)
            data = response.json()
            results = data.get("results", [])[offset:offset + limit]

            return {
                "results": [self._parse_entry_summary(item) for item in results],
                "total": int(response.headers.get("X-Total-Results", len(results))),
                "limit": limit,
                "offset": offset,
                "query": query
            }
        except Exception as e:
            logger.error(f"UniProt search failed: {e}")
            raise DatabaseError(f"Failed to search UniProt: {e}")
    
    def get_sequence(self, accession: str) -> str:
        """
        Get protein sequence for a UniProt accession
        
        Args:
            accession: UniProt accession number
            
        Returns:
            Protein sequence as string
        """
        try:
            entry = self._make_request("GET", f"{self.config.endpoint}/{accession}")
            return (entry.get("sequence") or {}).get("value", "")
        except Exception as e:
            logger.error(f"Failed to get sequence for {accession}: {e}")
            raise DatabaseError(f"Failed to get sequence: {e}")
    
    def get_go_annotations(self, accession: str) -> List[Dict[str, Any]]:
        """
        Get Gene Ontology annotations for a protein
        
        Args:
            accession: UniProt accession number
            
        Returns:
            List of GO annotations
        """
        try:
            entry = self._make_request("GET", f"{self.config.endpoint}/{accession}")
            return self._extract_go_annotations(entry)
        except Exception as e:
            logger.error(f"Failed to get GO annotations for {accession}: {e}")
            raise DatabaseError(f"Failed to get GO annotations: {e}")
    
    def get_pathways(self, accession: str) -> List[Dict[str, Any]]:
        """
        Get pathway information for a protein
        
        Args:
            accession: UniProt accession number
            
        Returns:
            List of pathway annotations
        """
        try:
            entry = self._make_request("GET", f"{self.config.endpoint}/{accession}")
            return self._extract_pathways(entry)
        except Exception as e:
            logger.error(f"Failed to get pathways for {accession}: {e}")
            raise DatabaseError(f"Failed to get pathways: {e}")
    
    def get_diseases(self, accession: str) -> List[Dict[str, Any]]:
        """
        Get disease associations for a protein
        
        Args:
            accession: UniProt accession number
            
        Returns:
            List of disease annotations
        """
        try:
            entry = self._make_request("GET", f"{self.config.endpoint}/{accession}")
            return self._extract_diseases(entry)
        except Exception as e:
            logger.error(f"Failed to get diseases for {accession}: {e}")
            raise DatabaseError(f"Failed to get diseases: {e}")
    
    def get_interactions(self, accession: str) -> List[Dict[str, Any]]:
        """
        Get protein-protein interactions for a protein
        
        Args:
            accession: UniProt accession number
            
        Returns:
            List of protein interactions
        """
        try:
            entry = self._make_request("GET", f"{self.config.endpoint}/{accession}")
            return self._extract_interactions(entry)
        except Exception as e:
            logger.error(f"Failed to get interactions for {accession}: {e}")
            raise DatabaseError(f"Failed to get interactions: {e}")
    
    def get_keywords(self, accession: str) -> List[str]:
        """
        Get keywords for a protein
        
        Args:
            accession: UniProt accession number
            
        Returns:
            List of keywords
        """
        try:
            entry = self._make_request("GET", f"{self.config.endpoint}/{accession}")
            return self._extract_keywords(entry)
        except Exception as e:
            logger.error(f"Failed to get keywords for {accession}: {e}")
            raise DatabaseError(f"Failed to get keywords: {e}")
    
    @staticmethod
    def _protein_name(data: Dict[str, Any]) -> str:
        """Read the recommended (or first submitted) protein name"""
        description = data.get("proteinDescription", {}) or {}
        recommended = description.get("recommendedName") or {}
        name = (recommended.get("fullName") or {}).get("value")
        if name:
            return name

        submitted = description.get("submissionNames") or []
        if submitted:
            return (submitted[0].get("fullName") or {}).get("value", "")
        return ""

    @staticmethod
    def _gene_names(data: Dict[str, Any]) -> List[str]:
        """Collect gene names and their synonyms"""
        names = []
        for gene in data.get("genes", []) or []:
            primary = (gene.get("geneName") or {}).get("value")
            if primary and primary not in names:
                names.append(primary)
            for synonym in gene.get("synonyms", []) or []:
                value = synonym.get("value")
                if value and value not in names:
                    names.append(value)
        return names

    def _parse_entry(self, data: Dict[str, Any]) -> UniProtEntry:
        """Parse UniProt entry data into UniProtEntry object"""
        organism = data.get("organism", {}) or {}
        sequence_block = data.get("sequence", {}) or {}
        sequence = sequence_block.get("value", "")

        return UniProtEntry(
            accession=data.get("primaryAccession", ""),
            entry_name=data.get("uniProtkbId", ""),
            protein_name=self._protein_name(data),
            gene_names=self._gene_names(data),
            organism=organism.get("scientificName", ""),
            organism_id=organism.get("taxonId", 0),
            taxonomy=organism.get("lineage", []) or [],
            sequence=sequence,
            length=sequence_block.get("length", len(sequence)),
            molecular_weight=sequence_block.get("molWeight"),
            pi=None,  # UniProt does not publish isoelectric point
            function=self._extract_function(data),
            pathways=self._extract_pathways(data),
            go_annotations=self._extract_go_annotations(data),
            keywords=self._extract_keywords(data),
            features=data.get("features", []) or [],
            diseases=self._extract_diseases(data),
            interactions=self._extract_interactions(data),
            references=data.get("references", []) or []
        )

    def _parse_entry_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a summary entry from search results"""
        return {
            "accession": data.get("primaryAccession", ""),
            "entry_name": data.get("uniProtkbId", ""),
            "protein_name": self._protein_name(data),
            "gene_names": self._gene_names(data),
            "organism": (data.get("organism", {}) or {}).get("scientificName", ""),
            "length": (data.get("sequence", {}) or {}).get("length", 0),
            "annotation_score": data.get("annotationScore"),
        }

    @staticmethod
    def _comment_texts(comment: Dict[str, Any]) -> str:
        """Join the free-text blocks of a comment into one string"""
        return " ".join(
            t.get("value", "") for t in comment.get("texts", []) or [] if t.get("value")
        )

    def _extract_function(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract protein function from annotations"""
        for comment in data.get("comments", []) or []:
            if comment.get("commentType") == "FUNCTION":
                text = self._comment_texts(comment)
                if text:
                    return text
        return None

    def _extract_diseases(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract disease associations from DISEASE comments

        A DISEASE comment carries a named disease, a free-text note, or both;
        the note lives under ``note.texts`` rather than directly on the comment.
        """
        diseases = []
        for comment in data.get("comments", []) or []:
            if comment.get("commentType") != "DISEASE":
                continue

            disease = comment.get("disease") or {}
            note = self._comment_texts(comment.get("note") or {})

            if not disease and not note:
                continue

            diseases.append({
                "disease_id": disease.get("diseaseId"),
                "disease_accession": disease.get("diseaseAccession"),
                "acronym": disease.get("acronym"),
                "description": disease.get("description"),
                "cross_reference": disease.get("diseaseCrossReference"),
                "note": note or None,
            })
        return diseases

    def _extract_interactions(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract binary interactions from INTERACTION comments"""
        interactions = []
        for comment in data.get("comments", []) or []:
            if comment.get("commentType") != "INTERACTION":
                continue
            for interaction in comment.get("interactions", []) or []:
                interactions.append({
                    "interactant_one": (interaction.get("interactantOne") or {}).get("uniProtKBAccession"),
                    "interactant_two": (interaction.get("interactantTwo") or {}).get("uniProtKBAccession"),
                    "gene_name": (interaction.get("interactantTwo") or {}).get("geneName"),
                    "experiments": interaction.get("numberOfExperiments"),
                    "organisms_differ": interaction.get("organismDiffer"),
                })
        return interactions

    @staticmethod
    def _cross_references(data: Dict[str, Any], databases) -> List[Dict[str, Any]]:
        """Collect cross-references belonging to any of `databases`"""
        refs = []
        for ref in data.get("uniProtKBCrossReferences", []) or []:
            if ref.get("database") not in databases:
                continue
            properties = {
                prop.get("key"): prop.get("value")
                for prop in ref.get("properties", []) or []
            }
            refs.append({
                "database": ref.get("database"),
                "id": ref.get("id"),
                "properties": properties,
            })
        return refs

    def _extract_go_annotations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract Gene Ontology annotations from cross-references"""
        annotations = []
        for ref in self._cross_references(data, {"GO"}):
            term = ref["properties"].get("GoTerm", "")
            aspect, _, name = term.partition(":")
            annotations.append({
                "id": ref["id"],
                "aspect": aspect,  # C (component), F (function), P (process)
                "term": name or term,
                "evidence": ref["properties"].get("GoEvidenceType"),
            })
        return annotations

    def _extract_pathways(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract pathway memberships from cross-references"""
        pathways = []
        for ref in self._cross_references(data, {"Reactome", "KEGG", "UniPathway"}):
            pathways.append({
                "database": ref["database"],
                "id": ref["id"],
                "name": ref["properties"].get("PathwayName"),
            })
        return pathways

    @staticmethod
    def _extract_keywords(data: Dict[str, Any]) -> List[str]:
        """Extract keyword names"""
        return [
            kw.get("name", "")
            for kw in data.get("keywords", []) or []
            if kw.get("name")
        ]


# Singleton instance
uniprot_client = UniProtClient()


# Convenience functions for direct use
def query_uniprot(accession: str, **kwargs) -> Dict[str, Any]:
    """Query UniProt database"""
    try:
        entry = uniprot_client.query(accession, **kwargs)
        return entry.__dict__
    except DatabaseError as e:
        return {"error": str(e), "accession": accession}


def search_uniprot(query: str, **kwargs) -> Dict[str, Any]:
    """Search UniProt database"""
    try:
        return uniprot_client.search(query, **kwargs)
    except DatabaseError as e:
        return {"error": str(e), "query": query}
