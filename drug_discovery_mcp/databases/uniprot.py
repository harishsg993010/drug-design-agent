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
            endpoint="https://www.ebi.ac.uk/proteins/api",
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
        url = f"{self.config.endpoint}/proteins/{accession}"
        
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
        sort: str = "score"
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
        url = f"{self.config.endpoint}/proteins"
        
        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "sort": sort
        }
        
        if fields:
            params["fields"] = ",".join(fields)
        
        try:
            data = self._make_request("GET", url, params=params)
            return {
                "results": [self._parse_entry_summary(item) for item in data.get("results", [])],
                "total": data.get("total", 0),
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
        url = f"{self.config.endpoint}/proteins/{accession}/sequence"
        
        try:
            data = self._make_request("GET", url)
            return data.get("sequence", "")
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
        url = f"{self.config.endpoint}/proteins/{accession}/annotations"
        
        try:
            data = self._make_request("GET", url)
            # Filter for GO annotations
            go_annotations = []
            for annotation in data.get("results", []):
                if annotation.get("type") == "Gene Ontology":
                    go_annotations.append(annotation)
            return go_annotations
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
        url = f"{self.config.endpoint}/proteins/{accession}/pathways"
        
        try:
            data = self._make_request("GET", url)
            return data.get("results", [])
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
        url = f"{self.config.endpoint}/proteins/{accession}/diseases"
        
        try:
            data = self._make_request("GET", url)
            return data.get("results", [])
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
        url = f"{self.config.endpoint}/proteins/{accession}/interactions"
        
        try:
            data = self._make_request("GET", url)
            return data.get("results", [])
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
        url = f"{self.config.endpoint}/proteins/{accession}/keywords"
        
        try:
            data = self._make_request("GET", url)
            return [kw.get("value") for kw in data.get("results", [])]
        except Exception as e:
            logger.error(f"Failed to get keywords for {accession}: {e}")
            raise DatabaseError(f"Failed to get keywords: {e}")
    
    def _parse_entry(self, data: Dict[str, Any]) -> UniProtEntry:
        """Parse UniProt entry data into UniProtEntry object"""
        db_references = data.get("dbReferences", [])
        
        # Extract gene names
        gene_names = []
        for ref in db_references:
            if ref.get("type") == "GeneID" or ref.get("type") == "GenBank":
                gene_names.extend(ref.get("properties", []))
        
        # Extract sequence
        sequence = ""
        if "sequence" in data:
            sequence = data["sequence"].get("sequence", "")
        elif "sequences" in data and len(data["sequences"]) > 0:
            sequence = data["sequences"][0].get("sequence", "")
        
        return UniProtEntry(
            accession=data.get("accession", data.get("primaryAccession", "")),
            entry_name=data.get("entryName", ""),
            protein_name=data.get("protein", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
            gene_names=gene_names,
            organism=data.get("organism", {}).get("scientificName", ""),
            organism_id=data.get("organism", {}).get("taxonId", 0),
            taxonomy=data.get("organism", {}).get("lineage", []),
            sequence=sequence,
            length=len(sequence),
            molecular_weight=data.get("protein", {}).get("molecularWeight", None),
            pi=data.get("protein", {}).get("pi", None),
            function=self._extract_function(data),
            pathways=data.get("pathways", []),
            go_annotations=data.get("goAnnotations", []),
            keywords=data.get("keywords", []),
            features=data.get("features", []),
            diseases=data.get("diseases", []),
            interactions=data.get("interactions", []),
            references=data.get("references", [])
        )
    
    def _parse_entry_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a summary entry from search results"""
        return {
            "accession": data.get("primaryAccession", ""),
            "entry_name": data.get("entryName", ""),
            "protein_name": data.get("protein", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
            "gene_names": data.get("genes", [{}])[0].get("geneName", {}).get("value", ""),
            "organism": data.get("organism", {}).get("scientificName", ""),
            "score": data.get("score", 0),
            "length": data.get("sequence", {}).get("length", 0)
        }
    
    def _extract_function(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract protein function from annotations"""
        comments = data.get("comments", [])
        for comment in comments:
            if comment.get("type") == "FUNCTION":
                return comment.get("text", [{}])[0].get("value", "")
        return None


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
