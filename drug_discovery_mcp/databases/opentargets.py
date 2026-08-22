"""
OpenTargets Database Client

Provides access to OpenTargets platform for target validation and disease association data.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from .base import DatabaseClient, DatabaseConfig, DatabaseError

logger = logging.getLogger(__name__)


@dataclass
class Target:
    """Represents a target from OpenTargets"""
    id: str
    name: str
    gene_symbol: str
    gene_id: Optional[str] = None
    protein_id: Optional[str] = None
    target_type: str = "PROTEIN"
    
    # Additional metadata
    description: Optional[str] = None
    synonyms: List[str] = None
    
    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = []


@dataclass
class Disease:
    """Represents a disease from OpenTargets"""
    id: str
    name: str
    description: Optional[str] = None
    
    # Classification
    disease_type: Optional[str] = None
    therapeutic_areas: List[str] = None
    
    def __post_init__(self):
        if self.therapeutic_areas is None:
            self.therapeutic_areas = []


@dataclass
class Association:
    """Represents a target-disease association"""
    target_id: str
    disease_id: str
    score: float
    
    # Evidence
    evidence: List[Dict[str, Any]] = None
    
    # Additional metadata
    mechanism: Optional[str] = None
    action_type: Optional[str] = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


class OpenTargetsClient(DatabaseClient):
    """
    Client for querying OpenTargets platform
    
    OpenTargets is a platform for target validation and disease association analysis.
    It provides:
    - Target information
    - Disease information
    - Target-disease associations
    - Evidence scoring
    - Drug information
    - Clinical trial data
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize OpenTargets client
        
        Args:
            config: Custom configuration
        """
        super().__init__(config or self.get_default_config())
    
    @classmethod
    def get_default_config(cls) -> DatabaseConfig:
        """Get default OpenTargets configuration"""
        return DatabaseConfig(
            endpoint="https://api.opentargets.io/v3",
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
        return "OpenTargets"
    
    def query_target(self, target_id: str) -> Target:
        """
        Query OpenTargets for a specific target
        
        Args:
            target_id: OpenTargets target ID (e.g., "ENSG00000123456")
            
        Returns:
            Target object with target information
        """
        url = f"{self.config.endpoint}/target/{target_id}"
        
        try:
            data = self._make_request("GET", url)
            return self._parse_target(data)
        except Exception as e:
            logger.error(f"OpenTargets target query failed for {target_id}: {e}")
            raise DatabaseError(f"Failed to query OpenTargets target {target_id}: {e}")
    
    def search_targets(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        gene_symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search OpenTargets targets
        
        Args:
            query: Search query
            limit: Maximum number of results
            offset: Pagination offset
            gene_symbol: Filter by gene symbol
            
        Returns:
            Dictionary with search results
        """
        url = f"{self.config.endpoint}/target/search"
        
        params = {
            "query": query,
            "limit": limit,
            "offset": offset
        }
        
        if gene_symbol:
            params["gene_symbol"] = gene_symbol
        
        try:
            data = self._make_request("GET", url, params=params)
            return {
                "results": [self._parse_target_summary(item) for item in data.get("results", [])],
                "total": data.get("total", 0),
                "limit": limit,
                "offset": offset
            }
        except Exception as e:
            logger.error(f"OpenTargets target search failed: {e}")
            raise DatabaseError(f"Failed to search OpenTargets targets: {e}")
    
    def query_disease(self, disease_id: str) -> Disease:
        """
        Query OpenTargets for a specific disease
        
        Args:
            disease_id: OpenTargets disease ID (e.g., "MONDO_0005737")
            
        Returns:
            Disease object with disease information
        """
        url = f"{self.config.endpoint}/disease/{disease_id}"
        
        try:
            data = self._make_request("GET", url)
            return self._parse_disease(data)
        except Exception as e:
            logger.error(f"OpenTargets disease query failed for {disease_id}: {e}")
            raise DatabaseError(f"Failed to query OpenTargets disease {disease_id}: {e}")
    
    def search_diseases(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Search OpenTargets diseases
        
        Args:
            query: Search query
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            Dictionary with search results
        """
        url = f"{self.config.endpoint}/disease/search"
        
        params = {
            "query": query,
            "limit": limit,
            "offset": offset
        }
        
        try:
            data = self._make_request("GET", url, params=params)
            return {
                "results": [self._parse_disease_summary(item) for item in data.get("results", [])],
                "total": data.get("total", 0),
                "limit": limit,
                "offset": offset
            }
        except Exception as e:
            logger.error(f"OpenTargets disease search failed: {e}")
            raise DatabaseError(f"Failed to search OpenTargets diseases: {e}")
    
    def get_associations(
        self,
        target_id: Optional[str] = None,
        disease_id: Optional[str] = None,
        limit: int = 100,
        score_threshold: float = 0.1
    ) -> List[Association]:
        """
        Get target-disease associations
        
        Args:
            target_id: Filter by target ID
            disease_id: Filter by disease ID
            limit: Maximum number of results
            score_threshold: Minimum association score
            
        Returns:
            List of Association objects
        """
        url = f"{self.config.endpoint}/association/search"
        
        params = {
            "limit": limit,
            "score_threshold": score_threshold
        }
        
        if target_id:
            params["target"] = target_id
        if disease_id:
            params["disease"] = disease_id
        
        try:
            data = self._make_request("GET", url, params=params)
            return [self._parse_association(item) for item in data.get("results", [])]
        except Exception as e:
            logger.error(f"OpenTargets association query failed: {e}")
            raise DatabaseError(f"Failed to query OpenTargets associations: {e}")
    
    def get_target_diseases(self, target_id: str) -> List[Association]:
        """
        Get diseases associated with a specific target
        
        Args:
            target_id: OpenTargets target ID
            
        Returns:
            List of Association objects
        """
        return self.get_associations(target_id=target_id)
    
    def get_disease_targets(self, disease_id: str) -> List[Association]:
        """
        Get targets associated with a specific disease
        
        Args:
            disease_id: OpenTargets disease ID
            
        Returns:
            List of Association objects
        """
        return self.get_associations(disease_id=disease_id)
    
    def get_evidence(
        self,
        target_id: str,
        disease_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get evidence for a specific target-disease association
        
        Args:
            target_id: OpenTargets target ID
            disease_id: OpenTargets disease ID
            
        Returns:
            List of evidence records
        """
        url = f"{self.config.endpoint}/evidence/{target_id}/{disease_id}"
        
        try:
            data = self._make_request("GET", url)
            return data.get("evidence", [])
        except Exception as e:
            logger.error(f"OpenTargets evidence query failed: {e}")
            raise DatabaseError(f"Failed to query OpenTargets evidence: {e}")
    
    def get_drugs(self, target_id: str) -> List[Dict[str, Any]]:
        """
        Get drugs associated with a target
        
        Args:
            target_id: OpenTargets target ID
            
        Returns:
            List of drug information
        """
        url = f"{self.config.endpoint}/target/{target_id}/drugs"
        
        try:
            data = self._make_request("GET", url)
            return data.get("drugs", [])
        except Exception as e:
            logger.error(f"OpenTargets drugs query failed: {e}")
            raise DatabaseError(f"Failed to query OpenTargets drugs: {e}")
    
    def get_clinical_trials(self, target_id: str) -> List[Dict[str, Any]]:
        """
        Get clinical trials associated with a target
        
        Args:
            target_id: OpenTargets target ID
            
        Returns:
            List of clinical trial information
        """
        url = f"{self.config.endpoint}/target/{target_id}/clinical_trials"
        
        try:
            data = self._make_request("GET", url)
            return data.get("clinical_trials", [])
        except Exception as e:
            logger.error(f"OpenTargets clinical trials query failed: {e}")
            raise DatabaseError(f"Failed to query OpenTargets clinical trials: {e}")
    
    def _parse_target(self, data: Dict[str, Any]) -> Target:
        """Parse target data into Target object"""
        target = data.get("target", data)
        
        return Target(
            id=target.get("id", ""),
            name=target.get("name", ""),
            gene_symbol=target.get("gene_symbol", ""),
            gene_id=target.get("gene_id", None),
            protein_id=target.get("protein_id", None),
            target_type=target.get("target_type", "PROTEIN"),
            description=target.get("description", None),
            synonyms=target.get("synonyms", [])
        )
    
    def _parse_target_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a target summary for search results"""
        return {
            "id": data.get("id", ""),
            "name": data.get("name", ""),
            "gene_symbol": data.get("gene_symbol", ""),
            "target_type": data.get("target_type", ""),
            "score": data.get("score", 0)
        }
    
    def _parse_disease(self, data: Dict[str, Any]) -> Disease:
        """Parse disease data into Disease object"""
        disease = data.get("disease", data)
        
        return Disease(
            id=disease.get("id", ""),
            name=disease.get("name", ""),
            description=disease.get("description", None),
            disease_type=disease.get("disease_type", None),
            therapeutic_areas=disease.get("therapeutic_areas", [])
        )
    
    def _parse_disease_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a disease summary for search results"""
        return {
            "id": data.get("id", ""),
            "name": data.get("name", ""),
            "disease_type": data.get("disease_type", ""),
            "score": data.get("score", 0)
        }
    
    def _parse_association(self, data: Dict[str, Any]) -> Association:
        """Parse association data into Association object"""
        association = data.get("association", data)
        
        return Association(
            target_id=association.get("target_id", ""),
            disease_id=association.get("disease_id", ""),
            score=association.get("score", 0.0),
            evidence=association.get("evidence", []),
            mechanism=association.get("mechanism", None),
            action_type=association.get("action_type", None)
        )


# Singleton instance
opentargets_client = OpenTargetsClient()


# Convenience functions for direct use
def query_opentargets(target_id: str, **kwargs) -> Dict[str, Any]:
    """Query OpenTargets database"""
    try:
        target = opentargets_client.query_target(target_id, **kwargs)
        return target.__dict__
    except DatabaseError as e:
        return {"error": str(e), "target_id": target_id}


def search_opentargets(query: str, **kwargs) -> Dict[str, Any]:
    """Search OpenTargets database"""
    try:
        return opentargets_client.search_targets(query, **kwargs)
    except DatabaseError as e:
        return {"error": str(e), "query": query}
