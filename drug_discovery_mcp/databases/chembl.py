"""
ChEMBL Database Client

Provides access to ChEMBL database for bioactivity data and drug-like properties.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from .base import DatabaseClient, DatabaseConfig, DatabaseError

logger = logging.getLogger(__name__)


@dataclass
class ChEMBLCompound:
    """Represents a ChEMBL compound"""
    compound_id: str
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    inchikey: Optional[str] = None
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None
    hba: Optional[int] = None  # Hydrogen bond acceptors
    hbd: Optional[int] = None  # Hydrogen bond donors
    tpsa: Optional[float] = None  # Topological polar surface area
    rotatable_bonds: Optional[int] = None
    heavy_atoms: Optional[int] = None
    aromatic_rings: Optional[int] = None
    fraction_csp3: Optional[float] = None
    qed: Optional[float] = None  # Quantitative estimate of drug-likeness
    
    # Additional properties
    pref_name: Optional[str] = None
    synonyms: List[str] = None
    compound_type: Optional[str] = None
    
    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = []


@dataclass
class ChEMBLTarget:
    """Represents a ChEMBL target"""
    target_id: str
    target_type: str
    pref_name: str
    organism: str
    taxonomy: List[str] = None
    synonyms: List[str] = None
    gene_name: Optional[str] = None
    protein_accession: Optional[str] = None
    
    def __post_init__(self):
        if self.taxonomy is None:
            self.taxonomy = []
        if self.synonyms is None:
            self.synonyms = []


@dataclass
class ChEMBLAssay:
    """Represents a ChEMBL assay"""
    assay_id: str
    assay_type: str
    description: str
    target_id: Optional[str] = None
    
    # Additional metadata
    assay_format: Optional[str] = None
    detection_technology: Optional[str] = None
    
    # Statistics
    num_compounds_tested: Optional[int] = None
    num_active_compounds: Optional[int] = None


@dataclass
class BioactivityRecord:
    """Represents a bioactivity measurement"""
    activity_id: str
    compound_id: str
    target_id: str
    assay_id: str
    
    # Activity data
    type: str  # IC50, Ki, EC50, etc.
    value: float
    unit: str
    relation: Optional[str] = None  # =, <, >, etc.
    
    # Additional metadata
    pchembl_value: Optional[float] = None  # Negative log of activity value
    standard_type: Optional[str] = None
    standard_relation: Optional[str] = None
    standard_value: Optional[float] = None
    standard_unit: Optional[str] = None
    
    # Reference
    publication: Optional[str] = None
    year: Optional[int] = None


class ChEMBLClient(DatabaseClient):
    """
    Client for querying ChEMBL database
    
    ChEMBL is a large-scale bioactivity database for drug discovery.
    It provides:
    - Compound structures and properties
    - Target information
    - Bioactivity measurements
    - Assay descriptions
    - Drug-target interactions
    - ADMET properties
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize ChEMBL client
        
        Args:
            config: Custom configuration
        """
        super().__init__(config or self.get_default_config())
    
    @classmethod
    def get_default_config(cls) -> DatabaseConfig:
        """Get default ChEMBL configuration"""
        return DatabaseConfig(
            endpoint="https://www.ebi.ac.uk/chembl/api/data",
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
        return "ChEMBL"
    
    def query_compound(self, compound_id: str) -> ChEMBLCompound:
        """
        Query ChEMBL for a specific compound
        
        Args:
            compound_id: ChEMBL compound ID (e.g., "CHEMBL123")
            
        Returns:
            ChEMBLCompound object with compound information
            
        Raises:
            DatabaseError: If the query fails
        """
        url = f"{self.config.endpoint}/molecule/{compound_id}"
        
        try:
            data = self._make_request("GET", url)
            return self._parse_compound(data)
        except Exception as e:
            logger.error(f"ChEMBL compound query failed for {compound_id}: {e}")
            raise DatabaseError(f"Failed to query ChEMBL compound {compound_id}: {e}")
    
    def search_compounds(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Search ChEMBL compounds
        
        Args:
            query: Search query (SMILES, compound name, etc.)
            limit: Maximum number of results
            offset: Pagination offset
            format: Response format
            
        Returns:
            Dictionary with search results
        """
        url = f"{self.config.endpoint}/molecule"
        
        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "format": format
        }
        
        try:
            data = self._make_request("GET", url, params=params)
            return {
                "results": [self._parse_compound_summary(item) for item in data.get("molecules", [])],
                "total": data.get("total", 0),
                "limit": limit,
                "offset": offset
            }
        except Exception as e:
            logger.error(f"ChEMBL compound search failed: {e}")
            raise DatabaseError(f"Failed to search ChEMBL compounds: {e}")
    
    def query_target(self, target_id: str) -> ChEMBLTarget:
        """
        Query ChEMBL for a specific target
        
        Args:
            target_id: ChEMBL target ID (e.g., "CHEMBL1234")
            
        Returns:
            ChEMBLTarget object with target information
        """
        url = f"{self.config.endpoint}/target/{target_id}"
        
        try:
            data = self._make_request("GET", url)
            return self._parse_target(data)
        except Exception as e:
            logger.error(f"ChEMBL target query failed for {target_id}: {e}")
            raise DatabaseError(f"Failed to query ChEMBL target {target_id}: {e}")
    
    def search_targets(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        target_type: Optional[str] = None,
        organism: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search ChEMBL targets
        
        Args:
            query: Search query
            limit: Maximum number of results
            offset: Pagination offset
            target_type: Filter by target type (e.g., "SINGLE PROTEIN")
            organism: Filter by organism
            
        Returns:
            Dictionary with search results
        """
        url = f"{self.config.endpoint}/target"
        
        params = {
            "query": query,
            "limit": limit,
            "offset": offset
        }
        
        if target_type:
            params["target_type"] = target_type
        if organism:
            params["organism"] = organism
        
        try:
            data = self._make_request("GET", url, params=params)
            return {
                "results": [self._parse_target_summary(item) for item in data.get("targets", [])],
                "total": data.get("total", 0),
                "limit": limit,
                "offset": offset
            }
        except Exception as e:
            logger.error(f"ChEMBL target search failed: {e}")
            raise DatabaseError(f"Failed to search ChEMBL targets: {e}")
    
    def query_assay(self, assay_id: str) -> ChEMBLAssay:
        """
        Query ChEMBL for a specific assay
        
        Args:
            assay_id: ChEMBL assay ID
            
        Returns:
            ChEMBLAssay object with assay information
        """
        url = f"{self.config.endpoint}/assay/{assay_id}"
        
        try:
            data = self._make_request("GET", url)
            return self._parse_assay(data)
        except Exception as e:
            logger.error(f"ChEMBL assay query failed for {assay_id}: {e}")
            raise DatabaseError(f"Failed to query ChEMBL assay {assay_id}: {e}")
    
    def get_bioactivities(
        self,
        compound_id: Optional[str] = None,
        target_id: Optional[str] = None,
        assay_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        activity_type: Optional[str] = None
    ) -> List[BioactivityRecord]:
        """
        Get bioactivity records
        
        Args:
            compound_id: Filter by compound ID
            target_id: Filter by target ID
            assay_id: Filter by assay ID
            limit: Maximum number of results
            offset: Pagination offset
            activity_type: Filter by activity type (IC50, Ki, etc.)
            
        Returns:
            List of bioactivity records
        """
        url = f"{self.config.endpoint}/activity"
        
        params = {
            "limit": limit,
            "offset": offset
        }
        
        if compound_id:
            params["molecule_chembl_id"] = compound_id
        if target_id:
            params["target_chembl_id"] = target_id
        if assay_id:
            params["assay_chembl_id"] = assay_id
        if activity_type:
            params["type"] = activity_type
        
        try:
            data = self._make_request("GET", url, params=params)
            return [self._parse_bioactivity(item) for item in data.get("activities", [])]
        except Exception as e:
            logger.error(f"ChEMBL bioactivity query failed: {e}")
            raise DatabaseError(f"Failed to query ChEMBL bioactivities: {e}")
    
    def get_compound_targets(self, compound_id: str) -> List[Dict[str, Any]]:
        """
        Get targets for a specific compound
        
        Args:
            compound_id: ChEMBL compound ID
            
        Returns:
            List of target information
        """
        url = f"{self.config.endpoint}/molecule/{compound_id}/targets"
        
        try:
            data = self._make_request("GET", url)
            return data.get("targets", [])
        except Exception as e:
            logger.error(f"Failed to get targets for compound {compound_id}: {e}")
            raise DatabaseError(f"Failed to get compound targets: {e}")
    
    def get_target_compounds(self, target_id: str) -> List[Dict[str, Any]]:
        """
        Get compounds for a specific target
        
        Args:
            target_id: ChEMBL target ID
            
        Returns:
            List of compound information
        """
        url = f"{self.config.endpoint}/target/{target_id}/molecules"
        
        try:
            data = self._make_request("GET", url)
            return data.get("molecules", [])
        except Exception as e:
            logger.error(f"Failed to get compounds for target {target_id}: {e}")
            raise DatabaseError(f"Failed to get target compounds: {e}")
    
    def get_similar_compounds(self, compound_id: str, similarity: float = 0.9) -> List[Dict[str, Any]]:
        """
        Get similar compounds to a given compound
        
        Args:
            compound_id: ChEMBL compound ID
            similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of similar compounds
        """
        url = f"{self.config.endpoint}/molecule/{compound_id}/similarity"
        
        params = {"similarity": similarity}
        
        try:
            data = self._make_request("GET", url, params=params)
            return data.get("molecules", [])
        except Exception as e:
            logger.error(f"Failed to get similar compounds for {compound_id}: {e}")
            raise DatabaseError(f"Failed to get similar compounds: {e}")
    
    def _parse_compound(self, data: Dict[str, Any]) -> ChEMBLCompound:
        """Parse compound data into ChEMBLCompound object"""
        molecule = data.get("molecule", data)
        
        return ChEMBLCompound(
            compound_id=molecule.get("chembl_id", ""),
            smiles=molecule.get("smiles", None),
            inchi=molecule.get("inchi", None),
            inchikey=molecule.get("inchikey", None),
            molecular_formula=molecule.get("molecule_formula", None),
            molecular_weight=molecule.get("molecular_weight", None),
            logp=molecule.get("logp", None),
            hba=molecule.get("hba", None),
            hbd=molecule.get("hbd", None),
            tpsa=molecule.get("tpsa", None),
            rotatable_bonds=molecule.get("rotatable_bonds", None),
            heavy_atoms=molecule.get("heavy_atoms", None),
            aromatic_rings=molecule.get("aromatic_rings", None),
            fraction_csp3=molecule.get("fraction_csp3", None),
            qed=molecule.get("qed", None),
            pref_name=molecule.get("pref_name", None),
            synonyms=molecule.get("synonyms", []),
            compound_type=molecule.get("molecule_type", None)
        )
    
    def _parse_compound_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a compound summary for search results"""
        return {
            "compound_id": data.get("chembl_id", ""),
            "pref_name": data.get("pref_name", ""),
            "smiles": data.get("smiles", ""),
            "molecular_weight": data.get("molecular_weight", 0),
            "logp": data.get("logp", 0),
            "similarity": data.get("similarity", 0)
        }
    
    def _parse_target(self, data: Dict[str, Any]) -> ChEMBLTarget:
        """Parse target data into ChEMBLTarget object"""
        target = data.get("target", data)
        
        return ChEMBLTarget(
            target_id=target.get("chembl_id", ""),
            target_type=target.get("target_type", ""),
            pref_name=target.get("pref_name", ""),
            organism=target.get("organism", ""),
            taxonomy=target.get("taxonomy", []),
            synonyms=target.get("synonyms", []),
            gene_name=target.get("gene_name", None),
            protein_accession=target.get("protein_accession", None)
        )
    
    def _parse_target_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a target summary for search results"""
        return {
            "target_id": data.get("chembl_id", ""),
            "pref_name": data.get("pref_name", ""),
            "target_type": data.get("target_type", ""),
            "organism": data.get("organism", "")
        }
    
    def _parse_assay(self, data: Dict[str, Any]) -> ChEMBLAssay:
        """Parse assay data into ChEMBLAssay object"""
        assay = data.get("assay", data)
        
        return ChEMBLAssay(
            assay_id=assay.get("chembl_id", ""),
            assay_type=assay.get("assay_type", ""),
            description=assay.get("description", ""),
            target_id=assay.get("target_chembl_id", None),
            assay_format=assay.get("assay_format", None),
            detection_technology=assay.get("detection_technology", None),
            num_compounds_tested=assay.get("num_compounds_tested", None),
            num_active_compounds=assay.get("num_active_compounds", None)
        )
    
    def _parse_bioactivity(self, data: Dict[str, Any]) -> BioactivityRecord:
        """Parse bioactivity data into BioactivityRecord object"""
        activity = data.get("activity", data)
        
        return BioactivityRecord(
            activity_id=activity.get("chembl_id", ""),
            compound_id=activity.get("molecule_chembl_id", ""),
            target_id=activity.get("target_chembl_id", ""),
            assay_id=activity.get("assay_chembl_id", ""),
            type=activity.get("type", ""),
            value=activity.get("value", 0.0),
            unit=activity.get("unit", ""),
            relation=activity.get("relation", None),
            pchembl_value=activity.get("pchembl_value", None),
            standard_type=activity.get("standard_type", None),
            standard_relation=activity.get("standard_relation", None),
            standard_value=activity.get("standard_value", None),
            standard_unit=activity.get("standard_unit", None),
            publication=activity.get("publication", None),
            year=activity.get("year", None)
        )


# Singleton instance
chembl_client = ChEMBLClient()


# Convenience functions for direct use
def query_chembl(compound_id: str, **kwargs) -> Dict[str, Any]:
    """Query ChEMBL database for a compound"""
    try:
        compound = chembl_client.query_compound(compound_id, **kwargs)
        return compound.__dict__
    except DatabaseError as e:
        return {"error": str(e), "compound_id": compound_id}


def search_chembl_compounds(query: str, **kwargs) -> Dict[str, Any]:
    """Search ChEMBL compounds"""
    try:
        return chembl_client.search_compounds(query, **kwargs)
    except DatabaseError as e:
        return {"error": str(e), "query": query}
