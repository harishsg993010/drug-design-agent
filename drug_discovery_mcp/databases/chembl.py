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
        url = f"{self.config.endpoint}/molecule/{compound_id}.json"
        
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
        url = f"{self.config.endpoint}/molecule.json"

        # ChEMBL has no free-text "query" parameter; it filters per field.
        params = {
            "pref_name__icontains": query,
            "limit": limit,
            "offset": offset,
        }

        try:
            data = self._make_request("GET", url, params=params)
            molecules = data.get("molecules", [])

            # Fall back to a synonym search when nothing matches the preferred name
            if not molecules:
                data = self._make_request("GET", url, params={
                    "molecule_synonyms__molecule_synonym__icontains": query,
                    "limit": limit,
                    "offset": offset,
                })
                molecules = data.get("molecules", [])

            page_meta = data.get("page_meta", {}) or {}
            return {
                "results": [self._parse_compound_summary(item) for item in molecules],
                "total": page_meta.get("total_count", len(molecules)),
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
        url = f"{self.config.endpoint}/target/{target_id}.json"
        
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
        url = f"{self.config.endpoint}/target.json"
        
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
                "total": (data.get("page_meta", {}) or {}).get("total_count", 0),
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
        url = f"{self.config.endpoint}/assay/{assay_id}.json"
        
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
        url = f"{self.config.endpoint}/activity.json"
        
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
        # ChEMBL exposes no molecule->targets route; targets are reached
        # through the activities that link the two.
        try:
            url = f"{self.config.endpoint}/activity.json"
            data = self._make_request("GET", url, params={
                "molecule_chembl_id": compound_id,
                "limit": 1000,
            })

            targets = {}
            for activity in data.get("activities", []) or []:
                target_id = activity.get("target_chembl_id")
                if not target_id or target_id in targets:
                    continue
                targets[target_id] = {
                    "target_chembl_id": target_id,
                    "target_pref_name": activity.get("target_pref_name"),
                    "target_organism": activity.get("target_organism"),
                }
            return list(targets.values())
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
        # ChEMBL exposes no target->molecules route; go through activities.
        try:
            url = f"{self.config.endpoint}/activity.json"
            data = self._make_request("GET", url, params={
                "target_chembl_id": target_id,
                "limit": 1000,
            })

            compounds = {}
            for activity in data.get("activities", []) or []:
                compound_id = activity.get("molecule_chembl_id")
                if not compound_id or compound_id in compounds:
                    continue
                compounds[compound_id] = {
                    "molecule_chembl_id": compound_id,
                    "pref_name": activity.get("molecule_pref_name"),
                    "canonical_smiles": activity.get("canonical_smiles"),
                }
            return list(compounds.values())
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
        # The similarity resource is addressed as /similarity/{smiles}/{percent},
        # so the compound has to be resolved to a structure first.
        try:
            compound = self.query_compound(compound_id)
            if not compound.smiles:
                raise DatabaseError(f"No structure available for {compound_id}")

            percent = int(round(similarity * 100))
            url = f"{self.config.endpoint}/similarity/{compound.smiles}/{percent}.json"
            data = self._make_request("GET", url)

            return [
                self._parse_compound_summary(m)
                for m in data.get("molecules", []) or []
                if m.get("molecule_chembl_id") != compound_id
            ]
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Failed to get similar compounds for {compound_id}: {e}")
            raise DatabaseError(f"Failed to get similar compounds: {e}")
    
    @staticmethod
    def _num(value: Any, cast=float) -> Optional[Any]:
        """
        Coerce a ChEMBL numeric field

        Most numeric properties come back as strings ("180.16"), so they are
        converted here rather than leaking strings into typed fields.
        """
        if value is None or value == "":
            return None
        try:
            return cast(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _synonyms(molecule: Dict[str, Any]) -> List[str]:
        """Collect distinct molecule synonyms"""
        names = []
        for entry in molecule.get("molecule_synonyms", []) or []:
            name = entry.get("molecule_synonym")
            if name and name not in names:
                names.append(name)
        return names

    def _parse_compound(self, data: Dict[str, Any]) -> ChEMBLCompound:
        """Parse compound data into ChEMBLCompound object"""
        molecule = data.get("molecule", data)
        structures = molecule.get("molecule_structures") or {}
        props = molecule.get("molecule_properties") or {}

        return ChEMBLCompound(
            compound_id=molecule.get("molecule_chembl_id", ""),
            smiles=structures.get("canonical_smiles"),
            inchi=structures.get("standard_inchi"),
            inchikey=structures.get("standard_inchi_key"),
            molecular_formula=props.get("full_molformula"),
            molecular_weight=self._num(props.get("full_mwt")),
            logp=self._num(props.get("alogp")),
            hba=self._num(props.get("hba"), int),
            hbd=self._num(props.get("hbd"), int),
            tpsa=self._num(props.get("psa")),
            rotatable_bonds=self._num(props.get("rtb"), int),
            heavy_atoms=self._num(props.get("heavy_atoms"), int),
            aromatic_rings=self._num(props.get("aromatic_rings"), int),
            fraction_csp3=None,  # not published in ChEMBL molecule_properties
            qed=self._num(props.get("qed_weighted")),
            pref_name=molecule.get("pref_name"),
            synonyms=self._synonyms(molecule),
            compound_type=molecule.get("molecule_type")
        )

    def _parse_compound_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a compound summary for search results"""
        structures = data.get("molecule_structures") or {}
        props = data.get("molecule_properties") or {}

        return {
            "compound_id": data.get("molecule_chembl_id", ""),
            "pref_name": data.get("pref_name") or "",
            "smiles": structures.get("canonical_smiles") or "",
            "molecular_weight": self._num(props.get("full_mwt")),
            "logp": self._num(props.get("alogp")),
            "max_phase": self._num(data.get("max_phase")),
            "similarity": self._num(data.get("similarity")),
        }
    
    def _parse_target(self, data: Dict[str, Any]) -> ChEMBLTarget:
        """Parse target data into ChEMBLTarget object"""
        target = data.get("target", data)
        components = target.get("target_components", []) or []

        synonyms, gene_name = [], None
        for component in components:
            for entry in component.get("target_component_synonyms", []) or []:
                name = entry.get("component_synonym")
                if name and name not in synonyms:
                    synonyms.append(name)
                if gene_name is None and entry.get("syn_type") == "GENE_SYMBOL":
                    gene_name = name

        accession = components[0].get("accession") if components else None
        tax_id = target.get("tax_id")

        return ChEMBLTarget(
            target_id=target.get("target_chembl_id", ""),
            target_type=target.get("target_type", ""),
            pref_name=target.get("pref_name", ""),
            organism=target.get("organism", ""),
            taxonomy=[str(tax_id)] if tax_id else [],
            synonyms=synonyms,
            gene_name=gene_name,
            protein_accession=accession
        )

    def _parse_target_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a target summary for search results"""
        components = data.get("target_components", []) or []

        return {
            "target_id": data.get("target_chembl_id", ""),
            "pref_name": data.get("pref_name") or "",
            "target_type": data.get("target_type") or "",
            "organism": data.get("organism") or "",
            "accession": components[0].get("accession") if components else None,
        }
    
    def _parse_assay(self, data: Dict[str, Any]) -> ChEMBLAssay:
        """Parse assay data into ChEMBLAssay object"""
        assay = data.get("assay", data)

        return ChEMBLAssay(
            assay_id=assay.get("assay_chembl_id", ""),
            assay_type=assay.get("assay_type_description") or assay.get("assay_type", ""),
            description=assay.get("description", ""),
            target_id=assay.get("target_chembl_id"),
            assay_format=assay.get("bao_format"),
            detection_technology=assay.get("assay_test_type"),
            num_compounds_tested=None,  # not published on the assay resource
            num_active_compounds=None
        )

    def _parse_bioactivity(self, data: Dict[str, Any]) -> BioactivityRecord:
        """Parse bioactivity data into BioactivityRecord object"""
        activity = data.get("activity", data)

        return BioactivityRecord(
            activity_id=str(activity.get("activity_id", "")),
            compound_id=activity.get("molecule_chembl_id", ""),
            target_id=activity.get("target_chembl_id", ""),
            assay_id=activity.get("assay_chembl_id", ""),
            type=activity.get("type") or activity.get("standard_type") or "",
            value=self._num(activity.get("value")) or 0.0,
            unit=activity.get("units") or "",
            relation=activity.get("relation"),
            pchembl_value=self._num(activity.get("pchembl_value")),
            standard_type=activity.get("standard_type"),
            standard_relation=activity.get("standard_relation"),
            standard_value=self._num(activity.get("standard_value")),
            standard_unit=activity.get("standard_units"),
            publication=activity.get("document_chembl_id"),
            year=self._num(activity.get("document_year"), int)
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
