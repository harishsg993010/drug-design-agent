"""
PubChem Database Client

Provides access to the PubChem PUG REST API for chemical compound information.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from urllib.parse import quote

from .base import DatabaseClient, DatabaseConfig, DatabaseError

logger = logging.getLogger(__name__)

# Properties requested for every compound lookup
_PROPERTIES = [
    "MolecularFormula",
    "MolecularWeight",
    "ExactMass",
    "SMILES",
    "ConnectivitySMILES",
    "InChI",
    "InChIKey",
    "IUPACName",
    "XLogP",
    "TPSA",
    "Charge",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "HeavyAtomCount",
]


@dataclass
class PubChemCompound:
    """Represents a PubChem compound"""
    compound_id: str  # CID
    name: Optional[str] = None
    iupac_name: Optional[str] = None
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    inchikey: Optional[str] = None
    formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    exact_mass: Optional[float] = None
    logp: Optional[float] = None
    tpsa: Optional[float] = None
    charge: Optional[int] = None
    hbd: Optional[int] = None
    hba: Optional[int] = None
    rotatable_bonds: Optional[int] = None
    heavy_atoms: Optional[int] = None
    synonyms: List[str] = field(default_factory=list)


class PubChemClient(DatabaseClient):
    """
    Client for querying PubChem

    PubChem is NCBI's public repository of chemical structures and their
    biological activities. This client covers:
    - Compound lookup by CID, name, SMILES or InChIKey
    - Computed physicochemical properties
    - Synonyms
    - Structure search (identity, similarity, substructure)
    - Bioassay activity summaries
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize PubChem client

        Args:
            config: Custom configuration
        """
        super().__init__(config or self.get_default_config())

    @classmethod
    def get_default_config(cls) -> DatabaseConfig:
        """Get default PubChem configuration"""
        return DatabaseConfig(
            endpoint="https://pubchem.ncbi.nlm.nih.gov/rest/pug",
            rate_limit=5,  # PubChem caps clients at 5 requests/second
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
        return "PubChem"

    # --- identifier resolution ------------------------------------------------

    def resolve_cid(self, identifier: str, namespace: str = "name") -> List[str]:
        """
        Resolve an identifier to PubChem CIDs

        Args:
            identifier: Value to resolve (compound name, SMILES, InChIKey, ...)
            namespace: PubChem namespace ("name", "smiles", "inchikey", "formula")

        Returns:
            List of matching CIDs, most relevant first
        """
        url = f"{self.config.endpoint}/compound/{namespace}/{quote(identifier, safe='')}/cids/JSON"

        try:
            data = self._make_request("GET", url)
            cids = (data.get("IdentifierList") or {}).get("CID") or []
            return [str(cid) for cid in cids]
        except DatabaseError as e:
            # PubChem answers an unknown identifier with 404
            if e.status_code == 404:
                return []
            raise

    # --- queries --------------------------------------------------------------

    def query_compound(self, compound_id: Union[str, int], namespace: str = "cid") -> PubChemCompound:
        """
        Query PubChem for a compound

        Args:
            compound_id: CID, or another identifier when `namespace` is given
            namespace: PubChem namespace ("cid", "name", "smiles", "inchikey")

        Returns:
            PubChemCompound with computed properties and synonyms
        """
        identifier = str(compound_id)

        try:
            if namespace != "cid":
                cids = self.resolve_cid(identifier, namespace)
                if not cids:
                    raise DatabaseError(f"No PubChem compound found for {identifier}")
                identifier = cids[0]

            properties = ",".join(_PROPERTIES)
            url = f"{self.config.endpoint}/compound/cid/{identifier}/property/{properties}/JSON"
            data = self._make_request("GET", url)

            rows = (data.get("PropertyTable") or {}).get("Properties") or []
            if not rows:
                raise DatabaseError(f"No properties returned for CID {identifier}")

            compound = self._parse_compound(rows[0])
            compound.synonyms = self.get_synonyms(identifier)
            if compound.synonyms:
                compound.name = compound.synonyms[0]
            return compound

        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"PubChem compound query failed for {compound_id}: {e}")
            raise DatabaseError(f"Failed to query PubChem compound {compound_id}: {e}")

    def get_synonyms(self, cid: Union[str, int], limit: int = 25) -> List[str]:
        """
        Get synonyms for a compound

        Args:
            cid: PubChem CID
            limit: Maximum number of synonyms

        Returns:
            List of synonyms
        """
        url = f"{self.config.endpoint}/compound/cid/{cid}/synonyms/JSON"

        try:
            data = self._make_request("GET", url)
            entries = (data.get("InformationList") or {}).get("Information") or []
            if not entries:
                return []
            return (entries[0].get("Synonym") or [])[:limit]
        except DatabaseError as e:
            if e.status_code == 404:
                return []
            logger.warning(f"Could not fetch synonyms for CID {cid}: {e}")
            return []

    def search_compounds(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search PubChem compounds by name

        Args:
            query: Compound name or partial name
            limit: Maximum number of results

        Returns:
            Dictionary with search results
        """
        try:
            cids = self.resolve_cid(query, "name")[:limit]

            return {
                "query": query,
                "total": len(cids),
                "results": [self._summary(cid) for cid in cids],
            }
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"PubChem compound search failed: {e}")
            raise DatabaseError(f"Failed to search PubChem compounds: {e}")

    def search_by_structure(
        self,
        smiles: str,
        search_type: str = "similarity",
        threshold: int = 90,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Search PubChem by chemical structure

        Args:
            smiles: Query structure as SMILES
            search_type: "similarity", "substructure" or "identity"
            threshold: Minimum Tanimoto similarity as a percentage (similarity only)
            limit: Maximum number of results

        Returns:
            Dictionary with search results
        """
        routes = {
            "similarity": "fastsimilarity_2d",
            "substructure": "fastsubstructure",
            "identity": "fastidentity",
        }
        if search_type not in routes:
            raise DatabaseError(
                f"Unknown search type: {search_type}. "
                f"Expected one of {', '.join(sorted(routes))}"
            )

        url = (
            f"{self.config.endpoint}/compound/{routes[search_type]}"
            f"/smiles/{quote(smiles, safe='')}/cids/JSON"
        )
        params: Dict[str, Any] = {"MaxRecords": limit}
        if search_type == "similarity":
            params["Threshold"] = threshold

        try:
            data = self._make_request("GET", url, params=params)
            cids = [str(c) for c in ((data.get("IdentifierList") or {}).get("CID") or [])]

            return {
                "query": smiles,
                "search_type": search_type,
                "total": len(cids),
                "results": [self._summary(cid) for cid in cids[:limit]],
            }
        except DatabaseError as e:
            if e.status_code == 404:
                return {"query": smiles, "search_type": search_type, "total": 0, "results": []}
            raise
        except Exception as e:
            logger.error(f"PubChem structure search failed: {e}")
            raise DatabaseError(f"Failed to search PubChem by structure: {e}")

    def get_assay_summary(self, cid: Union[str, int]) -> List[Dict[str, Any]]:
        """
        Get a summary of bioassay results for a compound

        Args:
            cid: PubChem CID

        Returns:
            List of assay summary records
        """
        url = f"{self.config.endpoint}/compound/cid/{cid}/assaysummary/JSON"

        try:
            data = self._make_request("GET", url)
            table = data.get("Table") or {}
            columns = table.get("Columns", {}).get("Column", []) or []
            rows = table.get("Row", []) or []

            return [
                dict(zip(columns, row.get("Cell", []) or []))
                for row in rows
            ]
        except DatabaseError as e:
            if e.status_code == 404:
                return []
            raise

    # --- parsing --------------------------------------------------------------

    def _summary(self, cid: str) -> Dict[str, Any]:
        """Fetch the compact property set used for search results"""
        properties = "MolecularFormula,MolecularWeight,SMILES,IUPACName,XLogP"
        url = f"{self.config.endpoint}/compound/cid/{cid}/property/{properties}/JSON"

        try:
            data = self._make_request("GET", url)
            rows = (data.get("PropertyTable") or {}).get("Properties") or []
        except DatabaseError as e:
            logger.warning(f"Could not fetch summary for CID {cid}: {e}")
            return {"compound_id": cid}

        row = rows[0] if rows else {}
        return {
            "compound_id": str(row.get("CID", cid)),
            "iupac_name": row.get("IUPACName"),
            "formula": row.get("MolecularFormula"),
            "molecular_weight": self._num(row.get("MolecularWeight")),
            "smiles": row.get("SMILES") or row.get("ConnectivitySMILES"),
            "logp": self._num(row.get("XLogP")),
        }

    def _parse_compound(self, data: Dict[str, Any]) -> PubChemCompound:
        """Parse a PubChem property row into a PubChemCompound"""
        return PubChemCompound(
            compound_id=str(data.get("CID", "")),
            iupac_name=data.get("IUPACName"),
            # PubChem now returns the stereo-aware structure as "SMILES"
            smiles=data.get("SMILES") or data.get("ConnectivitySMILES"),
            inchi=data.get("InChI"),
            inchikey=data.get("InChIKey"),
            formula=data.get("MolecularFormula"),
            molecular_weight=self._num(data.get("MolecularWeight")),
            exact_mass=self._num(data.get("ExactMass")),
            logp=self._num(data.get("XLogP")),
            tpsa=self._num(data.get("TPSA")),
            charge=self._num(data.get("Charge"), int),
            hbd=self._num(data.get("HBondDonorCount"), int),
            hba=self._num(data.get("HBondAcceptorCount"), int),
            rotatable_bonds=self._num(data.get("RotatableBondCount"), int),
            heavy_atoms=self._num(data.get("HeavyAtomCount"), int),
        )

    @staticmethod
    def _num(value: Any, cast=float) -> Optional[Any]:
        """
        Coerce a PubChem numeric field

        Masses come back as strings ("180.16"), so they are converted here
        rather than leaking strings into typed fields.
        """
        if value is None or value == "":
            return None
        try:
            return cast(value)
        except (TypeError, ValueError):
            return None


# Singleton instance
pubchem_client = PubChemClient()


# Convenience functions for direct use
def query_pubchem(compound_id: str, **kwargs) -> Dict[str, Any]:
    """Query PubChem database for compound information"""
    try:
        compound = pubchem_client.query_compound(compound_id, **kwargs)
        return compound.__dict__
    except DatabaseError as e:
        return {"error": str(e), "compound_id": compound_id}


def search_pubchem(query: str, **kwargs) -> Dict[str, Any]:
    """Search PubChem compounds by name"""
    try:
        return pubchem_client.search_compounds(query, **kwargs)
    except DatabaseError as e:
        return {"error": str(e), "query": query}
