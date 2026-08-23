"""
PDB Database Client

Provides access to Protein Data Bank (PDB) for 3D protein structure data.
"""

import logging
import gzip
import io
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

from .base import DatabaseClient, DatabaseConfig, DatabaseError

logger = logging.getLogger(__name__)


@dataclass
class PDBEntry:
    """Represents a PDB entry"""
    pdb_id: str
    title: str
    resolution: Optional[float] = None
    method: Optional[str] = None  # X-RAY DIFFRACTION, NMR, etc.
    chains: List[str] = None
    ligands: List[Dict[str, Any]] = None
    authors: List[str] = None
    release_date: Optional[str] = None
    experimental_data: Dict[str, Any] = None
    
    # Additional metadata
    classification: Optional[str] = None
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.chains is None:
            self.chains = []
        if self.ligands is None:
            self.ligands = []
        if self.authors is None:
            self.authors = []
        if self.keywords is None:
            self.keywords = []
        if self.experimental_data is None:
            self.experimental_data = {}


@dataclass
class PDBChain:
    """Represents a chain in a PDB structure"""
    pdb_id: str
    chain_id: str
    sequence: str
    num_residues: int
    entity_type: str  # Protein, DNA, RNA, etc.
    
    # Additional information
    description: Optional[str] = None
    uniprot_accession: Optional[str] = None


@dataclass
class Ligand:
    """Represents a ligand in a PDB structure"""
    pdb_id: str
    ligand_id: str  # 3-letter code
    chain: str
    residue_number: int
    name: str
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    formula: Optional[str] = None
    
    # Binding information
    binding_site: Optional[str] = None
    binding_affinity: Optional[float] = None


@dataclass
class PDBStructure:
    """Represents a parsed PDB structure"""
    pdb_id: str
    atoms: List[Dict[str, Any]] = None
    residues: List[Dict[str, Any]] = None
    chains: List[PDBChain] = None
    ligands: List[Ligand] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.atoms is None:
            self.atoms = []
        if self.residues is None:
            self.residues = []
        if self.chains is None:
            self.chains = []
        if self.ligands is None:
            self.ligands = []
        if self.metadata is None:
            self.metadata = {}


class PDBClient(DatabaseClient):
    """
    Client for querying Protein Data Bank (PDB)
    
    PDB is the primary repository for 3D structural data of proteins and nucleic acids.
    It provides:
    - Protein 3D structures
    - Ligand information
    - Experimental metadata
    - Structure validation reports
    - Electron density maps
    - Structure factors
    """
    
    # RCSB splits its API across three hosts: the data API (config.endpoint),
    # the search service, and a GraphQL endpoint used for batch lookups.
    SEARCH_ENDPOINT = "https://search.rcsb.org/rcsbsearch/v2/query"
    GRAPHQL_ENDPOINT = "https://data.rcsb.org/graphql"

    _ENTRY_QUERY = """
    query Entry($id: String!) {
      entry(entry_id: $id) {
        rcsb_id
        struct { title }
        exptl { method }
        refine { ls_d_res_high ls_R_factor_R_work ls_R_factor_R_free }
        reflns { number_obs }
        cell { length_a length_b length_c angle_alpha angle_beta angle_gamma }
        symmetry { space_group_name_H_M }
        rcsb_accession_info { initial_release_date }
        struct_keywords { pdbx_keywords text }
        audit_author { name }
        rcsb_entry_info {
          deposited_atom_count
          deposited_model_count
          molecular_weight
        }
        polymer_entities {
          rcsb_polymer_entity_container_identifiers { auth_asym_ids }
        }
        nonpolymer_entities {
          nonpolymer_comp { chem_comp { id name formula } }
          rcsb_nonpolymer_entity_container_identifiers { auth_asym_ids }
        }
      }
    }
    """

    def __init__(self, config: Optional[DatabaseConfig] = None, cache_dir: Optional[Path] = None):
        """
        Initialize PDB client

        Args:
            config: Custom configuration
            cache_dir: Directory for caching downloaded PDB files
        """
        super().__init__(config or self.get_default_config())
        self.cache_dir = cache_dir or Path("./cache/pdb")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._chem_comp_cache: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def get_default_config(cls) -> DatabaseConfig:
        """Get default PDB configuration"""
        return DatabaseConfig(
            endpoint="https://data.rcsb.org",
            rate_limit=20,
            timeout=60,
            retries=3,
            cache_enabled=True,
            cache_ttl=86400,  # 24 hours
            headers={
                "Accept": "application/json",
                "User-Agent": "DrugDiscoveryMCP/0.1.0"
            }
        )
    
    def get_name(self) -> str:
        """Get the name of this database"""
        return "PDB"

    def _entity_ids(self, pdb_id: str, key: str) -> List[str]:
        """
        List an entry's entity IDs of a given kind

        Args:
            pdb_id: PDB ID
            key: Container identifier key, e.g. "polymer_entity_ids"

        Returns:
            List of entity IDs (empty if the entry has none of that kind)
        """
        url = f"{self.config.endpoint}/rest/v1/core/entry/{pdb_id}"
        entry = self._make_request("GET", url)
        identifiers = entry.get("rcsb_entry_container_identifiers", {}) or {}
        return identifiers.get(key, []) or []

    def _fetch_entry_summaries(self, entry_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch summary metadata for several entries in one GraphQL round trip

        The search service returns bare identifiers, so titles and experimental
        metadata have to be looked up separately.

        Args:
            entry_ids: PDB IDs to look up

        Returns:
            List of summary dictionaries, in the order given
        """
        if not entry_ids:
            return []

        query = """
        query Entries($ids: [String!]!) {
          entries(entry_ids: $ids) {
            rcsb_id
            struct { title }
            exptl { method }
            refine { ls_d_res_high }
            rcsb_accession_info { initial_release_date }
            struct_keywords { pdbx_keywords }
          }
        }
        """
        payload = {"query": query, "variables": {"ids": entry_ids}}
        data = self._make_request("POST", self.GRAPHQL_ENDPOINT, data=payload)

        if data.get("errors"):
            messages = "; ".join(str(e.get("message", e)) for e in data["errors"])
            raise DatabaseError(f"GraphQL error: {messages}")

        entries = (data.get("data") or {}).get("entries") or []
        return [self._parse_entry_summary(e) for e in entries if e]
    
    def query(self, pdb_id: str) -> PDBEntry:
        """
        Query PDB for a specific entry
        
        Args:
            pdb_id: PDB ID (e.g., "1ABC")
            
        Returns:
            PDBEntry object with entry information
        """
        # A single GraphQL call resolves the entry together with its chain
        # identifiers and bound ligands; the REST entry document only carries
        # entity numbers, which are not chain names.
        payload = {
            "query": self._ENTRY_QUERY,
            "variables": {"id": pdb_id},
        }

        try:
            response = self._make_request("POST", self.GRAPHQL_ENDPOINT, data=payload)

            if response.get("errors"):
                messages = "; ".join(
                    str(e.get("message", e)) for e in response["errors"]
                )
                raise DatabaseError(f"GraphQL error: {messages}")

            entry = (response.get("data") or {}).get("entry")
            if not entry:
                raise DatabaseError(f"Entry not found: {pdb_id}")

            return self._parse_entry(entry)
        except Exception as e:
            logger.error(f"PDB query failed for {pdb_id}: {e}")
            raise DatabaseError(f"Failed to query PDB for {pdb_id}: {e}")
    
    def search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        sort: str = "score"
    ) -> Dict[str, Any]:
        """
        Search PDB database
        
        Args:
            query: Search query
            limit: Maximum number of results
            offset: Pagination offset
            sort: Sort by field
            
        Returns:
            Dictionary with search results
        """
        payload = {
            "query": {
                "type": "terminal",
                "service": "full_text",
                "parameters": {"value": query},
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": offset, "rows": limit},
                "results_content_type": ["experimental"],
                "scoring_strategy": "combined",
            },
        }

        try:
            data = self._make_request("POST", self.SEARCH_ENDPOINT, data=payload)
            hits = data.get("result_set", []) or []
            scores = {h.get("identifier"): h.get("score", 0) for h in hits}
            summaries = self._fetch_entry_summaries(list(scores))

            for summary in summaries:
                summary["score"] = scores.get(summary["pdb_id"], 0)

            return {
                "results": summaries,
                "total": data.get("total_count", 0),
                "limit": limit,
                "offset": offset
            }
        except Exception as e:
            logger.error(f"PDB search failed: {e}")
            raise DatabaseError(f"Failed to search PDB: {e}")
    
    def download_pdb_file(self, pdb_id: str, format: str = "pdb") -> str:
        """
        Download PDB file in specified format
        
        Args:
            pdb_id: PDB ID
            format: File format ("pdb", "mmcif", "xml", "bcif")
            
        Returns:
            Path to the downloaded file
        """
        # Check cache first
        cache_file = self.cache_dir / f"{pdb_id.lower()}.{format}"
        if cache_file.exists():
            logger.debug(f"Using cached PDB file: {cache_file}")
            return str(cache_file)
        
        url = f"https://files.rcsb.org/download/{pdb_id}.{format}"
        
        try:
            # For PDB files, we need to handle the response differently
            import requests
            
            response = requests.get(url, timeout=self.config.timeout)
            
            if response.status_code >= 400:
                raise DatabaseError(
                    f"Failed to download PDB file {pdb_id}: {response.status_code} - {response.text}",
                    status_code=response.status_code
                )
            
            # Save to cache
            content = response.content
            
            # Handle gzipped content
            if format == "bcif" or url.endswith(".gz"):
                content = gzip.decompress(content)
            
            with open(cache_file, 'wb') as f:
                f.write(content)
            
            logger.info(f"Downloaded PDB file: {cache_file}")
            return str(cache_file)
            
        except Exception as e:
            logger.error(f"Failed to download PDB file {pdb_id}: {e}")
            raise DatabaseError(f"Failed to download PDB file: {e}")
    
    def parse_pdb_file(self, pdb_id: str, file_path: Optional[str] = None) -> PDBStructure:
        """
        Parse a PDB file and extract structural information
        
        Args:
            pdb_id: PDB ID
            file_path: Path to PDB file (if not provided, will download)
            
        Returns:
            PDBStructure object with parsed data
        """
        if not file_path:
            file_path = self.download_pdb_file(pdb_id, format="pdb")
        
        try:
            # Use Biopython for parsing
            try:
                from Bio.PDB import PDBParser
                parser = PDBParser()
                structure = parser.get_structure(pdb_id, file_path)
                
                return self._parse_biopython_structure(pdb_id, structure)
            except ImportError:
                # Fallback to manual parsing
                return self._parse_pdb_file_manually(pdb_id, file_path)
                
        except Exception as e:
            logger.error(f"Failed to parse PDB file {pdb_id}: {e}")
            raise DatabaseError(f"Failed to parse PDB file: {e}")
    
    def get_chains(self, pdb_id: str) -> List[PDBChain]:
        """
        Get chain information for a PDB entry
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            List of PDBChain objects
        """
        try:
            chains = []
            for entity_id in self._entity_ids(pdb_id, "polymer_entity_ids"):
                url = f"{self.config.endpoint}/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
                entity = self._make_request("GET", url)
                # One polymer entity can be instantiated as several chains
                chains.extend(self._parse_chain(pdb_id, entity))
            return chains
        except Exception as e:
            logger.error(f"Failed to get chains for {pdb_id}: {e}")
            raise DatabaseError(f"Failed to get chains: {e}")
    
    def get_ligands(self, pdb_id: str) -> List[Ligand]:
        """
        Get ligand information for a PDB entry
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            List of Ligand objects
        """
        try:
            ligands = []
            for entity_id in self._entity_ids(pdb_id, "non_polymer_entity_ids"):
                url = f"{self.config.endpoint}/rest/v1/core/nonpolymer_entity/{pdb_id}/{entity_id}"
                entity = self._make_request("GET", url)
                ligands.extend(self._parse_ligand(pdb_id, entity))
            return ligands
        except Exception as e:
            logger.error(f"Failed to get ligands for {pdb_id}: {e}")
            raise DatabaseError(f"Failed to get ligands: {e}")
    
    def get_sequence(self, pdb_id: str, chain_id: str) -> str:
        """
        Get sequence for a specific chain in a PDB entry
        
        Args:
            pdb_id: PDB ID
            chain_id: Chain identifier
            
        Returns:
            Sequence as string
        """
        try:
            for chain in self.get_chains(pdb_id):
                if chain.chain_id == chain_id:
                    return chain.sequence
            raise DatabaseError(f"Chain {chain_id} not found in {pdb_id}")
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Failed to get sequence for {pdb_id}/{chain_id}: {e}")
            raise DatabaseError(f"Failed to get sequence: {e}")
    
    def get_structure_alignment(self, pdb_id1: str, pdb_id2: str) -> Dict[str, Any]:
        """
        Get structure alignment between two PDB entries
        
        Args:
            pdb_id1: First PDB ID
            pdb_id2: Second PDB ID
            
        Returns:
            Dictionary with alignment information
        """
        # RCSB exposes no public REST alignment endpoint, so superimpose the two
        # structures locally instead of calling out.
        from ..structural_biology.alignment import superimpose_structures

        try:
            result = superimpose_structures(pdb_id1, pdb_id2)
            if "error" in result:
                raise DatabaseError(result["error"])
            return {
                "rmsd": result.get("rmsd", 0),
                "aligned_residues": result.get("aligned_residues", 0),
                "transformation_matrix": result.get("transformation_matrix"),
                "translation_vector": result.get("translation_vector"),
                "residue_differences": result.get("residue_differences", []),
            }
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Failed to get structure alignment: {e}")
            raise DatabaseError(f"Failed to get structure alignment: {e}")
    
    def get_experimental_data(self, pdb_id: str) -> Dict[str, Any]:
        """
        Get experimental data for a PDB entry
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            Dictionary with experimental metadata
        """
        url = f"{self.config.endpoint}/rest/v1/core/entry/{pdb_id}"

        try:
            data = self._make_request("GET", url)
            return self._parse_experimental(data)
        except Exception as e:
            logger.error(f"Failed to get experimental data for {pdb_id}: {e}")
            raise DatabaseError(f"Failed to get experimental data: {e}")
    
    def search_by_uniprot(self, accession: str, limit: int = 10) -> Dict[str, Any]:
        """
        Find PDB entries whose polymer entities map to a UniProt accession

        Args:
            accession: UniProt accession, e.g. "P04637"
            limit: Maximum number of results

        Returns:
            Dictionary with search results
        """
        payload = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_polymer_entity_container_identifiers"
                                         ".reference_sequence_identifiers.database_accession",
                            "operator": "in",
                            "value": [accession],
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_polymer_entity_container_identifiers"
                                         ".reference_sequence_identifiers.database_name",
                            "operator": "exact_match",
                            "value": "UniProt",
                        },
                    },
                ],
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": limit}},
        }

        try:
            data = self._make_request("POST", self.SEARCH_ENDPOINT, data=payload)
            hits = data.get("result_set", []) or []
            entry_ids = [h.get("identifier") for h in hits if h.get("identifier")]

            return {
                "results": self._fetch_entry_summaries(entry_ids),
                "total": data.get("total_count", 0),
                "accession": accession,
            }
        except Exception as e:
            logger.error(f"Failed to search PDB by UniProt accession {accession}: {e}")
            raise DatabaseError(f"Failed to search PDB by UniProt accession: {e}")

    def search_by_sequence(
        self,
        sequence: str,
        limit: int = 10,
        identity_cutoff: float = 0.9,
        evalue_cutoff: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Search PDB by protein sequence
        
        Args:
            sequence: Protein sequence
            limit: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        payload = {
            "query": {
                "type": "terminal",
                "service": "sequence",
                "parameters": {
                    "sequence_type": "protein",
                    "value": sequence,
                    "identity_cutoff": identity_cutoff,
                    "evalue_cutoff": evalue_cutoff,
                },
            },
            "return_type": "polymer_entity",
            "request_options": {"paginate": {"start": 0, "rows": limit}},
        }

        try:
            data = self._make_request("POST", self.SEARCH_ENDPOINT, data=payload)
            hits = data.get("result_set", []) or []

            # Hits are "<entry>_<entity>"; enrich by the entry half
            entry_ids = []
            for hit in hits:
                entry_id = hit.get("identifier", "").split("_")[0]
                if entry_id and entry_id not in entry_ids:
                    entry_ids.append(entry_id)

            summaries = {s["pdb_id"]: s for s in self._fetch_entry_summaries(entry_ids)}

            results = []
            for hit in hits:
                identifier = hit.get("identifier", "")
                entry_id = identifier.split("_")[0]
                summary = dict(summaries.get(entry_id, {"pdb_id": entry_id}))
                summary["entity_id"] = identifier
                summary["score"] = hit.get("score", 0)
                results.append(summary)

            return {
                "results": results,
                "total": data.get("total_count", 0)
            }
        except Exception as e:
            logger.error(f"Failed to search by sequence: {e}")
            raise DatabaseError(f"Failed to search by sequence: {e}")
    
    @staticmethod
    def _first(items: Any, key: str) -> Any:
        """Read `key` off the first element of an RCSB list-valued category"""
        if isinstance(items, list) and items:
            return (items[0] or {}).get(key)
        if isinstance(items, dict):
            return items.get(key)
        return None

    @staticmethod
    def _date(value: Optional[str]) -> Optional[str]:
        """Trim an RCSB ISO timestamp down to its date part"""
        if not value:
            return None
        return str(value).split("T")[0]

    def _parse_entry(self, data: Dict[str, Any]) -> PDBEntry:
        """Parse an RCSB entry document into a PDBEntry"""
        keywords = data.get("struct_keywords", {}) or {}
        accession = data.get("rcsb_accession_info", {}) or {}

        keyword_text = keywords.get("text") or ""
        keyword_list = [k.strip() for k in keyword_text.split(",") if k.strip()]

        # Chain identifiers, flattened across polymer entities
        chains = []
        for entity in data.get("polymer_entities") or []:
            ids = (entity.get("rcsb_polymer_entity_container_identifiers") or {})
            for chain_id in ids.get("auth_asym_ids") or []:
                if chain_id not in chains:
                    chains.append(chain_id)

        ligands = []
        for entity in data.get("nonpolymer_entities") or []:
            chem = ((entity.get("nonpolymer_comp") or {}).get("chem_comp")) or {}
            ids = (entity.get("rcsb_nonpolymer_entity_container_identifiers") or {})
            ligands.append({
                "ligand_id": chem.get("id"),
                "name": chem.get("name"),
                "formula": chem.get("formula"),
                "chains": ids.get("auth_asym_ids") or [],
            })

        return PDBEntry(
            pdb_id=data.get("rcsb_id") or (data.get("entry", {}) or {}).get("id", ""),
            title=(data.get("struct", {}) or {}).get("title", ""),
            resolution=self._first(data.get("refine"), "ls_d_res_high"),
            method=self._first(data.get("exptl"), "method"),
            chains=chains,
            ligands=ligands,
            authors=[
                a.get("name", "")
                for a in data.get("audit_author", []) or []
                if a.get("name")
            ],
            release_date=self._date(accession.get("initial_release_date")),
            experimental_data=self._parse_experimental(data),
            classification=keywords.get("pdbx_keywords"),
            keywords=keyword_list,
        )

    def _parse_entry_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a summary entry, as returned by the batch GraphQL lookup"""
        accession = data.get("rcsb_accession_info") or {}
        keywords = data.get("struct_keywords") or {}

        return {
            "pdb_id": data.get("rcsb_id", ""),
            "title": (data.get("struct") or {}).get("title", ""),
            "resolution": self._first(data.get("refine"), "ls_d_res_high"),
            "method": self._first(data.get("exptl"), "method"),
            "release_date": self._date(accession.get("initial_release_date")),
            "classification": keywords.get("pdbx_keywords"),
            "score": data.get("score", 0),
        }

    def _parse_experimental(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract experimental metadata from an RCSB entry document"""
        cell = data.get("cell", {}) or {}
        symmetry = data.get("symmetry", {}) or {}
        reflns = data.get("reflns")
        entry_info = data.get("rcsb_entry_info", {}) or {}

        return {
            "method": self._first(data.get("exptl"), "method"),
            "resolution": self._first(data.get("refine"), "ls_d_res_high"),
            "r_factor_work": self._first(data.get("refine"), "ls_R_factor_R_work"),
            "r_factor_free": self._first(data.get("refine"), "ls_R_factor_R_free"),
            "space_group": symmetry.get("space_group_name_H_M"),
            "unit_cell": {
                "a": cell.get("length_a"),
                "b": cell.get("length_b"),
                "c": cell.get("length_c"),
                "alpha": cell.get("angle_alpha"),
                "beta": cell.get("angle_beta"),
                "gamma": cell.get("angle_gamma"),
            } if cell else {},
            "observed_reflections": self._first(reflns, "number_obs"),
            "deposited_atom_count": entry_info.get("deposited_atom_count"),
            "deposited_model_count": entry_info.get("deposited_model_count"),
            "molecular_weight": entry_info.get("molecular_weight"),
        }

    def _parse_chain(self, pdb_id: str, data: Dict[str, Any]) -> List[PDBChain]:
        """
        Parse a polymer entity into one PDBChain per instantiated chain

        RCSB models sequences per *entity*; a single entity can appear as
        several chains (e.g. the three p53 copies in 1TUP), so this returns a
        list rather than a single chain.
        """
        poly = data.get("entity_poly", {}) or {}
        entity = data.get("rcsb_polymer_entity", {}) or {}
        identifiers = data.get("rcsb_polymer_entity_container_identifiers", {}) or {}

        sequence = poly.get("pdbx_seq_one_letter_code_can") or ""
        references = identifiers.get("reference_sequence_identifiers", []) or []
        uniprot = next(
            (r.get("database_accession") for r in references
             if r.get("database_name") == "UniProt"),
            None,
        )
        if uniprot is None:
            uniprot_ids = identifiers.get("uniprot_ids") or []
            uniprot = uniprot_ids[0] if uniprot_ids else None

        auth_ids = identifiers.get("auth_asym_ids") or identifiers.get("asym_ids") or []

        return [
            PDBChain(
                pdb_id=pdb_id,
                chain_id=chain_id,
                sequence=sequence,
                num_residues=len(sequence),
                entity_type=poly.get("rcsb_entity_polymer_type", ""),
                description=entity.get("pdbx_description"),
                uniprot_accession=uniprot,
            )
            for chain_id in auth_ids
        ]

    def _parse_ligand(self, pdb_id: str, data: Dict[str, Any]) -> List[Ligand]:
        """
        Parse a non-polymer entity into one Ligand per chain it occupies

        Chemical identifiers (SMILES/InChI/formula) live on the chemical
        component definition, which is fetched separately and cached.
        """
        nonpoly = data.get("pdbx_entity_nonpoly", {}) or {}
        entity = data.get("rcsb_nonpolymer_entity", {}) or {}
        identifiers = data.get("rcsb_nonpolymer_entity_container_identifiers", {}) or {}

        comp_id = nonpoly.get("comp_id") or identifiers.get("nonpolymer_comp_id") or ""
        chem = self._get_chem_comp(comp_id) if comp_id else {}
        auth_ids = identifiers.get("auth_asym_ids") or identifiers.get("asym_ids") or [""]

        return [
            Ligand(
                pdb_id=pdb_id,
                ligand_id=comp_id,
                chain=chain_id,
                residue_number=0,
                name=nonpoly.get("name") or entity.get("pdbx_description") or "",
                smiles=chem.get("smiles"),
                inchi=chem.get("inchi"),
                formula=chem.get("formula"),
            )
            for chain_id in auth_ids
        ]

    def _get_chem_comp(self, comp_id: str) -> Dict[str, Any]:
        """
        Look up a chemical component definition (SMILES, InChI, formula)

        Results are memoised per client because the same component (ZN, HOH,
        ATP...) recurs across entries.
        """
        if comp_id in self._chem_comp_cache:
            return self._chem_comp_cache[comp_id]

        try:
            url = f"{self.config.endpoint}/rest/v1/core/chemcomp/{comp_id}"
            data = self._make_request("GET", url)
        except DatabaseError as e:
            logger.warning(f"Could not fetch chemical component {comp_id}: {e}")
            self._chem_comp_cache[comp_id] = {}
            return {}

        chem_comp = data.get("chem_comp", {}) or {}
        descriptor = data.get("rcsb_chem_comp_descriptor", {}) or {}

        result = {
            "formula": chem_comp.get("formula"),
            "formula_weight": chem_comp.get("formula_weight"),
            "smiles": descriptor.get("SMILES_stereo") or descriptor.get("SMILES"),
            "inchi": descriptor.get("InChI"),
            "inchikey": descriptor.get("InChIKey"),
        }
        self._chem_comp_cache[comp_id] = result
        return result
    
    def _parse_biopython_structure(self, pdb_id: str, structure) -> PDBStructure:
        """Parse Biopython structure into PDBStructure"""
        chains = []
        ligands = []
        atoms = []
        residues = []
        
        for model in structure:
            for chain in model:
                chain_data = PDBChain(
                    pdb_id=pdb_id,
                    chain_id=chain.id,
                    sequence="",
                    num_residues=0,
                    entity_type="PROTEIN"
                )
                chains.append(chain_data)
                
                for residue in chain:
                    residue_data = {
                        "chain": chain.id,
                        "residue_id": residue.id,
                        "residue_name": residue.get_resname(),
                        "residue_number": residue.id[1],
                        "insertion_code": residue.id[2] if len(residue.id) > 2 else None
                    }
                    residues.append(residue_data)
                    
                    for atom in residue:
                        atom_data = {
                            "chain": chain.id,
                            "residue_id": residue.id,
                            "residue_name": residue.get_resname(),
                            "atom_id": atom.id,
                            "atom_name": atom.get_name(),
                            "coordinates": [atom.coord[0], atom.coord[1], atom.coord[2]],
                            "bfactor": atom.get_bfactor(),
                            "occupancy": atom.get_occupancy(),
                            "element": atom.element
                        }
                        atoms.append(atom_data)
        
        return PDBStructure(
            pdb_id=pdb_id,
            atoms=atoms,
            residues=residues,
            chains=chains,
            ligands=ligands,
            metadata={"source": "biopython"}
        )
    
    def _parse_pdb_file_manually(self, pdb_id: str, file_path: str) -> PDBStructure:
        """Manual PDB file parsing (fallback when Biopython is not available)"""
        atoms = []
        residues = []
        chains = []
        ligands = []
        
        current_chain = None
        current_residue = None
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('HEADER'):
                    # Extract header information
                    continue
                elif line.startswith('ATOM') or line.startswith('HETATM'):
                    # Parse atom record
                    record_type = line[0:6].strip()
                    atom_number = int(line[6:11].strip())
                    atom_name = line[12:16].strip()
                    alt_loc = line[16].strip()
                    residue_name = line[17:20].strip()
                    chain_id = line[21].strip()
                    residue_number = int(line[22:26].strip())
                    insertion_code = line[26].strip()
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    occupancy = float(line[54:60].strip())
                    bfactor = float(line[60:66].strip())
                    element = line[76:78].strip()
                    charge = line[78:80].strip()
                    
                    atom_data = {
                        "record_type": record_type,
                        "atom_number": atom_number,
                        "atom_name": atom_name,
                        "alt_loc": alt_loc,
                        "residue_name": residue_name,
                        "chain_id": chain_id,
                        "residue_number": residue_number,
                        "insertion_code": insertion_code,
                        "coordinates": [x, y, z],
                        "occupancy": occupancy,
                        "bfactor": bfactor,
                        "element": element,
                        "charge": charge
                    }
                    atoms.append(atom_data)
                    
                    # Update current chain and residue
                    if chain_id and chain_id not in [c.chain_id for c in chains]:
                        chains.append(PDBChain(
                            pdb_id=pdb_id,
                            chain_id=chain_id,
                            sequence="",
                            num_residues=0,
                            entity_type="PROTEIN"
                        ))
                    
                    residue_key = f"{chain_id}_{residue_number}_{insertion_code}"
                    if residue_key not in [f"{r['chain']}_{r['residue_number']}_{r.get('insertion_code', '')}" 
                                           for r in residues]:
                        residues.append({
                            "chain": chain_id,
                            "residue_name": residue_name,
                            "residue_number": residue_number,
                            "insertion_code": insertion_code
                        })
                    
                    # Check for ligands (HETATM records with non-standard residues)
                    if record_type == "HETATM" and residue_name not in ["HOH", "DOD", "NA", "CL"]:
                        ligand_key = f"{chain_id}_{residue_number}_{residue_name}"
                        if ligand_key not in [f"{l.chain}_{l.residue_number}_{l.name}" for l in ligands]:
                            ligands.append(Ligand(
                                pdb_id=pdb_id,
                                ligand_id=residue_name,
                                chain=chain_id,
                                residue_number=residue_number,
                                name=residue_name,
                                smiles=None,
                                inchi=None,
                                formula=None
                            ))
        
        return PDBStructure(
            pdb_id=pdb_id,
            atoms=atoms,
            residues=residues,
            chains=chains,
            ligands=ligands,
            metadata={"source": "manual", "num_atoms": len(atoms), "num_residues": len(residues)}
        )


# Singleton instance
pdb_client = PDBClient()


# Convenience functions for direct use
def query_pdb(pdb_id: str, **kwargs) -> Dict[str, Any]:
    """Query PDB database"""
    try:
        entry = pdb_client.query(pdb_id, **kwargs)
        return entry.__dict__
    except DatabaseError as e:
        return {"error": str(e), "pdb_id": pdb_id}


def download_pdb(pdb_id: str, format: str = "pdb") -> str:
    """Download PDB file"""
    try:
        return pdb_client.download_pdb_file(pdb_id, format)
    except DatabaseError as e:
        return {"error": str(e), "pdb_id": pdb_id}


def parse_pdb(pdb_id: str, file_path: Optional[str] = None) -> Dict[str, Any]:
    """Parse PDB file"""
    try:
        structure = pdb_client.parse_pdb_file(pdb_id, file_path)
        return {
            "pdb_id": structure.pdb_id,
            "num_atoms": len(structure.atoms),
            "num_residues": len(structure.residues),
            "num_chains": len(structure.chains),
            "num_ligands": len(structure.ligands),
            "chains": [chain.__dict__ for chain in structure.chains],
            "ligands": [ligand.__dict__ for ligand in structure.ligands]
        }
    except DatabaseError as e:
        return {"error": str(e), "pdb_id": pdb_id}
