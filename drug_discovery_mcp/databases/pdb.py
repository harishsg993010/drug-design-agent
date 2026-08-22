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
    
    def query(self, pdb_id: str) -> PDBEntry:
        """
        Query PDB for a specific entry
        
        Args:
            pdb_id: PDB ID (e.g., "1ABC")
            
        Returns:
            PDBEntry object with entry information
        """
        url = f"{self.config.endpoint}/pdb/v1/entries/{pdb_id}"
        
        try:
            data = self._make_request("GET", url)
            return self._parse_entry(data)
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
        url = f"{self.config.endpoint}/pdb/v1/entries"
        
        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "sort": sort
        }
        
        try:
            data = self._make_request("GET", url, params=params)
            return {
                "results": [self._parse_entry_summary(item) for item in data.get("results", [])],
                "total": data.get("total", 0),
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
        
        url = f"{self.config.endpoint}/pdb/v1/files/{pdb_id}.{format}"
        
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
        url = f"{self.config.endpoint}/pdb/v1/entries/{pdb_id}/chains"
        
        try:
            data = self._make_request("GET", url)
            return [self._parse_chain(pdb_id, item) for item in data.get("results", [])]
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
        url = f"{self.config.endpoint}/pdb/v1/entries/{pdb_id}/ligands"
        
        try:
            data = self._make_request("GET", url)
            return [self._parse_ligand(pdb_id, item) for item in data.get("results", [])]
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
        url = f"{self.config.endpoint}/pdb/v1/entries/{pdb_id}/chains/{chain_id}/sequence"
        
        try:
            data = self._make_request("GET", url)
            return data.get("sequence", "")
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
        url = f"{self.config.endpoint}/pdb/v1/align/{pdb_id1}/{pdb_id2}"
        
        try:
            data = self._make_request("GET", url)
            return {
                "rmsd": data.get("rmsd", 0),
                "aligned_residues": data.get("aligned_residues", 0),
                "sequence_identity": data.get("sequence_identity", 0),
                "alignment": data.get("alignment", [])
            }
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
        url = f"{self.config.endpoint}/pdb/v1/entries/{pdb_id}/experimental"
        
        try:
            data = self._make_request("GET", url)
            return data
        except Exception as e:
            logger.error(f"Failed to get experimental data for {pdb_id}: {e}")
            raise DatabaseError(f"Failed to get experimental data: {e}")
    
    def search_by_sequence(self, sequence: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search PDB by protein sequence
        
        Args:
            sequence: Protein sequence
            limit: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        url = f"{self.config.endpoint}/pdb/v1/sequence"
        
        params = {
            "sequence": sequence,
            "limit": limit
        }
        
        try:
            data = self._make_request("GET", url, params=params)
            return {
                "results": [self._parse_entry_summary(item) for item in data.get("results", [])],
                "total": data.get("total", 0)
            }
        except Exception as e:
            logger.error(f"Failed to search by sequence: {e}")
            raise DatabaseError(f"Failed to search by sequence: {e}")
    
    def _parse_entry(self, data: Dict[str, Any]) -> PDBEntry:
        """Parse PDB entry data"""
        return PDBEntry(
            pdb_id=data.get("pdb_id", ""),
            title=data.get("title", ""),
            resolution=data.get("resolution", None),
            method=data.get("method", None),
            chains=data.get("chains", []),
            ligands=data.get("ligands", []),
            authors=data.get("authors", []),
            release_date=data.get("release_date", None),
            experimental_data=data.get("experimental_data", {}),
            classification=data.get("classification", None),
            keywords=data.get("keywords", [])
        )
    
    def _parse_entry_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a summary entry from search results"""
        return {
            "pdb_id": data.get("pdb_id", ""),
            "title": data.get("title", ""),
            "resolution": data.get("resolution", None),
            "method": data.get("method", None),
            "release_date": data.get("release_date", None),
            "score": data.get("score", 0)
        }
    
    def _parse_chain(self, pdb_id: str, data: Dict[str, Any]) -> PDBChain:
        """Parse chain data"""
        return PDBChain(
            pdb_id=pdb_id,
            chain_id=data.get("chain_id", ""),
            sequence=data.get("sequence", ""),
            num_residues=data.get("num_residues", 0),
            entity_type=data.get("entity_type", ""),
            description=data.get("description", None),
            uniprot_accession=data.get("uniprot_accession", None)
        )
    
    def _parse_ligand(self, pdb_id: str, data: Dict[str, Any]) -> Ligand:
        """Parse ligand data"""
        return Ligand(
            pdb_id=pdb_id,
            ligand_id=data.get("ligand_id", ""),
            chain=data.get("chain", ""),
            residue_number=data.get("residue_number", 0),
            name=data.get("name", ""),
            smiles=data.get("smiles", None),
            inchi=data.get("inchi", None),
            formula=data.get("formula", None),
            binding_site=data.get("binding_site", None),
            binding_affinity=data.get("binding_affinity", None)
        )
    
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
