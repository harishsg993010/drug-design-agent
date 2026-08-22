"""
PDB File Parser

Provides tools for parsing, reading, and manipulating PDB files.
"""

import logging
import gzip
import io
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

from .base import StructuralBiologyBase, StructuralBiologyError

logger = logging.getLogger(__name__)


@dataclass
class Atom:
    """Represents an atom in a PDB structure"""
    atom_number: int
    atom_name: str
    alt_loc: str
    residue_name: str
    chain_id: str
    residue_number: int
    insertion_code: str
    x: float
    y: float
    z: float
    occupancy: float
    bfactor: float
    element: str
    charge: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "atom_number": self.atom_number,
            "atom_name": self.atom_name,
            "alt_loc": self.alt_loc,
            "residue_name": self.residue_name,
            "chain_id": self.chain_id,
            "residue_number": self.residue_number,
            "insertion_code": self.insertion_code,
            "coordinates": [self.x, self.y, self.z],
            "occupancy": self.occupancy,
            "bfactor": self.bfactor,
            "element": self.element,
            "charge": self.charge
        }


@dataclass
class Residue:
    """Represents a residue in a PDB structure"""
    residue_name: str
    chain_id: str
    residue_number: int
    insertion_code: str
    atoms: List[Atom] = None
    
    def __post_init__(self):
        if self.atoms is None:
            self.atoms = []
    
    def get_coordinates(self) -> List[List[float]]:
        """Get coordinates of all atoms in the residue"""
        return [[atom.x, atom.y, atom.z] for atom in self.atoms]
    
    def get_center(self) -> List[float]:
        """Get the geometric center of the residue"""
        coords = self.get_coordinates()
        if not coords:
            return [0.0, 0.0, 0.0]
        
        n = len(coords)
        center = [0.0, 0.0, 0.0]
        for coord in coords:
            center[0] += coord[0]
            center[1] += coord[1]
            center[2] += coord[2]
        
        return [c / n for c in center]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "residue_name": self.residue_name,
            "chain_id": self.chain_id,
            "residue_number": self.residue_number,
            "insertion_code": self.insertion_code,
            "num_atoms": len(self.atoms),
            "atoms": [atom.to_dict() for atom in self.atoms]
        }


@dataclass
class Chain:
    """Represents a chain in a PDB structure"""
    chain_id: str
    residues: List[Residue] = None
    sequence: str = ""
    
    def __post_init__(self):
        if self.residues is None:
            self.residues = []
    
    def get_sequence(self) -> str:
        """Get the amino acid sequence of the chain"""
        if self.sequence:
            return self.sequence
        
        # Extract from residues
        seq = ""
        for residue in self.residues:
            # Use three-letter code for amino acids
            aa_code = residue.residue_name
            # Convert to one-letter code if it's a standard amino acid
            aa_map = {
                "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E",
                "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
                "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
                "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
                "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
            }
            seq += aa_map.get(aa_code, "X")
        
        self.sequence = seq
        return seq
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "chain_id": self.chain_id,
            "num_residues": len(self.residues),
            "sequence": self.get_sequence(),
            "residues": [residue.to_dict() for residue in self.residues]
        }


@dataclass
class Ligand:
    """Represents a ligand in a PDB structure"""
    ligand_id: str
    chain_id: str
    residue_number: int
    name: str
    atoms: List[Atom] = None
    
    def __post_init__(self):
        if self.atoms is None:
            self.atoms = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "ligand_id": self.ligand_id,
            "chain_id": self.chain_id,
            "residue_number": self.residue_number,
            "name": self.name,
            "num_atoms": len(self.atoms),
            "atoms": [atom.to_dict() for atom in self.atoms]
        }


@dataclass
class PDBStructure:
    """Represents a parsed PDB structure"""
    pdb_id: str
    header: Dict[str, Any] = None
    chains: List[Chain] = None
    ligands: List[Ligand] = None
    atoms: List[Atom] = None
    residues: List[Residue] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.header is None:
            self.header = {}
        if self.chains is None:
            self.chains = []
        if self.ligands is None:
            self.ligands = []
        if self.atoms is None:
            self.atoms = []
        if self.residues is None:
            self.residues = []
        if self.metadata is None:
            self.metadata = {}
    
    def get_chain(self, chain_id: str) -> Optional[Chain]:
        """Get a specific chain by ID"""
        for chain in self.chains:
            if chain.chain_id == chain_id:
                return chain
        return None
    
    def get_residue(self, chain_id: str, residue_number: int, insertion_code: str = "") -> Optional[Residue]:
        """Get a specific residue by chain and residue number"""
        chain = self.get_chain(chain_id)
        if chain is None:
            return None
        
        for residue in chain.residues:
            if residue.residue_number == residue_number and residue.insertion_code == insertion_code:
                return residue
        return None
    
    def get_atoms_by_element(self, element: str) -> List[Atom]:
        """Get all atoms of a specific element"""
        return [atom for atom in self.atoms if atom.element == element]
    
    def get_atoms_by_residue_name(self, residue_name: str) -> List[Atom]:
        """Get all atoms in residues with a specific name"""
        return [atom for atom in self.atoms if atom.residue_name == residue_name]
    
    def get_coordinates(self) -> List[List[float]]:
        """Get coordinates of all atoms"""
        return [[atom.x, atom.y, atom.z] for atom in self.atoms]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pdb_id": self.pdb_id,
            "header": self.header,
            "num_chains": len(self.chains),
            "num_residues": len(self.residues),
            "num_atoms": len(self.atoms),
            "num_ligands": len(self.ligands),
            "chains": [chain.to_dict() for chain in self.chains],
            "ligands": [ligand.to_dict() for ligand in self.ligands],
            "metadata": self.metadata
        }


class PDBParser(StructuralBiologyBase):
    """
    Parser for PDB files
    
    Provides functionality to:
    - Parse PDB files from strings or files
    - Extract structural information (atoms, residues, chains, ligands)
    - Convert between different representations
    - Validate PDB files
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize PDB parser"""
        super().__init__()
        self.cache_dir = cache_dir or Path("./cache/pdb")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_file(self, file_path: Union[str, Path]) -> PDBStructure:
        """
        Parse a PDB file from disk
        
        Args:
            file_path: Path to PDB file
            
        Returns:
            PDBStructure object with parsed data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise StructuralBiologyError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            return self.parse_string(content, file_path.stem)
            
        except Exception as e:
            logger.error(f"Failed to parse PDB file {file_path}: {e}")
            raise StructuralBiologyError(f"Failed to parse PDB file: {e}")
    
    def parse_string(self, pdb_content: str, pdb_id: Optional[str] = None) -> PDBStructure:
        """
        Parse a PDB file from string
        
        Args:
            pdb_content: PDB file content as string
            pdb_id: PDB ID (optional, extracted from header if not provided)
            
        Returns:
            PDBStructure object with parsed data
        """
        try:
            # Use Biopython if available
            if self._biopython_available:
                return self._parse_with_biopython(pdb_content, pdb_id)
            else:
                return self._parse_manually(pdb_content, pdb_id)
                
        except Exception as e:
            logger.error(f"Failed to parse PDB string: {e}")
            raise StructuralBiologyError(f"Failed to parse PDB string: {e}")
    
    def _parse_with_biopython(self, pdb_content: str, pdb_id: Optional[str]) -> PDBStructure:
        """Parse PDB using Biopython"""
        PDBParser = self._get_biopython()
        
        # Create a file-like object
        pdb_file = io.StringIO(pdb_content)
        
        # Parse the structure
        parser = PDBParser()
        structure = parser.get_structure(pdb_id or "unknown", pdb_file)
        
        return self._convert_biopython_structure(structure, pdb_id)
    
    def _parse_manually(self, pdb_content: str, pdb_id: Optional[str]) -> PDBStructure:
        """Parse PDB manually (fallback when Biopython is not available)"""
        structure = PDBStructure(pdb_id=pdb_id or "unknown")
        
        current_chain = None
        current_residue = None
        
        for line in pdb_content.split('\n'):
            line = line.strip()
            if len(line) < 6:
                continue
            
            record_type = line[0:6].strip()
            
            # Parse header
            if record_type == "HEADER":
                structure.header = self._parse_header_line(line)
                
            # Parse atoms and hetero atoms
            elif record_type in ["ATOM", "HETATM"]:
                atom = self._parse_atom_line(line)
                structure.atoms.append(atom)
                
                # Create or update chain
                chain_id = atom.chain_id
                if chain_id not in [c.chain_id for c in structure.chains]:
                    structure.chains.append(Chain(chain_id=chain_id))
                
                current_chain = structure.get_chain(chain_id)
                
                # Create or update residue
                residue_key = f"{chain_id}_{atom.residue_number}_{atom.insertion_code}"
                residue_exists = False
                
                for residue in current_chain.residues:
                    if (residue.residue_number == atom.residue_number and 
                        residue.insertion_code == atom.insertion_code):
                        current_residue = residue
                        residue_exists = True
                        break
                
                if not residue_exists:
                    current_residue = Residue(
                        residue_name=atom.residue_name,
                        chain_id=chain_id,
                        residue_number=atom.residue_number,
                        insertion_code=atom.insertion_code
                    )
                    current_chain.residues.append(current_residue)
                
                current_residue.atoms.append(atom)
                
                # Check if this is a ligand (HETATM with non-standard residue)
                if record_type == "HETATM" and atom.residue_name not in ["HOH", "DOD", "NA", "CL"]:
                    ligand_key = f"{chain_id}_{atom.residue_number}_{atom.residue_name}"
                    ligand_exists = False
                    
                    for ligand in structure.ligands:
                        if (ligand.chain_id == chain_id and 
                            ligand.residue_number == atom.residue_number and
                            ligand.name == atom.residue_name):
                            ligand_exists = True
                            break
                    
                    if not ligand_exists:
                        ligand = Ligand(
                            ligand_id=atom.residue_name,
                            chain_id=chain_id,
                            residue_number=atom.residue_number,
                            name=atom.residue_name
                        )
                        structure.ligands.append(ligand)
                    
                    # Add atom to ligand
                    for lg in structure.ligands:
                        if (lg.chain_id == chain_id and 
                            lg.residue_number == atom.residue_number and
                            lg.name == atom.residue_name):
                            lg.atoms.append(atom)
                            break
            
            # Parse other records
            elif record_type == "SEQRES":
                # Sequence information
                pass
            elif record_type == "HELIX":
                # Helix information
                pass
            elif record_type == "SHEET":
                # Sheet information
                pass
            elif record_type == "CONECT":
                # Connectivity information
                pass
        
        # Extract metadata
        structure.metadata = {
            "num_models": 1,  # Simple PDB files have 1 model
            "source": "manual"
        }
        
        return structure
    
    def _parse_header_line(self, line: str) -> Dict[str, Any]:
        """Parse HEADER line"""
        # HEADER format: COLUMNS 1-6: "HEADER"
        # COLUMNS 11-50: Classification
        # COLUMNS 51-60: Date
        # COLUMNS 61-66: ID code
        
        header = {}
        header["classification"] = line[10:50].strip()
        header["date"] = line[50:60].strip()
        header["id_code"] = line[60:66].strip()
        
        return header
    
    def _parse_atom_line(self, line: str) -> Atom:
        """Parse ATOM or HETATM line"""
        # ATOM format:
        # COLUMNS 1-6: "ATOM " or "HETATM"
        # COLUMNS 7-11: Atom serial number
        # COLUMNS 13-16: Atom name
        # COLUMN 17: Alternate location indicator
        # COLUMNS 18-20: Residue name
        # COLUMN 22: Chain identifier
        # COLUMNS 23-26: Residue sequence number
        # COLUMN 27: Code for insertions of residues
        # COLUMNS 31-38: X coordinate
        # COLUMNS 39-46: Y coordinate
        # COLUMNS 47-54: Z coordinate
        # COLUMNS 55-60: Occupancy
        # COLUMNS 61-66: Temperature factor
        # COLUMNS 77-78: Element symbol
        # COLUMNS 79-80: Charge
        
        try:
            atom_number = int(line[6:11].strip())
        except:
            atom_number = 0
        
        atom_name = line[12:16].strip()
        alt_loc = line[16].strip()
        residue_name = line[17:20].strip()
        chain_id = line[21].strip()
        
        try:
            residue_number = int(line[22:26].strip())
        except:
            residue_number = 0
        
        insertion_code = line[26].strip()
        
        try:
            x = float(line[30:38].strip())
        except:
            x = 0.0
        
        try:
            y = float(line[38:46].strip())
        except:
            y = 0.0
        
        try:
            z = float(line[46:54].strip())
        except:
            z = 0.0
        
        try:
            occupancy = float(line[54:60].strip())
        except:
            occupancy = 1.0
        
        try:
            bfactor = float(line[60:66].strip())
        except:
            bfactor = 0.0
        
        element = line[76:78].strip()
        charge = line[78:80].strip()
        
        return Atom(
            atom_number=atom_number,
            atom_name=atom_name,
            alt_loc=alt_loc,
            residue_name=residue_name,
            chain_id=chain_id,
            residue_number=residue_number,
            insertion_code=insertion_code,
            x=x,
            y=y,
            z=z,
            occupancy=occupancy,
            bfactor=bfactor,
            element=element,
            charge=charge
        )
    
    def _convert_biopython_structure(self, structure, pdb_id: Optional[str]) -> PDBStructure:
        """Convert Biopython structure to PDBStructure"""
        pdb_structure = PDBStructure(pdb_id=pdb_id or "unknown")
        
        # Extract header information
        if structure.header:
            pdb_structure.header = {
                "classification": structure.header.classification,
                "deposition_date": str(structure.header.deposition_date),
                "id_code": structure.header.id_code
            }
        
        # Extract chains, residues, and atoms
        for model in structure:
            for chain in model:
                chain_data = Chain(chain_id=chain.id)
                pdb_structure.chains.append(chain_data)
                
                for residue in chain:
                    residue_data = Residue(
                        residue_name=residue.get_resname(),
                        chain_id=chain.id,
                        residue_number=residue.id[1],
                        insertion_code=residue.id[2] if len(residue.id) > 2 else ""
                    )
                    chain_data.residues.append(residue_data)
                    
                    for atom in residue:
                        atom_data = Atom(
                            atom_number=atom.serial_number,
                            atom_name=atom.name,
                            alt_loc=atom.altloc,
                            residue_name=residue.get_resname(),
                            chain_id=chain.id,
                            residue_number=residue.id[1],
                            insertion_code=residue.id[2] if len(residue.id) > 2 else "",
                            x=atom.coord[0],
                            y=atom.coord[1],
                            z=atom.coord[2],
                            occupancy=atom.occupancy,
                            bfactor=atom.bfactor,
                            element=atom.element,
                            charge=""
                        )
                        residue_data.atoms.append(atom_data)
                        pdb_structure.atoms.append(atom_data)
                    
                    pdb_structure.residues.append(residue_data)
                
                # Extract ligands (HETATM residues)
                for residue in chain:
                    if residue.id[0] != " ":  # Not a standard residue
                        ligand_data = Ligand(
                            ligand_id=residue.get_resname(),
                            chain_id=chain.id,
                            residue_number=residue.id[1],
                            name=residue.get_resname()
                        )
                        
                        for atom in residue:
                            atom_data = Atom(
                                atom_number=atom.serial_number,
                                atom_name=atom.name,
                                alt_loc=atom.altloc,
                                residue_name=residue.get_resname(),
                                chain_id=chain.id,
                                residue_number=residue.id[1],
                                insertion_code=residue.id[2] if len(residue.id) > 2 else "",
                                x=atom.coord[0],
                                y=atom.coord[1],
                                z=atom.coord[2],
                                occupancy=atom.occupancy,
                                bfactor=atom.bfactor,
                                element=atom.element,
                                charge=""
                            )
                            ligand_data.atoms.append(atom_data)
                        
                        pdb_structure.ligands.append(ligand_data)
        
        # Extract metadata
        pdb_structure.metadata = {
            "num_models": len(list(structure.get_models())),
            "source": "biopython"
        }
        
        return pdb_structure


# Singleton instance
_pdb_parser = PDBParser()


def download_pdb(pdb_id: str, format: str = "pdb", cache_dir: Optional[Path] = None) -> str:
    """
    Download PDB file
    
    Args:
        pdb_id: PDB ID
        format: File format ("pdb", "mmcif", "xml", "bcif")
        cache_dir: Cache directory
        
    Returns:
        Path to the downloaded file
    """
    from ..databases.pdb import PDBClient
    client = PDBClient()
    return client.download_pdb_file(pdb_id, format)


def parse_pdb(pdb_id: str, file_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Parse PDB file
    
    Args:
        pdb_id: PDB ID
        file_path: Path to PDB file (optional)
        **kwargs: Additional options
        
    Returns:
        Dictionary with parsed structure
    """
    try:
        if file_path:
            structure = _pdb_parser.parse_file(file_path)
        else:
            # Download and parse
            local_path = download_pdb(pdb_id)
            structure = _pdb_parser.parse_file(local_path)
        
        return structure.to_dict()
        
    except StructuralBiologyError as e:
        return {"error": str(e), "pdb_id": pdb_id}


def query_pdb(pdb_id: str, **kwargs) -> Dict[str, Any]:
    """
    Query PDB database
    
    Args:
        pdb_id: PDB ID
        **kwargs: Additional options
        
    Returns:
        Dictionary with PDB entry information
    """
    from ..databases.pdb import PDBClient
    client = PDBClient()
    
    try:
        entry = client.query(pdb_id)
        return entry.__dict__
    except Exception as e:
        return {"error": str(e), "pdb_id": pdb_id}
