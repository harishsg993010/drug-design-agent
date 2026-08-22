"""
Binding Site Analysis Module

Provides tools for analyzing protein binding sites and molecular interactions.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from .base import StructuralBiologyBase, StructuralBiologyError
from .pdb_parser import PDBParser

logger = logging.getLogger(__name__)


@dataclass
class Interaction:
    """Represents a molecular interaction"""
    interaction_type: str  # hydrogen_bond, hydrophobic, ionic, etc.
    residue1: str  # Residue 1 (chain:residue_number)
    atom1: str  # Atom in residue 1
    residue2: str  # Residue 2 (chain:residue_number)
    atom2: str  # Atom in residue 2
    distance: float  # Distance in angstroms
    angle: Optional[float] = None  # Angle in degrees (for hydrogen bonds)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "type": self.interaction_type,
            "residue1": self.residue1,
            "atom1": self.atom1,
            "residue2": self.residue2,
            "atom2": self.atom2,
            "distance": self.distance
        }
        if self.angle is not None:
            result["angle"] = self.angle
        return result


@dataclass
class BindingSite:
    """Represents a binding site"""
    pdb_id: str
    chain_id: str
    ligand_id: str
    residue_number: int
    
    # Binding site properties
    volume: Optional[float] = None  # Volume in Å³
    surface_area: Optional[float] = None  # Surface area in Å²
    hydrophobicity: Optional[float] = None  # Hydrophobicity score
    
    # Interactions
    interactions: List[Interaction] = None
    
    # Residues in binding site
    residues: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.interactions is None:
            self.interactions = []
        if self.residues is None:
            self.residues = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pdb_id": self.pdb_id,
            "chain_id": self.chain_id,
            "ligand_id": self.ligand_id,
            "residue_number": self.residue_number,
            "volume": self.volume,
            "surface_area": self.surface_area,
            "hydrophobicity": self.hydrophobicity,
            "interactions": [i.to_dict() for i in self.interactions],
            "residues": self.residues
        }


class BindingSiteAnalyzer(StructuralBiologyBase):
    """
    Tools for analyzing protein binding sites
    
    Provides functionality to:
    - Identify binding sites
    - Analyze binding site properties
    - Detect molecular interactions
    - Compare binding sites
    - Calculate binding site metrics
    """
    
    def __init__(self):
        """Initialize binding site analyzer"""
        super().__init__()
        self.pdb_parser = PDBParser()
    
    def analyze(
        self,
        pdb_id: str,
        chain_id: Optional[str] = None,
        residue_number: Optional[int] = None,
        ligand_id: Optional[str] = None
    ) -> BindingSite:
        """
        Analyze a binding site in a PDB structure
        
        Args:
            pdb_id: PDB ID
            chain_id: Chain ID (optional)
            residue_number: Residue number of the ligand (optional)
            ligand_id: Ligand ID (optional)
            
        Returns:
            BindingSite object with analysis results
        """
        try:
            # Parse the structure
            structure = self.pdb_parser.parse_string(self._download_pdb(pdb_id), pdb_id)
            
            # Find the ligand
            ligand = self._find_ligand(structure, chain_id, residue_number, ligand_id)
            
            if ligand is None:
                raise StructuralBiologyError(f"Ligand not found in structure {pdb_id}")
            
            # Identify binding site residues
            binding_site_residues = self._identify_binding_site_residues(structure, ligand)
            
            # Find interactions
            interactions = self._find_interactions(structure, ligand, binding_site_residues)
            
            # Calculate binding site properties
            volume = self._calculate_binding_site_volume(binding_site_residues)
            surface_area = self._calculate_binding_site_surface_area(binding_site_residues)
            hydrophobicity = self._calculate_binding_site_hydrophobicity(binding_site_residues)
            
            # Create binding site object
            binding_site = BindingSite(
                pdb_id=pdb_id,
                chain_id=ligand.chain_id,
                ligand_id=ligand.ligand_id,
                residue_number=ligand.residue_number,
                volume=volume,
                surface_area=surface_area,
                hydrophobicity=hydrophobicity,
                interactions=interactions,
                residues=[
                    {
                        "chain": r.chain_id,
                        "residue_number": r.residue_number,
                        "residue_name": r.residue_name,
                        "insertion_code": r.insertion_code
                    }
                    for r in binding_site_residues
                ]
            )
            
            return binding_site
            
        except Exception as e:
            logger.error(f"Failed to analyze binding site in {pdb_id}: {e}")
            raise StructuralBiologyError(f"Failed to analyze binding site: {e}")
    
    def find_interactions(
        self,
        pdb_id: str,
        chain_id: Optional[str] = None,
        residue_number: Optional[int] = None,
        ligand_id: Optional[str] = None,
        distance_threshold: float = 5.0
    ) -> List[Interaction]:
        """
        Find molecular interactions in a PDB structure
        
        Args:
            pdb_id: PDB ID
            chain_id: Chain ID (optional)
            residue_number: Residue number of the ligand (optional)
            ligand_id: Ligand ID (optional)
            distance_threshold: Maximum distance for interaction (Å)
            
        Returns:
            List of Interaction objects
        """
        try:
            # Parse the structure
            structure = self.pdb_parser.parse_string(self._download_pdb(pdb_id), pdb_id)
            
            # Find the ligand
            ligand = self._find_ligand(structure, chain_id, residue_number, ligand_id)
            
            if ligand is None:
                raise StructuralBiologyError(f"Ligand not found in structure {pdb_id}")
            
            # Identify binding site residues
            binding_site_residues = self._identify_binding_site_residues(structure, ligand)
            
            # Find interactions
            interactions = self._find_interactions(structure, ligand, binding_site_residues, distance_threshold)
            
            return interactions
            
        except Exception as e:
            logger.error(f"Failed to find interactions in {pdb_id}: {e}")
            raise StructuralBiologyError(f"Failed to find interactions: {e}")
    
    def compare_binding_sites(
        self,
        pdb_id1: str,
        pdb_id2: str,
        ligand_id1: Optional[str] = None,
        ligand_id2: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare binding sites between two structures
        
        Args:
            pdb_id1: First PDB ID
            pdb_id2: Second PDB ID
            ligand_id1: Ligand ID in first structure (optional)
            ligand_id2: Ligand ID in second structure (optional)
            
        Returns:
            Dictionary with comparison results
        """
        try:
            site1 = self.analyze(pdb_id1, ligand_id=ligand_id1)
            site2 = self.analyze(pdb_id2, ligand_id=ligand_id2)
            
            # Compare interactions
            comparison = {
                "pdb_id1": pdb_id1,
                "pdb_id2": pdb_id2,
                "site1": site1.to_dict(),
                "site2": site2.to_dict(),
                "similarity": self._compare_binding_sites(site1, site2)
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare binding sites: {e}")
            raise StructuralBiologyError(f"Failed to compare binding sites: {e}")
    
    def _download_pdb(self, pdb_id: str) -> str:
        """Download PDB file content"""
        from ..databases.pdb import PDBClient
        client = PDBClient()
        file_path = client.download_pdb_file(pdb_id, format="pdb")
        
        with open(file_path, 'r') as f:
            return f.read()
    
    def _find_ligand(
        self,
        structure: Any,
        chain_id: Optional[str],
        residue_number: Optional[int],
        ligand_id: Optional[str]
    ) -> Optional[Any]:
        """Find a ligand in the structure"""
        # If specific ligand info provided, find it
        if chain_id and residue_number:
            for ligand in structure.ligands:
                if (ligand.chain_id == chain_id and 
                    ligand.residue_number == residue_number):
                    return ligand
        
        if ligand_id:
            for ligand in structure.ligands:
                if ligand.ligand_id == ligand_id or ligand.name == ligand_id:
                    return ligand
        
        # If no specific info, return first ligand
        if structure.ligands:
            return structure.ligands[0]
        
        return None
    
    def _identify_binding_site_residues(self, structure: Any, ligand: Any, distance: float = 8.0) -> List[Any]:
        """Identify residues in the binding site"""
        binding_site_residues = []
        
        # Get ligand center
        ligand_coords = np.array([
            [atom.x, atom.y, atom.z] for atom in ligand.atoms
        ])
        ligand_center = np.mean(ligand_coords, axis=0)
        
        # Find residues within distance threshold of ligand
        for chain in structure.chains:
            for residue in chain.residues:
                # Skip the ligand itself
                if (residue.chain_id == ligand.chain_id and 
                    residue.residue_number == ligand.residue_number):
                    continue
                
                # Get residue center
                residue_coords = np.array([
                    [atom.x, atom.y, atom.z] for atom in residue.atoms
                ])
                residue_center = np.mean(residue_coords, axis=0)
                
                # Calculate distance
                dist = np.linalg.norm(residue_center - ligand_center)
                
                if dist <= distance:
                    binding_site_residues.append(residue)
        
        return binding_site_residues
    
    def _find_interactions(
        self,
        structure: Any,
        ligand: Any,
        binding_site_residues: List[Any],
        distance_threshold: float = 5.0
    ) -> List[Interaction]:
        """Find interactions between ligand and binding site residues"""
        interactions = []
        
        # Get all atoms
        ligand_atoms = ligand.atoms
        residue_atoms = []
        for residue in binding_site_residues:
            residue_atoms.extend(residue.atoms)
        
        # Find close contacts
        for ligand_atom in ligand_atoms:
            for residue_atom in residue_atoms:
                distance = self._calculate_distance(ligand_atom, residue_atom)
                
                if distance <= distance_threshold:
                    # Determine interaction type
                    interaction_type = self._determine_interaction_type(ligand_atom, residue_atom, distance)
                    
                    if interaction_type:
                        interaction = Interaction(
                            interaction_type=interaction_type,
                            residue1=f"{ligand.chain_id}:{ligand.residue_number}",
                            atom1=ligand_atom.atom_name,
                            residue2=f"{residue_atom.chain_id}:{residue_atom.residue_number}",
                            atom2=residue_atom.atom_name,
                            distance=distance
                        )
                        
                        # Calculate angle for hydrogen bonds
                        if interaction_type == "hydrogen_bond":
                            angle = self._calculate_hbond_angle(ligand_atom, residue_atom)
                            if angle is not None:
                                interaction.angle = angle
                        
                        interactions.append(interaction)
        
        return interactions
    
    def _calculate_distance(self, atom1: Any, atom2: Any) -> float:
        """Calculate distance between two atoms"""
        return np.sqrt(
            (atom1.x - atom2.x) ** 2 + 
            (atom1.y - atom2.y) ** 2 + 
            (atom1.z - atom2.z) ** 2
        )
    
    def _determine_interaction_type(
        self,
        atom1: Any,
        atom2: Any,
        distance: float
    ) -> Optional[str]:
        """Determine the type of interaction between two atoms"""
        # Hydrogen bond (N/O to H)
        if self._is_hydrogen_bond(atom1, atom2):
            return "hydrogen_bond"
        
        # Hydrophobic contact
        if self._is_hydrophobic(atom1, atom2):
            return "hydrophobic"
        
        # Ionic interaction
        if self._is_ionic(atom1, atom2):
            return "ionic"
        
        # Van der Waals
        if distance < 4.0:  # Typical van der Waals distance
            return "van_der_waals"
        
        return None
    
    def _is_hydrogen_bond(self, atom1: Any, atom2: Any) -> bool:
        """Check if two atoms form a hydrogen bond"""
        # Hydrogen bond: donor (N/O) to acceptor (N/O) with H in between
        donor_atoms = ["N", "O"]
        acceptor_atoms = ["N", "O"]
        
        # Check if one is donor and other is acceptor
        is_donor1 = atom1.atom_name.strip() in donor_atoms
        is_acceptor2 = atom2.atom_name.strip() in acceptor_atoms
        is_donor2 = atom2.atom_name.strip() in donor_atoms
        is_acceptor1 = atom1.atom_name.strip() in acceptor_atoms
        
        return (is_donor1 and is_acceptor2) or (is_donor2 and is_acceptor1)
    
    def _is_hydrophobic(self, atom1: Any, atom2: Any) -> bool:
        """Check if two atoms form a hydrophobic contact"""
        hydrophobic_atoms = ["C", "H"]
        return (atom1.element in hydrophobic_atoms and atom2.element in hydrophobic_atoms)
    
    def _is_ionic(self, atom1: Any, atom2: Any) -> bool:
        """Check if two atoms form an ionic interaction"""
        charged_atoms = {"N": [1, -1], "O": [-1, 0], "S": [-1, 0]}  # Simplified
        
        # Check if atoms have opposite charges
        charge1 = self._get_atomic_charge(atom1)
        charge2 = self._get_atomic_charge(atom2)
        
        return charge1 is not None and charge2 is not None and charge1 * charge2 < 0
    
    def _get_atomic_charge(self, atom: Any) -> Optional[int]:
        """Get the charge of an atom"""
        if atom.charge:
            try:
                return int(atom.charge)
            except:
                pass
        
        # Default charges for common atoms
        default_charges = {
            "N": 0, "O": 0, "C": 0, "H": 0,
            "NH": 1, "OH": 0, "CH": 0,
            "NZ": 1, "OZ": -1, "SZ": -1
        }
        
        return default_charges.get(atom.atom_name.strip())
    
    def _calculate_hbond_angle(self, atom1: Any, atom2: Any) -> Optional[float]:
        """Calculate the angle for a hydrogen bond"""
        # This is a simplified version - in reality, we'd need to identify the hydrogen
        # and calculate the donor-H-acceptor angle
        return 150.0  # Typical hydrogen bond angle
    
    def _calculate_binding_site_volume(self, residues: List[Any]) -> float:
        """Calculate the volume of a binding site"""
        # This is a simplified calculation
        # In reality, we'd use a proper volume calculation algorithm
        if not residues:
            return 0.0
        
        # Count atoms and estimate volume
        num_atoms = sum(len(r.atoms) for r in residues)
        return num_atoms * 20.0  # Rough estimate
    
    def _calculate_binding_site_surface_area(self, residues: List[Any]) -> float:
        """Calculate the surface area of a binding site"""
        # Simplified calculation
        if not residues:
            return 0.0
        
        num_atoms = sum(len(r.atoms) for r in residues)
        return num_atoms * 15.0  # Rough estimate
    
    def _calculate_binding_site_hydrophobicity(self, residues: List[Any]) -> float:
        """Calculate the hydrophobicity of a binding site"""
        if not residues:
            return 0.0
        
        # Count hydrophobic residues
        hydrophobic_residues = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR"}
        total_residues = len(residues)
        hydrophobic_count = sum(
            1 for r in residues if r.residue_name in hydrophobic_residues
        )
        
        return hydrophobic_count / total_residues if total_residues > 0 else 0.0
    
    def _compare_binding_sites(self, site1: BindingSite, site2: BindingSite) -> Dict[str, Any]:
        """Compare two binding sites"""
        # Compare number of interactions
        num_interactions1 = len(site1.interactions)
        num_interactions2 = len(site2.interactions)
        
        # Compare interaction types
        types1 = {i.interaction_type for i in site1.interactions}
        types2 = {i.interaction_type for i in site2.interactions}
        
        return {
            "num_interactions_similarity": 1 - abs(num_interactions1 - num_interactions2) / max(num_interactions1, num_interactions2, 1),
            "interaction_types_match": len(types1 & types2) / max(len(types1 | types2), 1),
            "volume_ratio": site1.volume / site2.volume if site2.volume > 0 else 0,
            "hydrophobicity_difference": abs(site1.hydrophobicity - site2.hydrophobicity)
        }


# Singleton instance
_binding_site_analyzer = BindingSiteAnalyzer()


def analyze_binding_site(
    pdb_id: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze a binding site in a PDB structure
    
    Args:
        pdb_id: PDB ID
        **kwargs: Additional options (chain_id, residue_number, ligand_id)
        
    Returns:
        Dictionary with binding site analysis
    """
    try:
        result = _binding_site_analyzer.analyze(pdb_id, **kwargs)
        return result.to_dict()
    except StructuralBiologyError as e:
        return {"error": str(e), "pdb_id": pdb_id}


def find_interactions(
    pdb_id: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Find molecular interactions in a PDB structure
    
    Args:
        pdb_id: PDB ID
        **kwargs: Additional options
        
    Returns:
        Dictionary with list of interactions
    """
    try:
        interactions = _binding_site_analyzer.find_interactions(pdb_id, **kwargs)
        return {"pdb_id": pdb_id, "interactions": [i.to_dict() for i in interactions]}
    except StructuralBiologyError as e:
        return {"error": str(e), "pdb_id": pdb_id}
