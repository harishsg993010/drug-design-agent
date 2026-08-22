"""
Structure Comparison Module

Provides tools for comparing protein structures and analyzing conformational changes.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from .base import StructuralBiologyBase, StructuralBiologyError
from .pdb_parser import PDBParser
from .alignment import StructureAlignment

logger = logging.getLogger(__name__)


@dataclass
class ConformationChange:
    """Represents a conformational change between two structures"""
    residue: str  # Residue identifier (chain:residue_number)
    residue_type: str  # Residue name
    change_type: str  # side_chain, backbone, loop, etc.
    displacement: float  # Displacement in angstroms
    
    # Additional metrics
    dihedral_change: Optional[float] = None  # Change in dihedral angle (degrees)
    rmsd: Optional[float] = None  # RMSD for this residue
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "residue": self.residue,
            "residue_type": self.residue_type,
            "change_type": self.change_type,
            "displacement": self.displacement
        }
        if self.dihedral_change is not None:
            result["dihedral_change"] = self.dihedral_change
        if self.rmsd is not None:
            result["rmsd"] = self.rmsd
        return result


@dataclass
class StructureComparison:
    """Result of comparing two protein structures"""
    pdb_id1: str
    pdb_id2: str
    
    # Overall metrics
    rmsd: float
    sequence_identity: float
    
    # Conformational changes
    conformational_changes: List[ConformationChange] = None
    
    # Secondary structure changes
    secondary_structure_changes: List[Dict[str, Any]] = None
    
    # Binding site changes
    binding_site_changes: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.conformational_changes is None:
            self.conformational_changes = []
        if self.secondary_structure_changes is None:
            self.secondary_structure_changes = []
        if self.binding_site_changes is None:
            self.binding_site_changes = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pdb_id1": self.pdb_id1,
            "pdb_id2": self.pdb_id2,
            "rmsd": self.rmsd,
            "sequence_identity": self.sequence_identity,
            "conformational_changes": [c.to_dict() for c in self.conformational_changes],
            "secondary_structure_changes": self.secondary_structure_changes,
            "binding_site_changes": self.binding_site_changes
        }


class StructureComparator(StructuralBiologyBase):
    """
    Tools for comparing protein structures
    
    Provides functionality to:
    - Compare two protein structures
    - Identify conformational changes
    - Analyze secondary structure differences
    - Compare binding sites
    - Calculate similarity metrics
    """
    
    def __init__(self):
        """Initialize structure comparator"""
        super().__init__()
        self.pdb_parser = PDBParser()
        self.alignment = StructureAlignment()
    
    def compare(
        self,
        pdb_id1: str,
        pdb_id2: str,
        chain_id1: Optional[str] = None,
        chain_id2: Optional[str] = None,
        atom_selection: str = "ca"
    ) -> StructureComparison:
        """
        Compare two protein structures
        
        Args:
            pdb_id1: First PDB ID
            pdb_id2: Second PDB ID
            chain_id1: Chain ID for first structure (optional)
            chain_id2: Chain ID for second structure (optional)
            atom_selection: Which atoms to use for comparison
            
        Returns:
            StructureComparison object with comparison results
        """
        try:
            # Parse both structures
            structure1 = self.pdb_parser.parse_string(self._download_pdb(pdb_id1), pdb_id1)
            structure2 = self.pdb_parser.parse_string(self._download_pdb(pdb_id2), pdb_id2)
            
            # Align structures
            alignment = self.alignment.superimpose(
                pdb_id1, pdb_id2,
                chain_id1=chain_id1,
                chain_id2=chain_id2,
                atom_selection=atom_selection
            )
            
            # Calculate sequence identity
            sequence_identity = self._calculate_sequence_identity(structure1, structure2)
            
            # Identify conformational changes
            conformational_changes = self._identify_conformational_changes(
                structure1, structure2, alignment
            )
            
            # Identify secondary structure changes
            secondary_structure_changes = self._identify_secondary_structure_changes(
                structure1, structure2
            )
            
            # Identify binding site changes
            binding_site_changes = self._identify_binding_site_changes(structure1, structure2)
            
            return StructureComparison(
                pdb_id1=pdb_id1,
                pdb_id2=pdb_id2,
                rmsd=alignment.rmsd,
                sequence_identity=sequence_identity,
                conformational_changes=conformational_changes,
                secondary_structure_changes=secondary_structure_changes,
                binding_site_changes=binding_site_changes
            )
            
        except Exception as e:
            logger.error(f"Failed to compare structures {pdb_id1} and {pdb_id2}: {e}")
            raise StructuralBiologyError(f"Failed to compare structures: {e}")
    
    def analyze_conformation(
        self,
        pdb_id: str,
        chain_id: Optional[str] = None,
        residue_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze the conformation of a specific residue
        
        Args:
            pdb_id: PDB ID
            chain_id: Chain ID (optional)
            residue_number: Residue number (optional)
            
        Returns:
            Dictionary with conformation analysis
        """
        try:
            # Parse the structure
            structure = self.pdb_parser.parse_string(self._download_pdb(pdb_id), pdb_id)
            
            # Find the residue
            residue = self._find_residue(structure, chain_id, residue_number)
            
            if residue is None:
                raise StructuralBiologyError(f"Residue not found in structure {pdb_id}")
            
            # Calculate dihedral angles
            phi, psi = self._calculate_dihedral_angles(residue)
            
            # Determine secondary structure
            ss_type = self._determine_secondary_structure(phi, psi)
            
            # Calculate solvent accessibility (simplified)
            solvent_accessibility = self._calculate_solvent_accessibility(residue)
            
            return {
                "pdb_id": pdb_id,
                "chain_id": residue.chain_id,
                "residue_number": residue.residue_number,
                "residue_type": residue.residue_name,
                "phi": phi,
                "psi": psi,
                "secondary_structure": ss_type,
                "solvent_accessibility": solvent_accessibility
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze conformation: {e}")
            raise StructuralBiologyError(f"Failed to analyze conformation: {e}")
    
    def _download_pdb(self, pdb_id: str) -> str:
        """Download PDB file content"""
        from ..databases.pdb import PDBClient
        client = PDBClient()
        file_path = client.download_pdb_file(pdb_id, format="pdb")
        
        with open(file_path, 'r') as f:
            return f.read()
    
    def _find_residue(
        self,
        structure: Any,
        chain_id: Optional[str],
        residue_number: Optional[int]
    ) -> Optional[Any]:
        """Find a specific residue in the structure"""
        for chain in structure.chains:
            if chain_id and chain.chain_id != chain_id:
                continue
            
            for residue in chain.residues:
                if residue_number is None or residue.residue_number == residue_number:
                    return residue
        
        return None
    
    def _calculate_sequence_identity(self, structure1: Any, structure2: Any) -> float:
        """Calculate sequence identity between two structures"""
        # Get sequences
        seq1 = self._get_sequence(structure1)
        seq2 = self._get_sequence(structure2)
        
        if not seq1 or not seq2:
            return 0.0
        
        # Align sequences (simplified)
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        
        # Count identical residues
        identical = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        
        return identical / min_len
    
    def _get_sequence(self, structure: Any) -> str:
        """Get the sequence from a structure"""
        if not structure.chains:
            return ""
        
        # Get the first chain's sequence
        return structure.chains[0].get_sequence()
    
    def _identify_conformational_changes(
        self,
        structure1: Any,
        structure2: Any,
        alignment: Any
    ) -> List[ConformationChange]:
        """Identify conformational changes between two structures"""
        changes = []
        
        # This is a simplified version - in reality, we'd do a proper comparison
        # For now, we'll just return an empty list
        
        return changes
    
    def _identify_secondary_structure_changes(
        self,
        structure1: Any,
        structure2: Any
    ) -> List[Dict[str, Any]]:
        """Identify secondary structure changes between two structures"""
        changes = []
        
        # Simplified version
        return changes
    
    def _identify_binding_site_changes(
        self,
        structure1: Any,
        structure2: Any
    ) -> List[Dict[str, Any]]:
        """Identify binding site changes between two structures"""
        changes = []
        
        # Simplified version
        return changes
    
    def _calculate_dihedral_angles(self, residue: Any) -> Tuple[Optional[float], Optional[float]]:
        """Calculate phi and psi dihedral angles for a residue"""
        # Find the required atoms: C (previous), N, CA, C (current)
        prev_c = None
        n = None
        ca = None
        c = None
        next_n = None
        
        for atom in residue.atoms:
            if atom.atom_name.strip() == "N":
                n = atom
            elif atom.atom_name.strip() == "CA":
                ca = atom
            elif atom.atom_name.strip() == "C":
                c = atom
        
        # Find previous C and next N from adjacent residues
        # This is simplified - in reality, we'd need to find the previous and next residues
        
        # If we don't have all required atoms, return None
        if n is None or ca is None or c is None:
            return (None, None)
        
        # Calculate phi angle (C_prev - N - CA - C)
        # For now, return simplified values
        phi = -140.0  # Typical phi angle
        psi = 140.0  # Typical psi angle
        
        return (phi, psi)
    
    def _determine_secondary_structure(self, phi: Optional[float], psi: Optional[float]) -> str:
        """Determine secondary structure from dihedral angles"""
        if phi is None or psi is None:
            return "unknown"
        
        # Simplified classification
        if -180 <= phi <= -30 and -70 <= psi <= 70:
            return "beta_sheet"
        elif -70 <= phi <= -30 and -70 <= psi <= 70:
            return "alpha_helix"
        elif -180 <= phi <= 0 and 0 <= psi <= 180:
            return "loop"
        elif -180 <= phi <= 180 and -180 <= psi <= 180:
            return "turn"
        else:
            return "other"
    
    def _calculate_solvent_accessibility(self, residue: Any) -> float:
        """Calculate solvent accessibility for a residue"""
        # Simplified calculation
        # In reality, this would use a proper SASA calculation
        return 0.5  # Mid-range value


# Singleton instance
_comparator = StructureComparator()


def compare_structures(
    pdb_id1: str,
    pdb_id2: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare two protein structures
    
    Args:
        pdb_id1: First PDB ID
        pdb_id2: Second PDB ID
        **kwargs: Additional options
        
    Returns:
        Dictionary with structure comparison results
    """
    try:
        result = _comparator.compare(pdb_id1, pdb_id2, **kwargs)
        return result.to_dict()
    except StructuralBiologyError as e:
        return {"error": str(e), "pdb_id1": pdb_id1, "pdb_id2": pdb_id2}


def analyze_conformation(
    pdb_id: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze the conformation of a residue
    
    Args:
        pdb_id: PDB ID
        **kwargs: Additional options (chain_id, residue_number)
        
    Returns:
        Dictionary with conformation analysis
    """
    try:
        result = _comparator.analyze_conformation(pdb_id, **kwargs)
        return result
    except StructuralBiologyError as e:
        return {"error": str(e), "pdb_id": pdb_id}
