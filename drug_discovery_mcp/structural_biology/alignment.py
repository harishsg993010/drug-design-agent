"""
Structure Alignment Module

Provides tools for superimposing and aligning protein structures.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from .base import StructuralBiologyBase, StructuralBiologyError
from .pdb_parser import PDBParser

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Result of a structure alignment"""
    pdb_id1: str
    pdb_id2: str
    rmsd: float
    aligned_residues: int
    transformation_matrix: Optional[np.ndarray] = None
    translation_vector: Optional[np.ndarray] = None
    residue_differences: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.residue_differences is None:
            self.residue_differences = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "pdb_id1": self.pdb_id1,
            "pdb_id2": self.pdb_id2,
            "rmsd": self.rmsd,
            "aligned_residues": self.aligned_residues
        }
        
        if self.transformation_matrix is not None:
            result["transformation_matrix"] = self.transformation_matrix.tolist()
        if self.translation_vector is not None:
            result["translation_vector"] = self.translation_vector.tolist()
        if self.residue_differences:
            result["residue_differences"] = self.residue_differences
        
        return result


class StructureAlignment(StructuralBiologyBase):
    """
    Tools for aligning and superimposing protein structures
    
    Provides functionality to:
    - Superimpose two protein structures
    - Calculate RMSD between structures
    - Align specific chains or residues
    - Compare conformations
    """
    
    def __init__(self):
        """Initialize structure alignment"""
        super().__init__()
        self.pdb_parser = PDBParser()
    
    def superimpose(
        self,
        pdb_id1: str,
        pdb_id2: str,
        chain_id1: Optional[str] = None,
        chain_id2: Optional[str] = None,
        atom_selection: str = "ca"  # "ca" for C-alpha, "all" for all atoms, "backbone" for backbone
    ) -> AlignmentResult:
        """
        Superimpose two protein structures
        
        Args:
            pdb_id1: First PDB ID
            pdb_id2: Second PDB ID
            chain_id1: Chain ID for first structure (None for first chain)
            chain_id2: Chain ID for second structure (None for first chain)
            atom_selection: Which atoms to use for alignment ("ca", "all", "backbone")
            
        Returns:
            AlignmentResult with alignment information
        """
        try:
            # Parse both structures
            structure1 = self.pdb_parser.parse_string(self._download_pdb(pdb_id1), pdb_id1)
            structure2 = self.pdb_parser.parse_string(self._download_pdb(pdb_id2), pdb_id2)
            
            # Get atoms for alignment
            atoms1, coords1 = self._get_atoms_for_alignment(structure1, chain_id1, atom_selection)
            atoms2, coords2 = self._get_atoms_for_alignment(structure2, chain_id2, atom_selection)
            
            if len(coords1) == 0 or len(coords2) == 0:
                raise StructuralBiologyError("No atoms found for alignment")
            
            # Align structures using Kabsch algorithm
            rmsd, rotation, translation = self._kabsch_align(coords1, coords2)
            
            # Apply transformation to second structure
            aligned_coords2 = self._apply_transformation(coords2, rotation, translation)
            
            # Calculate residue differences
            residue_differences = self._calculate_residue_differences(
                structure1, structure2, rotation, translation
            )
            
            return AlignmentResult(
                pdb_id1=pdb_id1,
                pdb_id2=pdb_id2,
                rmsd=rmsd,
                aligned_residues=len(coords1),
                transformation_matrix=rotation,
                translation_vector=translation,
                residue_differences=residue_differences
            )
            
        except Exception as e:
            logger.error(f"Failed to superimpose structures {pdb_id1} and {pdb_id2}: {e}")
            raise StructuralBiologyError(f"Failed to superimpose structures: {e}")
    
    def calculate_rmsd(
        self,
        pdb_id1: str,
        pdb_id2: str,
        chain_id1: Optional[str] = None,
        chain_id2: Optional[str] = None,
        atom_selection: str = "ca"
    ) -> float:
        """
        Calculate RMSD between two structures
        
        Args:
            pdb_id1: First PDB ID
            pdb_id2: Second PDB ID
            chain_id1: Chain ID for first structure
            chain_id2: Chain ID for second structure
            atom_selection: Which atoms to use for RMSD calculation
            
        Returns:
            RMSD value
        """
        try:
            result = self.superimpose(
                pdb_id1, pdb_id2, 
                chain_id1=chain_id1, 
                chain_id2=chain_id2,
                atom_selection=atom_selection
            )
            return result.rmsd
            
        except Exception as e:
            logger.error(f"Failed to calculate RMSD: {e}")
            raise StructuralBiologyError(f"Failed to calculate RMSD: {e}")
    
    def align_chains(
        self,
        pdb_id1: str,
        chain_id1: str,
        pdb_id2: str,
        chain_id2: str,
        atom_selection: str = "ca"
    ) -> AlignmentResult:
        """
        Align specific chains from two structures
        
        Args:
            pdb_id1: First PDB ID
            chain_id1: Chain ID in first structure
            pdb_id2: Second PDB ID
            chain_id2: Chain ID in second structure
            atom_selection: Which atoms to use for alignment
            
        Returns:
            AlignmentResult with alignment information
        """
        return self.superimpose(
            pdb_id1, pdb_id2,
            chain_id1=chain_id1,
            chain_id2=chain_id2,
            atom_selection=atom_selection
        )
    
    def _download_pdb(self, pdb_id: str) -> str:
        """Download PDB file content"""
        from ..databases.pdb import PDBClient
        client = PDBClient()
        file_path = client.download_pdb_file(pdb_id, format="pdb")
        
        with open(file_path, 'r') as f:
            return f.read()
    
    def _get_atoms_for_alignment(
        self,
        structure: Any,
        chain_id: Optional[str],
        atom_selection: str
    ) -> Tuple[List[Any], np.ndarray]:
        """Get atoms and coordinates for alignment"""
        atoms = []
        coords = []
        
        # Get the specified chain or first chain
        target_chain = None
        if chain_id:
            for chain in structure.chains:
                if chain.chain_id == chain_id:
                    target_chain = chain
                    break
        else:
            if structure.chains:
                target_chain = structure.chains[0]
        
        if not target_chain:
            return atoms, np.array(coords)
        
        # Select atoms based on atom_selection
        for residue in target_chain.residues:
            for atom in residue.atoms:
                if atom_selection == "ca" and atom.atom_name.strip() == "CA":
                    atoms.append(atom)
                    coords.append([atom.x, atom.y, atom.z])
                elif atom_selection == "backbone":
                    if atom.atom_name.strip() in ["N", "CA", "C", "O"]:
                        atoms.append(atom)
                        coords.append([atom.x, atom.y, atom.z])
                elif atom_selection == "all":
                    atoms.append(atom)
                    coords.append([atom.x, atom.y, atom.z])
        
        return atoms, np.array(coords)
    
    def _kabsch_align(self, coords1: np.ndarray, coords2: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Align two sets of coordinates using the Kabsch algorithm
        
        Args:
            coords1: First set of coordinates (N x 3)
            coords2: Second set of coordinates (N x 3)
            
        Returns:
            Tuple of (RMSD, rotation_matrix, translation_vector)
        """
        # Center the coordinates
        centroid1 = np.mean(coords1, axis=0)
        centroid2 = np.mean(coords2, axis=0)
        
        centered1 = coords1 - centroid1
        centered2 = coords2 - centroid2
        
        # Calculate covariance matrix
        covariance = np.dot(centered1.T, centered2)
        
        # Singular value decomposition
        U, S, Vt = np.linalg.svd(covariance)
        
        # Calculate rotation matrix
        rotation = np.dot(Vt.T, U.T)
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(rotation) < 0:
            Vt[-1, :] *= -1
            rotation = np.dot(Vt.T, U.T)
        
        # Calculate RMSD
        aligned_coords2 = np.dot(centered2, rotation.T)
        rmsd = np.sqrt(np.mean(np.sum((centered1 - aligned_coords2) ** 2, axis=1)))
        
        # Calculate translation vector
        translation = centroid1 - np.dot(rotation, centroid2)
        
        return rmsd, rotation, translation
    
    def _apply_transformation(self, coords: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """Apply rotation and translation to coordinates"""
        return np.dot(coords, rotation.T) + translation
    
    def _calculate_residue_differences(
        self,
        structure1: Any,
        structure2: Any,
        rotation: np.ndarray,
        translation: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Calculate differences between corresponding residues"""
        differences = []
        
        # This is a simplified version - in a real implementation, we'd properly map residues
        # between the two structures
        
        # For now, just return empty list
        return differences


# Singleton instance
_alignment = StructureAlignment()


def superimpose_structures(
    pdb_id1: str,
    pdb_id2: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Superimpose two protein structures
    
    Args:
        pdb_id1: First PDB ID
        pdb_id2: Second PDB ID
        **kwargs: Additional options (chain_id1, chain_id2, atom_selection)
        
    Returns:
        Dictionary with alignment results
    """
    try:
        result = _alignment.superimpose(pdb_id1, pdb_id2, **kwargs)
        return result.to_dict()
    except StructuralBiologyError as e:
        return {"error": str(e), "pdb_id1": pdb_id1, "pdb_id2": pdb_id2}


def calculate_rmsd(
    pdb_id1: str,
    pdb_id2: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Calculate RMSD between two structures
    
    Args:
        pdb_id1: First PDB ID
        pdb_id2: Second PDB ID
        **kwargs: Additional options
        
    Returns:
        Dictionary with RMSD value
    """
    try:
        rmsd = _alignment.calculate_rmsd(pdb_id1, pdb_id2, **kwargs)
        return {"pdb_id1": pdb_id1, "pdb_id2": pdb_id2, "rmsd": rmsd}
    except StructuralBiologyError as e:
        return {"error": str(e), "pdb_id1": pdb_id1, "pdb_id2": pdb_id2}
