"""
Structural Biology Tools Module

Provides a unified interface for all structural biology operations.
This is the main entry point for structural biology tools from the MCP server.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from .pdb_parser import PDBParser, download_pdb, parse_pdb, query_pdb
from .alignment import StructureAlignment, superimpose_structures, calculate_rmsd
from .binding_site import BindingSiteAnalyzer, analyze_binding_site, find_interactions
from .comparison import StructureComparator, compare_structures, analyze_conformation
from .base import StructuralBiologyError

logger = logging.getLogger(__name__)


class StructuralBiologyTools:
    """
    Unified interface for all structural biology operations
    
    This class provides a single entry point for all structural biology tools,
    making it easy to access different structural biology functions through a
    consistent interface.
    """
    
    def __init__(self):
        """Initialize all structural biology components"""
        self.pdb_parser = PDBParser()
        self.alignment = StructureAlignment()
        self.binding_site = BindingSiteAnalyzer()
        self.comparator = StructureComparator()
    
    def initialize(self):
        """Initialize all structural biology components"""
        logger.info("Initializing structural biology tools")
        # All components are initialized in __init__
    
    async def close(self):
        """Close all structural biology components"""
        logger.info("Closing structural biology tools")
        # No cleanup needed for structural biology tools
    
    # PDB methods
    def download_pdb(self, pdb_id: str, **kwargs) -> str:
        """Download PDB file"""
        return download_pdb(pdb_id, **kwargs)
    
    def parse_pdb(self, pdb_id: str, **kwargs) -> Dict[str, Any]:
        """Parse PDB file"""
        return parse_pdb(pdb_id, **kwargs)
    
    def query_pdb(self, pdb_id: str, **kwargs) -> Dict[str, Any]:
        """Query PDB database"""
        return query_pdb(pdb_id, **kwargs)
    
    # Alignment methods
    def superimpose_structures(self, pdb_id1: str, pdb_id2: str, **kwargs) -> Dict[str, Any]:
        """Superimpose two protein structures"""
        return superimpose_structures(pdb_id1, pdb_id2, **kwargs)
    
    def calculate_rmsd(self, pdb_id1: str, pdb_id2: str, **kwargs) -> Dict[str, Any]:
        """Calculate RMSD between two structures"""
        return calculate_rmsd(pdb_id1, pdb_id2, **kwargs)
    
    # Binding site methods
    def analyze_binding_site(self, pdb_id: str, **kwargs) -> Dict[str, Any]:
        """Analyze binding site"""
        return analyze_binding_site(pdb_id, **kwargs)
    
    def find_interactions(self, pdb_id: str, **kwargs) -> Dict[str, Any]:
        """Find molecular interactions"""
        return find_interactions(pdb_id, **kwargs)
    
    # Comparison methods
    def compare_structures(self, pdb_id1: str, pdb_id2: str, **kwargs) -> Dict[str, Any]:
        """Compare two structures"""
        return compare_structures(pdb_id1, pdb_id2, **kwargs)
    
    def analyze_conformation(self, pdb_id: str, **kwargs) -> Dict[str, Any]:
        """Analyze protein conformation"""
        return analyze_conformation(pdb_id, **kwargs)
    
    # Combined methods
    def analyze_structure(self, pdb_id: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a protein structure
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        results = {
            "pdb_id": pdb_id,
            "entry": self.query_pdb(pdb_id),
            "structure": self.parse_pdb(pdb_id),
            "binding_sites": self.analyze_binding_site(pdb_id),
            "interactions": self.find_interactions(pdb_id)
        }
        
        return results
    
    def compare_binding_sites(self, pdb_id1: str, pdb_id2: str, **kwargs) -> Dict[str, Any]:
        """
        Compare binding sites between two structures
        
        Args:
            pdb_id1: First PDB ID
            pdb_id2: Second PDB ID
            **kwargs: Additional options
            
        Returns:
            Dictionary with comparison results
        """
        return {
            "structure1": self.analyze_structure(pdb_id1),
            "structure2": self.analyze_structure(pdb_id2),
            "alignment": self.superimpose_structures(pdb_id1, pdb_id2),
            "rmsd": self.calculate_rmsd(pdb_id1, pdb_id2)
        }
    
    def extract_ligand(self, pdb_id: str, ligand_name: str, **kwargs) -> Dict[str, Any]:
        """
        Extract ligand from a PDB structure
        
        Args:
            pdb_id: PDB ID
            ligand_name: Name of the ligand to extract
            **kwargs: Additional options
            
        Returns:
            Dictionary with ligand information
        """
        try:
            structure = self.parse_pdb(pdb_id)
            
            # Find the ligand
            ligands = structure.get("ligands", [])
            for ligand in ligands:
                if ligand.get("name") == ligand_name or ligand.get("ligand_id") == ligand_name:
                    return {
                        "pdb_id": pdb_id,
                        "ligand": ligand,
                        "found": True
                    }
            
            return {
                "pdb_id": pdb_id,
                "ligand_name": ligand_name,
                "found": False,
                "error": f"Ligand {ligand_name} not found in structure"
            }
            
        except Exception as e:
            logger.error(f"Failed to extract ligand {ligand_name} from {pdb_id}: {e}")
            return {"error": str(e), "pdb_id": pdb_id, "ligand_name": ligand_name}
    
    def analyze_solvent_accessibility(self, pdb_id: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze solvent accessibility of a protein structure
        
        Args:
            pdb_id: PDB ID
            **kwargs: Additional options
            
        Returns:
            Dictionary with solvent accessibility analysis
        """
        try:
            # This would use a proper SASA calculation in a full implementation
            # For now, we'll provide a simplified version
            
            structure = self.parse_pdb(pdb_id)
            atoms = structure.get("atoms", [])
            
            # Count atoms by element
            element_counts = {}
            for atom in atoms:
                element = atom.get("element", "")
                element_counts[element] = element_counts.get(element, 0) + 1
            
            return {
                "pdb_id": pdb_id,
                "total_atoms": len(atoms),
                "element_counts": element_counts,
                "sasa_approximate": len(atoms) * 20,  # Rough estimate
                "note": "This is a simplified analysis. For accurate SASA calculation, use specialized tools."
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze solvent accessibility for {pdb_id}: {e}")
            return {"error": str(e), "pdb_id": pdb_id}
