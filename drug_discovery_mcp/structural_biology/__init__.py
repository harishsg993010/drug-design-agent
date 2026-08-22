"""
Structural Biology Module

This module provides tools for analyzing protein structures and molecular interactions.
Inspired by BIOMNI framework's structural biology tools.

Features:
- PDB file parsing and manipulation
- Structure superimposition and alignment
- Binding site analysis
- Molecular docking preparation
- Protein-ligand interaction analysis
- Solvent accessibility analysis
- Secondary structure analysis
- Protein-protein interaction analysis
"""

from .base import StructuralBiologyError
from .pdb_parser import PDBParser, download_pdb, parse_pdb, query_pdb
from .alignment import StructureAlignment, superimpose_structures, calculate_rmsd
from .binding_site import BindingSiteAnalyzer, analyze_binding_site, find_interactions
from .comparison import StructureComparator, compare_structures, analyze_conformation
from .tools import StructuralBiologyTools

__all__ = [
    "StructuralBiologyError",
    "PDBParser",
    "download_pdb",
    "parse_pdb",
    "query_pdb",
    "StructureAlignment",
    "superimpose_structures",
    "calculate_rmsd",
    "BindingSiteAnalyzer",
    "analyze_binding_site",
    "find_interactions",
    "StructureComparator",
    "compare_structures",
    "analyze_conformation",
    "StructuralBiologyTools",
]
