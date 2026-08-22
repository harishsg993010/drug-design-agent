"""
Base Structural Biology Module

Provides the foundation for structural biology operations with error handling
and common functionality.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class StructuralBiologyError(Exception):
    """Base exception for structural biology errors"""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        return f"Structural Biology Error: {self.message}"


@dataclass
class StructureData:
    """Base class for structural data"""
    pdb_id: Optional[str] = None
    chain_id: Optional[str] = None
    residue_id: Optional[str] = None
    
    def __post_init__(self):
        pass


class StructuralBiologyBase:
    """
    Base class for structural biology tools
    
    Provides common functionality for structural biology operations including:
    - Biopython integration
    - PDB file handling
    - Coordinate manipulation
    - Error handling
    """
    
    def __init__(self):
        """Initialize the structural biology base"""
        self._biopython_available = self._check_biopython()
        if not self._biopython_available:
            logger.warning("Biopython is not available. Some structural biology functions may not work.")
    
    def _check_biopython(self) -> bool:
        """Check if Biopython is available"""
        try:
            import Bio
            from Bio.PDB import PDBParser
            return True
        except ImportError:
            return False
    
    def _get_biopython(self):
        """Get Biopython module with error checking"""
        if not self._biopython_available:
            raise StructuralBiologyError(
                "Biopython is not installed. Please install it with: pip install biopython"
            )
        
        try:
            from Bio.PDB import PDBParser
            return PDBParser
        except ImportError as e:
            raise StructuralBiologyError(f"Failed to import Biopython: {e}")
    
    def _get_numpy(self):
        """Get numpy module with error checking"""
        try:
            import numpy as np
            return np
        except ImportError as e:
            raise StructuralBiologyError(f"Failed to import numpy: {e}")
    
    def _get_scipy(self):
        """Get scipy module with error checking"""
        try:
            import scipy
            return scipy
        except ImportError as e:
            raise StructuralBiologyError(f"Failed to import scipy: {e}")
