"""
Base Cheminformatics Module

Provides the foundation for cheminformatics operations with error handling
and common functionality.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CheminformaticsError(Exception):
    """Base exception for cheminformatics errors"""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        return f"Cheminformatics Error: {self.message}"


@dataclass
class MolecularData:
    """Base class for molecular data"""
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    inchikey: Optional[str] = None
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    
    def __post_init__(self):
        pass


class CheminformaticsBase:
    """
    Base class for cheminformatics tools
    
    Provides common functionality for cheminformatics operations including:
    - RDKit integration
    - Error handling
    - Result validation
    - Caching
    """
    
    def __init__(self):
        """Initialize the cheminformatics base"""
        self._rdkit_available = self._check_rdkit()
        if not self._rdkit_available:
            logger.warning("RDKit is not available. Some cheminformatics functions may not work.")
    
    def _check_rdkit(self) -> bool:
        """Check if RDKit is available"""
        try:
            import rdkit  # noqa: F401
            from rdkit import Chem  # noqa: F401
            return True
        except ImportError:
            return False
    
    def _get_rdkit(self):
        """
        Get the RDKit ``Chem`` module with error checking
        
        The sub-modules used across this package (``Crippen``, ``Lipinski``,
        ``rdMolDescriptors``, ``Descriptors`` and ``AllChem``) are not bound on
        ``rdkit.Chem`` until they are imported, so import them here to make
        ``Chem.<submodule>`` attribute access work for callers.
        """
        if not self._rdkit_available:
            raise CheminformaticsError(
                "RDKit is not installed. Please install it with: conda install -c conda-forge rdkit"
            )
        
        try:
            from rdkit import Chem
            from rdkit.Chem import (  # noqa: F401
                AllChem,
                Crippen,
                Descriptors,
                Lipinski,
                rdMolDescriptors,
            )
            return Chem
        except ImportError as e:
            raise CheminformaticsError(f"Failed to import RDKit: {e}")
    
    def _validate_molecule(self, mol: Any) -> bool:
        """Validate that a molecule object is valid"""
        if mol is None:
            return False
        
        try:
            Chem = self._get_rdkit()
            return Chem.MolToSmiles(mol) is not None
        except:
            return False
    
    def _sanitize_smiles(self, smiles: str) -> str:
        """Sanitize a SMILES string"""
        try:
            Chem = self._get_rdkit()
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol)
            return smiles
        except Exception as e:
            logger.warning(f"Failed to sanitize SMILES: {e}")
            return smiles
    
    def _smiles_to_mol(self, smiles: str) -> Any:
        """Convert SMILES to RDKit molecule"""
        try:
            Chem = self._get_rdkit()
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise CheminformaticsError(f"Invalid SMILES: {smiles}")
            return mol
        except Exception as e:
            raise CheminformaticsError(f"Failed to convert SMILES to molecule: {e}")
    
    def _mol_to_smiles(self, mol: Any) -> str:
        """Convert RDKit molecule to SMILES"""
        try:
            Chem = self._get_rdkit()
            smiles = Chem.MolToSmiles(mol)
            if smiles is None:
                raise CheminformaticsError("Failed to convert molecule to SMILES")
            return smiles
        except Exception as e:
            raise CheminformaticsError(f"Failed to convert molecule to SMILES: {e}")
    
    def _get_molecular_formula(self, mol: Any) -> str:
        """Get molecular formula from molecule"""
        try:
            Chem = self._get_rdkit()
            return Chem.rdMolDescriptors.CalcMolFormula(mol)
        except Exception as e:
            logger.warning(f"Failed to get molecular formula: {e}")
            return ""
    
    def _get_molecular_weight(self, mol: Any) -> float:
        """Get molecular weight from molecule"""
        try:
            Chem = self._get_rdkit()
            return Chem.rdMolDescriptors.CalcExactMolWt(mol)
        except Exception as e:
            logger.warning(f"Failed to get molecular weight: {e}")
            return 0.0
