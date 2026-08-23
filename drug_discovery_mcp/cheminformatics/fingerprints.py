"""
Molecular Fingerprint Tools

Generates various types of molecular fingerprints for similarity searching
and machine learning applications.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from .base import CheminformaticsBase, CheminformaticsError

logger = logging.getLogger(__name__)


class FingerprintType(Enum):
    """Types of molecular fingerprints"""
    MORGAN = "morgan"
    RDKIT = "rdkit"
    ATOM_PAIR = "atom_pair"
    TOPOLOGICAL_TORSION = "topological_torsion"
    MACCS = "maccs"
    DAYLIGHT = "daylight"
    AVALON = "avalon"


@dataclass
class Fingerprint:
    """Represents a molecular fingerprint"""
    smiles: str
    fingerprint_type: str
    fingerprint: List[int]  # Binary fingerprint as list of 0s and 1s
    bit_length: int
    radius: Optional[int] = None
    
    # Metadata
    num_bits_set: int = 0
    density: float = 0.0
    
    def __post_init__(self):
        self.fingerprint = list(self.fingerprint) if self.fingerprint else []
        self.num_bits_set = sum(self.fingerprint)
        self.bit_length = len(self.fingerprint)
        if self.bit_length > 0:
            self.density = self.num_bits_set / self.bit_length
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "smiles": self.smiles,
            "fingerprint_type": self.fingerprint_type,
            "bit_length": self.bit_length,
            "radius": self.radius,
            "num_bits_set": self.num_bits_set,
            "density": self.density,
            "fingerprint": self.fingerprint
        }
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array(self.fingerprint)
    
    def to_hex(self) -> str:
        """Convert fingerprint to hexadecimal string"""
        # Convert binary list to bytes, then to hex
        byte_array = bytearray()
        for i in range(0, len(self.fingerprint), 8):
            byte_bits = self.fingerprint[i:i+8]
            if len(byte_bits) < 8:
                byte_bits += [0] * (8 - len(byte_bits))
            byte_val = int(''.join(map(str, byte_bits)), 2)
            byte_array.append(byte_val)
        
        return byte_array.hex()


class FingerprintTools(CheminformaticsBase):
    """
    Tools for generating molecular fingerprints
    
    Provides various types of fingerprints used in cheminformatics:
    - Morgan fingerprints (ECFP-like)
    - RDKit fingerprints
    - Atom pair fingerprints
    - Topological torsion fingerprints
    - MACCS keys
    
    Fingerprints are used for:
    - Molecular similarity searching
    - Substructure searching
    - Machine learning
    - Clustering
    """
    
    def __init__(self):
        """Initialize fingerprint tools"""
        super().__init__()
    
    def calculate(
        self,
        smiles: str,
        fingerprint_type: str = "morgan",
        radius: int = 2,
        bit_length: int = 2048,
        **kwargs
    ) -> Fingerprint:
        """
        Calculate fingerprint for a SMILES string
        
        Args:
            smiles: SMILES string
            fingerprint_type: Type of fingerprint ("morgan", "rdkit", "atom_pair", "topological_torsion", "maccs")
            radius: Radius for Morgan fingerprints
            bit_length: Length of the fingerprint (number of bits)
            **kwargs: Additional fingerprint-specific options
            
        Returns:
            Fingerprint object
        """
        try:
            Chem = self._get_rdkit()
            
            # Sanitize SMILES
            smiles = self._sanitize_smiles(smiles)
            mol = self._smiles_to_mol(smiles)
            
            fingerprint_type = fingerprint_type.lower()
            
            if fingerprint_type == "morgan":
                fingerprint = self._calculate_morgan_fingerprint(mol, radius, bit_length)
            elif fingerprint_type == "rdkit":
                fingerprint = self._calculate_rdkit_fingerprint(mol, bit_length)
            elif fingerprint_type == "atom_pair":
                fingerprint = self._calculate_atom_pair_fingerprint(mol, bit_length)
            elif fingerprint_type == "topological_torsion" or fingerprint_type == "tt":
                fingerprint = self._calculate_topological_torsion_fingerprint(mol, bit_length)
            elif fingerprint_type == "maccs":
                fingerprint = self._calculate_maccs_fingerprint(mol)
            else:
                raise ValueError(f"Unknown fingerprint type: {fingerprint_type}")
            
            return Fingerprint(
                smiles=smiles,
                fingerprint_type=fingerprint_type,
                fingerprint=fingerprint,
                bit_length=bit_length,
                radius=radius if fingerprint_type == "morgan" else None
            )
            
        except Exception as e:
            logger.error(f"Fingerprint calculation failed: {e}")
            raise CheminformaticsError(f"Failed to calculate fingerprint: {e}")
    
    def calculate_batch(
        self,
        smiles_list: List[str],
        fingerprint_type: str = "morgan",
        radius: int = 2,
        bit_length: int = 2048,
        **kwargs
    ) -> List[Fingerprint]:
        """
        Calculate fingerprints for multiple SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            fingerprint_type: Type of fingerprint
            radius: Radius for Morgan fingerprints
            bit_length: Length of the fingerprint
            **kwargs: Additional options
            
        Returns:
            List of Fingerprint objects
        """
        fingerprints = []
        for smiles in smiles_list:
            try:
                fp = self.calculate(
                    smiles,
                    fingerprint_type=fingerprint_type,
                    radius=radius,
                    bit_length=bit_length,
                    **kwargs
                )
                fingerprints.append(fp)
            except Exception as e:
                logger.error(f"Failed to calculate fingerprint for {smiles}: {e}")
                fingerprints.append(Fingerprint(
                    smiles=smiles,
                    fingerprint_type=fingerprint_type,
                    fingerprint=[],
                    bit_length=bit_length,
                    radius=radius if fingerprint_type == "morgan" else None
                ))
        
        return fingerprints
    
    def get_fingerprint_info(self, fingerprint_type: str) -> Dict[str, Any]:
        """
        Get information about a fingerprint type
        
        Args:
            fingerprint_type: Type of fingerprint
            
        Returns:
            Dictionary with fingerprint information
        """
        info = {
            "morgan": {
                "name": "Morgan Fingerprint",
                "description": "Circular fingerprint based on Morgan algorithm (ECFP-like)",
                "radius": "Number of bonds to consider",
                "bit_length": "Number of bits in the fingerprint",
                "use_cases": ["similarity searching", "clustering", "machine learning"],
                "default_radius": 2,
                "default_bit_length": 2048
            },
            "rdkit": {
                "name": "RDKit Fingerprint",
                "description": "Path-based fingerprint using RDKit's default algorithm",
                "bit_length": "Number of bits in the fingerprint",
                "use_cases": ["substructure searching", "similarity searching"],
                "default_bit_length": 2048
            },
            "atom_pair": {
                "name": "Atom Pair Fingerprint",
                "description": "Fingerprint based on atom pairs and their distances",
                "bit_length": "Number of bits in the fingerprint",
                "use_cases": ["similarity searching", "pharmacophore matching"],
                "default_bit_length": 2048
            },
            "topological_torsion": {
                "name": "Topological Torsion Fingerprint",
                "description": "Fingerprint based on topological torsion paths",
                "bit_length": "Number of bits in the fingerprint",
                "use_cases": ["3D similarity", "conformer-independent comparison"],
                "default_bit_length": 2048
            },
            "maccs": {
                "name": "MACCS Keys",
                "description": "166-bit structural key fingerprint",
                "bit_length": 166,
                "use_cases": ["substructure searching", "similarity searching"],
                "default_bit_length": 166
            }
        }
        
        return info.get(fingerprint_type.lower(), {"error": f"Unknown fingerprint type: {fingerprint_type}"})
    
    # Individual fingerprint calculation methods
    
    def _calculate_morgan_fingerprint(self, mol: Any, radius: int = 2, bit_length: int = 2048) -> List[int]:
        """Calculate Morgan fingerprint (ECFP-like)"""
        Chem = self._get_rdkit()
        
        # Generate Morgan fingerprint
        fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            radius,
            nBits=bit_length
        )
        
        # Convert to list of bits
        return [1 if fp.GetBit(i) else 0 for i in range(bit_length)]
    
    def _calculate_rdkit_fingerprint(self, mol: Any, bit_length: int = 2048) -> List[int]:
        """Calculate RDKit fingerprint"""
        Chem = self._get_rdkit()
        
        # Generate RDKit fingerprint
        fp = Chem.RDKFingerprint(mol, fpSize=bit_length)
        
        # Convert to list of bits
        return [1 if fp.GetBit(i) else 0 for i in range(bit_length)]
    
    def _calculate_atom_pair_fingerprint(self, mol: Any, bit_length: int = 2048) -> List[int]:
        """Calculate atom pair fingerprint"""
        Chem = self._get_rdkit()
        
        # Generate atom pair fingerprint
        fp = Chem.rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
            mol,
            nBits=bit_length,
            minLength=1,
            maxLength=30
        )
        
        # Convert to list of bits
        return [1 if fp.GetBit(i) else 0 for i in range(bit_length)]
    
    def _calculate_topological_torsion_fingerprint(self, mol: Any, bit_length: int = 2048) -> List[int]:
        """Calculate topological torsion fingerprint"""
        Chem = self._get_rdkit()
        
        # Generate topological torsion fingerprint
        fp = Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
            mol,
            nBits=bit_length
        )
        
        # Convert to list of bits
        return [1 if fp.GetBit(i) else 0 for i in range(bit_length)]
    
    def _calculate_maccs_fingerprint(self, mol: Any) -> List[int]:
        """Calculate MACCS keys fingerprint"""
        Chem = self._get_rdkit()
        
        # Generate MACCS keys
        fp = Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        
        # Convert to list of bits (MACCS keys are 167 bits; bit 0 is unused)
        return [1 if fp.GetBit(i) else 0 for i in range(fp.GetNumBits())]


# Singleton instance
_fingerprint_tools = FingerprintTools()


def calculate_fingerprint(
    smiles: str,
    fingerprint_type: str = "morgan",
    **kwargs
) -> Dict[str, Any]:
    """
    Calculate fingerprint for a SMILES string
    
    Args:
        smiles: SMILES string
        fingerprint_type: Type of fingerprint ("morgan", "rdkit", "atom_pair", "topological_torsion", "maccs")
        **kwargs: Additional options (radius, bit_length)
        
    Returns:
        Dictionary with fingerprint result
    """
    try:
        result = _fingerprint_tools.calculate(smiles, fingerprint_type=fingerprint_type, **kwargs)
        return result.to_dict()
    except CheminformaticsError as e:
        return {"error": str(e), "smiles": smiles}
