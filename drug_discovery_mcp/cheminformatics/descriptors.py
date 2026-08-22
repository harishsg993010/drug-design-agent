"""
Molecular Descriptor Calculator

Calculates various molecular descriptors and properties for drug discovery.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from .base import CheminformaticsBase, CheminformaticsError

logger = logging.getLogger(__name__)


@dataclass
class MolecularDescriptors:
    """Container for molecular descriptors"""
    smiles: str
    molecular_weight: float = 0.0
    logp: float = 0.0
    logd: Optional[float] = None
    hba: int = 0  # Hydrogen bond acceptors
    hbd: int = 0  # Hydrogen bond donors
    tpsa: float = 0.0  # Topological polar surface area
    rotatable_bonds: int = 0
    heavy_atoms: int = 0
    aromatic_rings: int = 0
    fraction_csp3: float = 0.0
    num_heteroatoms: int = 0
    num_aromatic_rings: int = 0
    num_saturated_rings: int = 0
    num_rotatable_bonds: int = 0
    num_hydrogens: int = 0
    num_carbons: int = 0
    num_oxygens: int = 0
    num_nitrogens: int = 0
    num_sulfurs: int = 0
    num_halogens: int = 0
    num_phosphorus: int = 0
    
    # Drug-likeness metrics
    qed: Optional[float] = None  # Quantitative estimate of drug-likeness
    lipinski_violations: int = 0
    
    # Additional properties
    molar_refractivity: Optional[float] = None
    density: Optional[float] = None
    boiling_point: Optional[float] = None
    melting_point: Optional[float] = None
    
    # Metadata
    descriptor_names: List[str] = None
    
    def __post_init__(self):
        if self.descriptor_names is None:
            self.descriptor_names = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "smiles": self.smiles,
            "molecular_weight": self.molecular_weight,
            "logp": self.logp,
            "logd": self.logd,
            "hba": self.hba,
            "hbd": self.hbd,
            "tpsa": self.tpsa,
            "rotatable_bonds": self.rotatable_bonds,
            "heavy_atoms": self.heavy_atoms,
            "aromatic_rings": self.aromatic_rings,
            "fraction_csp3": self.fraction_csp3,
            "num_heteroatoms": self.num_heteroatoms,
            "num_aromatic_rings": self.num_aromatic_rings,
            "num_saturated_rings": self.num_saturated_rings,
            "num_rotatable_bonds": self.num_rotatable_bonds,
            "num_hydrogens": self.num_hydrogens,
            "num_carbons": self.num_carbons,
            "num_oxygens": self.num_oxygens,
            "num_nitrogens": self.num_nitrogens,
            "num_sulfurs": self.num_sulfurs,
            "num_halogens": self.num_halogens,
            "num_phosphorus": self.num_phosphorus,
            "qed": self.qed,
            "lipinski_violations": self.lipinski_violations,
            "molar_refractivity": self.molar_refractivity,
            "descriptor_names": self.descriptor_names
        }


class DescriptorCalculator(CheminformaticsBase):
    """
    Calculator for molecular descriptors
    
    Calculates a comprehensive set of molecular descriptors used in drug discovery:
    - Physicochemical properties (molecular weight, logP, etc.)
    - Structural properties (HBA, HBD, TPSA, etc.)
    - Drug-likeness metrics
    - Atom and bond counts
    """
    
    def __init__(self):
        """Initialize the descriptor calculator"""
        super().__init__()
        self._descriptor_functions = {
            "molecular_weight": self._calculate_molecular_weight,
            "logp": self._calculate_logp,
            "logd": self._calculate_logd,
            "hba": self._calculate_hba,
            "hbd": self._calculate_hbd,
            "tpsa": self._calculate_tpsa,
            "rotatable_bonds": self._calculate_rotatable_bonds,
            "heavy_atoms": self._calculate_heavy_atoms,
            "aromatic_rings": self._calculate_aromatic_rings,
            "fraction_csp3": self._calculate_fraction_csp3,
            "num_heteroatoms": self._calculate_num_heteroatoms,
            "num_aromatic_rings": self._calculate_num_aromatic_rings,
            "num_saturated_rings": self._calculate_num_saturated_rings,
            "num_rotatable_bonds": self._calculate_num_rotatable_bonds,
            "num_hydrogens": self._calculate_num_hydrogens,
            "num_carbons": self._calculate_num_carbons,
            "num_oxygens": self._calculate_num_oxygens,
            "num_nitrogens": self._calculate_num_nitrogens,
            "num_sulfurs": self._calculate_num_sulfurs,
            "num_halogens": self._calculate_num_halogens,
            "num_phosphorus": self._calculate_num_phosphorus,
            "qed": self._calculate_qed,
            "lipinski_violations": self._calculate_lipinski_violations,
            "molar_refractivity": self._calculate_molar_refractivity,
        }
    
    def calculate(self, smiles: str, descriptors: Optional[List[str]] = None) -> MolecularDescriptors:
        """
        Calculate molecular descriptors for a SMILES string
        
        Args:
            smiles: SMILES string
            descriptors: List of specific descriptors to calculate (None for all)
            
        Returns:
            MolecularDescriptors object with calculated properties
            
        Raises:
            CheminformaticsError: If descriptor calculation fails
        """
        try:
            Chem = self._get_rdkit()
            
            # Sanitize SMILES
            smiles = self._sanitize_smiles(smiles)
            mol = self._smiles_to_mol(smiles)
            
            # Create result object
            result = MolecularDescriptors(smiles=smiles)
            
            # Calculate requested descriptors
            if descriptors is None:
                descriptors = list(self._descriptor_functions.keys())
            
            for desc_name in descriptors:
                if desc_name in self._descriptor_functions:
                    try:
                        value = self._descriptor_functions[desc_name](mol)
                        setattr(result, desc_name, value)
                        result.descriptor_names.append(desc_name)
                    except Exception as e:
                        logger.warning(f"Failed to calculate {desc_name}: {e}")
                        setattr(result, desc_name, None)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate descriptors for {smiles}: {e}")
            raise CheminformaticsError(f"Failed to calculate descriptors: {e}")
    
    def calculate_batch(self, smiles_list: List[str]) -> List[MolecularDescriptors]:
        """
        Calculate descriptors for multiple SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of MolecularDescriptors objects
        """
        results = []
        for smiles in smiles_list:
            try:
                descriptors = self.calculate(smiles)
                results.append(descriptors)
            except Exception as e:
                logger.error(f"Failed to calculate descriptors for {smiles}: {e}")
                results.append(MolecularDescriptors(smiles=smiles))
        
        return results
    
    # Individual descriptor calculation methods
    
    def _calculate_molecular_weight(self, mol: Any) -> float:
        """Calculate molecular weight"""
        Chem = self._get_rdkit()
        return Chem.rdMolDescriptor.CalcExactMolWt(mol)
    
    def _calculate_logp(self, mol: Any) -> float:
        """Calculate octanol/water partition coefficient (logP)"""
        Chem = self._get_rdkit()
        return Chem.Crippen.MolLogP(mol)
    
    def _calculate_logd(self, mol: Any) -> Optional[float]:
        """Calculate distribution coefficient (logD at pH 7.4)"""
        Chem = self._get_rdkit()
        try:
            return Chem.Crippen.MolLogD(mol)
        except:
            return None
    
    def _calculate_hba(self, mol: Any) -> int:
        """Calculate number of hydrogen bond acceptors"""
        Chem = self._get_rdkit()
        return Chem.Lipinski.NumHAcceptors(mol)
    
    def _calculate_hbd(self, mol: Any) -> int:
        """Calculate number of hydrogen bond donors"""
        Chem = self._get_rdkit()
        return Chem.Lipinski.NumHDonors(mol)
    
    def _calculate_tpsa(self, mol: Any) -> float:
        """Calculate topological polar surface area"""
        Chem = self._get_rdkit()
        return Chem.rdMolDescriptor.CalcTPSA(mol)
    
    def _calculate_rotatable_bonds(self, mol: Any) -> int:
        """Calculate number of rotatable bonds"""
        Chem = self._get_rdkit()
        return Chem.Lipinski.NumRotatableBonds(mol)
    
    def _calculate_heavy_atoms(self, mol: Any) -> int:
        """Calculate number of heavy atoms (non-hydrogen)"""
        Chem = self._get_rdkit()
        return Chem.rdMolDescriptor.CalcNumHeavyAtoms(mol)
    
    def _calculate_aromatic_rings(self, mol: Any) -> int:
        """Calculate number of aromatic rings"""
        Chem = self._get_rdkit()
        return Chem.Lipinski.NumAromaticRings(mol)
    
    def _calculate_fraction_csp3(self, mol: Any) -> float:
        """Calculate fraction of sp3 hybridized carbons"""
        Chem = self._get_rdkit()
        return Chem.Lipinski.FractionCSP3(mol)
    
    def _calculate_num_heteroatoms(self, mol: Any) -> int:
        """Calculate number of heteroatoms (non-carbon, non-hydrogen)"""
        Chem = self._get_rdkit()
        return Chem.rdMolDescriptor.CalcNumHeteroAtoms(mol)
    
    def _calculate_num_aromatic_rings(self, mol: Any) -> int:
        """Calculate number of aromatic rings"""
        return self._calculate_aromatic_rings(mol)
    
    def _calculate_num_saturated_rings(self, mol: Any) -> int:
        """Calculate number of saturated rings"""
        Chem = self._get_rdkit()
        return Chem.rdMolDescriptor.CalcNumSaturatedRings(mol)
    
    def _calculate_num_rotatable_bonds(self, mol: Any) -> int:
        """Calculate number of rotatable bonds"""
        return self._calculate_rotatable_bonds(mol)
    
    def _calculate_num_hydrogens(self, mol: Any) -> int:
        """Calculate number of hydrogen atoms"""
        Chem = self._get_rdkit()
        return Chem.rdMolDescriptor.CalcNumHBA(mol) + Chem.rdMolDescriptor.CalcNumHBD(mol)
    
    def _calculate_num_carbons(self, mol: Any) -> int:
        """Calculate number of carbon atoms"""
        Chem = self._get_rdkit()
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:  # Carbon
                count += 1
        return count
    
    def _calculate_num_oxygens(self, mol: Any) -> int:
        """Calculate number of oxygen atoms"""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 8:  # Oxygen
                count += 1
        return count
    
    def _calculate_num_nitrogens(self, mol: Any) -> int:
        """Calculate number of nitrogen atoms"""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:  # Nitrogen
                count += 1
        return count
    
    def _calculate_num_sulfurs(self, mol: Any) -> int:
        """Calculate number of sulfur atoms"""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 16:  # Sulfur
                count += 1
        return count
    
    def _calculate_num_halogens(self, mol: Any) -> int:
        """Calculate number of halogen atoms (F, Cl, Br, I)"""
        halogens = {9, 17, 35, 53}  # F, Cl, Br, I
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in halogens:
                count += 1
        return count
    
    def _calculate_num_phosphorus(self, mol: Any) -> int:
        """Calculate number of phosphorus atoms"""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:  # Phosphorus
                count += 1
        return count
    
    def _calculate_qed(self, mol: Any) -> Optional[float]:
        """Calculate Quantitative Estimate of Drug-likeness (QED)"""
        Chem = self._get_rdkit()
        try:
            from rdkit.Chem import QED
            return QED.qed(mol)
        except ImportError:
            logger.warning("QED module not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to calculate QED: {e}")
            return None
    
    def _calculate_lipinski_violations(self, mol: Any) -> int:
        """Calculate number of Lipinski rule violations"""
        Chem = self._get_rdkit()
        violations = 0
        
        # Molecular weight > 500
        if Chem.rdMolDescriptor.CalcExactMolWt(mol) > 500:
            violations += 1
        
        # logP > 5
        if Chem.Crippen.MolLogP(mol) > 5:
            violations += 1
        
        # HBA > 10
        if Chem.Lipinski.NumHAcceptors(mol) > 10:
            violations += 1
        
        # HBD > 5
        if Chem.Lipinski.NumHDonors(mol) > 5:
            violations += 1
        
        # Rotatable bonds > 10
        if Chem.Lipinski.NumRotatableBonds(mol) > 10:
            violations += 1
        
        return violations
    
    def _calculate_molar_refractivity(self, mol: Any) -> Optional[float]:
        """Calculate molar refractivity"""
        Chem = self._get_rdkit()
        try:
            return Chem.rdMolDescriptor.CalcMolarRefractivity(mol)
        except Exception as e:
            logger.warning(f"Failed to calculate molar refractivity: {e}")
            return None


# Singleton instance
_descriptor_calculator = DescriptorCalculator()


def calculate_descriptors(smiles: str, descriptors: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calculate molecular descriptors for a SMILES string
    
    Args:
        smiles: SMILES string
        descriptors: List of specific descriptors to calculate (None for all)
        
    Returns:
        Dictionary with calculated descriptors
    """
    try:
        result = _descriptor_calculator.calculate(smiles, descriptors)
        return result.to_dict()
    except CheminformaticsError as e:
        return {"error": str(e), "smiles": smiles}
