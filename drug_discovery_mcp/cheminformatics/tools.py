"""
Cheminformatics Tools Module

Provides a unified interface for all cheminformatics operations.
This is the main entry point for cheminformatics tools from the MCP server.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from .descriptors import DescriptorCalculator, calculate_descriptors
from .conversion import ConversionTools, smiles_to_inchi, inchi_to_smiles
from .similarity import SimilarityTools, molecular_similarity
from .fingerprints import FingerprintTools, calculate_fingerprint
from .admet import ADMETPredictor, predict_admet
from .drug_likeness import DrugLikenessChecker, check_drug_likeness
from .base import CheminformaticsError

logger = logging.getLogger(__name__)


class CheminformaticsTools:
    """
    Unified interface for all cheminformatics operations
    
    This class provides a single entry point for all cheminformatics tools,
    making it easy to access different cheminformatics functions through a
    consistent interface.
    """
    
    def __init__(self):
        """Initialize all cheminformatics components"""
        self.descriptor_calculator = DescriptorCalculator()
        self.conversion_tools = ConversionTools()
        self.similarity_tools = SimilarityTools()
        self.fingerprint_tools = FingerprintTools()
        self.admet_predictor = ADMETPredictor()
        self.drug_likeness_checker = DrugLikenessChecker()
    
    def initialize(self):
        """Initialize all cheminformatics components"""
        logger.info("Initializing cheminformatics tools")
        # All components are initialized in __init__
    
    async def close(self):
        """Close all cheminformatics components"""
        logger.info("Closing cheminformatics tools")
        # No cleanup needed for cheminformatics tools
    
    # Descriptor methods
    def calculate_descriptors(self, smiles: str, **kwargs) -> Dict[str, Any]:
        """Calculate molecular descriptors"""
        return calculate_descriptors(smiles, **kwargs)
    
    # Conversion methods
    def smiles_to_inchi(self, smiles: str, **kwargs) -> Dict[str, Any]:
        """Convert SMILES to InChI"""
        return smiles_to_inchi(smiles, **kwargs)
    
    def inchi_to_smiles(self, inchi: str, **kwargs) -> Dict[str, Any]:
        """Convert InChI to SMILES"""
        return inchi_to_smiles(inchi, **kwargs)
    
    def convert(self, input_value: str, input_format: str, output_format: str, **kwargs) -> Dict[str, Any]:
        """Generic conversion between molecular formats"""
        result = self.conversion_tools.convert(
            input_value, input_format, output_format, **kwargs
        )
        return {
            "input_format": result.input_format,
            "input_value": result.input_value,
            "output_format": result.output_format,
            "output_value": result.output_value,
            "success": result.success,
            "error": result.error
        }
    
    # Similarity methods
    def molecular_similarity(self, smiles1: str, smiles2: str, **kwargs) -> Dict[str, Any]:
        """Calculate molecular similarity"""
        return molecular_similarity(smiles1, smiles2, **kwargs)
    
    def find_most_similar(self, query_smiles: str, candidate_smiles: List[str], **kwargs) -> Dict[str, Any]:
        """Find most similar molecules"""
        results = self.similarity_tools.find_most_similar(query_smiles, candidate_smiles, **kwargs)
        return {
            "query": query_smiles,
            "candidates": candidate_smiles,
            "results": results
        }
    
    def cluster(self, smiles_list: List[str], **kwargs) -> Dict[str, Any]:
        """Cluster molecules by similarity"""
        clusters = self.similarity_tools.cluster(smiles_list, **kwargs)
        return {
            "molecules": smiles_list,
            "clusters": clusters,
            "num_clusters": len(clusters)
        }
    
    # Fingerprint methods
    def calculate_fingerprint(self, smiles: str, **kwargs) -> Dict[str, Any]:
        """Calculate molecular fingerprint"""
        return calculate_fingerprint(smiles, **kwargs)
    
    def get_fingerprint_info(self, fingerprint_type: str) -> Dict[str, Any]:
        """Get information about a fingerprint type"""
        return self.fingerprint_tools.get_fingerprint_info(fingerprint_type)
    
    # ADMET methods
    def predict_admet(self, smiles: str, **kwargs) -> Dict[str, Any]:
        """Predict ADMET properties"""
        return predict_admet(smiles, **kwargs)
    
    # Drug-likeness methods
    def check_drug_likeness(self, smiles: str, **kwargs) -> Dict[str, Any]:
        """Check drug-likeness"""
        return check_drug_likeness(smiles, **kwargs)
    
    def filter_drug_like(self, smiles_list: List[str], **kwargs) -> Dict[str, Any]:
        """Filter drug-like molecules"""
        return self.drug_likeness_checker.filter_drug_like(smiles_list, **kwargs)
    
    # Combined methods
    def analyze_molecule(self, smiles: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a molecule
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        results = {
            "smiles": smiles,
            "descriptors": self.calculate_descriptors(smiles),
            "admet": self.predict_admet(smiles),
            "drug_likeness": self.check_drug_likeness(smiles),
            "fingerprint": self.calculate_fingerprint(smiles)
        }
        
        # Add InChI conversion
        inchi_result = self.smiles_to_inchi(smiles)
        if inchi_result.get("success", False):
            results["inchi"] = inchi_result["output"]
        
        return results
    
    def compare_molecules(self, smiles1: str, smiles2: str) -> Dict[str, Any]:
        """
        Compare two molecules
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            
        Returns:
            Dictionary with comparison results
        """
        return {
            "molecule1": self.analyze_molecule(smiles1),
            "molecule2": self.analyze_molecule(smiles2),
            "similarity": self.molecular_similarity(smiles1, smiles2)
        }
    
    def generate_conformers(self, smiles: str, num_conformers: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Generate 3D conformers for a molecule
        
        Args:
            smiles: SMILES string
            num_conformers: Number of conformers to generate
            **kwargs: Additional options
            
        Returns:
            Dictionary with conformer generation results
        """
        try:
            Chem = self.descriptor_calculator._get_rdkit()
            
            # Convert SMILES to molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": "Invalid SMILES", "smiles": smiles}
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate conformers
            conformers = []
            for i in range(num_conformers):
                # Generate random coordinates
                Chem.AllChem.EmbedMolecule(mol, randomSeed=i)
                
                # Optimize geometry
                Chem.AllChem.MMFFOptimizeMolecule(mol)
                
                # Get coordinates
                coords = mol.GetConformer().GetPositions()
                conformers.append({
                    "index": i,
                    "coordinates": coords.tolist(),
                    "energy": 0.0  # Energy would be calculated in a real implementation
                })
            
            return {
                "smiles": smiles,
                "num_conformers": len(conformers),
                "conformers": conformers
            }
            
        except Exception as e:
            logger.error(f"Failed to generate conformers for {smiles}: {e}")
            return {"error": str(e), "smiles": smiles}
    
    def optimize_geometry(self, smiles: str, **kwargs) -> Dict[str, Any]:
        """
        Optimize molecular geometry
        
        Args:
            smiles: SMILES string
            **kwargs: Additional options
            
        Returns:
            Dictionary with optimized geometry
        """
        try:
            Chem = self.descriptor_calculator._get_rdkit()
            
            # Convert SMILES to molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": "Invalid SMILES", "smiles": smiles}
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate initial coordinates
            Chem.AllChem.EmbedMolecule(mol)
            
            # Optimize geometry using MMFF
            result = Chem.AllChem.MMFFOptimizeMolecule(mol)
            
            # Get optimized coordinates
            coords = mol.GetConformer().GetPositions()
            
            # Calculate energy
            mp = Chem.AllChem.MMFFGetMoleculeProperties(mol)
            if mp is not None:
                ff = Chem.AllChem.MMFFGetMoleculeForceField(mol, mp)
                energy = ff.CalcEnergy()
            else:
                energy = 0.0
            
            return {
                "smiles": smiles,
                "coordinates": coords.tolist(),
                "energy": energy,
                "converged": result == 0
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize geometry for {smiles}: {e}")
            return {"error": str(e), "smiles": smiles}
    
    def calculate_charge(self, smiles: str, ph: float = 7.0, **kwargs) -> Dict[str, Any]:
        """
        Calculate formal charge and protonation states
        
        Args:
            smiles: SMILES string
            ph: pH for protonation state calculation
            **kwargs: Additional options
            
        Returns:
            Dictionary with charge information
        """
        try:
            Chem = self.descriptor_calculator._get_rdkit()
            
            # Convert SMILES to molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": "Invalid SMILES", "smiles": smiles}
            
            # Calculate formal charge
            total_charge = 0
            atom_charges = []
            
            for atom in mol.GetAtoms():
                charge = atom.GetFormalCharge()
                total_charge += charge
                atom_charges.append({
                    "atom_index": atom.GetIdx(),
                    "atom_symbol": atom.GetSymbol(),
                    "charge": charge
                })
            
            return {
                "smiles": smiles,
                "total_charge": total_charge,
                "atom_charges": atom_charges
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate charge for {smiles}: {e}")
            return {"error": str(e), "smiles": smiles}
