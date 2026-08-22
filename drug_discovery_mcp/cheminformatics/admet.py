"""
ADMET Property Predictor

Predicts Absorption, Distribution, Metabolism, Excretion, and Toxicity properties
for drug discovery applications.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from .base import CheminformaticsBase, CheminformaticsError
from .descriptors import DescriptorCalculator

logger = logging.getLogger(__name__)


@dataclass
class ADMETProperties:
    """Container for ADMET properties"""
    smiles: str
    
    # Absorption
    caco2_permeability: Optional[float] = None  # Caco-2 cell permeability (nm/s)
    human_intestinal_absorption: Optional[float] = None  # % absorbed
    pgp_substrate: Optional[bool] = None  # P-glycoprotein substrate
    pgp_inhibitor: Optional[bool] = None  # P-glycoprotein inhibitor
    
    # Distribution
    vd: Optional[float] = None  # Volume of distribution (L/kg)
    blood_brain_barrier: Optional[float] = None  # BBB penetration (log BB)
    central_nervous_system: Optional[float] = None  # CNS penetration
    plasma_protein_binding: Optional[float] = None  # % bound to plasma proteins
    
    # Metabolism
    cyp450_2c9_substrate: Optional[bool] = None  # CYP2C9 substrate
    cyp450_2d6_substrate: Optional[bool] = None  # CYP2D6 substrate
    cyp450_3a4_substrate: Optional[bool] = None  # CYP3A4 substrate
    cyp450_2c9_inhibitor: Optional[bool] = None  # CYP2C9 inhibitor
    cyp450_2d6_inhibitor: Optional[bool] = None  # CYP2D6 inhibitor
    cyp450_3a4_inhibitor: Optional[bool] = None  # CYP3A4 inhibitor
    half_life: Optional[float] = None  # Half-life (hours)
    
    # Excretion
    renal_excretion: Optional[float] = None  # % excreted renally
    biliary_excretion: Optional[float] = None  # % excreted biliary
    
    # Toxicity
    ames_test: Optional[bool] = None  # Ames test (mutagenicity)
    carcinogenicity: Optional[bool] = None  # Carcinogenicity
    herg: Optional[bool] = None  # hERG inhibition (cardiotoxicity)
    ld50: Optional[float] = None  # Median lethal dose (mg/kg)
    
    # Additional properties
    bioavailabilty: Optional[float] = None  # Oral bioavailability (%)
    clearance: Optional[float] = None  # Clearance (mL/min/kg)
    
    # Drug-likeness
    lipinski_violations: int = 0
    ghose_violations: int = 0
    veber_violations: int = 0
    egan_violations: int = 0
    mute_violations: int = 0
    
    # Metadata
    method: str = "predicted"
    confidence: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.confidence is None:
            self.confidence = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "smiles": self.smiles,
            "method": self.method,
            "absorption": {
                "caco2_permeability": self.caco2_permeability,
                "human_intestinal_absorption": self.human_intestinal_absorption,
                "pgp_substrate": self.pgp_substrate,
                "pgp_inhibitor": self.pgp_inhibitor
            },
            "distribution": {
                "vd": self.vd,
                "blood_brain_barrier": self.blood_brain_barrier,
                "central_nervous_system": self.central_nervous_system,
                "plasma_protein_binding": self.plasma_protein_binding
            },
            "metabolism": {
                "cyp450_2c9_substrate": self.cyp450_2c9_substrate,
                "cyp450_2d6_substrate": self.cyp450_2d6_substrate,
                "cyp450_3a4_substrate": self.cyp450_3a4_substrate,
                "cyp450_2c9_inhibitor": self.cyp450_2c9_inhibitor,
                "cyp450_2d6_inhibitor": self.cyp450_2d6_inhibitor,
                "cyp450_3a4_inhibitor": self.cyp450_3a4_inhibitor,
                "half_life": self.half_life
            },
            "excretion": {
                "renal_excretion": self.renal_excretion,
                "biliary_excretion": self.biliary_excretion
            },
            "toxicity": {
                "ames_test": self.ames_test,
                "carcinogenicity": self.carcinogenicity,
                "herg": self.herg,
                "ld50": self.ld50
            },
            "drug_likeness": {
                "lipinski_violations": self.lipinski_violations,
                "ghose_violations": self.ghose_violations,
                "veber_violations": self.veber_violations,
                "egan_violations": self.egan_violations,
                "mute_violations": self.mute_violations
            },
            "additional": {
                "bioavailability": self.bioavailabilty,
                "clearance": self.clearance
            },
            "confidence": self.confidence
        }
        return result


class ADMETPredictor(CheminformaticsBase):
    """
    Predictor for ADMET properties
    
    Uses various models and rules to predict:
    - Absorption properties
    - Distribution properties
    - Metabolism properties
    - Excretion properties
    - Toxicity properties
    - Drug-likeness
    
    Note: Some properties require external models or databases.
    This implementation provides rule-based predictions where possible.
    """
    
    def __init__(self):
        """Initialize ADMET predictor"""
        super().__init__()
        self.descriptor_calculator = DescriptorCalculator()
    
    def predict(self, smiles: str, **kwargs) -> ADMETProperties:
        """
        Predict ADMET properties for a SMILES string
        
        Args:
            smiles: SMILES string
            **kwargs: Additional prediction options
            
        Returns:
            ADMETProperties object with predicted properties
        """
        try:
            Chem = self._get_rdkit()
            
            # Sanitize SMILES
            smiles = self._sanitize_smiles(smiles)
            mol = self._smiles_to_mol(smiles)
            
            # Calculate basic descriptors
            descriptors = self.descriptor_calculator.calculate(smiles)
            
            # Create result object
            result = ADMETProperties(smiles=smiles)
            
            # Predict absorption properties
            result = self._predict_absorption(result, mol, descriptors)
            
            # Predict distribution properties
            result = self._predict_distribution(result, mol, descriptors)
            
            # Predict metabolism properties
            result = self._predict_metabolism(result, mol, descriptors)
            
            # Predict excretion properties
            result = self._predict_excretion(result, mol, descriptors)
            
            # Predict toxicity properties
            result = self._predict_toxicity(result, mol, descriptors)
            
            # Predict drug-likeness
            result = self._predict_drug_likeness(result, mol, descriptors)
            
            # Predict additional properties
            result = self._predict_additional(result, mol, descriptors)
            
            return result
            
        except Exception as e:
            logger.error(f"ADMET prediction failed for {smiles}: {e}")
            raise CheminformaticsError(f"Failed to predict ADMET properties: {e}")
    
    def predict_batch(self, smiles_list: List[str]) -> List[ADMETProperties]:
        """
        Predict ADMET properties for multiple SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of ADMETProperties objects
        """
        results = []
        for smiles in smiles_list:
            try:
                admet = self.predict(smiles)
                results.append(admet)
            except Exception as e:
                logger.error(f"Failed to predict ADMET for {smiles}: {e}")
                results.append(ADMETProperties(smiles=smiles, method="error"))
        
        return results
    
    # Individual prediction methods
    
    def _predict_absorption(self, result: ADMETProperties, mol: Any, descriptors: Any) -> ADMETProperties:
        """Predict absorption properties"""
        Chem = self._get_rdkit()
        
        # Human intestinal absorption (rule-based)
        # Based on molecular weight and logP
        mw = descriptors.molecular_weight
        logp = descriptors.logp
        hba = descriptors.hba
        hbd = descriptors.hbd
        
        # Simple rule-based prediction for intestinal absorption
        if mw < 500 and logp < 5 and hba <= 10 and hbd <= 5:
            result.human_intestinal_absorption = 0.9  # 90%
        elif mw < 500 and (logp >= 5 or hba > 10 or hbd > 5):
            result.human_intestinal_absorption = 0.7  # 70%
        else:
            result.human_intestinal_absorption = 0.5  # 50%
        
        # Caco-2 permeability (rule-based)
        # Based on logP and HBD
        if logp > 1.5 and hbd <= 2:
            result.caco2_permeability = 100.0  # High permeability (nm/s)
        elif logp > 0 and hbd <= 5:
            result.caco2_permeability = 50.0  # Medium permeability
        else:
            result.caco2_permeability = 10.0  # Low permeability
        
        # P-gp substrate prediction (rule-based)
        # Based on molecular weight and logP
        if mw > 400 and logp > 4:
            result.pgp_substrate = True
        else:
            result.pgp_substrate = False
        
        # P-gp inhibitor prediction
        result.pgp_inhibitor = False  # Simplified
        
        return result
    
    def _predict_distribution(self, result: ADMETProperties, mol: Any, descriptors: Any) -> ADMETProperties:
        """Predict distribution properties"""
        Chem = self._get_rdkit()
        
        # Volume of distribution (rule-based)
        logp = descriptors.logp
        
        if logp > 3:
            result.vd = 5.0  # High Vd (L/kg)
        elif logp > 0:
            result.vd = 1.5  # Medium Vd
        else:
            result.vd = 0.5  # Low Vd
        
        # Blood-brain barrier penetration (log BB)
        # Based on logP and TPSA
        tpsa = descriptors.tpsa
        
        if logp > 1.5 and tpsa < 90:
            result.blood_brain_barrier = 0.5  # log BB = 0.5 (BBB+)
        elif logp > 0.5 and tpsa < 140:
            result.blood_brain_barrier = 0.0  # log BB = 0 (BBB+/-)
        else:
            result.blood_brain_barrier = -1.0  # log BB = -1 (BBB-)
        
        # CNS penetration
        if result.blood_brain_barrier > 0:
            result.central_nervous_system = 1.0  # High CNS penetration
        else:
            result.central_nervous_system = 0.0  # Low CNS penetration
        
        # Plasma protein binding (rule-based)
        if logp > 2:
            result.plasma_protein_binding = 95.0  # 95%
        elif logp > 0:
            result.plasma_protein_binding = 90.0  # 90%
        else:
            result.plasma_protein_binding = 70.0  # 70%
        
        return result
    
    def _predict_metabolism(self, result: ADMETProperties, mol: Any, descriptors: Any) -> ADMETProperties:
        """Predict metabolism properties"""
        Chem = self._get_rdkit()
        
        # CYP450 substrate predictions (simplified)
        # These are typically predicted using machine learning models
        # Here we use simple rules
        
        # Check for common CYP450 substrate features
        result.cyp450_2c9_substrate = False
        result.cyp450_2d6_substrate = False
        result.cyp450_3a4_substrate = True  # Most drugs are CYP3A4 substrates
        
        # CYP450 inhibitor predictions
        result.cyp450_2c9_inhibitor = False
        result.cyp450_2d6_inhibitor = False
        result.cyp450_3a4_inhibitor = False
        
        # Half-life (rule-based)
        mw = descriptors.molecular_weight
        if mw < 300:
            result.half_life = 2.0  # Short half-life (hours)
        elif mw < 500:
            result.half_life = 6.0  # Medium half-life
        else:
            result.half_life = 12.0  # Long half-life
        
        return result
    
    def _predict_excretion(self, result: ADMETProperties, mol: Any, descriptors: Any) -> ADMETProperties:
        """Predict excretion properties"""
        Chem = self._get_rdkit()
        
        # Renal excretion (rule-based)
        mw = descriptors.molecular_weight
        logp = descriptors.logp
        
        if mw < 300 and logp < 1:
            result.renal_excretion = 70.0  # 70% renal excretion
        elif mw < 500:
            result.renal_excretion = 30.0  # 30% renal excretion
        else:
            result.renal_excretion = 10.0  # 10% renal excretion
        
        # Biliary excretion
        result.biliary_excretion = 100.0 - result.renal_excretion
        
        return result
    
    def _predict_toxicity(self, result: ADMETProperties, mol: Any, descriptors: Any) -> ADMETProperties:
        """Predict toxicity properties"""
        Chem = self._get_rdkit()
        
        # Ames test (mutagenicity) - simplified
        # In reality, this requires specific models
        result.ames_test = False
        
        # Carcinogenicity - simplified
        result.carcinogenicity = False
        
        # hERG inhibition (cardiotoxicity) - rule-based
        logp = descriptors.logp
        basic_nitrogen = self._has_basic_nitrogen(mol)
        
        if logp > 3 and basic_nitrogen:
            result.herg = True  # Potential hERG inhibitor
        else:
            result.herg = False
        
        # LD50 - simplified (in reality, this varies greatly)
        result.ld50 = 1000.0  # mg/kg (typical for many drugs)
        
        return result
    
    def _predict_drug_likeness(self, result: ADMETProperties, mol: Any, descriptors: Any) -> ADMETProperties:
        """Predict drug-likeness using various rule sets"""
        
        # Lipinski's Rule of Five
        mw = descriptors.molecular_weight
        logp = descriptors.logp
        hba = descriptors.hba
        hbd = descriptors.hbd
        
        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if hba > 10:
            violations += 1
        if hbd > 5:
            violations += 1
        result.lipinski_violations = violations
        
        # Ghose filter
        violations = 0
        if mw < 160 or mw > 480:
            violations += 1
        if logp < 0.4 or logp > 5.6:
            violations += 1
        if descriptors.num_carbons < 4 or descriptors.num_carbons > 36:
            violations += 1
        if descriptors.num_heteroatoms < 1 or descriptors.num_heteroatoms > 12:
            violations += 1
        result.ghose_violations = violations
        
        # Veber rules
        violations = 0
        if descriptors.rotatable_bonds > 10:
            violations += 1
        if descriptors.tpsa > 140:
            violations += 1
        result.veber_violations = violations
        
        # Egan rules
        violations = 0
        if logp > 5.88 and descriptors.tpsa > 131.6:
            violations += 1
        result.egan_violations = violations
        
        # MUE rules
        violations = 0
        if mw < 150 or mw > 500:
            violations += 1
        if logp < -2 or logp > 5:
            violations += 1
        if descriptors.tpsa < 40 or descriptors.tpsa > 130:
            violations += 1
        if descriptors.rotatable_bonds > 10:
            violations += 1
        result.mute_violations = violations
        
        return result
    
    def _predict_additional(self, result: ADMETProperties, mol: Any, descriptors: Any) -> ADMETProperties:
        """Predict additional properties"""
        
        # Oral bioavailability (rule-based)
        # Based on Lipinski violations and other factors
        if result.lipinski_violations == 0:
            result.bioavailabilty = 80.0  # 80%
        elif result.lipinski_violations <= 2:
            result.bioavailabilty = 50.0  # 50%
        else:
            result.bioavailabilty = 20.0  # 20%
        
        # Clearance (rule-based)
        if descriptors.molecular_weight < 300:
            result.clearance = 30.0  # High clearance (mL/min/kg)
        elif descriptors.molecular_weight < 500:
            result.clearance = 15.0  # Medium clearance
        else:
            result.clearance = 5.0  # Low clearance
        
        return result
    
    def _has_basic_nitrogen(self, mol: Any) -> bool:
        """Check if molecule has basic nitrogen atoms"""
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:  # Nitrogen
                # Check if it's basic (has lone pair and is not positively charged)
                if atom.GetFormalCharge() <= 0:
                    # Check number of hydrogens (simplified)
                    if atom.GetTotalNumHs() > 0:
                        return True
        return False


# Singleton instance
_admet_predictor = ADMETPredictor()


def predict_admet(smiles: str, **kwargs) -> Dict[str, Any]:
    """
    Predict ADMET properties for a SMILES string
    
    Args:
        smiles: SMILES string
        **kwargs: Additional prediction options
        
    Returns:
        Dictionary with ADMET properties
    """
    try:
        result = _admet_predictor.predict(smiles, **kwargs)
        return result.to_dict()
    except CheminformaticsError as e:
        return {"error": str(e), "smiles": smiles}
