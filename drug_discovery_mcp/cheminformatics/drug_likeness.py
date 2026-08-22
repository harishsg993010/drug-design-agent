"""
Drug-Likeness Checker

Evaluates molecules for drug-like properties using various rule sets
and scoring systems.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

from .base import CheminformaticsBase, CheminformaticsError
from .descriptors import DescriptorCalculator

logger = logging.getLogger(__name__)


@dataclass
class DrugLikenessResult:
    """Result of drug-likeness evaluation"""
    smiles: str
    
    # Overall assessment
    is_drug_like: bool = False
    overall_score: float = 0.0
    
    # Individual rule sets
    lipinski: Dict[str, Any] = None
    ghose: Dict[str, Any] = None
    veber: Dict[str, Any] = None
    egan: Dict[str, Any] = None
    mute: Dict[str, Any] = None
    
    # Scoring systems
    qed: Optional[float] = None  # Quantitative Estimate of Drug-likeness
    
    # Additional metrics
    synthetic_accessibility: Optional[float] = None
    natural_product_likeness: Optional[float] = None
    
    # Recommendations
    recommendations: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.lipinski is None:
            self.lipinski = {}
        if self.ghose is None:
            self.ghose = {}
        if self.veber is None:
            self.veber = {}
        if self.egan is None:
            self.egan = {}
        if self.mute is None:
            self.mute = {}
        if self.recommendations is None:
            self.recommendations = []
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "smiles": self.smiles,
            "is_drug_like": self.is_drug_like,
            "overall_score": self.overall_score,
            "lipinski": self.lipinski,
            "ghose": self.ghose,
            "veber": self.veber,
            "egan": self.egan,
            "mute": self.mute,
            "qed": self.qed,
            "synthetic_accessibility": self.synthetic_accessibility,
            "natural_product_likeness": self.natural_product_likeness,
            "recommendations": self.recommendations,
            "warnings": self.warnings
        }


class DrugLikenessChecker(CheminformaticsBase):
    """
    Checker for drug-likeness properties
    
    Evaluates molecules using various rule sets and scoring systems:
    - Lipinski's Rule of Five
    - Ghose filter
    - Veber rules
    - Egan rules
    - MUE rules
    - Quantitative Estimate of Drug-likeness (QED)
    - Synthetic Accessibility (SA) score
    - Natural Product Likeness (NPL) score
    
    Provides recommendations for improving drug-likeness.
    """
    
    def __init__(self):
        """Initialize drug-likeness checker"""
        super().__init__()
        self.descriptor_calculator = DescriptorCalculator()
    
    def check(self, smiles: str, **kwargs) -> DrugLikenessResult:
        """
        Check drug-likeness for a SMILES string
        
        Args:
            smiles: SMILES string
            **kwargs: Additional checking options
            
        Returns:
            DrugLikenessResult with evaluation results
        """
        try:
            Chem = self._get_rdkit()
            
            # Sanitize SMILES
            smiles = self._sanitize_smiles(smiles)
            mol = self._smiles_to_mol(smiles)
            
            # Calculate descriptors
            descriptors = self.descriptor_calculator.calculate(smiles)
            
            # Create result object
            result = DrugLikenessResult(smiles=smiles)
            
            # Evaluate using different rule sets
            result.lipinski = self._evaluate_lipinski(mol, descriptors)
            result.ghose = self._evaluate_ghose(mol, descriptors)
            result.veber = self._evaluate_veber(mol, descriptors)
            result.egan = self._evaluate_egan(mol, descriptors)
            result.mute = self._evaluate_mute(mol, descriptors)
            
            # Calculate scoring systems
            result.qed = self._calculate_qed(mol)
            result.synthetic_accessibility = self._calculate_synthetic_accessibility(mol)
            result.natural_product_likeness = self._calculate_natural_product_likeness(mol)
            
            # Generate recommendations and warnings
            result.recommendations = self._generate_recommendations(result)
            result.warnings = self._generate_warnings(result)
            
            # Calculate overall score
            result.overall_score = self._calculate_overall_score(result)
            
            # Determine if drug-like
            result.is_drug_like = result.overall_score >= 0.5
            
            return result
            
        except Exception as e:
            logger.error(f"Drug-likeness check failed for {smiles}: {e}")
            raise CheminformaticsError(f"Failed to check drug-likeness: {e}")
    
    def check_batch(self, smiles_list: List[str]) -> List[DrugLikenessResult]:
        """
        Check drug-likeness for multiple SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of DrugLikenessResult objects
        """
        results = []
        for smiles in smiles_list:
            try:
                result = self.check(smiles)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to check drug-likeness for {smiles}: {e}")
                results.append(DrugLikenessResult(smiles=smiles))
        
        return results
    
    def filter_drug_like(
        self,
        smiles_list: List[str],
        min_score: float = 0.5,
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        Filter a list of SMILES to find drug-like molecules
        
        Args:
            smiles_list: List of SMILES strings
            min_score: Minimum overall score to be considered drug-like
            strict: If True, require all rule sets to pass
            
        Returns:
            Dictionary with filtered results
        """
        results = []
        for smiles in smiles_list:
            try:
                result = self.check(smiles)
                
                if strict:
                    # All rule sets must pass
                    passes_all = (
                        result.lipinski.get("passes", False) and
                        result.ghose.get("passes", False) and
                        result.veber.get("passes", False) and
                        result.egan.get("passes", False) and
                        result.mute.get("passes", False)
                    )
                    if passes_all:
                        results.append(result)
                else:
                    # Use overall score
                    if result.overall_score >= min_score:
                        results.append(result)
                        
            except Exception as e:
                logger.error(f"Failed to check drug-likeness for {smiles}: {e}")
        
        # Sort by overall score (descending)
        results.sort(key=lambda x: x.overall_score, reverse=True)
        
        return {
            "total": len(smiles_list),
            "drug_like": len(results),
            "results": [r.to_dict() for r in results]
        }
    
    # Individual evaluation methods
    
    def _evaluate_lipinski(self, mol: Any, descriptors: Any) -> Dict[str, Any]:
        """Evaluate using Lipinski's Rule of Five"""
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
        
        return {
            "name": "Lipinski's Rule of Five",
            "passes": violations <= 1,
            "violations": violations,
            "max_violations": 1,
            "details": {
                "molecular_weight": {"value": mw, "limit": 500, "passes": mw <= 500},
                "logp": {"value": logp, "limit": 5, "passes": logp <= 5},
                "hba": {"value": hba, "limit": 10, "passes": hba <= 10},
                "hbd": {"value": hbd, "limit": 5, "passes": hbd <= 5}
            }
        }
    
    def _evaluate_ghose(self, mol: Any, descriptors: Any) -> Dict[str, Any]:
        """Evaluate using Ghose filter"""
        mw = descriptors.molecular_weight
        logp = descriptors.logp
        num_c = descriptors.num_carbons
        num_hetero = descriptors.num_heteroatoms
        
        violations = 0
        if mw < 160 or mw > 480:
            violations += 1
        if logp < 0.4 or logp > 5.6:
            violations += 1
        if num_c < 4 or num_c > 36:
            violations += 1
        if num_hetero < 1 or num_hetero > 12:
            violations += 1
        
        return {
            "name": "Ghose Filter",
            "passes": violations == 0,
            "violations": violations,
            "max_violations": 0,
            "details": {
                "molecular_weight": {"value": mw, "min": 160, "max": 480, "passes": 160 <= mw <= 480},
                "logp": {"value": logp, "min": 0.4, "max": 5.6, "passes": 0.4 <= logp <= 5.6},
                "num_carbons": {"value": num_c, "min": 4, "max": 36, "passes": 4 <= num_c <= 36},
                "num_heteroatoms": {"value": num_hetero, "min": 1, "max": 12, "passes": 1 <= num_hetero <= 12}
            }
        }
    
    def _evaluate_veber(self, mol: Any, descriptors: Any) -> Dict[str, Any]:
        """Evaluate using Veber rules"""
        rot_bonds = descriptors.rotatable_bonds
        tpsa = descriptors.tpsa
        
        violations = 0
        if rot_bonds > 10:
            violations += 1
        if tpsa > 140:
            violations += 1
        
        return {
            "name": "Veber Rules",
            "passes": violations <= 1,
            "violations": violations,
            "max_violations": 1,
            "details": {
                "rotatable_bonds": {"value": rot_bonds, "limit": 10, "passes": rot_bonds <= 10},
                "tpsa": {"value": tpsa, "limit": 140, "passes": tpsa <= 140}
            }
        }
    
    def _evaluate_egan(self, mol: Any, descriptors: Any) -> Dict[str, Any]:
        """Evaluate using Egan rules"""
        logp = descriptors.logp
        tpsa = descriptors.tpsa
        
        # Egan rule: logP <= 5.88 and TPSA <= 131.6
        passes = logp <= 5.88 and tpsa <= 131.6
        
        return {
            "name": "Egan Rules",
            "passes": passes,
            "violations": 0 if passes else 1,
            "max_violations": 0,
            "details": {
                "logp": {"value": logp, "limit": 5.88, "passes": logp <= 5.88},
                "tpsa": {"value": tpsa, "limit": 131.6, "passes": tpsa <= 131.6}
            }
        }
    
    def _evaluate_mute(self, mol: Any, descriptors: Any) -> Dict[str, Any]:
        """Evaluate using MUE rules"""
        mw = descriptors.molecular_weight
        logp = descriptors.logp
        tpsa = descriptors.tpsa
        rot_bonds = descriptors.rotatable_bonds
        
        violations = 0
        if mw < 150 or mw > 500:
            violations += 1
        if logp < -2 or logp > 5:
            violations += 1
        if tpsa < 40 or tpsa > 130:
            violations += 1
        if rot_bonds > 10:
            violations += 1
        
        return {
            "name": "MUE Rules",
            "passes": violations == 0,
            "violations": violations,
            "max_violations": 0,
            "details": {
                "molecular_weight": {"value": mw, "min": 150, "max": 500, "passes": 150 <= mw <= 500},
                "logp": {"value": logp, "min": -2, "max": 5, "passes": -2 <= logp <= 5},
                "tpsa": {"value": tpsa, "min": 40, "max": 130, "passes": 40 <= tpsa <= 130},
                "rotatable_bonds": {"value": rot_bonds, "limit": 10, "passes": rot_bonds <= 10}
            }
        }
    
    def _calculate_qed(self, mol: Any) -> Optional[float]:
        """Calculate Quantitative Estimate of Drug-likeness (QED)"""
        try:
            Chem = self._get_rdkit()
            from rdkit.Chem import QED
            return QED.qed(mol)
        except ImportError:
            logger.warning("QED module not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to calculate QED: {e}")
            return None
    
    def _calculate_synthetic_accessibility(self, mol: Any) -> Optional[float]:
        """Calculate Synthetic Accessibility (SA) score"""
        try:
            Chem = self._get_rdkit()
            from rdkit.Chem import Descriptors
            return Descriptors.SA(mol)
        except ImportError:
            logger.warning("SA descriptor not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to calculate SA score: {e}")
            return None
    
    def _calculate_natural_product_likeness(self, mol: Any) -> Optional[float]:
        """Calculate Natural Product Likeness (NPL) score"""
        try:
            Chem = self._get_rdkit()
            from rdkit.Chem import Descriptors
            return Descriptors.NP(mol)
        except ImportError:
            logger.warning("NP descriptor not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to calculate NP score: {e}")
            return None
    
    def _generate_recommendations(self, result: DrugLikenessResult) -> List[str]:
        """Generate recommendations for improving drug-likeness"""
        recommendations = []
        
        # Lipinski recommendations
        if not result.lipinski.get("passes", True):
            details = result.lipinski.get("details", {})
            if not details.get("molecular_weight", {}).get("passes", True):
                recommendations.append("Reduce molecular weight below 500 Da")
            if not details.get("logp", {}).get("passes", True):
                recommendations.append("Reduce lipophilicity (logP) below 5")
            if not details.get("hba", {}).get("passes", True):
                recommendations.append("Reduce number of hydrogen bond acceptors below 10")
            if not details.get("hbd", {}).get("passes", True):
                recommendations.append("Reduce number of hydrogen bond donors below 5")
        
        # Veber recommendations
        if not result.veber.get("passes", True):
            details = result.veber.get("details", {})
            if not details.get("rotatable_bonds", {}).get("passes", True):
                recommendations.append("Reduce number of rotatable bonds below 10")
            if not details.get("tpsa", {}).get("passes", True):
                recommendations.append("Reduce topological polar surface area below 140 Å²")
        
        # QED recommendations
        if result.qed is not None and result.qed < 0.5:
            recommendations.append("Improve overall drug-likeness (QED score)")
        
        # SA recommendations
        if result.synthetic_accessibility is not None and result.synthetic_accessibility > 5:
            recommendations.append("Improve synthetic accessibility")
        
        # Remove duplicates
        recommendations = list(set(recommendations))
        
        return recommendations
    
    def _generate_warnings(self, result: DrugLikenessResult) -> List[str]:
        """Generate warnings for drug-likeness issues"""
        warnings = []
        
        # Lipinski warnings
        if result.lipinski.get("violations", 0) > 1:
            warnings.append(f"Multiple Lipinski violations ({result.lipinski.get('violations', 0)})")
        
        # Ghose warnings
        if not result.ghose.get("passes", True):
            warnings.append("Fails Ghose filter")
        
        # Veber warnings
        if result.veber.get("violations", 0) > 1:
            warnings.append(f"Multiple Veber violations ({result.veber.get('violations', 0)})")
        
        # Egan warnings
        if not result.egan.get("passes", True):
            warnings.append("Fails Egan rules")
        
        # MUE warnings
        if not result.mute.get("passes", True):
            warnings.append("Fails MUE rules")
        
        # QED warnings
        if result.qed is not None and result.qed < 0.3:
            warnings.append(f"Low QED score ({result.qed:.2f})")
        
        # SA warnings
        if result.synthetic_accessibility is not None and result.synthetic_accessibility > 7:
            warnings.append(f"High synthetic complexity (SA score: {result.synthetic_accessibility:.2f})")
        
        # NPL warnings
        if result.natural_product_likeness is not None and result.natural_product_likeness < -0.5:
            warnings.append(f"Low natural product likeness (NPL score: {result.natural_product_likeness:.2f})")
        
        return warnings
    
    def _calculate_overall_score(self, result: DrugLikenessResult) -> float:
        """Calculate overall drug-likeness score (0-1)"""
        score = 0.0
        
        # Rule-based scoring
        if result.lipinski.get("passes", False):
            score += 0.2
        if result.ghose.get("passes", False):
            score += 0.2
        if result.veber.get("passes", False):
            score += 0.15
        if result.egan.get("passes", False):
            score += 0.15
        if result.mute.get("passes", False):
            score += 0.15
        
        # QED score (0-1)
        if result.qed is not None:
            score += result.qed * 0.15
        
        # Penalize for warnings
        score -= len(result.warnings) * 0.05
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return score


# Singleton instance
_drug_likeness_checker = DrugLikenessChecker()


def check_drug_likeness(smiles: str, **kwargs) -> Dict[str, Any]:
    """
    Check drug-likeness for a SMILES string
    
    Args:
        smiles: SMILES string
        **kwargs: Additional checking options
        
    Returns:
        Dictionary with drug-likeness evaluation results
    """
    try:
        result = _drug_likeness_checker.check(smiles, **kwargs)
        return result.to_dict()
    except CheminformaticsError as e:
        return {"error": str(e), "smiles": smiles}
