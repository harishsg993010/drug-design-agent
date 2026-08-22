"""
Cheminformatics Module

This module provides cheminformatics tools for molecular analysis and manipulation.
Inspired by BIOMNI framework's cheminformatics tools.

Features:
- Molecular descriptor calculation
- SMILES/InChI conversion
- Molecular similarity and clustering
- ADMET property prediction
- Drug-likeness scoring
- Molecular fingerprinting
- 2D and 3D structure generation
- Reaction prediction
- Chemical space analysis
"""

from .base import CheminformaticsError
from .descriptors import DescriptorCalculator, calculate_descriptors
from .conversion import ConversionTools, smiles_to_inchi, inchi_to_smiles
from .similarity import SimilarityTools, molecular_similarity
from .fingerprints import FingerprintTools, calculate_fingerprint
from .admet import ADMETPredictor, predict_admet
from .drug_likeness import DrugLikenessChecker, check_drug_likeness
from .tools import CheminformaticsTools

__all__ = [
    "CheminformaticsError",
    "DescriptorCalculator",
    "calculate_descriptors",
    "ConversionTools",
    "smiles_to_inchi",
    "inchi_to_smiles",
    "SimilarityTools",
    "molecular_similarity",
    "FingerprintTools",
    "calculate_fingerprint",
    "ADMETPredictor",
    "predict_admet",
    "DrugLikenessChecker",
    "check_drug_likeness",
    "CheminformaticsTools",
]
