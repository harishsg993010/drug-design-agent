"""
Molecular Similarity Tools

Calculates molecular similarity and distance metrics between compounds.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from .base import CheminformaticsBase, CheminformaticsError
from .fingerprints import FingerprintTools

logger = logging.getLogger(__name__)


class SimilarityMetric(Enum):
    """Similarity metrics"""
    TANIMOTO = "tanimoto"
    DICE = "dice"
    COSINE = "cosine"
    JACCARD = "jaccard"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class DistanceMetric(Enum):
    """Distance metrics"""
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"


@dataclass
class SimilarityResult:
    """Result of a similarity calculation"""
    smiles1: str
    smiles2: str
    similarity: float
    metric: str
    fingerprint_type: str
    distance: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "smiles1": self.smiles1,
            "smiles2": self.smiles2,
            "similarity": self.similarity,
            "metric": self.metric,
            "fingerprint_type": self.fingerprint_type
        }
        if self.distance is not None:
            result["distance"] = self.distance
        return result


class SimilarityTools(CheminformaticsBase):
    """
    Tools for calculating molecular similarity
    
    Provides various similarity and distance metrics for comparing molecules:
    - Tanimoto similarity (Jaccard similarity for fingerprints)
    - Dice similarity
    - Cosine similarity
    - Euclidean distance
    - Manhattan distance
    
    Uses molecular fingerprints for comparison.
    """
    
    def __init__(self):
        """Initialize similarity tools"""
        super().__init__()
        self.fingerprint_tools = FingerprintTools()
    
    def calculate(
        self,
        smiles1: str,
        smiles2: str,
        metric: str = "tanimoto",
        fingerprint_type: str = "morgan",
        radius: int = 2,
        bit_length: int = 2048
    ) -> SimilarityResult:
        """
        Calculate similarity between two SMILES strings
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            metric: Similarity metric ("tanimoto", "dice", "cosine", "jaccard")
            fingerprint_type: Type of fingerprint ("morgan", "rdkit", "atom_pair", "topological_torsion")
            radius: Fingerprint radius (for Morgan fingerprints)
            bit_length: Fingerprint bit length
            
        Returns:
            SimilarityResult with similarity score
        """
        try:
            Chem = self._get_rdkit()
            
            # Sanitize SMILES
            smiles1 = self._sanitize_smiles(smiles1)
            smiles2 = self._sanitize_smiles(smiles2)
            
            # Generate fingerprints
            fp1 = self.fingerprint_tools.calculate(
                smiles1, 
                fingerprint_type=fingerprint_type, 
                radius=radius, 
                bit_length=bit_length
            )
            fp2 = self.fingerprint_tools.calculate(
                smiles2, 
                fingerprint_type=fingerprint_type, 
                radius=radius, 
                bit_length=bit_length
            )
            
            # Convert to numpy arrays
            fp1_array = np.array(fp1.fingerprint)
            fp2_array = np.array(fp2.fingerprint)
            
            # Calculate similarity based on metric
            metric = metric.lower()
            
            if metric == "tanimoto" or metric == "jaccard":
                similarity = self._tanimoto_similarity(fp1_array, fp2_array)
            elif metric == "dice":
                similarity = self._dice_similarity(fp1_array, fp2_array)
            elif metric == "cosine":
                similarity = self._cosine_similarity(fp1_array, fp2_array)
            elif metric == "euclidean":
                similarity = self._euclidean_similarity(fp1_array, fp2_array)
            elif metric == "manhattan":
                similarity = self._manhattan_similarity(fp1_array, fp2_array)
            else:
                raise ValueError(f"Unknown similarity metric: {metric}")
            
            # Calculate distance for distance-based metrics
            distance = None
            if metric in ["euclidean", "manhattan"]:
                distance = 1 - similarity  # Convert similarity to distance
            
            return SimilarityResult(
                smiles1=smiles1,
                smiles2=smiles2,
                similarity=similarity,
                metric=metric,
                fingerprint_type=fingerprint_type,
                distance=distance
            )
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            raise CheminformaticsError(f"Failed to calculate similarity: {e}")
    
    def calculate_batch(
        self,
        smiles_list: List[str],
        metric: str = "tanimoto",
        fingerprint_type: str = "morgan",
        radius: int = 2,
        bit_length: int = 2048
    ) -> List[List[float]]:
        """
        Calculate pairwise similarity matrix for a list of SMILES
        
        Args:
            smiles_list: List of SMILES strings
            metric: Similarity metric
            fingerprint_type: Type of fingerprint
            radius: Fingerprint radius
            bit_length: Fingerprint bit length
            
        Returns:
            Similarity matrix (n x n) where matrix[i][j] is similarity between smiles_list[i] and smiles_list[j]
        """
        n = len(smiles_list)
        similarity_matrix = [[0.0] * n for _ in range(n)]
        
        # Generate fingerprints for all molecules
        fingerprints = []
        for smiles in smiles_list:
            fp = self.fingerprint_tools.calculate(
                smiles,
                fingerprint_type=fingerprint_type,
                radius=radius,
                bit_length=bit_length
            )
            fingerprints.append(np.array(fp.fingerprint))
        
        # Calculate pairwise similarities
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    metric_lower = metric.lower()
                    if metric_lower == "tanimoto" or metric_lower == "jaccard":
                        sim = self._tanimoto_similarity(fingerprints[i], fingerprints[j])
                    elif metric_lower == "dice":
                        sim = self._dice_similarity(fingerprints[i], fingerprints[j])
                    elif metric_lower == "cosine":
                        sim = self._cosine_similarity(fingerprints[i], fingerprints[j])
                    else:
                        sim = self._tanimoto_similarity(fingerprints[i], fingerprints[j])
                    
                    similarity_matrix[i][j] = sim
                    similarity_matrix[j][i] = sim  # Symmetric
        
        return similarity_matrix
    
    def find_most_similar(
        self,
        query_smiles: str,
        candidate_smiles: List[str],
        metric: str = "tanimoto",
        fingerprint_type: str = "morgan",
        radius: int = 2,
        bit_length: int = 2048,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find most similar molecules to a query from a list of candidates
        
        Args:
            query_smiles: Query SMILES string
            candidate_smiles: List of candidate SMILES strings
            metric: Similarity metric
            fingerprint_type: Type of fingerprint
            radius: Fingerprint radius
            bit_length: Fingerprint bit length
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with similarity scores, sorted by similarity (descending)
        """
        results = []
        
        for candidate in candidate_smiles:
            try:
                result = self.calculate(
                    query_smiles,
                    candidate,
                    metric=metric,
                    fingerprint_type=fingerprint_type,
                    radius=radius,
                    bit_length=bit_length
                )
                results.append({
                    "smiles": candidate,
                    "similarity": result.similarity,
                    "metric": result.metric,
                    "fingerprint_type": result.fingerprint_type
                })
            except Exception as e:
                logger.warning(f"Failed to calculate similarity with {candidate}: {e}")
                results.append({
                    "smiles": candidate,
                    "similarity": 0.0,
                    "metric": metric,
                    "fingerprint_type": fingerprint_type,
                    "error": str(e)
                })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results[:limit]
    
    def cluster(
        self,
        smiles_list: List[str],
        threshold: float = 0.7,
        metric: str = "tanimoto",
        fingerprint_type: str = "morgan",
        radius: int = 2,
        bit_length: int = 2048
    ) -> List[List[int]]:
        """
        Cluster molecules based on similarity
        
        Args:
            smiles_list: List of SMILES strings
            threshold: Similarity threshold for clustering
            metric: Similarity metric
            fingerprint_type: Type of fingerprint
            radius: Fingerprint radius
            bit_length: Fingerprint bit length
            
        Returns:
            List of clusters, where each cluster is a list of indices into smiles_list
        """
        n = len(smiles_list)
        if n == 0:
            return []
        
        # Calculate similarity matrix
        sim_matrix = self.calculate_batch(
            smiles_list,
            metric=metric,
            fingerprint_type=fingerprint_type,
            radius=radius,
            bit_length=bit_length
        )
        
        # Perform clustering (simple threshold-based clustering)
        clusters = []
        assigned = [False] * n
        
        for i in range(n):
            if not assigned[i]:
                # Start new cluster
                cluster = [i]
                assigned[i] = True
                
                # Find all molecules similar to i
                for j in range(i + 1, n):
                    if not assigned[j] and sim_matrix[i][j] >= threshold:
                        cluster.append(j)
                        assigned[j] = True
                
                clusters.append(cluster)
        
        return clusters
    
    # Similarity calculation methods
    
    def _tanimoto_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate Tanimoto similarity (Jaccard similarity for fingerprints)"""
        intersection = np.sum(np.logical_and(fp1, fp2))
        union = np.sum(np.logical_or(fp1, fp2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _dice_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate Dice similarity"""
        intersection = np.sum(np.logical_and(fp1, fp2))
        sum_fp = np.sum(fp1) + np.sum(fp2)
        
        if sum_fp == 0:
            return 0.0
        
        return (2 * intersection) / sum_fp
    
    def _cosine_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate Cosine similarity"""
        dot_product = np.dot(fp1, fp2)
        norm1 = np.linalg.norm(fp1)
        norm2 = np.linalg.norm(fp2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _euclidean_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate Euclidean similarity (inverse of distance)"""
        distance = np.linalg.norm(fp1 - fp2)
        max_distance = np.sqrt(fp1.size)  # Maximum possible distance
        
        if max_distance == 0:
            return 1.0
        
        return 1 - (distance / max_distance)
    
    def _manhattan_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate Manhattan similarity (inverse of distance)"""
        distance = np.sum(np.abs(fp1 - fp2))
        max_distance = fp1.size  # Maximum possible distance
        
        if max_distance == 0:
            return 1.0
        
        return 1 - (distance / max_distance)


# Singleton instance
_similarity_tools = SimilarityTools()


def molecular_similarity(
    smiles1: str,
    smiles2: str,
    metric: str = "tanimoto",
    **kwargs
) -> Dict[str, Any]:
    """
    Calculate molecular similarity between two SMILES strings
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        metric: Similarity metric ("tanimoto", "dice", "cosine", "jaccard")
        **kwargs: Additional options (fingerprint_type, radius, bit_length)
        
    Returns:
        Dictionary with similarity result
    """
    try:
        result = _similarity_tools.calculate(smiles1, smiles2, metric=metric, **kwargs)
        return result.to_dict()
    except CheminformaticsError as e:
        return {"error": str(e), "smiles1": smiles1, "smiles2": smiles2}
