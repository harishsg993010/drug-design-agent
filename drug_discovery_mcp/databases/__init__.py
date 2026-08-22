"""
Database Clients Module

This module provides database client tools for querying biomedical databases.
Inspired by BIOMNI framework's database tools.

Supported databases:
- UniProt: Protein sequence, function, and annotation
- ChEMBL: Bioactivity data and drug-like properties
- OpenTargets: Target validation and disease association
- PDB: Protein 3D structures
- KEGG: Pathway and metabolic information
- PubChem: Chemical compound database
- NCBI: Genetic and genomic data
"""

from .base import DatabaseClient, DatabaseError
from .uniprot import UniProtClient
from .chembl import ChEMBLClient
from .pdb import PDBClient
from .opentargets import OpenTargetsClient
from .kegg import KEGGClient
from .pubchem import PubChemClient
from .ncbi import NCBIClient
from .tools import DatabaseTools

__all__ = [
    "DatabaseClient",
    "DatabaseError",
    "UniProtClient",
    "ChEMBLClient",
    "PDBClient",
    "OpenTargetsClient",
    "KEGGClient",
    "PubChemClient",
    "NCBIClient",
    "DatabaseTools",
]
