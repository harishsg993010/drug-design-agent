"""
Test Module for Drug Discovery MCP Server

This module contains tests for all components of the Drug Discovery MCP Server.
"""

from .test_databases import *
from .test_cheminformatics import *
from .test_structural_biology import *
from .test_tasks import *
from .test_server import *

__all__ = [
    # Database tests
    "test_uniprot_query",
    "test_chembl_query",
    "test_pdb_query",
    "test_opentargets_query",
    
    # Cheminformatics tests
    "test_descriptor_calculation",
    "test_conversion",
    "test_similarity",
    "test_fingerprints",
    "test_admet_prediction",
    "test_drug_likeness",
    
    # Structural biology tests
    "test_pdb_parsing",
    "test_structure_alignment",
    "test_binding_site_analysis",
    "test_structure_comparison",
    
    # Task tests
    "test_task_creation",
    "test_task_execution",
    "test_task_evaluation",
    "test_task_registry",
    
    # Server tests
    "test_server_initialization",
    "test_server_routes",
    "test_tool_registration",
]
