"""
Drug Discovery MCP Server

An MCP (Model Context Protocol) server for assisting with early-stage drug discovery tasks,
inspired by the DrugDiscoveryBench benchmark and BIOMNI framework.

This package provides:
- 200+ biomedical and drug discovery tools across 22 domains
- Database clients for UniProt, ChEMBL, OpenTargets, PDB, etc.
- Cheminformatics tools using RDKit
- Structural biology analysis tools
- Patent and literature mining capabilities
- Task management system for DrugDiscoveryBench-style tasks
- Rubric-based evaluation system
"""

__version__ = "0.1.0"
__author__ = "Harish G"
__email__ = "harishsg993010@gmail.com"

# Import core modules
from .server import DrugDiscoveryMCPServer
from .client import DrugDiscoveryClient
from .config import settings, DrugDiscoveryConfig

# Import domain modules
from . import databases
from . import cheminformatics
from . import structural_biology
from . import patent_mining
from . import target_identification
from . import hit_identification
from . import sar_analysis

# Import task system
from .tasks import TaskRunner, Task
from .evaluation import Evaluator, Rubric, RubricCriterion

__all__ = [
    # Core
    "__version__",
    "__author__",
    "__email__",
    "DrugDiscoveryMCPServer",
    "DrugDiscoveryClient",
    "settings",
    "DrugDiscoveryConfig",
    # Domains
    "databases",
    "cheminformatics",
    "structural_biology",
    "patent_mining",
    "target_identification",
    "hit_identification",
    "sar_analysis",
    # Task system
    "TaskRunner",
    "Task",
    "Evaluator",
    "Rubric",
    "RubricCriterion",
]
