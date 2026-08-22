"""
Task Definition Module

Defines the Task class and related enumerations for managing drug discovery tasks.
"""

import logging
import uuid
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TaskCategory(Enum):
    """Categories of drug discovery tasks"""
    STRUCTURAL_REASONING = "structural_reasoning"
    DATABASE_SCREENING = "database_screening"
    PATENT_MINING = "patent_mining"
    TARGET_ID_GENETICS = "target_id_genetics"
    CHEMINFORMATICS = "cheminformatics"
    SAR_AFFINITY = "sar_affinity"
    MOLECULAR_BIOLOGY = "molecular_biology"
    ADJACENT_BIOMEDICAL = "adjacent_biomedical"


class TaskStatus(Enum):
    """Status of a task"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TaskDifficulty(Enum):
    """Difficulty level of a task"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class TaskResult:
    """Result of a task execution"""
    task_id: str
    status: TaskStatus
    answer: Optional[Any] = None
    score: Optional[float] = None
    passed: bool = False
    execution_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "answer": self.answer,
            "score": self.score,
            "passed": self.passed,
            "execution_time": self.execution_time,
            "error": self.error,
            "metadata": self.metadata
        }


@dataclass
class Task:
    """
    Represents a drug discovery task
    
    A task is a specific, verifiable problem that an AI agent needs to solve.
    Tasks are inspired by the DrugDiscoveryBench benchmark and include:
    - A unique identifier
    - A category (e.g., structural reasoning, database screening)
    - A prompt/description
    - Expected answer format
    - Rubric for evaluation
    - Required tools and resources
    """
    
    # Task identification
    id: str
    name: str
    category: TaskCategory
    
    # Task content
    prompt: str
    description: str = ""
    
    # Expected output
    answer_format: str = "text"  # text, json, csv, smiles, etc.
    expected_answer: Optional[Any] = None
    
    # Task metadata
    difficulty: TaskDifficulty = TaskDifficulty.MEDIUM
    estimated_time: float = 300.0  # seconds
    max_time: float = 1200.0  # seconds (20 minutes)
    
    # Task requirements
    required_tools: List[str] = field(default_factory=list)
    required_databases: List[str] = field(default_factory=list)
    required_resources: List[str] = field(default_factory=list)
    
    # Evaluation
    rubric: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    
    # Status tracking
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Execution context
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize task with default values"""
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Ensure lists are initialized
        if self.required_tools is None:
            self.required_tools = []
        if self.required_databases is None:
            self.required_databases = []
        if self.required_resources is None:
            self.required_resources = []
        if self.rubric is None:
            self.rubric = {}
        if self.context is None:
            self.context = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "prompt": self.prompt,
            "description": self.description,
            "answer_format": self.answer_format,
            "expected_answer": self.expected_answer,
            "difficulty": self.difficulty.value,
            "estimated_time": self.estimated_time,
            "max_time": self.max_time,
            "required_tools": self.required_tools,
            "required_databases": self.required_databases,
            "required_resources": self.required_resources,
            "rubric": self.rubric,
            "weight": self.weight,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "context": self.context
        }
    
    def to_json(self) -> str:
        """Convert task to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create task from dictionary"""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            category=TaskCategory(data.get("category", "cheminformatics")),
            prompt=data.get("prompt", ""),
            description=data.get("description", ""),
            answer_format=data.get("answer_format", "text"),
            expected_answer=data.get("expected_answer"),
            difficulty=TaskDifficulty(data.get("difficulty", "medium")),
            estimated_time=data.get("estimated_time", 300.0),
            max_time=data.get("max_time", 1200.0),
            required_tools=data.get("required_tools", []),
            required_databases=data.get("required_databases", []),
            required_resources=data.get("required_resources", []),
            rubric=data.get("rubric", {}),
            weight=data.get("weight", 1.0),
            status=TaskStatus(data.get("status", "pending")),
            context=data.get("context", {})
        )
    
    def validate(self) -> bool:
        """Validate that the task has all required fields"""
        required_fields = ["id", "name", "category", "prompt"]
        for field in required_fields:
            if not getattr(self, field):
                logger.error(f"Task {self.id} is missing required field: {field}")
                return False
        return True
    
    def get_required_tools(self) -> List[str]:
        """Get list of required tools"""
        return self.required_tools
    
    def get_required_databases(self) -> List[str]:
        """Get list of required databases"""
        return self.required_databases
    
    def get_required_resources(self) -> List[str]:
        """Get list of required resources"""
        return self.required_resources
    
    def set_status(self, status: TaskStatus) -> None:
        """Update task status"""
        self.status = status
        self.updated_at = datetime.now()
    
    def add_context(self, key: str, value: Any) -> None:
        """Add context information"""
        self.context[key] = value
        self.updated_at = datetime.now()


# Example tasks from DrugDiscoveryBench

def create_example_tasks() -> List[Task]:
    """Create example tasks based on DrugDiscoveryBench"""
    
    tasks = []
    
    # Structural Reasoning Task
    tasks.append(Task(
        id="e3c637",
        name="Net Charge of Ligand in Crystal Structure",
        category=TaskCategory.STRUCTURAL_REASONING,
        prompt="""
        We are studying the binding of an inhibitor to HIV reverse transcriptase (RT).
        In the PDB structure 5J1E, there is a ligand with ID 6FT bound to HIV-RT.
        
        Question: What is the net charge of the ligand 6FT as it appears in the actual crystal structure?
        
        Provide your answer as a single integer in the file /workspace/answer.md.
        """,
        description="Determine the net charge of ligand 6FT in PDB structure 5J1E",
        answer_format="integer",
        expected_answer=-1,
        difficulty=TaskDifficulty.MEDIUM,
        estimated_time=180.0,
        required_tools=["pdb_parser", "charge_calculator"],
        required_databases=["pdb"],
        rubric={
            "outcome": {
                "correct_answer": {"weight": 30, "expected": -1},
                "format": {"weight": 10, "expected": "integer"}
            },
            "process": {
                "downloads_pdb": {"weight": 15, "expected": True},
                "identifies_ligand": {"weight": 15, "expected": True},
                "calculates_charge": {"weight": 10, "expected": True}
            }
        },
        weight=1.0
    ))
    
    # Database Screening Task
    tasks.append(Task(
        id="db_screen_001",
        name="Find Potent Inhibitors of EGFR",
        category=TaskCategory.DATABASE_SCREENING,
        prompt="""
        We are looking for potent inhibitors of EGFR (Epidermal Growth Factor Receptor).
        
        Task: Search the ChEMBL database for compounds with IC50 < 10 nM against EGFR (target CHEMBL203).
        Return the top 5 compounds sorted by IC50 (ascending), including:
        - ChEMBL compound ID
        - SMILES
        - IC50 value and unit
        - Target name
        
        Provide your answer as a JSON array in the file /workspace/answer.md.
        """,
        description="Search ChEMBL for potent EGFR inhibitors",
        answer_format="json",
        expected_answer=[
            {"compound_id": "CHEMBL123", "smiles": "CCO", "ic50": 1.0, "unit": "nM", "target": "EGFR"},
            # More compounds would be here in a real task
        ],
        difficulty=TaskDifficulty.MEDIUM,
        estimated_time=300.0,
        required_tools=["chembl_search", "filter_compounds"],
        required_databases=["chembl"],
        rubric={
            "outcome": {
                "correct_format": {"weight": 15, "expected": "json"},
                "has_5_compounds": {"weight": 10, "expected": True},
                "sorted_by_ic50": {"weight": 10, "expected": True},
                "ic50_under_10nm": {"weight": 15, "expected": True}
            },
            "process": {
                "queries_chembl": {"weight": 10, "expected": True},
                "filters_by_target": {"weight": 10, "expected": True},
                "filters_by_ic50": {"weight": 10, "expected": True}
            }
        },
        weight=1.0
    ))
    
    # Target ID & Genetics Task
    tasks.append(Task(
        id="target_id_001",
        name="Identify Targets for Alzheimer's Disease",
        category=TaskCategory.TARGET_ID_GENETICS,
        prompt="""
        We are researching potential drug targets for Alzheimer's disease.
        
        Task: Use OpenTargets to find genes associated with Alzheimer's disease (MONDO_0004975).
        Return the top 5 targets sorted by association score (descending), including:
        - Target ID
        - Gene symbol
        - Target name
        - Association score
        - Evidence count
        
        Provide your answer as a JSON array in the file /workspace/answer.md.
        """,
        description="Find targets for Alzheimer's disease using OpenTargets",
        answer_format="json",
        expected_answer=[
            {"target_id": "ENSG00000123456", "gene_symbol": "APP", "target_name": "Amyloid precursor protein", "score": 0.95, "evidence_count": 10},
            # More targets would be here in a real task
        ],
        difficulty=TaskDifficulty.MEDIUM,
        estimated_time=300.0,
        required_tools=["opentargets_search", "filter_targets"],
        required_databases=["opentargets"],
        rubric={
            "outcome": {
                "correct_format": {"weight": 15, "expected": "json"},
                "has_5_targets": {"weight": 10, "expected": True},
                "sorted_by_score": {"weight": 10, "expected": True},
                "alzheimer_association": {"weight": 15, "expected": True}
            },
            "process": {
                "queries_opentargets": {"weight": 10, "expected": True},
                "filters_by_disease": {"weight": 10, "expected": True},
                "includes_evidence": {"weight": 10, "expected": True}
            }
        },
        weight=1.0
    ))
    
    # Cheminformatics Task
    tasks.append(Task(
        id="chem_001",
        name="Calculate Molecular Descriptors",
        category=TaskCategory.CHEMINFORMATICS,
        prompt="""
        We have a compound with SMILES: CC(=O)OC1=CC=CC=C1C(=O)O
        
        Task: Calculate the following molecular descriptors:
        - Molecular weight
        - logP
        - Number of hydrogen bond acceptors (HBA)
        - Number of hydrogen bond donors (HBD)
        - Topological polar surface area (TPSA)
        - Number of rotatable bonds
        - Number of aromatic rings
        
        Provide your answer as a JSON object in the file /workspace/answer.md with the format:
        {
            "molecular_weight": <value>,
            "logp": <value>,
            "hba": <value>,
            "hbd": <value>,
            "tpsa": <value>,
            "rotatable_bonds": <value>,
            "aromatic_rings": <value>
        }
        """,
        description="Calculate molecular descriptors for aspirin",
        answer_format="json",
        expected_answer={
            "molecular_weight": 180.16,
            "logp": 1.19,
            "hba": 4,
            "hbd": 1,
            "tpsa": 63.6,
            "rotatable_bonds": 2,
            "aromatic_rings": 1
        },
        difficulty=TaskDifficulty.EASY,
        estimated_time=60.0,
        required_tools=["descriptor_calculator"],
        required_databases=[],
        rubric={
            "outcome": {
                "correct_format": {"weight": 10, "expected": "json"},
                "molecular_weight": {"weight": 5, "expected": 180.16, "tolerance": 0.1},
                "logp": {"weight": 5, "expected": 1.19, "tolerance": 0.1},
                "hba": {"weight": 5, "expected": 4},
                "hbd": {"weight": 5, "expected": 1},
                "tpsa": {"weight": 5, "expected": 63.6, "tolerance": 0.5},
                "rotatable_bonds": {"weight": 5, "expected": 2},
                "aromatic_rings": {"weight": 5, "expected": 1}
            },
            "process": {
                "calculates_descriptors": {"weight": 10, "expected": True}
            }
        },
        weight=1.0
    ))
    
    # SAR / Affinity Task
    tasks.append(Task(
        id="sar_001",
        name="Rank Compounds by Binding Affinity",
        category=TaskCategory.SAR_AFFINITY,
        prompt="""
        We have the following compounds with their IC50 values against a target:
        - Compound A: CC1=CC=CC=C1, IC50 = 100 nM
        - Compound B: CC1=CC=C(C=C1)C, IC50 = 10 nM
        - Compound C: CC1=CC=CC=C1O, IC50 = 1 nM
        - Compound D: CC1=CC=C(C=C1)O, IC50 = 50 nM
        - Compound E: CC1=CC=CC=C1N, IC50 = 500 nM
        
        Task: Rank the compounds by binding affinity (from most potent to least potent).
        
        Provide your answer as a JSON array in the file /workspace/answer.md with the format:
        ["CompoundX", "CompoundY", ...]
        """,
        description="Rank compounds by binding affinity",
        answer_format="json",
        expected_answer=["Compound C", "Compound B", "Compound D", "Compound A", "Compound E"],
        difficulty=TaskDifficulty.EASY,
        estimated_time=60.0,
        required_tools=["affinity_ranking"],
        required_databases=[],
        rubric={
            "outcome": {
                "correct_format": {"weight": 10, "expected": "json"},
                "correct_ranking": {"weight": 30, "expected": ["Compound C", "Compound B", "Compound D", "Compound A", "Compound E"]}
            },
            "process": {
                "understands_affinity": {"weight": 10, "expected": True},
                "sorts_correctly": {"weight": 10, "expected": True}
            }
        },
        weight=1.0
    ))
    
    # Lead Optimization Task (from DrugDiscoveryBench example)
    tasks.append(Task(
        id="lead_opt_kras",
        name="KRAS G12C Lead Optimization",
        category=TaskCategory.STRUCTURAL_REASONING,
        prompt="""
        We're comparing two lead candidates, compound A and compound B, of a KRAS G12C drug development campaign.
        We manage to obtain co-crystal structure data of KRAS G12C with compound A (6USX.pdb) and compound B (6UT0.pdb).
        Experimentally, we know that compound B is more potent. We suspect that a significant conformational change 
        of one specific residue X occurs, when compound B is used instead of compound A. This allows for more 
        hydrophobic contact of residue X with compound B.
        
        Provide the name of residue X, as 3-letter abbreviation with residue number (from the PDB file).
        
        Provide your answer in the file /workspace/answer.md.
        """,
        description="Identify residue with conformational change between KRAS G12C structures",
        answer_format="text",
        expected_answer="THR58",
        difficulty=TaskDifficulty.HARD,
        estimated_time=600.0,
        required_tools=["pdb_parser", "structure_alignment", "conformation_analysis"],
        required_databases=["pdb"],
        rubric={
            "outcome": {
                "correct_residue": {"weight": 30, "expected": "THR58"},
                "format": {"weight": 10, "expected": "3-letter abbreviation with residue number"}
            },
            "process": {
                "obtains_structures": {"weight": 20, "expected": True},
                "superimposes_structures": {"weight": 20, "expected": True},
                "identifies_conformational_changes": {"weight": 20, "expected": True},
                "verifies_hydrophobic_contact": {"weight": 25, "expected": True}
            }
        },
        weight=1.0
    ))
    
    return tasks
