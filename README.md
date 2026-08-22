# Drug Discovery MCP Server

An MCP (Model Context Protocol) server for assisting Mistral Vibe with early-stage drug discovery tasks, inspired by the DrugDiscoveryBench benchmark and BIOMNI framework.

## Overview

This server provides 200+ biomedical and drug discovery tools across 22 domains, enabling AI agents to:
- Query biological databases (UniProt, ChEMBL, OpenTargets, PDB, etc.)
- Perform cheminformatics calculations (RDKit)
- Analyze protein structures and molecular interactions
- Mine patents and scientific literature
- Execute multi-step drug discovery workflows

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Drug Discovery MCP Server                  │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │   Database   │  │ Cheminform-  │  │  Structural Bio-   │  │
│  │   Clients    │  │   atics      │  │    logy Tools      │  │
│  └──────────────┘  └──────────────┘  └────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  Patent &     │  │  Target ID &  │  │   Task Management  │  │
│  │ Literature    │  │  Genetics     │  │   & Evaluation     │  │
│  └──────────────┘  └──────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Features

### Database Clients (Domain 1)
- **UniProt**: Protein sequence, function, and annotation
- **ChEMBL**: Bioactivity data and drug-like properties
- **OpenTargets**: Target validation and disease association
- **PDB**: Protein 3D structures
- **KEGG**: Pathway and metabolic information
- **PubChem**: Chemical compound database
- **NCBI**: Genetic and genomic data

### Cheminformatics (Domain 2)
- Molecular descriptor calculation
- SMILES/InChI conversion
- Molecular similarity and clustering
- ADMET property prediction
- Drug-likeness scoring
- Molecular fingerprinting

### Structural Biology (Domain 3)
- PDB file parsing and manipulation
- Structure superimposition and alignment
- Binding site analysis
- Molecular docking preparation
- Protein-ligand interaction analysis

### Patent & Literature Mining (Domain 4)
- Patent database searching
- Scientific literature retrieval
- Text mining and entity extraction
- Citation network analysis

### Target Identification & Validation (Domain 5)
- Gene-disease association analysis
- Target tractability assessment
- Genetic evidence evaluation
- Pathway analysis

### Hit Identification (Domain 6)
- Virtual screening
- Similarity searching
- Pharmacophore matching
- Database filtering

### Hit-to-Lead & SAR (Domain 7)
- Structure-activity relationship analysis
- Lead optimization workflows
- Bioisostere replacement
- Scaffold hopping

## Installation

```bash
# Clone the repository
git clone https://github.com/harishsg993010/drug-design-agent.git
cd drug-design-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Install RDKit (required for cheminformatics)
conda install -c conda-forge rdkit

# Start the MCP server
python -m drug_discovery_mcp.server
```

## Usage

### As MCP Server

```bash
# Start server (default port 8080)
python -m drug_discovery_mcp.server --port 8080

# Or with custom configuration
python -m drug_discovery_mcp.server --config config.json --log-level DEBUG
```

### Direct Python Usage

```python
from drug_discovery_mcp import DrugDiscoveryClient

# Initialize client
client = DrugDiscoveryClient()

# Query UniProt
protein_info = client.query_uniprot("P12345")

# Calculate molecular descriptors
from rdkit import Chem
mol = Chem.MolFromSmiles("CCO")
descriptors = client.calculate_descriptors(mol)

# Search ChEMBL
bioactivity = client.query_chembl("CHEMBL123")
```

## Configuration

Create a `config.json` file:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "max_workers": 10,
    "timeout": 300
  },
  "databases": {
    "uniprot": {
      "endpoint": "https://www.ebi.ac.uk/proteins/api",
      "rate_limit": 10
    },
    "chembl": {
      "endpoint": "https://www.ebi.ac.uk/chembl/api",
      "rate_limit": 10
    },
    "pdb": {
      "endpoint": "https://data.rcsb.org",
      "rate_limit": 20
    }
  },
  "cache": {
    "enabled": true,
    "directory": "./cache",
    "expiry_days": 7
  },
  "logging": {
    "level": "INFO",
    "file": "./logs/server.log"
  }
}
```

## API Reference

### Database Tools

#### `query_uniprot(accession: str) -> dict`
Query UniProt database for protein information.

**Parameters:**
- `accession`: UniProt accession number (e.g., "P12345")

**Returns:**
```json
{
  "accession": "P12345",
  "entry_name": "PROTEIN_HUMAN",
  "protein_name": "Example protein",
  "gene_name": "EXPG",
  "organism": "Homo sapiens",
  "sequence": "MA...",
  "function": "Example function",
  "pathways": [...],
  "go_annotations": [...],
  "diseases": [...]
}
```

#### `query_chembl(compound_id: str) -> dict`
Query ChEMBL database for compound bioactivity data.

**Parameters:**
- `compound_id`: ChEMBL compound ID (e.g., "CHEMBL123")

**Returns:**
```json
{
  "compound_id": "CHEMBL123",
  "smiles": "CCO",
  "molecular_weight": 46.07,
  "logp": 0.32,
  "targets": [...],
  "assays": [...],
  "bioactivities": [...]
}
```

#### `query_pdb(pdb_id: str) -> dict`
Query PDB database for protein structure information.

**Parameters:**
- `pdb_id`: PDB ID (e.g., "1ABC")

**Returns:**
```json
{
  "pdb_id": "1ABC",
  "title": "Structure of example protein",
  "resolution": 1.8,
  "method": "X-RAY DIFFRACTION",
  "chains": [...],
  "ligands": [...],
  "experimental_data": {...}
}
```

### Cheminformatics Tools

#### `calculate_descriptors(smiles: str) -> dict`
Calculate molecular descriptors for a SMILES string.

**Parameters:**
- `smiles`: SMILES string (e.g., "CCO")

**Returns:**
```json
{
  "smiles": "CCO",
  "molecular_weight": 46.07,
  "logp": 0.32,
  "hba": 1,
  "hbd": 0,
  "tpsa": 20.23,
  "rotatable_bonds": 0,
  "heavy_atoms": 2,
  "aromatic_rings": 0,
  "fraction_csp3": 0.0,
  "qed": 0.99
}
```

#### `smiles_to_inchi(smiles: str) -> dict`
Convert SMILES to InChI and InChIKey.

**Parameters:**
- `smiles`: SMILES string

**Returns:**
```json
{
  "smiles": "CCO",
  "inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
  "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
}
```

#### `molecular_similarity(smiles1: str, smiles2: str, metric: str = "tanimoto") -> float`
Calculate molecular similarity between two SMILES strings.

**Parameters:**
- `smiles1`: First SMILES string
- `smiles2`: Second SMILES string
- `metric`: Similarity metric ("tanimoto", "dice", "cosine")

**Returns:**
```json
{
  "similarity": 0.85,
  "metric": "tanimoto",
  "fingerprint_type": "morgan"
}
```

### Structural Biology Tools

#### `superimpose_structures(pdb_id1: str, pdb_id2: str) -> dict`
Superimpose two protein structures and calculate RMSD.

**Parameters:**
- `pdb_id1`: First PDB ID
- `pdb_id2`: Second PDB ID

**Returns:**
```json
{
  "pdb_id1": "1ABC",
  "pdb_id2": "2DEF",
  "rmsd": 1.2,
  "aligned_residues": 200,
  "transformation_matrix": [...],
  "residue_differences": [...]
}
```

#### `analyze_binding_site(pdb_id: str, chain: str, residue: str) -> dict`
Analyze binding site interactions for a specific residue.

**Parameters:**
- `pdb_id`: PDB ID
- `chain`: Chain identifier
- `residue`: Residue number

**Returns:**
```json
{
  "pdb_id": "1ABC",
  "chain": "A",
  "residue": "58",
  "residue_type": "THR",
  "interactions": [
    {
      "type": "hydrogen_bond",
      "partner": "LIG1",
      "distance": 2.8,
      "angle": 165.5
    }
  ],
  "accessibility": 0.45,
  "conformation": "extended"
}
```

## Task System

The server includes a task management system for running DrugDiscoveryBench-style tasks:

### Task Categories

1. **Structural Reasoning** (19 tasks)
   - Protein structure analysis
   - Binding site comparison
   - Conformational changes

2. **Database Screening** (14 tasks)
   - Compound database searching
   - Bioactivity filtering
   - Target identification

3. **Patent Mining** (13 tasks)
   - Patent text analysis
   - Chemical entity extraction
   - Prior art searching

4. **Target ID & Genetics** (12 tasks)
   - Gene-disease association
   - Target validation
   - Genetic evidence analysis

5. **Cheminformatics** (10 tasks)
   - Molecular property calculation
   - Structure manipulation
   - Drug-likeness assessment

6. **SAR / Affinity Ranking** (7 tasks)
   - Structure-activity relationship
   - Binding affinity prediction
   - Compound ranking

7. **Molecular Biology** (7 tasks)
   - Sequence analysis
   - Protein function prediction
   - Pathway analysis

### Running Tasks

```python
from drug_discovery_mcp.tasks import TaskRunner

# Initialize task runner
runner = TaskRunner()

# List available tasks
tasks = runner.list_tasks()

# Run a specific task
result = runner.run_task("lead_optimization_kras_g12c")

# Run with custom parameters
result = runner.run_task(
    "target_identification",
    parameters={"gene": "TP53", "disease": "cancer"}
)
```

## Evaluation System

The server includes a rubric-based evaluation system for grading task outputs:

```python
from drug_discovery_mcp.evaluation import Evaluator

evaluator = Evaluator()

# Evaluate a task result
score = evaluator.evaluate(
    task_id="lead_optimization_kras_g12c",
    answer={"residue": "THR58"},
    rubric=[
        {"criterion": "correct_residue", "weight": 30, "expected": "THR58"},
        {"criterion": "process_check", "weight": 20, "expected": True},
    ]
)

print(f"Score: {score['total']}%")
print(f"Passed: {score['passed']}")
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License

## Acknowledgments

- Inspired by [DrugDiscoveryBench](https://github.com/scaleapi/DrugDiscoveryBench)
- Built on [BIOMNI](https://github.com/snap-stanford/BIOMNI) framework
- Uses [RDKit](https://www.rdkit.org/) for cheminformatics
- MCP protocol from [Model Context Protocol](https://github.com/modelcontextprotocol)

## Contact

For questions or support, please open an issue on GitHub.
