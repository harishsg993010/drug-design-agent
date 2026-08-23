"""
Model Context Protocol server for Drug Discovery

Exposes the drug discovery toolset over MCP using the official Python SDK
(https://github.com/modelcontextprotocol/python-sdk).

Every tool below is a plain synchronous function. The SDK runs sync tools in a
worker thread, so the blocking network and RDKit calls underneath do not stall
the event loop.

Run it over stdio (the default, and what MCP clients such as Claude Desktop
expect)::

    drug-discovery-mcp

or over HTTP::

    drug-discovery-mcp --transport streamable-http --port 8080
"""

import argparse
import logging
from typing import Any, Dict, List, Optional

from mcp.server import Server as MCPServer

from . import __version__
from . import cheminformatics, databases, structural_biology

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
Tools for early-stage drug discovery, in three groups:

- Databases: look up proteins (UniProt), bioactive compounds (ChEMBL, PubChem),
  structures (PDB), target-disease evidence (OpenTargets), pathways (KEGG) and
  genes (NCBI).
- Cheminformatics: molecular descriptors, format conversion, fingerprints and
  similarity, ADMET prediction, drug-likeness filters, and 3D conformers.
- Structural biology: download and parse PDB entries, superimpose structures,
  compute RMSD, and analyse binding sites, contacts and solvent accessibility.

Compounds are addressed by SMILES unless a database accession is called for.
Structures are addressed by four-character PDB ID.
"""


def build_server(name: str = "drug-discovery") -> MCPServer:
    """
    Build the MCP server with every drug discovery tool registered

    Args:
        name: Server name advertised to clients

    Returns:
        A configured MCPServer, ready to run over any transport
    """
    server = MCPServer(
        name=name,
        title="Drug Discovery",
        version=__version__,
        instructions=INSTRUCTIONS,
    )

    db = databases.DatabaseTools()
    chem = cheminformatics.CheminformaticsTools()
    struct = structural_biology.StructuralBiologyTools()

    # ------------------------------------------------------------------
    # Databases
    # ------------------------------------------------------------------

    @server.tool()
    def query_uniprot(accession: str) -> Dict[str, Any]:
        """
        Look up a protein in UniProt.

        Returns the protein name, gene names, organism, sequence, function,
        keywords, GO terms, pathways, disease associations and interactions.

        Args:
            accession: UniProt accession, e.g. "P04637" for human p53
        """
        return db.query_uniprot(accession)

    @server.tool()
    def query_chembl(compound_id: str) -> Dict[str, Any]:
        """
        Look up a compound in ChEMBL.

        Returns the structure (SMILES/InChI), computed physicochemical
        properties and preferred name.

        Args:
            compound_id: ChEMBL ID, e.g. "CHEMBL25" for aspirin
        """
        return db.query_chembl(compound_id)

    @server.tool()
    def query_pdb(pdb_id: str) -> Dict[str, Any]:
        """
        Look up a structure entry in the Protein Data Bank.

        Returns the title, experimental method, resolution, chain identifiers,
        bound ligands, authors and release date.

        Args:
            pdb_id: Four-character PDB ID, e.g. "1TUP"
        """
        return db.query_pdb(pdb_id)

    @server.tool()
    def query_opentargets(target_id: str) -> Dict[str, Any]:
        """
        Look up a target in the OpenTargets Platform.

        Returns the approved symbol and name, biotype, UniProt cross-reference,
        function description and synonyms.

        Args:
            target_id: Ensembl gene ID, e.g. "ENSG00000141510" for TP53
        """
        return db.query_opentargets(target_id)

    @server.tool()
    def query_kegg(pathway_id: str) -> Dict[str, Any]:
        """
        Look up a pathway in KEGG.

        Returns the pathway name, description, class, and its member genes,
        compounds, drugs and related pathways.

        Args:
            pathway_id: KEGG pathway ID, e.g. "hsa04110" for the human cell cycle
        """
        return db.query_kegg(pathway_id)

    @server.tool()
    def query_pubchem(compound_id: str, namespace: str = "cid") -> Dict[str, Any]:
        """
        Look up a compound in PubChem.

        Returns the structure, computed properties and synonyms.

        Args:
            compound_id: The identifier to look up, e.g. "2244" or "aspirin"
            namespace: How to interpret compound_id -- "cid", "name", "smiles"
                or "inchikey"
        """
        return db.query_pubchem(compound_id, namespace=namespace)

    @server.tool()
    def query_ncbi(gene_id: str) -> Dict[str, Any]:
        """
        Look up a gene in NCBI Gene.

        Returns the symbol, description, organism, chromosomal location,
        aliases and RefSeq summary.

        Args:
            gene_id: NCBI Gene ID, e.g. "7157" for TP53
        """
        return db.query_ncbi(gene_id)

    @server.tool()
    def search_compounds(
        query: str,
        databases: Optional[List[str]] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Search for compounds by name across chemical databases.

        Args:
            query: Compound name or partial name, e.g. "aspirin"
            databases: Which databases to search; defaults to ChEMBL and PubChem
            limit: Maximum results per database
        """
        return db.search_compounds(
            query,
            databases=databases or ["chembl", "pubchem"],
            limit=limit,
        )

    @server.tool()
    def search_proteins(
        query: str,
        databases: Optional[List[str]] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Search for proteins by name or gene symbol.

        Args:
            query: Protein or gene name, e.g. "p53"
            databases: Which databases to search; defaults to UniProt
            limit: Maximum results per database
        """
        return db.search_proteins(
            query,
            databases=databases or ["uniprot"],
            limit=limit,
        )

    @server.tool()
    def search_patents(query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search chemical patents.

        Not implemented yet -- returns an empty result set with a message.

        Args:
            query: Search terms
            limit: Maximum number of results
        """
        return db.search_patents(query, limit=limit)

    # ------------------------------------------------------------------
    # Cheminformatics
    # ------------------------------------------------------------------

    @server.tool()
    def calculate_descriptors(
        smiles: str,
        descriptors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute molecular descriptors for a structure.

        Covers molecular weight, logP, TPSA, hydrogen bond donors and
        acceptors, rotatable bonds, ring counts and atom composition.

        Args:
            smiles: Molecule as SMILES, e.g. "CC(=O)Oc1ccccc1C(=O)O"
            descriptors: Restrict the result to these descriptor names;
                omit for all of them
        """
        return chem.calculate_descriptors(smiles, descriptors=descriptors)

    @server.tool()
    def smiles_to_inchi(smiles: str) -> Dict[str, Any]:
        """
        Convert a SMILES string to InChI.

        Args:
            smiles: Molecule as SMILES
        """
        return chem.smiles_to_inchi(smiles)

    @server.tool()
    def inchi_to_smiles(inchi: str) -> Dict[str, Any]:
        """
        Convert an InChI string to SMILES.

        Args:
            inchi: Molecule as InChI
        """
        return chem.inchi_to_smiles(inchi)

    @server.tool()
    def molecular_similarity(
        smiles1: str,
        smiles2: str,
        metric: str = "tanimoto",
        fingerprint_type: str = "morgan",
        radius: int = 2,
        bit_length: int = 2048,
    ) -> Dict[str, Any]:
        """
        Measure structural similarity between two molecules.

        Args:
            smiles1: First molecule as SMILES
            smiles2: Second molecule as SMILES
            metric: Similarity coefficient -- "tanimoto", "dice", "cosine"
            fingerprint_type: "morgan", "rdkit", "atom_pair",
                "topological_torsion" or "maccs"
            radius: Neighbourhood radius for Morgan fingerprints
            bit_length: Fingerprint length in bits
        """
        return chem.molecular_similarity(
            smiles1,
            smiles2,
            metric=metric,
            fingerprint_type=fingerprint_type,
            radius=radius,
            bit_length=bit_length,
        )

    @server.tool()
    def calculate_fingerprint(
        smiles: str,
        fingerprint_type: str = "morgan",
        radius: int = 2,
        bit_length: int = 2048,
    ) -> Dict[str, Any]:
        """
        Compute a structural fingerprint for a molecule.

        Args:
            smiles: Molecule as SMILES
            fingerprint_type: "morgan", "rdkit", "atom_pair",
                "topological_torsion" or "maccs"
            radius: Neighbourhood radius for Morgan fingerprints
            bit_length: Fingerprint length in bits (ignored for MACCS)
        """
        return chem.calculate_fingerprint(
            smiles,
            fingerprint_type=fingerprint_type,
            radius=radius,
            bit_length=bit_length,
        )

    @server.tool()
    def predict_admet(smiles: str) -> Dict[str, Any]:
        """
        Predict ADMET properties for a molecule.

        Estimates absorption, distribution, metabolism, excretion and toxicity
        from computed physicochemical properties. These are rule-based
        estimates, not experimental measurements.

        Args:
            smiles: Molecule as SMILES
        """
        return chem.predict_admet(smiles)

    @server.tool()
    def check_drug_likeness(smiles: str) -> Dict[str, Any]:
        """
        Score a molecule against drug-likeness filters.

        Applies Lipinski's Rule of Five, plus the Ghose, Veber, Egan and Muegge
        filters, reporting each violation.

        Args:
            smiles: Molecule as SMILES
        """
        return chem.check_drug_likeness(smiles)

    @server.tool()
    def generate_conformers(smiles: str, num_conformers: int = 10) -> Dict[str, Any]:
        """
        Generate 3D conformers for a molecule.

        Args:
            smiles: Molecule as SMILES
            num_conformers: How many conformers to embed
        """
        return chem.generate_conformers(smiles, num_conformers=num_conformers)

    @server.tool()
    def optimize_geometry(smiles: str) -> Dict[str, Any]:
        """
        Embed a molecule in 3D and minimise its geometry with MMFF.

        Args:
            smiles: Molecule as SMILES
        """
        return chem.optimize_geometry(smiles)

    @server.tool()
    def calculate_charge(smiles: str, ph: float = 7.0) -> Dict[str, Any]:
        """
        Compute formal charge and per-atom charges for a molecule.

        Args:
            smiles: Molecule as SMILES
            ph: pH at which to consider protonation state
        """
        return chem.calculate_charge(smiles, ph=ph)

    # ------------------------------------------------------------------
    # Structural biology
    # ------------------------------------------------------------------

    @server.tool()
    def download_pdb(pdb_id: str, format: str = "pdb") -> str:
        """
        Download a structure file from RCSB and cache it locally.

        Args:
            pdb_id: Four-character PDB ID
            format: File format -- "pdb", "mmcif", "xml" or "bcif"

        Returns:
            Path to the cached file
        """
        return struct.download_pdb(pdb_id, format=format)

    @server.tool()
    def parse_pdb(pdb_id: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse a structure into chains, residues, atoms and ligands.

        Args:
            pdb_id: Four-character PDB ID
            file_path: Parse this local file instead of downloading the entry
        """
        return struct.parse_pdb(pdb_id, file_path=file_path)

    @server.tool()
    def superimpose_structures(
        pdb_id1: str,
        pdb_id2: str,
        chain_id1: Optional[str] = None,
        chain_id2: Optional[str] = None,
        atom_selection: str = "ca",
    ) -> Dict[str, Any]:
        """
        Superimpose two structures and report the fit.

        Atoms are paired by residue number, so structures of different lengths
        align on the residues they share.

        Args:
            pdb_id1: First PDB ID
            pdb_id2: Second PDB ID
            chain_id1: Chain to use from the first structure; omit for the
                first chain that has atoms matching the selection
            chain_id2: Chain to use from the second structure
            atom_selection: Which atoms to fit on -- "ca", "backbone" or "all"
        """
        return struct.superimpose_structures(
            pdb_id1,
            pdb_id2,
            chain_id1=chain_id1,
            chain_id2=chain_id2,
            atom_selection=atom_selection,
        )

    @server.tool()
    def calculate_rmsd(
        pdb_id1: str,
        pdb_id2: str,
        chain_id1: Optional[str] = None,
        chain_id2: Optional[str] = None,
        atom_selection: str = "ca",
    ) -> Dict[str, Any]:
        """
        Compute the RMSD between two structures after optimal superposition.

        Args:
            pdb_id1: First PDB ID
            pdb_id2: Second PDB ID
            chain_id1: Chain to use from the first structure
            chain_id2: Chain to use from the second structure
            atom_selection: Which atoms to fit on -- "ca", "backbone" or "all"
        """
        return struct.calculate_rmsd(
            pdb_id1,
            pdb_id2,
            chain_id1=chain_id1,
            chain_id2=chain_id2,
            atom_selection=atom_selection,
        )

    @server.tool()
    def analyze_binding_site(
        pdb_id: str,
        chain_id: Optional[str] = None,
        residue_number: Optional[int] = None,
        ligand_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Describe a binding site: its residues and their contacts.

        Args:
            pdb_id: Four-character PDB ID
            chain_id: Restrict to this chain
            residue_number: Centre the site on this residue
            ligand_id: Centre the site on this ligand, e.g. "OHT"
        """
        return struct.analyze_binding_site(
            pdb_id,
            chain_id=chain_id,
            residue_number=residue_number,
            ligand_id=ligand_id,
        )

    @server.tool()
    def find_interactions(
        pdb_id: str,
        chain_id: Optional[str] = None,
        residue_number: Optional[int] = None,
        ligand_id: Optional[str] = None,
        distance_threshold: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Find non-covalent contacts in a structure.

        Args:
            pdb_id: Four-character PDB ID
            chain_id: Restrict to this chain
            residue_number: Restrict to contacts of this residue
            ligand_id: Restrict to contacts of this ligand
            distance_threshold: Maximum contact distance in Angstroms
        """
        return struct.find_interactions(
            pdb_id,
            chain_id=chain_id,
            residue_number=residue_number,
            ligand_id=ligand_id,
            distance_threshold=distance_threshold,
        )

    @server.tool()
    def analyze_conformation(
        pdb_id: str,
        chain_id: Optional[str] = None,
        residue_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Report backbone conformation, including dihedral angles.

        Args:
            pdb_id: Four-character PDB ID
            chain_id: Restrict to this chain
            residue_number: Restrict to this residue
        """
        return struct.analyze_conformation(
            pdb_id,
            chain_id=chain_id,
            residue_number=residue_number,
        )

    @server.tool()
    def compare_structures(
        pdb_id1: str,
        pdb_id2: str,
        chain_id1: Optional[str] = None,
        chain_id2: Optional[str] = None,
        atom_selection: str = "ca",
    ) -> Dict[str, Any]:
        """
        Compare two structures and summarise where they differ.

        Args:
            pdb_id1: First PDB ID
            pdb_id2: Second PDB ID
            chain_id1: Chain to use from the first structure
            chain_id2: Chain to use from the second structure
            atom_selection: Which atoms to compare on -- "ca", "backbone" or "all"
        """
        return struct.compare_structures(
            pdb_id1,
            pdb_id2,
            chain_id1=chain_id1,
            chain_id2=chain_id2,
            atom_selection=atom_selection,
        )

    @server.tool()
    def extract_ligand(pdb_id: str, ligand_name: str) -> Dict[str, Any]:
        """
        Pull a bound ligand out of a structure.

        Args:
            pdb_id: Four-character PDB ID
            ligand_name: Ligand chemical component ID, e.g. "OHT"
        """
        return struct.extract_ligand(pdb_id, ligand_name)

    @server.tool()
    def analyze_solvent_accessibility(pdb_id: str) -> Dict[str, Any]:
        """
        Estimate per-residue solvent accessibility.

        Args:
            pdb_id: Four-character PDB ID
        """
        return struct.analyze_solvent_accessibility(pdb_id)

    return server


def main() -> None:
    """Entry point for the MCP server"""
    parser = argparse.ArgumentParser(
        description="Drug Discovery MCP server",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport to serve on (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address for the HTTP transports",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP transports",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    args = parser.parse_args()

    # stdio carries the protocol itself, so logs must go to stderr and never
    # to stdout
    import sys
    logging.basicConfig(level=args.log_level, stream=sys.stderr)

    server = build_server()

    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
