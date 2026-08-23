"""
Database Tools Module

Provides a unified interface for all database operations.
This is the main entry point for database queries from the MCP server.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from .uniprot import UniProtClient, query_uniprot, search_uniprot
from .chembl import ChEMBLClient, query_chembl, search_chembl_compounds
from .pdb import PDBClient, query_pdb, download_pdb, parse_pdb
from .opentargets import OpenTargetsClient, query_opentargets, search_opentargets
from .kegg import KEGGClient, query_kegg, search_kegg
from .pubchem import PubChemClient, query_pubchem, search_pubchem
from .ncbi import NCBIClient, query_ncbi, search_ncbi
from .base import DatabaseError

logger = logging.getLogger(__name__)


class DatabaseTools:
    """
    Unified interface for all database operations
    
    This class provides a single entry point for all database queries,
    making it easy to access different databases through a consistent interface.
    """
    
    def __init__(self):
        """Initialize all database clients"""
        self.uniprot = UniProtClient()
        self.chembl = ChEMBLClient()
        self.pdb = PDBClient()
        self.opentargets = OpenTargetsClient()
        self.kegg = KEGGClient()
        self.pubchem = PubChemClient()
        self.ncbi = NCBIClient()
    
    def initialize(self):
        """Initialize all database clients"""
        logger.info("Initializing database clients")
        # All clients are initialized in __init__
    
    async def close(self):
        """Close all database clients"""
        logger.info("Closing database clients")
        self.uniprot.close()
        self.chembl.close()
        self.pdb.close()
        self.opentargets.close()
        self.kegg.close()
        self.pubchem.close()
        self.ncbi.close()
    
    # UniProt methods
    def query_uniprot(self, accession: str, **kwargs) -> Dict[str, Any]:
        """Query UniProt database"""
        return query_uniprot(accession, **kwargs)
    
    def search_uniprot(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search UniProt database"""
        return search_uniprot(query, **kwargs)
    
    # ChEMBL methods
    def query_chembl(self, compound_id: str, **kwargs) -> Dict[str, Any]:
        """Query ChEMBL database for a compound"""
        return query_chembl(compound_id, **kwargs)
    
    def search_chembl(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search ChEMBL compounds"""
        return search_chembl_compounds(query, **kwargs)
    
    # PDB methods
    def query_pdb(self, pdb_id: str, **kwargs) -> Dict[str, Any]:
        """Query PDB database"""
        return query_pdb(pdb_id, **kwargs)
    
    def download_pdb(self, pdb_id: str, format: str = "pdb") -> str:
        """Download PDB file"""
        return download_pdb(pdb_id, format)
    
    def parse_pdb(self, pdb_id: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Parse PDB file"""
        return parse_pdb(pdb_id, file_path)
    
    # OpenTargets methods
    def query_opentargets(self, target_id: str, **kwargs) -> Dict[str, Any]:
        """Query OpenTargets database for a target"""
        return query_opentargets(target_id, **kwargs)
    
    def search_opentargets(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search OpenTargets targets"""
        return search_opentargets(query, **kwargs)
    
    # KEGG methods
    def query_kegg(self, pathway_id: str, **kwargs) -> Dict[str, Any]:
        """Query KEGG database for pathway information"""
        return query_kegg(pathway_id, **kwargs)

    def search_kegg(self, database: str, query: str, **kwargs) -> Dict[str, Any]:
        """Search a KEGG database by keyword"""
        return search_kegg(database, query, **kwargs)

    # PubChem methods
    def query_pubchem(self, compound_id: str, **kwargs) -> Dict[str, Any]:
        """Query PubChem database for compound information"""
        return query_pubchem(compound_id, **kwargs)

    def search_pubchem(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search PubChem compounds by name"""
        return search_pubchem(query, **kwargs)

    # NCBI methods
    def query_ncbi(self, gene_id: str, **kwargs) -> Dict[str, Any]:
        """Query NCBI database for gene information"""
        return query_ncbi(gene_id, **kwargs)

    def search_ncbi(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search NCBI Gene"""
        return search_ncbi(query, **kwargs)
    
    # Generic search method
    def search(
        self,
        database: str,
        query: str,
        limit: int = 10,
        offset: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generic search across databases
        
        Args:
            database: Database name ("uniprot", "chembl", "pdb", etc.)
            query: Search query
            limit: Maximum number of results
            offset: Pagination offset
            **kwargs: Additional database-specific parameters
            
        Returns:
            Dictionary with search results
        """
        database = database.lower()
        
        if database == "uniprot":
            return self.search_uniprot(query, limit=limit, offset=offset, **kwargs)
        elif database == "chembl":
            return self.search_chembl(query, limit=limit, offset=offset, **kwargs)
        elif database == "pdb":
            return self.pdb.search(query, limit=limit, offset=offset, **kwargs)
        elif database == "opentargets":
            return self.opentargets.search_targets(query, limit=limit, offset=offset, **kwargs)
        elif database == "pubchem":
            return self.pubchem.search_compounds(query, limit=limit, **kwargs)
        elif database == "ncbi":
            return self.ncbi.search_genes(query, limit=limit, **kwargs)
        else:
            raise ValueError(f"Unknown database: {database}")
    
    # Generic query method
    def query(
        self,
        database: str,
        identifier: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generic query across databases
        
        Args:
            database: Database name ("uniprot", "chembl", "pdb", etc.)
            identifier: Database identifier (accession, compound ID, PDB ID, etc.)
            **kwargs: Additional database-specific parameters
            
        Returns:
            Dictionary with query results
        """
        database = database.lower()
        
        if database == "uniprot":
            return self.query_uniprot(identifier, **kwargs)
        elif database == "chembl":
            return self.query_chembl(identifier, **kwargs)
        elif database == "pdb":
            return self.query_pdb(identifier, **kwargs)
        elif database == "opentargets":
            return self.query_opentargets(identifier, **kwargs)
        elif database == "kegg":
            return self.query_kegg(identifier, **kwargs)
        elif database == "pubchem":
            return self.query_pubchem(identifier, **kwargs)
        elif database == "ncbi":
            return self.query_ncbi(identifier, **kwargs)
        else:
            raise ValueError(f"Unknown database: {database}")
    
    def search_compounds(
        self,
        query: str,
        databases: List[str] = ["chembl", "pubchem"],
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for compounds across multiple databases
        
        Args:
            query: Search query (SMILES, compound name, etc.)
            databases: List of databases to search
            limit: Maximum number of results per database
            
        Returns:
            Dictionary with combined search results
        """
        results = {}
        
        for db in databases:
            try:
                if db.lower() == "chembl":
                    results["chembl"] = self.search_chembl(query, limit=limit)
                elif db.lower() == "pubchem":
                    results["pubchem"] = self.pubchem.search_compounds(query, limit=limit)
            except Exception as e:
                logger.error(f"Search failed for {db}: {e}")
                results[db] = {"error": str(e)}
        
        return {
            "query": query,
            "databases": databases,
            "results": results
        }
    
    def search_proteins(
        self,
        query: str,
        databases: List[str] = ["uniprot"],
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for proteins across multiple databases
        
        Args:
            query: Search query (protein name, gene name, etc.)
            databases: List of databases to search
            limit: Maximum number of results per database
            
        Returns:
            Dictionary with combined search results
        """
        results = {}
        
        for db in databases:
            try:
                if db.lower() == "uniprot":
                    results["uniprot"] = self.search_uniprot(query, limit=limit)
                elif db.lower() == "pdb":
                    results["pdb"] = self.pdb.search(query, limit=limit)
            except Exception as e:
                logger.error(f"Search failed for {db}: {e}")
                results[db] = {"error": str(e)}
        
        return {
            "query": query,
            "databases": databases,
            "results": results
        }
    
    def search_patents(
        self,
        query: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search patents (placeholder - will be implemented)
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        # This will be implemented when patent mining tools are added
        return {
            "query": query,
            "results": [],
            "message": "Patent search not yet implemented"
        }
    
    def get_protein_info(self, accession: str) -> Dict[str, Any]:
        """
        Get comprehensive protein information from multiple databases
        
        Args:
            accession: UniProt accession number
            
        Returns:
            Dictionary with combined protein information
        """
        try:
            uniprot_data = self.query_uniprot(accession)

            # Get related PDB structures. This needs a structured lookup on the
            # entity's UniProt cross-reference -- a full-text search for the
            # accession does not match.
            pdb_results = self.pdb.search_by_uniprot(accession, limit=5)
            
            return {
                "uniprot": uniprot_data,
                "pdb_structures": pdb_results.get("results", []),
                "source": "uniprot"
            }
        except Exception as e:
            logger.error(f"Failed to get protein info for {accession}: {e}")
            return {"error": str(e), "accession": accession}
    
    def get_compound_info(self, compound_id: str) -> Dict[str, Any]:
        """
        Get comprehensive compound information from multiple databases
        
        Args:
            compound_id: ChEMBL compound ID or SMILES
            
        Returns:
            Dictionary with combined compound information
        """
        try:
            # Try ChEMBL first
            chembl_data = self.query_chembl(compound_id)
            
            # Get bioactivities
            bioactivities = self.chembl.get_bioactivities(compound_id=compound_id, limit=10)
            
            return {
                "chembl": chembl_data,
                "bioactivities": [ba.__dict__ for ba in bioactivities],
                "source": "chembl"
            }
        except Exception as e:
            logger.error(f"Failed to get compound info for {compound_id}: {e}")
            return {"error": str(e), "compound_id": compound_id}
    
    def get_structure_info(self, pdb_id: str) -> Dict[str, Any]:
        """
        Get comprehensive structure information
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            Dictionary with structure information
        """
        try:
            pdb_data = self.query_pdb(pdb_id)
            chains = self.pdb.get_chains(pdb_id)
            ligands = self.pdb.get_ligands(pdb_id)
            
            return {
                "pdb": pdb_data,
                "chains": [chain.__dict__ for chain in chains],
                "ligands": [ligand.__dict__ for ligand in ligands],
                "source": "pdb"
            }
        except Exception as e:
            logger.error(f"Failed to get structure info for {pdb_id}: {e}")
            return {"error": str(e), "pdb_id": pdb_id}
