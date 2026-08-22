"""
Client Module for Drug Discovery MCP Server

Provides a client interface for interacting with the Drug Discovery MCP Server.
This can be used both locally and remotely.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from .config import settings, get_config

logger = logging.getLogger(__name__)


class DrugDiscoveryClient:
    """
    Client for interacting with Drug Discovery MCP Server
    
    This client provides:
    - Local execution of drug discovery tools
    - Remote HTTP API access to MCP server
    - Task management
    - Evaluation capabilities
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        """
        Initialize the client
        
        Args:
            base_url: Base URL for the MCP server (e.g., "http://localhost:8080")
            api_key: API key for authentication (if required)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or f"http://{settings.server.host}:{settings.server.port}"
        self.api_key = api_key
        self.timeout = timeout
        self.session = httpx.AsyncClient(timeout=timeout)
        
        # Initialize local tools
        self._init_local_tools()
        
        logger.info(f"DrugDiscoveryClient initialized with base URL: {self.base_url}")
    
    def _init_local_tools(self):
        """Initialize local tool instances"""
        try:
            from .databases import DatabaseTools
            self.databases = DatabaseTools()
            self.databases.initialize()
        except Exception as e:
            logger.warning(f"Failed to initialize database tools: {e}")
            self.databases = None
        
        try:
            from .cheminformatics import CheminformaticsTools
            self.cheminformatics = CheminformaticsTools()
            self.cheminformatics.initialize()
        except Exception as e:
            logger.warning(f"Failed to initialize cheminformatics tools: {e}")
            self.cheminformatics = None
        
        try:
            from .structural_biology import StructuralBiologyTools
            self.structural_biology = StructuralBiologyTools()
            self.structural_biology.initialize()
        except Exception as e:
            logger.warning(f"Failed to initialize structural biology tools: {e}")
            self.structural_biology = None
        
        try:
            from .tasks import TaskRunner
            self.tasks = TaskRunner()
        except Exception as e:
            logger.warning(f"Failed to initialize task runner: {e}")
            self.tasks = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        import asyncio
        if asyncio.get_event_loop().is_running():
            asyncio.run(self.close())
        else:
            self.close()
    
    async def close(self):
        """Close the client"""
        await self.session.aclose()
        
        if self.databases:
            await self.databases.close()
        if self.cheminformatics:
            await self.cheminformatics.close()
        if self.structural_biology:
            await self.structural_biology.close()
        
        logger.info("DrugDiscoveryClient closed")
    
    # Local execution methods
    
    def query_uniprot(self, accession: str, **kwargs) -> Dict[str, Any]:
        """Query UniProt database (local execution)"""
        if self.databases:
            return self.databases.query_uniprot(accession, **kwargs)
        return {"error": "Database tools not available"}
    
    def query_chembl(self, compound_id: str, **kwargs) -> Dict[str, Any]:
        """Query ChEMBL database (local execution)"""
        if self.databases:
            return self.databases.query_chembl(compound_id, **kwargs)
        return {"error": "Database tools not available"}
    
    def query_pdb(self, pdb_id: str, **kwargs) -> Dict[str, Any]:
        """Query PDB database (local execution)"""
        if self.databases:
            return self.databases.query_pdb(pdb_id, **kwargs)
        return {"error": "Database tools not available"}
    
    def calculate_descriptors(self, smiles: str, **kwargs) -> Dict[str, Any]:
        """Calculate molecular descriptors (local execution)"""
        if self.cheminformatics:
            return self.cheminformatics.calculate_descriptors(smiles, **kwargs)
        return {"error": "Cheminformatics tools not available"}
    
    def molecular_similarity(self, smiles1: str, smiles2: str, **kwargs) -> Dict[str, Any]:
        """Calculate molecular similarity (local execution)"""
        if self.cheminformatics:
            return self.cheminformatics.molecular_similarity(smiles1, smiles2, **kwargs)
        return {"error": "Cheminformatics tools not available"}
    
    def superimpose_structures(self, pdb_id1: str, pdb_id2: str, **kwargs) -> Dict[str, Any]:
        """Superimpose two protein structures (local execution)"""
        if self.structural_biology:
            return self.structural_biology.superimpose_structures(pdb_id1, pdb_id2, **kwargs)
        return {"error": "Structural biology tools not available"}
    
    def run_task(self, task_id: str, **kwargs) -> Dict[str, Any]:
        """Run a task (local execution)"""
        if self.tasks:
            result = self.tasks.run_task(task_id, **kwargs)
            return result.to_dict()
        return {"error": "Task runner not available"}
    
    # Remote API methods
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        json: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make an HTTP request to the MCP server"""
        url = f"{self.base_url}{endpoint}"
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            if method.upper() == "GET":
                response = await self.session.get(url, params=params, headers=headers)
            elif method.upper() == "POST":
                response = await self.session.post(url, json=json or data, headers=headers)
            elif method.upper() == "PUT":
                response = await self.session.put(url, json=json or data, headers=headers)
            elif method.upper() == "DELETE":
                response = await self.session.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Check for errors
            if response.status_code >= 400:
                error_data = {"error": response.text}
                try:
                    error_data = response.json()
                except:
                    pass
                raise Exception(f"API request failed: {response.status_code} - {error_data}")
            
            # Try to parse JSON response
            try:
                return response.json()
            except:
                return {"response": response.text}
                
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        return await self._make_request("GET", "/health")
    
    async def list_tools(self) -> Dict[str, Any]:
        """List all available tools"""
        return await self._make_request("GET", "/tools")
    
    async def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get information about a specific tool"""
        return await self._make_request("GET", f"/tools/{tool_name}")
    
    async def call_tool(self, tool_name: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Call a specific tool"""
        return await self._make_request("POST", f"/call/{tool_name}", json={"params": params or {}})
    
    async def batch_call(self, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute multiple tool calls in batch"""
        return await self._make_request("POST", "/batch", json={"calls": calls})
    
    async def list_categories(self) -> Dict[str, Any]:
        """List all tool categories"""
        return await self._make_request("GET", "/categories")
    
    # Convenience methods for remote execution
    
    async def remote_query_uniprot(self, accession: str) -> Dict[str, Any]:
        """Query UniProt via remote API"""
        return await self.call_tool("query_uniprot", {"accession": accession})
    
    async def remote_query_chembl(self, compound_id: str) -> Dict[str, Any]:
        """Query ChEMBL via remote API"""
        return await self.call_tool("query_chembl", {"compound_id": compound_id})
    
    async def remote_query_pdb(self, pdb_id: str) -> Dict[str, Any]:
        """Query PDB via remote API"""
        return await self.call_tool("query_pdb", {"pdb_id": pdb_id})
    
    async def remote_calculate_descriptors(self, smiles: str) -> Dict[str, Any]:
        """Calculate descriptors via remote API"""
        return await self.call_tool("calculate_descriptors", {"smiles": smiles})
    
    async def remote_molecular_similarity(self, smiles1: str, smiles2: str) -> Dict[str, Any]:
        """Calculate similarity via remote API"""
        return await self.call_tool("molecular_similarity", {"smiles1": smiles1, "smiles2": smiles2})
    
    # Task management via remote API
    
    async def remote_list_tasks(self) -> Dict[str, Any]:
        """List tasks via remote API"""
        return await self._make_request("GET", "/tasks")
    
    async def remote_run_task(self, task_id: str, **kwargs) -> Dict[str, Any]:
        """Run a task via remote API"""
        return await self._make_request("POST", f"/tasks/{task_id}/run", json=kwargs)
    
    async def remote_get_task_result(self, task_id: str) -> Dict[str, Any]:
        """Get task result via remote API"""
        return await self._make_request("GET", f"/tasks/{task_id}/result")
    
    # Hybrid methods (try local first, then remote)
    
    async def query_uniprot_hybrid(self, accession: str, **kwargs) -> Dict[str, Any]:
        """Query UniProt with local fallback to remote"""
        try:
            return self.query_uniprot(accession, **kwargs)
        except Exception as e:
            logger.warning(f"Local UniProt query failed, trying remote: {e}")
            return await self.remote_query_uniprot(accession)
    
    async def query_chembl_hybrid(self, compound_id: str, **kwargs) -> Dict[str, Any]:
        """Query ChEMBL with local fallback to remote"""
        try:
            return self.query_chembl(compound_id, **kwargs)
        except Exception as e:
            logger.warning(f"Local ChEMBL query failed, trying remote: {e}")
            return await self.remote_query_chembl(compound_id)
    
    async def query_pdb_hybrid(self, pdb_id: str, **kwargs) -> Dict[str, Any]:
        """Query PDB with local fallback to remote"""
        try:
            return self.query_pdb(pdb_id, **kwargs)
        except Exception as e:
            logger.warning(f"Local PDB query failed, trying remote: {e}")
            return await self.remote_query_pdb(pdb_id)


# Singleton instance
_client = DrugDiscoveryClient()


def get_client() -> DrugDiscoveryClient:
    """Get the global client instance"""
    return _client


def set_client(client: DrugDiscoveryClient) -> None:
    """Set the global client instance"""
    global _client
    _client = client
