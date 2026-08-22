"""
MCP Server implementation for Drug Discovery

This module provides the main MCP server that exposes all drug discovery tools
via the Model Context Protocol.
"""

import asyncio
import json
import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import aiohttp
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings, get_config
from . import databases, cheminformatics, structural_biology

# Configure logging
logger = logging.getLogger(__name__)


class MCPRequest(BaseModel):
    """Base model for MCP requests"""
    method: str = Field(..., description="The method/tool to call")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the method")
    id: Optional[str] = Field(default=None, description="Request ID")


class MCPResponse(BaseModel):
    """Base model for MCP responses"""
    id: Optional[str] = Field(default=None, description="Request ID")
    result: Any = Field(default=None, description="Result of the operation")
    error: Optional[str] = Field(default=None, description="Error message if any")
    success: bool = Field(default=True, description="Whether the operation succeeded")


class ToolInfo(BaseModel):
    """Information about a tool/function"""
    name: str
    description: str
    parameters: Dict[str, Any]
    category: str
    tags: List[str] = Field(default_factory=list)


class DrugDiscoveryMCPServer:
    """
    Main MCP Server for Drug Discovery
    
    This server provides access to all drug discovery tools via MCP protocol.
    It supports both HTTP/REST and MCP-native transport protocols.
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        config: Optional[Any] = None,
    ):
        """
        Initialize the MCP server
        
        Args:
            host: Server host (defaults to config)
            port: Server port (defaults to config)
            config: Custom configuration
        """
        self.config = config or get_config()
        self.host = host or self.config.server.host
        self.port = port or self.config.server.port
        
        self.app = FastAPI(
            title="Drug Discovery MCP Server",
            description="MCP server for early-stage drug discovery tasks",
            version="0.1.0",
        )
        
        # Initialize all tool modules
        self._initialize_tools()
        
        # Register API routes
        self._register_routes()
        
        # Tool registry
        self.tools: Dict[str, Any] = {}
        self.tool_info: Dict[str, ToolInfo] = {}
        
        # Register all tools
        self._register_all_tools()
        
        logger.info(f"MCP Server initialized on {self.host}:{self.port}")
        logger.info(f"Registered {len(self.tools)} tools")
    
    def _initialize_tools(self):
        """Initialize all tool modules"""
        # Database tools
        self.db_tools = databases.DatabaseTools()
        
        # Cheminformatics tools
        self.chem_tools = cheminformatics.CheminformaticsTools()
        
        # Structural biology tools
        self.struct_tools = structural_biology.StructuralBiologyTools()
        
        # Additional tool modules will be added here
        # self.patent_tools = patent_mining.PatentMiningTools()
        # self.target_tools = target_identification.TargetIdentificationTools()
        # self.hit_tools = hit_identification.HitIdentificationTools()
        # self.sar_tools = sar_analysis.SARAnalysisTools()
    
    def _register_routes(self):
        """Register API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "name": "Drug Discovery MCP Server",
                "version": "0.1.0",
                "description": "MCP server for early-stage drug discovery",
                "tools_count": len(self.tools),
                "docs": "/docs",
                "health": "/health"
            }
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "tools_loaded": len(self.tools)}
        
        @self.app.get("/tools")
        async def list_tools():
            """List all available tools"""
            return {
                "tools": list(self.tool_info.values()),
                "count": len(self.tools)
            }
        
        @self.app.get("/tools/{tool_name}")
        async def get_tool_info(tool_name: str):
            """Get information about a specific tool"""
            if tool_name not in self.tool_info:
                raise HTTPException(status_code=404, detail="Tool not found")
            return self.tool_info[tool_name]
        
        @self.app.post("/mcp")
        async def handle_mcp_request(request: MCPRequest):
            """Handle MCP protocol requests"""
            return await self._handle_request(request)
        
        @self.app.post("/call/{tool_name}")
        async def call_tool(tool_name: str, request: Request):
            """Call a specific tool by name"""
            try:
                params = await request.json()
                return await self._call_tool(tool_name, params.get("params", {}))
            except Exception as e:
                logger.error(f"Error calling tool {tool_name}: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/categories")
        async def list_categories():
            """List all tool categories"""
            categories = {}
            for tool_name, tool in self.tool_info.items():
                category = tool.category
                if category not in categories:
                    categories[category] = []
                categories[category].append(tool_name)
            return categories
        
        @self.app.post("/batch")
        async def batch_call(request: Request):
            """Execute multiple tool calls in batch"""
            try:
                data = await request.json()
                calls = data.get("calls", [])
                results = []
                
                for call in calls:
                    tool_name = call.get("tool")
                    params = call.get("params", {})
                    result = await self._call_tool(tool_name, params)
                    results.append(result)
                
                return {"results": results, "count": len(results)}
            except Exception as e:
                logger.error(f"Batch call error: {e}")
                raise HTTPException(status_code=400, detail=str(e))
    
    def _register_all_tools(self):
        """Register all available tools"""
        
        # Database tools
        db_tools = [
            ("query_uniprot", self.db_tools.query_uniprot, "Database", ["uniprot", "protein"]),
            ("query_chembl", self.db_tools.query_chembl, "Database", ["chembl", "bioactivity"]),
            ("query_pdb", self.db_tools.query_pdb, "Database", ["pdb", "structure"]),
            ("query_opentargets", self.db_tools.query_opentargets, "Database", ["target", "disease"]),
            ("query_kegg", self.db_tools.query_kegg, "Database", ["kegg", "pathway"]),
            ("query_pubchem", self.db_tools.query_pubchem, "Database", ["pubchem", "compound"]),
            ("query_ncbi", self.db_tools.query_ncbi, "Database", ["ncbi", "genomics"]),
            ("search_compounds", self.db_tools.search_compounds, "Database", ["search", "compound"]),
            ("search_proteins", self.db_tools.search_proteins, "Database", ["search", "protein"]),
            ("search_patents", self.db_tools.search_patents, "Database", ["search", "patent"]),
        ]
        
        for name, func, category, tags in db_tools:
            self._register_tool(name, func, category, tags)
        
        # Cheminformatics tools
        chem_tools = [
            ("calculate_descriptors", self.chem_tools.calculate_descriptors, "Cheminformatics", ["descriptors", "molecular"]),
            ("smiles_to_inchi", self.chem_tools.smiles_to_inchi, "Cheminformatics", ["conversion", "inchi"]),
            ("inchi_to_smiles", self.chem_tools.inchi_to_smiles, "Cheminformatics", ["conversion", "smiles"]),
            ("molecular_similarity", self.chem_tools.molecular_similarity, "Cheminformatics", ["similarity", "tanimoto"]),
            ("calculate_fingerprint", self.chem_tools.calculate_fingerprint, "Cheminformatics", ["fingerprint", "morgan"]),
            ("predict_admet", self.chem_tools.predict_admet, "Cheminformatics", ["admet", "drug-likeness"]),
            ("check_drug_likeness", self.chem_tools.check_drug_likeness, "Cheminformatics", ["drug-likeness", "ro5"]),
            ("generate_conformers", self.chem_tools.generate_conformers, "Cheminformatics", ["conformers", "3d"]),
            ("optimize_geometry", self.chem_tools.optimize_geometry, "Cheminformatics", ["optimization", "geometry"]),
            ("calculate_charge", self.chem_tools.calculate_charge, "Cheminformatics", ["charge", "formal"]),
        ]
        
        for name, func, category, tags in chem_tools:
            self._register_tool(name, func, category, tags)
        
        # Structural biology tools
        struct_tools = [
            ("superimpose_structures", self.struct_tools.superimpose_structures, "Structural Biology", ["superimpose", "rmsd"]),
            ("analyze_binding_site", self.struct_tools.analyze_binding_site, "Structural Biology", ["binding", "interaction"]),
            ("download_pdb", self.struct_tools.download_pdb, "Structural Biology", ["pdb", "download"]),
            ("parse_pdb", self.struct_tools.parse_pdb, "Structural Biology", ["pdb", "parse"]),
            ("calculate_rmsd", self.struct_tools.calculate_rmsd, "Structural Biology", ["rmsd", "alignment"]),
            ("find_interactions", self.struct_tools.find_interactions, "Structural Biology", ["interactions", "contacts"]),
            ("analyze_conformation", self.struct_tools.analyze_conformation, "Structural Biology", ["conformation", "dihedral"]),
            ("compare_structures", self.struct_tools.compare_structures, "Structural Biology", ["compare", "difference"]),
            ("extract_ligand", self.struct_tools.extract_ligand, "Structural Biology", ["ligand", "extract"]),
            ("analyze_solvent_accessibility", self.struct_tools.analyze_solvent_accessibility, "Structural Biology", ["sasa", "accessibility"]),
        ]
        
        for name, func, category, tags in struct_tools:
            self._register_tool(name, func, category, tags)
        
        # Additional tool categories will be added here
        # Patent mining, Target identification, Hit identification, SAR analysis
    
    def _register_tool(self, name: str, func: Any, category: str, tags: List[str]):
        """Register a single tool"""
        self.tools[name] = func
        
        # Extract docstring and parameters
        docstring = func.__doc__ or ""
        lines = docstring.strip().split('\n')
        description = lines[0] if lines else ""
        
        # Simple parameter extraction (improved in future versions)
        params = {}
        if func.__code__.co_varnames:
            for var in func.__code__.co_varnames[:func.__code__.co_argcount]:
                if var != 'self':
                    params[var] = {"type": "any", "description": ""}
        
        self.tool_info[name] = ToolInfo(
            name=name,
            description=description,
            parameters={"type": "object", "properties": params},
            category=category,
            tags=tags
        )
        
        logger.debug(f"Registered tool: {name} ({category})")
    
    async def _handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle an MCP request"""
        try:
            if request.method not in self.tools:
                return MCPResponse(
                    id=request.id,
                    success=False,
                    error=f"Tool '{request.method}' not found"
                )
            
            func = self.tools[request.method]
            
            # Call the function with parameters
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(**request.params)
                else:
                    result = func(**request.params)
                
                return MCPResponse(
                    id=request.id,
                    success=True,
                    result=result
                )
            except Exception as e:
                logger.error(f"Error executing {request.method}: {e}")
                logger.error(traceback.format_exc())
                return MCPResponse(
                    id=request.id,
                    success=False,
                    error=str(e)
                )
                
        except Exception as e:
            logger.error(f"Request handling error: {e}")
            return MCPResponse(
                id=request.id,
                success=False,
                error=str(e)
            )
    
    async def _call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool"""
        try:
            if tool_name not in self.tools:
                return {"error": f"Tool '{tool_name}' not found", "success": False}
            
            func = self.tools[tool_name]
            
            if asyncio.iscoroutinefunction(func):
                result = await func(**params)
            else:
                result = func(**params)
            
            return {"result": result, "success": True, "tool": tool_name}
            
        except Exception as e:
            logger.error(f"Tool call error {tool_name}: {e}")
            return {"error": str(e), "success": False, "tool": tool_name}
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        """Application lifespan manager"""
        # Startup
        logger.info("Starting Drug Discovery MCP Server")
        
        # Initialize all components
        self._initialize_components()
        
        yield
        
        # Shutdown
        logger.info("Shutting down Drug Discovery MCP Server")
        await self._cleanup_components()
    
    def _initialize_components(self):
        """Initialize all server components"""
        # Initialize database connections
        self.db_tools.initialize()
        
        # Initialize cheminformatics
        self.chem_tools.initialize()
        
        # Initialize structural biology
        self.struct_tools.initialize()
        
        logger.info("All components initialized")
    
    async def _cleanup_components(self):
        """Clean up server components"""
        # Close database connections
        await self.db_tools.close()
        
        logger.info("Components cleaned up")
    
    def run(self):
        """Run the server"""
        import uvicorn
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            reload=self.config.server.debug,
            workers=self.config.server.max_workers,
            timeout_keep_alive=self.config.server.timeout,
        )
    
    async def start_async(self):
        """Start the server asynchronously"""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            reload=self.config.server.debug,
            workers=self.config.server.max_workers,
        )
        server = uvicorn.Server(config)
        await server.start()


def main():
    """Main entry point for the MCP server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Drug Discovery MCP Server")
    parser.add_argument("--host", type=str, default=None, help="Server host")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    parser.add_argument("--config", type=str, default=None, help="Configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Load custom config if provided
    config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
                # Create config from dict (simplified for now)
                from .config import DrugDiscoveryConfig
                config = DrugDiscoveryConfig(**config_data)
    
    # Create and run server
    server = DrugDiscoveryMCPServer(
        host=args.host,
        port=args.port,
        config=config
    )
    
    if args.debug:
        server.config.server.debug = True
    
    server.run()


if __name__ == "__main__":
    main()
