"""
Configuration management for Drug Discovery MCP Server
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Configuration for external database connections"""
    
    uniprot: Dict[str, Any] = Field(
        default={"endpoint": "https://rest.uniprot.org/uniprotkb", "rate_limit": 10}
    )
    chembl: Dict[str, Any] = Field(
        default={"endpoint": "https://www.ebi.ac.uk/chembl/api/data", "rate_limit": 10}
    )
    pdb: Dict[str, Any] = Field(
        default={"endpoint": "https://data.rcsb.org", "rate_limit": 20}
    )
    open_targets: Dict[str, Any] = Field(
        default={"endpoint": "https://api.platform.opentargets.org/api/v4/graphql", "rate_limit": 10}
    )
    kegg: Dict[str, Any] = Field(
        default={"endpoint": "https://rest.kegg.jp", "rate_limit": 10}
    )
    pubchem: Dict[str, Any] = Field(
        default={"endpoint": "https://pubchem.ncbi.nlm.nih.gov/rest/pug", "rate_limit": 10}
    )
    ncbi: Dict[str, Any] = Field(
        default={"endpoint": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils", "rate_limit": 10}
    )


class CacheConfig(BaseSettings):
    """Configuration for caching system"""
    
    enabled: bool = True
    directory: Path = Path("./cache")
    expiry_days: int = 7
    max_size_mb: int = 1024  # 1GB


class LoggingConfig(BaseSettings):
    """Configuration for logging"""
    
    level: str = "INFO"
    file: Optional[Path] = Path("./logs/server.log")
    console: bool = True
    rotation: str = "10 MB"
    retention: str = "7 days"


class ServerConfig(BaseSettings):
    """Configuration for MCP server"""
    
    host: str = "0.0.0.0"
    port: int = 8080
    max_workers: int = 10
    timeout: int = 300  # 5 minutes
    debug: bool = False
    cors_origins: List[str] = Field(default=["*"])


class DrugDiscoveryConfig(BaseSettings):
    """Main configuration for Drug Discovery MCP Server"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )
    
    server: ServerConfig = Field(default_factory=ServerConfig)
    databases: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Additional settings
    data_directory: Path = Path("./data")
    temp_directory: Path = Path("./temp")
    max_file_size_mb: int = 100
    
    @field_validator("data_directory", "temp_directory", mode="before")
    @classmethod
    def create_directories(cls, v: Any) -> Path:
        """Ensure directories exist"""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global settings instance
settings = DrugDiscoveryConfig()


def get_config() -> DrugDiscoveryConfig:
    """Get the global configuration instance"""
    return settings


def reload_config() -> DrugDiscoveryConfig:
    """Reload configuration from environment and files"""
    global settings
    settings = DrugDiscoveryConfig()
    return settings
