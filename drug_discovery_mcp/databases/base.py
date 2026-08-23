"""
Base Database Client

Provides the foundation for all database clients with common functionality
like caching, rate limiting, and error handling.
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

import aiohttp
import requests
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from cachetools import TTLCache, cached

from ..config import settings

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database errors"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


@dataclass
class DatabaseConfig:
    """Configuration for a database client"""
    endpoint: str
    rate_limit: int = 10
    timeout: int = 30
    retries: int = 3
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    api_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)


class DatabaseClient(ABC):
    """
    Base class for database clients
    
    Provides common functionality for all database clients including:
    - HTTP request handling
    - Rate limiting
    - Caching
    - Error handling
    - Retry logic
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize the database client
        
        Args:
            config: Database configuration
        """
        self.config = config or self.get_default_config()
        self.session: Optional[aiohttp.ClientSession] = None
        self._cache: Optional[TTLCache] = None
        self._last_request_time: float = 0
        self._request_count: int = 0
        
        if self.config.cache_enabled:
            self._cache = TTLCache(
                maxsize=1000,
                ttl=self.config.cache_ttl
            )
        
        # Initialize HTTP session
        self._init_session()
    
    @classmethod
    @abstractmethod
    def get_default_config(cls) -> DatabaseConfig:
        """Get default configuration for this database"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this database"""
        pass
    
    def _init_session(self):
        """Initialize HTTP session"""
        pass
    
    def _get_cache_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate a cache key for the given arguments"""
        key_data = {"args": args, "kwargs": kwargs}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _rate_limit_check(self) -> None:
        """Check rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        # Reset counter if enough time has passed
        if time_since_last > 60:  # 1 minute window
            self._request_count = 0
        
        # Check if we've hit the rate limit
        if self._request_count >= self.config.rate_limit:
            sleep_time = max(0, 60 - time_since_last)
            logger.warning(f"Rate limit hit for {self.get_name()}, sleeping for {sleep_time:.1f}s")
            time.sleep(sleep_time)
            self._request_count = 0
        
        self._last_request_time = current_time
        self._request_count += 1
    
    def _make_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with retry logic
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            params: Query parameters
            data: Request body
            headers: Request headers
            timeout: Request timeout
            
        Returns:
            Response as dictionary

        Raises:
            DatabaseError: If the request fails
        """
        response = self._request_raw(method, url, params, data, headers, timeout)

        try:
            return response.json()
        except Exception:
            return {"response": response.text}

    def _request_raw(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: Optional[int] = None,
    ) -> requests.Response:
        """
        Make an HTTP request and return the raw response

        Use this instead of ``_make_request`` when the caller needs response
        headers (pagination totals) or a non-object JSON body (a bare array).

        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            params: Query parameters
            data: Request body
            headers: Request headers
            timeout: Request timeout

        Returns:
            The requests.Response

        Raises:
            DatabaseError: If the request fails
        """
        timeout = timeout or self.config.timeout
        headers = headers or {}
        headers.update(self.config.headers)
        
        # Add API key if configured
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        # Retry only on transport errors -- an HTTP 4xx/5xx is surfaced as a
        # DatabaseError below and must not be retried.
        retry_strategy = Retrying(
            stop=stop_after_attempt(max(1, self.config.retries)),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type(requests.exceptions.RequestException),
            reraise=True,
        )
        
        try:
            self._rate_limit_check()
            
            # This method is synchronous, so always use the sync client.
            # (Async callers should await _make_async_request instead -- calling
            # asyncio.run() here would blow up inside a running event loop.)
            response = retry_strategy(
                lambda: requests.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=headers,
                    timeout=timeout,
                )
            )
            
            if response.status_code >= 400:
                try:
                    error_data = response.json()
                except Exception:
                    error_data = {"error": response.text}

                raise DatabaseError(
                    message=f"Request failed: {response.status_code} - {response.text}",
                    status_code=response.status_code,
                    details=error_data
                )

            return response

        except Exception as e:
            logger.error(f"Request error for {self.get_name()}: {e}")
            raise DatabaseError(
                message=f"Request failed: {str(e)}",
                details={"url": url, "method": method}
            )
    
    async def _make_async_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Make an async HTTP request"""
        timeout = timeout or self.config.timeout
        headers = headers or {}
        headers.update(self.config.headers)
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            self._rate_limit_check()
            
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                
                if response.status >= 400:
                    try:
                        error_data = await response.json()
                    except:
                        error_data = {"error": await response.text()}
                    
                    raise DatabaseError(
                        message=f"Request failed: {response.status} - {await response.text()}",
                        status_code=response.status,
                        details=error_data
                    )
                
                try:
                    return await response.json()
                except:
                    return {"response": await response.text()}
                    
        except Exception as e:
            logger.error(f"Async request error for {self.get_name()}: {e}")
            raise DatabaseError(
                message=f"Request failed: {str(e)}",
                details={"url": url, "method": method}
            )
    
    def close(self):
        """Close the client and clean up resources"""
        if self.session is not None:
            # ClientSession.close() is a coroutine
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self.session.close())
            else:
                loop.create_task(self.session.close())
        self.session = None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        self.close()
