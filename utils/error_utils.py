"""
Error Utilities Module
--------------------
Centralized utilities for error handling.
"""

import os
import traceback
from typing import Any, Callable, TypeVar, Dict, List, Optional, Awaitable
from functools import wraps
import asyncio

# Import centralized utilities
from utils.logging_utils import get_logger

# Set up logging
logger = get_logger(__name__)

# Type variables for generic function signatures
T = TypeVar('T')
R = TypeVar('R')

class ErrorUtils:
    """Centralized utility functions for error handling."""
    
    @staticmethod
    def log_exception(e: Exception, message: str) -> None:
        """Log an exception with a custom message."""
        logger.error(f"{message}: {str(e)}", exc_info=True)
    
    @staticmethod
    def handle_exceptions(default_return: Any = None):
        """Decorator for synchronous functions to handle exceptions and return a default value."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    func_name = func.__name__
                    ErrorUtils.log_exception(e, f"Error in {func_name}")
                    return default_return
            return wrapper
        return decorator
    
    @staticmethod
    def handle_async_exceptions(default_return: Any = None):
        """Decorator for async functions to handle exceptions and return a default value."""
        def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    func_name = func.__name__
                    ErrorUtils.log_exception(e, f"Error in async {func_name}")
                    return default_return
            return wrapper
        return decorator
    
    @staticmethod
    def with_fallback(fallback_func: Callable[..., T]):
        """Decorator to use a fallback function if the main function fails."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    func_name = func.__name__
                    ErrorUtils.log_exception(e, f"Error in {func_name}, using fallback")
                    return fallback_func(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def with_async_fallback(fallback_func: Callable[..., Awaitable[T]]):
        """Decorator to use an async fallback function if the main async function fails."""
        def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    func_name = func.__name__
                    ErrorUtils.log_exception(e, f"Error in async {func_name}, using fallback")
                    return await fallback_func(*args, **kwargs)
            return wrapper
        return decorator 