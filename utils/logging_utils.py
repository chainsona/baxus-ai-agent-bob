"""
Logging Utilities Module
----------------------
Centralized logging configuration for the entire application.
"""

import os
import logging
import colorlog
from typing import Optional

# Default logging format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def configure_logging(
    module_name: str,
    level: Optional[str] = None,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger for the specified module.
    
    Args:
        module_name: Name of the module requesting the logger
        level: Logging level (defaults to LOG_LEVEL env var or "INFO")
        format_str: Logging format string (defaults to standard format)
        
    Returns:
        Configured logger instance
    """
    # Get logging level from environment or use default
    log_level_str = level or os.getenv("LOG_LEVEL", "INFO")
    
    # Convert string level to logging level
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # Use provided format or default
    log_format = format_str or DEFAULT_FORMAT
    
    # Get the module-specific logger
    logger = logging.getLogger(module_name)
    
    # Only configure if it has no handlers yet
    if not logger.handlers:
        # Create colored formatter
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s' + log_format,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        
        # Create and add handler with formatter
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Set level and prevent propagation
        logger.setLevel(log_level)
        logger.propagate = False
    
    return logger

def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for the specified module using the centralized configuration.
    
    Args:
        module_name: Name of the module requesting the logger
        
    Returns:
        Configured logger instance
    """
    return configure_logging(module_name) 