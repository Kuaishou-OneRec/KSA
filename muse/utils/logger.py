"""
Logging Configuration and Utilities.

This module provides logging setup and configuration for the muse package.
It configures structured logging with appropriate handlers and formatters
for both console and file output.

The default configuration:
- Logs to stdout at INFO level
- Uses the "muse" logger namespace
- Does not propagate to root logger
- Preserves existing loggers

Functions:
    init_logger: Initialize a named logger
    _configure_logging: Configure logging from dict config

Constants:
    DEFAULT_LOGGING_CONFIG: Default logging configuration dict

Example:
    >>> from muse.utils.logger import init_logger
    >>> logger = init_logger("muse.training")
    >>> logger.info("Training started")
    >>> logger.warning("Learning rate is very high")
"""
import logging
from logging.config import dictConfig

DEFAULT_LOGGING_CONFIG = {
    "handlers": {
        "muse": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "muse": {
            "handlers": ["muse"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "version": 1,
    "disable_existing_loggers": False
}

def init_logger(name: str) -> logging.Logger:
    """
    Initialize and return a named logger.
    
    Creates a logger with the specified name. The logger will use the
    configuration set by _configure_logging (called at module import).
    
    Args:
        name (str): Logger name, typically module name (e.g., "muse.training.checkpoint")
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        >>> logger = init_logger(__name__)
        >>> logger.info("Module initialized")
        >>> logger.debug("Detailed debug information")
    """
    logger = logging.getLogger(name)
    return logger

def _configure_logging(config: dict = DEFAULT_LOGGING_CONFIG):
    """
    Configure logging from a dictionary configuration.
    
    Internal function called at module import to set up the logging system.
    Uses Python's dictConfig for flexible logging configuration.
    
    Args:
        config (dict): Logging configuration dictionary following Python's
            logging.config.dictConfig format. Defaults to DEFAULT_LOGGING_CONFIG.
    """
    dictConfig(config)

_configure_logging(DEFAULT_LOGGING_CONFIG)