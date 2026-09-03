"""
Universal logging & tracing wrapper.
Provides logfire-compatible interface with graceful fallback to standard Python logging.
"""

import os
import sys
import logging
from contextlib import contextmanager

# Configure base logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("incident_system")

class DummySpan:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class LogfireFallback:
    """Provides logfire methods (info, warning, error, span, etc.) without requiring logfire package."""
    
    @staticmethod
    def configure(*args, **kwargs):
        pass
        
    @staticmethod
    def info(msg: str, **kwargs):
        extra = f" | {kwargs}" if kwargs else ""
        logger.info(f"{msg}{extra}")
        
    @staticmethod
    def warning(msg: str, **kwargs):
        extra = f" | {kwargs}" if kwargs else ""
        logger.warning(f"{msg}{extra}")
        
    @staticmethod
    def error(msg: str, **kwargs):
        extra = f" | {kwargs}" if kwargs else ""
        logger.error(f"{msg}{extra}")
        
    @staticmethod
    def debug(msg: str, **kwargs):
        extra = f" | {kwargs}" if kwargs else ""
        logger.debug(f"{msg}{extra}")
        
    @staticmethod
    @contextmanager
    def span(name: str, **kwargs):
        extra = f" ({kwargs})" if kwargs else ""
        logger.info(f"▶ SPAN: {name}{extra}")
        try:
            yield
        finally:
            logger.info(f"◀ END SPAN: {name}")

try:
    import logfire as _real_logfire
    # Test if configure is callable
    logfire = _real_logfire
except ImportError:
    logfire = LogfireFallback()
