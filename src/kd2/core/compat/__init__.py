"""Compatibility layer for external algorithms.

This module provides bridging functions for format conversions
between kd2's Python AST-based IR and external formats like
prefix notation (used by DISCOVER, etc.).
"""

from kd2.core.compat.prefix import prefix_to_python, python_to_prefix

__all__ = ["python_to_prefix", "prefix_to_python"]
