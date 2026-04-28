"""Linear solve module for kd2.

This module provides sparse solvers for the equation y = theta @ xi.

Classes:
    SolveResult: Structured result from solver
    SparseSolver: Abstract base class for solvers
    LeastSquaresSolver: Standard least squares solver
    STRidgeSolver: Sequential Threshold Ridge regression solver
"""

from kd2.core.linear_solve.base import SolveResult, SparseSolver
from kd2.core.linear_solve.least_squares import LeastSquaresSolver
from kd2.core.linear_solve.stridge import STRidgeSolver

__all__ = [
    "SolveResult",
    "SparseSolver",
    "LeastSquaresSolver",
    "STRidgeSolver",
]
