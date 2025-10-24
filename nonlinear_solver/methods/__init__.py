"""
Métodos numéricos para resolução de sistemas não lineares.
"""

from .newton import NewtonSolver
from .iteration import IterationSolver
from .gradient import GradientSolver

__all__ = ['NewtonSolver', 'IterationSolver', 'GradientSolver']
