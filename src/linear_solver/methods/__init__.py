"""
Métodos para resolução de sistemas lineares.
"""

from .conjugate_gradient import ConjugateGradientSolver
from .gauss_seidel import GaussSeidelSolver
from .jacobi import JacobiSolver
from .preconditioned_cg import PreconditionedConjugateGradientSolver

__all__ = [
    "JacobiSolver",
    "GaussSeidelSolver",
    "ConjugateGradientSolver",
    "PreconditionedConjugateGradientSolver",
]
