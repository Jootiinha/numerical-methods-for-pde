"""
Métodos para resolução de sistemas lineares.
"""

from .cgs import CGSSolver
from .conjugate_gradient import ConjugateGradientSolver
from .gauss_seidel import GaussSeidelSolver
from .jacobi import JacobiSolver
from .preconditioned_cg import PreconditionedConjugateGradientSolver

__all__ = [
    "JacobiSolver",
    "GaussSeidelSolver",
    "ConjugateGradientSolver",
    "PreconditionedConjugateGradientSolver",
    "CGSSolver",
]
