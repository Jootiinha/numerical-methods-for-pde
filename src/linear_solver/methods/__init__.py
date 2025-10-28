"""
Métodos para resolução de sistemas lineares.
"""

from .jacobi import JacobiSolver
from .gauss_seidel import GaussSeidelSolver
from .conjugate_gradient import ConjugateGradientSolver
from .preconditioned_cg import PreconditionedConjugateGradientSolver

__all__ = [
    'JacobiSolver',
    'GaussSeidelSolver',
    'ConjugateGradientSolver',
    'PreconditionedConjugateGradientSolver'
]
