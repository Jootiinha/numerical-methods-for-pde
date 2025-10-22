"""
Métodos para resolução de sistemas lineares.
"""

from .jacobi import JacobiSolver
from .gauss_seidel import GaussSeidelSolver
from .jacobi_order2 import JacobiOrder2Solver
from .gauss_seidel_order2 import GaussSeidelOrder2Solver
from .conjugate_gradient import ConjugateGradientSolver
from .preconditioned_cg import PreconditionedConjugateGradientSolver

__all__ = [
    'JacobiSolver',
    'GaussSeidelSolver', 
    'JacobiOrder2Solver',
    'GaussSeidelOrder2Solver',
    'ConjugateGradientSolver',
    'PreconditionedConjugateGradientSolver'
]
