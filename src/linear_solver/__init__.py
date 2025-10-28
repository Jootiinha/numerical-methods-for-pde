"""
Biblioteca para resolução de sistemas lineares usando métodos numéricos iterativos.

Métodos implementados:
- Jacobi
- Gauss-Seidel  
- Jacobi de ordem 2
- Gauss-Seidel de ordem 2
- Gradiente Conjugado
"""

__version__ = "1.0.0"
__author__ = "João Monteiro"

from .base import LinearSolver
from .methods import (
    JacobiSolver, GaussSeidelSolver,
    JacobiOrder2Solver, GaussSeidelOrder2Solver,
    ConjugateGradientSolver, PreconditionedConjugateGradientSolver
)
from .utils import CSVMatrixLoader, MatrixValidator, MatrixGenerator

__all__ = [
    'LinearSolver',
    'JacobiSolver', 
    'GaussSeidelSolver',
    'JacobiOrder2Solver',
    'GaussSeidelOrder2Solver', 
    'ConjugateGradientSolver',
    'PreconditionedConjugateGradientSolver',
    'CSVMatrixLoader',
    'MatrixValidator',
    'MatrixGenerator'
]
