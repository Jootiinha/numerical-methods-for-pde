"""
Biblioteca para resolução de sistemas lineares usando métodos numéricos iterativos.

Métodos implementados:
- Jacobi (com suporte a Ordem 2)
- Gauss-Seidel (com suporte a SOR e Ordem 2)
- Gradiente Conjugado
- Gradiente Conjugado Pré-condicionado
"""

__version__ = "1.0.0"
__author__ = "João Monteiro - joaocrm@id.uff.br"

from .base import LinearSolver
from .methods import (
    JacobiSolver, GaussSeidelSolver,
    ConjugateGradientSolver, PreconditionedConjugateGradientSolver
)
from .utils import CSVMatrixLoader, MatrixValidator, MatrixGenerator

__all__ = [
    'LinearSolver',
    'JacobiSolver',
    'GaussSeidelSolver',
    'ConjugateGradientSolver',
    'PreconditionedConjugateGradientSolver',
    'CSVMatrixLoader',
    'MatrixValidator',
    'MatrixGenerator'
]
