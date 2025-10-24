"""
Módulo para resolução de sistemas de equações não lineares.

Este módulo contém métodos numéricos para resolver sistemas da forma F(x) = 0,
onde F é um sistema de funções não lineares.

Métodos disponíveis:
- Método de Newton-Raphson
- Método da Iteração
- Método do Gradiente

Exemplos de uso:
    from nonlinear_solver import NewtonSolver, IterationSolver, GradientSolver

    # Definir sistema de equações e jacobiano
    def system_function(x):
        # Implementar F(x) = 0
        pass
    
    def jacobian_function(x):
        # Implementar jacobiano de F
        pass
    
    # Resolver usando Newton
    solver = NewtonSolver(tolerance=1e-6)
    solution, info = solver.solve(system_function, jacobian_function, x0)
"""

from .base import NonLinearSolver
from .methods.newton import NewtonSolver
from .methods.iteration import IterationSolver
from .methods.gradient import GradientSolver

__all__ = [
    'NonLinearSolver',
    'NewtonSolver', 
    'IterationSolver',
    'GradientSolver'
]
