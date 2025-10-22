"""
Método de Jacobi de ordem 2 (Jacobi Relaxado) para resolução de sistemas lineares.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from ..base import LinearSolver


class JacobiOrder2Solver(LinearSolver):
    """
    Método de Jacobi de ordem 2 (Método de Jacobi Relaxado ou SOR-Jacobi).
    
    Utiliza uma combinação das duas últimas iterações para acelerar a convergência:
    x^(k+1) = ω₁ * x_jacobi^(k+1) + ω₂ * x^(k) + ω₃ * x^(k-1)
    
    onde x_jacobi^(k+1) é a iteração padrão do Jacobi.
    """
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 1000, 
                 omega1: float = 1.0, omega2: float = 0.0, omega3: float = 0.0):
        """
        Inicializa o método Jacobi de ordem 2.
        
        Args:
            tolerance: Tolerância para convergência
            max_iterations: Número máximo de iterações  
            omega1: Peso para a iteração Jacobi atual (padrão: 1.0)
            omega2: Peso para a iteração anterior (padrão: 0.0) 
            omega3: Peso para a iteração de duas posições atrás (padrão: 0.0)
        """
        super().__init__(tolerance, max_iterations)
        self.omega1 = omega1
        self.omega2 = omega2
        self.omega3 = omega3
        
        # Normalizar pesos para garantir estabilidade
        total_weight = omega1 + omega2 + omega3
        if abs(total_weight - 1.0) > 1e-10:
            self.omega1 /= total_weight
            self.omega2 /= total_weight  
            self.omega3 /= total_weight
    
    def get_method_name(self) -> str:
        return f"Jacobi Ordem 2 (ω₁={self.omega1:.2f}, ω₂={self.omega2:.2f}, ω₃={self.omega3:.2f})"
    
    def solve(self, A: np.ndarray, b: np.ndarray, 
              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resolve o sistema usando Jacobi de ordem 2.
        """
        self._validate_inputs(A, b)
        
        diagonal = np.diag(A)
        if np.any(np.abs(diagonal) < 1e-14):
            raise ValueError("Matriz A deve ter diagonal principal não-nula")
        
        n = A.shape[0]
        x = self._get_initial_guess(A, x0)
        x_prev = x.copy()  # x^(k-1)
        x_jacobi = np.zeros(n)
        
        self.convergence_history = []
        
        for iteration in range(self.max_iterations):
            # Calcular iteração Jacobi padrão
            for i in range(n):
                sum_ax = sum(A[i, j] * x[j] for j in range(n) if j != i)
                x_jacobi[i] = (b[i] - sum_ax) / A[i, i]
            
            # Aplicar combinação de alta ordem
            if iteration == 0:
                # Primeira iteração: usar apenas Jacobi padrão
                x_new = x_jacobi.copy()
            else:
                # Combinar as três últimas aproximações
                x_new = (self.omega1 * x_jacobi + 
                        self.omega2 * x + 
                        self.omega3 * x_prev)
            
            # Calcular erro e verificar convergência
            error = np.linalg.norm(x_new - x, ord=np.inf)
            self.convergence_history.append(error)
            
            if self._check_convergence(x_new, x):
                info = {
                    'converged': True,
                    'iterations': iteration + 1,
                    'final_error': error,
                    'method': self.get_method_name(),
                    'convergence_history': self.convergence_history.copy(),
                    'parameters': {'omega1': self.omega1, 'omega2': self.omega2, 'omega3': self.omega3}
                }
                return x_new.copy(), info
            
            # Atualizar para próxima iteração
            x_prev, x = x, x_new
        
        info = {
            'converged': False,
            'iterations': self.max_iterations,
            'final_error': self.convergence_history[-1],
            'method': self.get_method_name(),
            'convergence_history': self.convergence_history.copy(),
            'parameters': {'omega1': self.omega1, 'omega2': self.omega2, 'omega3': self.omega3}
        }
        return x.copy(), info
