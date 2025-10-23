"""
Método de Gauss-Seidel de ordem 2 (SOR de ordem 2) para resolução de sistemas lineares.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from ..base import LinearSolver


class GaussSeidelOrder2Solver(LinearSolver):
    """
    Método de Gauss-Seidel de ordem 2 (SOR - Successive Over-Relaxation de ordem 2).
    
    Combina o método SOR tradicional com informações de iterações anteriores:
    x^(k+1) = ω₁ * x_sor^(k+1) + ω₂ * x^(k) + ω₃ * x^(k-1)
    
    onde x_sor^(k+1) é o resultado do Gauss-Seidel relaxado.
    """
    
    def __init__(self, tolerance: float = 1e-4, max_iterations: int = 1000,
                 relaxation_factor: float = 1.0,
                 omega1: float = 1.0, omega2: float = 0.0, omega3: float = 0.0):
        """
        Inicializa o método Gauss-Seidel de ordem 2.
        
        Args:
            tolerance: Tolerância para convergência
            max_iterations: Número máximo de iterações
            relaxation_factor: Fator de relaxação para SOR (ω ∈ (0,2))
            omega1: Peso para a iteração SOR atual
            omega2: Peso para a iteração anterior  
            omega3: Peso para a iteração de duas posições atrás
        """
        super().__init__(tolerance, max_iterations)
        
        if not 0 < relaxation_factor < 2:
            raise ValueError("Fator de relaxação deve estar no intervalo (0, 2)")
            
        self.relaxation_factor = relaxation_factor
        self.omega1 = omega1
        self.omega2 = omega2
        self.omega3 = omega3
        
        # Normalizar pesos
        total_weight = omega1 + omega2 + omega3
        if abs(total_weight - 1.0) > 1e-10:
            self.omega1 /= total_weight
            self.omega2 /= total_weight
            self.omega3 /= total_weight
    
    def get_method_name(self) -> str:
        return (f"Gauss-Seidel Ordem 2 (ω_relax={self.relaxation_factor:.2f}, "
               f"ω₁={self.omega1:.2f}, ω₂={self.omega2:.2f}, ω₃={self.omega3:.2f})")
    
    def solve(self, A: np.ndarray, b: np.ndarray, 
              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resolve o sistema usando Gauss-Seidel de ordem 2.
        """
        self._validate_inputs(A, b)
        
        diagonal = np.diag(A)
        if np.any(np.abs(diagonal) < 1e-14):
            raise ValueError("Matriz A deve ter diagonal principal não-nula")
        
        n = A.shape[0]
        x = self._get_initial_guess(A, x0)
        x_prev = x.copy()
        
        self.convergence_history = []
        residual_history = []
        
        for iteration in range(self.max_iterations):
            x_old = x.copy()
            x_sor = x.copy()
            
            # Aplicar Gauss-Seidel com relaxação (SOR)
            for i in range(n):
                sum_lower = sum(A[i, j] * x_sor[j] for j in range(i))
                sum_upper = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
                
                x_gauss_seidel = (b[i] - sum_lower - sum_upper) / A[i, i]
                
                # Aplicar relaxação
                x_sor[i] = ((1 - self.relaxation_factor) * x_old[i] + 
                           self.relaxation_factor * x_gauss_seidel)
            
            # Aplicar combinação de alta ordem
            if iteration == 0:
                x_new = x_sor.copy()
            else:
                x_new = (self.omega1 * x_sor + 
                        self.omega2 * x + 
                        self.omega3 * x_prev)
            
            # Calcular erro e resíduo
            error = np.linalg.norm(x_new - x, ord=np.inf)
            residual = np.linalg.norm(A @ x_new - b)
            
            self.convergence_history.append(error)
            residual_history.append(residual)
            
            if self._check_convergence(x_new, x):
                info = {
                    'converged': True,
                    'iterations': iteration + 1,
                    'final_error': error,
                    'final_residual': residual,
                    'method': self.get_method_name(),
                    'convergence_history': self.convergence_history.copy(),
                    'residual_history': residual_history.copy(),
                    'parameters': {
                        'relaxation_factor': self.relaxation_factor,
                        'omega1': self.omega1, 
                        'omega2': self.omega2, 
                        'omega3': self.omega3
                    }
                }
                return x_new.copy(), info
            
            # Atualizar para próxima iteração
            x_prev, x = x, x_new
        
        final_residual = np.linalg.norm(A @ x - b)
        info = {
            'converged': False,
            'iterations': self.max_iterations,
            'final_error': self.convergence_history[-1],
            'final_residual': final_residual,
            'method': self.get_method_name(),
            'convergence_history': self.convergence_history.copy(),
            'residual_history': residual_history.copy(),
            'parameters': {
                'relaxation_factor': self.relaxation_factor,
                'omega1': self.omega1,
                'omega2': self.omega2, 
                'omega3': self.omega3
            }
        }
        return x.copy(), info
