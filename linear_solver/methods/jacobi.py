"""
Método iterativo de Jacobi para resolução de sistemas lineares.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from ..base import LinearSolver


class JacobiSolver(LinearSolver):
    """
    Método iterativo de Jacobi para resolução de sistemas lineares.
    
    O método de Jacobi resolve o sistema Ax = b através da decomposição:
    A = D + L + U, onde D é diagonal, L triangular inferior, U triangular superior.
    
    Fórmula iterativa: x^(k+1) = D^(-1) * (b - (L + U) * x^(k))
    """
    
    def get_method_name(self) -> str:
        return "Jacobi"
    
    def solve(self, A: np.ndarray, b: np.ndarray, 
              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resolve o sistema usando o método de Jacobi.
        
        Args:
            A: Matriz de coeficientes (deve ter diagonal não-nula)
            b: Vetor de termos independentes
            x0: Aproximação inicial (padrão: vetor nulo)
            
        Returns:
            Tupla (solução, informações_convergência)
        """
        self._validate_inputs(A, b)
        
        # Verificar se a diagonal principal é não-nula
        diagonal = np.diag(A)
        if np.any(np.abs(diagonal) < 1e-14):
            raise ValueError("Matriz A deve ter diagonal principal não-nula")
        
        n = A.shape[0]
        x = self._get_initial_guess(A, x0)
        x_new = np.zeros(n)
        
        self.convergence_history = []
        
        for iteration in range(self.max_iterations):
            # Aplicar fórmula iterativa de Jacobi
            for i in range(n):
                sum_ax = sum(A[i, j] * x[j] for j in range(n) if j != i)
                x_new[i] = (b[i] - sum_ax) / A[i, i]
            
            # Calcular erro e verificar convergência
            error = np.linalg.norm(x_new - x, ord=np.inf)
            self.convergence_history.append(error)
            
            if self._check_convergence(x_new, x):
                info = {
                    'converged': True,
                    'iterations': iteration + 1,
                    'final_error': error,
                    'method': self.get_method_name(),
                    'convergence_history': self.convergence_history.copy()
                }
                return x_new.copy(), info
            
            x, x_new = x_new, x
        
        # Não convergiu
        info = {
            'converged': False,
            'iterations': self.max_iterations,
            'final_error': self.convergence_history[-1],
            'method': self.get_method_name(),
            'convergence_history': self.convergence_history.copy()
        }
        return x.copy(), info
