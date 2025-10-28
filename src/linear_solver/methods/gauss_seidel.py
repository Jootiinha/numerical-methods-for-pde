"""
Método iterativo de Gauss-Seidel para resolução de sistemas lineares.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from ..base import LinearSolver


class GaussSeidelSolver(LinearSolver):
    """
    Método iterativo de Gauss-Seidel (e SOR) para sistemas lineares.
    Pode ser configurado como SOR (Successive Over-Relaxation) e/ou Ordem 2.
    """

    def __init__(self, tolerance: float = 1e-4, max_iterations: int = 1000,
                 relaxation_factor: float = 1.0,
                 omega1: float = 1.0, omega2: float = 0.0, omega3: float = 0.0):
        """
        Inicializa o método.

        Args:
            relaxation_factor: Fator de relaxação (ω). ω=1 é Gauss-Seidel.
            omega1, omega2, omega3: Pesos para o método de ordem 2.
        """
        super().__init__(tolerance, max_iterations)
        if not 0 < relaxation_factor < 2:
            raise ValueError("Fator de relaxação (ω) deve estar em (0, 2)")
        
        self.relaxation_factor = relaxation_factor
        self.omega1 = omega1
        self.omega2 = omega2
        self.omega3 = omega3
        self.is_sor = relaxation_factor != 1.0
        self.is_order2 = not (omega1 == 1.0 and omega2 == 0.0 and omega3 == 0.0)

    def get_method_name(self) -> str:
        if self.is_order2:
            return (f"Gauss-Seidel Ordem 2 (ω_relax={self.relaxation_factor:.2f}, "
                    f"ω₁={self.omega1:.2f}, ω₂={self.omega2:.2f}, ω₃={self.omega3:.2f})")
        if self.is_sor:
            return f"SOR (ω={self.relaxation_factor:.2f})"
        return "Gauss-Seidel"

    def solve(self, A: np.ndarray, b: np.ndarray,
              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
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

            # Loop principal do Gauss-Seidel/SOR (vetorização não é trivial)
            for i in range(n):
                sum_lower = np.dot(A[i, :i], x_sor[:i])
                sum_upper = np.dot(A[i, i + 1:], x_old[i + 1:])
                
                x_gs = (b[i] - sum_lower - sum_upper) / A[i, i]
                x_sor[i] = (1 - self.relaxation_factor) * x_old[i] + self.relaxation_factor * x_gs

            if self.is_order2:
                if iteration == 0:
                    x_new = x_sor
                else:
                    x_new = self.omega1 * x_sor + self.omega2 * x + self.omega3 * x_prev
            else:
                x_new = x_sor

            error = np.linalg.norm(x_new - x, ord=np.inf)
            residual = np.linalg.norm(A @ x_new - b)
            self.convergence_history.append(error)
            residual_history.append(residual)

            if self._check_convergence(x_new, x):
                return self._create_convergence_info(
                    converged=True, iterations=iteration + 1, solution=x_new,
                    final_error=error, final_residual=residual,
                    residual_history=residual_history
                )

            x_prev, x = x, x_new

        final_residual = np.linalg.norm(A @ x - b)
        return self._create_convergence_info(
            converged=False, iterations=self.max_iterations, solution=x,
            final_error=self.convergence_history[-1], final_residual=final_residual,
            residual_history=residual_history
        )

    def _create_convergence_info(self, converged: bool, iterations: int, solution: np.ndarray,
                                 final_error: float, final_residual: float, residual_history: list) -> Tuple[np.ndarray, Dict[str, Any]]:
        info = {
            'converged': converged, 'iterations': iterations,
            'final_error': final_error, 'final_residual': final_residual,
            'method': self.get_method_name(),
            'convergence_history': self.convergence_history.copy(),
            'residual_history': residual_history.copy()
        }
        if self.is_sor or self.is_order2:
            info['parameters'] = {'relaxation_factor': self.relaxation_factor}
        if self.is_order2:
            info['parameters'].update({'omega1': self.omega1, 'omega2': self.omega2, 'omega3': self.omega3})
            
        return solution.copy(), info

    def get_iteration_matrix(self, A: np.ndarray) -> np.ndarray:
        """Retorna a matriz de iteração para Gauss-Seidel/SOR."""
        L = np.tril(A, -1)
        U = np.triu(A, 1)
        D = np.diag(np.diag(A))
        
        # Matriz de iteração para SOR
        D_plus_omega_L = D + self.relaxation_factor * L
        if np.linalg.det(D_plus_omega_L) == 0:
            raise ValueError("Matriz (D + ωL) é singular, não é possível calcular a matriz de iteração.")
            
        D_plus_omega_L_inv = np.linalg.inv(D_plus_omega_L)
        rhs = (1 - self.relaxation_factor) * D - self.relaxation_factor * U
        M_sor = D_plus_omega_L_inv @ rhs
        
        if self.is_order2:
            I = np.eye(A.shape[0])
            return self.omega1 * M_sor + self.omega2 * I + self.omega3 * np.linalg.matrix_power(M_sor, 2)
            
        return M_sor
