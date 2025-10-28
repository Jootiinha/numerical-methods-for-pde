"""
Método iterativo de Jacobi para resolução de sistemas lineares.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..base import LinearSolver


class JacobiSolver(LinearSolver):
    """
    Método iterativo de Jacobi para resolução de sistemas lineares.
    Pode ser configurado como Jacobi Relaxado (ordem 2) através dos pesos omega.

    O método de Jacobi resolve o sistema Ax = b através da decomposição:
    A = D + L + U, onde D é diagonal, L triangular inferior, U triangular superior.

    Fórmula iterativa padrão: x^(k+1) = D^(-1) * (b - (L + U) * x^(k))
    Fórmula de ordem 2: x_new = ω₁*x_jacobi + ω₂*x_old + ω₃*x_older
    """

    def __init__(
        self,
        tolerance: float = 1e-4,
        max_iterations: int = 1000,
        omega1: float = 1.0,
        omega2: float = 0.0,
        omega3: float = 0.0,
    ):
        """
        Inicializa o método Jacobi.

        Args:
            tolerance: Tolerância para convergência.
            max_iterations: Número máximo de iterações.
            omega1: Peso para a iteração Jacobi atual (padrão: 1.0).
            omega2: Peso para a iteração anterior (padrão: 0.0).
            omega3: Peso para a iteração de duas posições atrás (padrão: 0.0).
        """
        super().__init__(tolerance, max_iterations)
        self.omega1 = omega1
        self.omega2 = omega2
        self.omega3 = omega3
        self.is_order2 = not (omega1 == 1.0 and omega2 == 0.0 and omega3 == 0.0)

    def get_method_name(self) -> str:
        if not self.is_order2:
            return "Jacobi"
        return (
            f"Jacobi Ordem 2 (ω₁={self.omega1:.2f}, ω₂={self.omega2:.2f}, "
            f"ω₃={self.omega3:.2f})"
        )

    def solve(
        self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resolve o sistema usando o método de Jacobi (ou Jacobi Ordem 2).
        """
        self._validate_inputs(A, b)

        diagonal_values = np.diag(A)
        if np.any(np.abs(diagonal_values) < 1e-14):
            raise ValueError("Matriz A deve ter diagonal principal não-nula")

        x = self._get_initial_guess(A, x0)
        x_prev = x.copy()

        self.convergence_history = []
        residual_history = []

        # Matriz L+U (off-diagonal)
        L_plus_U = A - np.diag(diagonal_values)

        for iteration in range(self.max_iterations):
            # Fórmula de Jacobi vetorizada: x_jacobi = D⁻¹ * (b - (L+U)x)
            x_jacobi = (b - (L_plus_U @ x)) / diagonal_values

            if self.is_order2:
                if iteration == 0:
                    x_new = x_jacobi
                else:
                    x_new = (
                        self.omega1 * x_jacobi + self.omega2 * x + self.omega3 * x_prev
                    )
            else:
                x_new = x_jacobi

            error = float(np.linalg.norm(x_new - x, ord=np.inf))
            residual = float(np.linalg.norm(A @ x_new - b))

            self.convergence_history.append(error)
            residual_history.append(residual)

            if self._check_convergence(x_new, x):
                return self._create_convergence_info(
                    converged=True,
                    iterations=iteration + 1,
                    solution=x_new,
                    final_error=error,
                    final_residual=residual,
                    residual_history=residual_history,
                )

            x_prev, x = x, x_new

        final_residual = float(np.linalg.norm(A @ x - b))
        return self._create_convergence_info(
            converged=False,
            iterations=self.max_iterations,
            solution=x,
            final_error=self.convergence_history[-1],
            final_residual=final_residual,
            residual_history=residual_history,
        )

    def _create_convergence_info(
        self,
        converged: bool,
        iterations: int,
        solution: np.ndarray,
        final_error: float,
        final_residual: float,
        residual_history: list,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        info = {
            "converged": converged,
            "iterations": iterations,
            "final_error": final_error,
            "final_residual": final_residual,
            "method": self.get_method_name(),
            "convergence_history": self.convergence_history.copy(),
            "residual_history": residual_history.copy(),
        }
        if self.is_order2:
            info["parameters"] = {
                "omega1": self.omega1,
                "omega2": self.omega2,
                "omega3": self.omega3,
            }

        return solution.copy(), info

    def get_iteration_matrix(self, A: np.ndarray) -> np.ndarray:
        """Retorna a matriz de iteração para o método de Jacobi."""
        diagonal_values = np.diag(A)
        if np.any(np.abs(diagonal_values) < 1e-14):
            raise ValueError("Matriz A deve ter diagonal principal não-nula")

        D_inv = np.diag(1 / diagonal_values)
        L_plus_U = A - np.diag(diagonal_values)

        M_jacobi = -D_inv @ L_plus_U

        if self.is_order2:
            # Para ordem 2, a "matriz de iteração" não é fixa, mas esta é
            # uma aproximação linear
            identity_matrix = np.eye(A.shape[0])
            iteration_matrix = (
                self.omega1 * M_jacobi
                + self.omega2 * identity_matrix
                + self.omega3 * np.linalg.matrix_power(M_jacobi, 2)
            )
            return np.asarray(iteration_matrix)

        return np.asarray(M_jacobi)
