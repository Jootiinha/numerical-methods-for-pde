"""
Método do Gradiente Conjugado Quadrado (CGS) para sistemas lineares.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..base import LinearSolver


class CGSSolver(LinearSolver):
    """
    Método do Gradiente Conjugado Quadrado (CGS) para sistemas lineares.

    Este método é uma variante do BiCG (Gradiente Conjugado Bi-Estabilizado)
    e é adequado para matrizes não simétricas.
    Pode sofrer de convergência irregular e grandes resíduos intermediários.

    Algoritmo (conforme implementado):
    1.  Inicializar x₀, r₀ = b - Ax₀
    2.  Escolher r̃₀ (e.g., r̃₀ = r₀)
    3.  p₀ = u₀ = r₀
    4.  Para k = 0, 1, ...:
        -   αₖ = (r̃₀ᵀ rₖ) / (r̃₀ᵀ A pₖ)
        -   qₖ = uₖ - αₖ A pₖ
        -   xₖ₊₁ = xₖ + αₖ (uₖ + qₖ)
        -   rₖ₊₁ = b - A xₖ₊₁  (recalculado para maior precisão)
        -   βₖ = (r̃₀ᵀ rₖ₊₁) / (r̃₀ᵀ rₖ)
        -   uₖ₊₁ = rₖ₊₁ + βₖ qₖ
        -   pₖ₊₁ = uₖ₊₁ + βₖ (qₖ + βₖ pₖ)
    """

    def get_method_name(self) -> str:
        return "Gradiente Conjugado Quadrado (CGS)"

    def solve(
        self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resolve o sistema usando o método CGS.

        Args:
            A: Matriz de coeficientes (pode ser não simétrica).
            b: Vetor de termos independentes.
            x0: Aproximação inicial (padrão: vetor nulo).

        Returns:
            Tupla (solução, informações_convergência).
        """
        self._validate_inputs(A, b)
        x = self._get_initial_guess(A, x0)

        r = b - A @ x
        r_tilde = r.copy()  # Escolha padrão para o vetor sombra

        u = r.copy()
        p = r.copy()

        self.convergence_history = []
        residual_history = []

        for iteration in range(self.max_iterations):
            x_old = x.copy()

            # ρₖ = r̃₀ᵀ rₖ
            rho = np.dot(r_tilde, r)
            if abs(rho) < 1e-15:
                break  # Breakdown do método

            # v = A pₖ
            v = A @ p

            # αₖ = ρₖ / (r̃₀ᵀ v)
            r_tilde_v = np.dot(r_tilde, v)
            if abs(r_tilde_v) < 1e-15:
                break  # Breakdown do método

            alpha = rho / r_tilde_v

            # qₖ = uₖ - αₖ v
            q = u - alpha * v

            # xₖ₊₁ = xₖ + αₖ (uₖ + qₖ)
            x = x + alpha * (u + q)

            # rₖ₊₁ = rₖ - αₖ A (uₖ + qₖ)
            # Otimização: A(u+q) = A(r+βq) + A(u-αv) ... é complexo.
            # Recalcular é mais seguro.
            r_new = b - A @ x

            # ρₖ₊₁ = r̃₀ᵀ rₖ₊₁
            rho_new = np.dot(r_tilde, r_new)

            # βₖ = ρₖ₊₁ / ρₖ
            beta = rho_new / rho

            # uₖ₊₁ = rₖ₊₁ + βₖ qₖ
            u = r_new + beta * q

            # pₖ₊₁ = uₖ₊₁ + βₖ (qₖ + βₖ pₖ)
            p = u + beta * (q + beta * p)

            r = r_new

            error = float(np.linalg.norm(x - x_old, ord=np.inf))
            residual = float(np.linalg.norm(r))
            self.convergence_history.append(error)
            residual_history.append(residual)

            if residual < self.tolerance:
                return self._create_convergence_info(
                    converged=True,
                    iterations=iteration + 1,
                    solution=x,
                    final_error=error,
                    final_residual=residual,
                    residual_history=residual_history,
                )

        final_residual = float(np.linalg.norm(b - A @ x))
        return self._create_convergence_info(
            converged=False,
            iterations=self.max_iterations,
            solution=x,
            final_error=self.convergence_history[-1] if self.convergence_history else float("inf"),
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
        return solution.copy(), info
