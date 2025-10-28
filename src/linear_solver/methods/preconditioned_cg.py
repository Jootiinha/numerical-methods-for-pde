"""
Método do Gradiente Conjugado Precondicionado para resolução de sistemas lineares.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .conjugate_gradient import ConjugateGradientSolver


class PreconditionedConjugateGradientSolver(ConjugateGradientSolver):
    """
    Método do Gradiente Conjugado Precondicionado (PCG).

        Utiliza um precondicionador M para acelerar a convergência do CG
    tradicional. O precondicionador deve ser uma aproximação de A que
    seja fácil de inverter.
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        preconditioner: Optional[str] = "jacobi",
        check_symmetric: bool = True,
        check_positive_definite: bool = True,
    ):
        """
        Inicializa o PCG.

        Args:
            tolerance: Tolerância para convergência
            max_iterations: Número máximo de iterações
            preconditioner: Tipo de precondicionador ("jacobi", "ssor", ou
            None)
            check_symmetric: Se deve verificar se a matriz é simétrica
            check_positive_definite: Se deve verificar se a matriz é
            positiva definida
        """
        super().__init__(
            tolerance,
            max_iterations,
            check_symmetric,
            check_positive_definite,
        )
        self.preconditioner_type = preconditioner

    def get_method_name(self) -> str:
        if self.preconditioner_type:
            return (
                "Gradiente Conjugado Precondicionado " f"({self.preconditioner_type})"
            )
        return "Gradiente Conjugado Precondicionado"

    def _apply_preconditioner(self, A: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Aplica o precondicionador ao vetor r.

        Args:
            A: Matriz do sistema
            r: Vetor a ser precondicionado

        Returns:
            Vetor precondicionado M⁻¹r
        """
        if self.preconditioner_type == "jacobi":
            # Precondicionador de Jacobi: M = diag(A)
            diagonal = np.diag(A)
            return np.asarray(r / diagonal)

        elif self.preconditioner_type == "ssor":
            # Precondicionador SSOR simplificado
            # Para demonstração, usar apenas a diagonal (como Jacobi)
            diagonal = np.diag(A)
            return np.asarray(r / diagonal)

        else:
            # Sem precondicionador
            return np.asarray(r.copy())

    def solve(
        self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resolve o sistema usando PCG.
        """
        self._validate_inputs(A, b)
        self._check_matrix_properties(A)

        x = self._get_initial_guess(A, x0)

        # Inicialização do algoritmo PCG
        r = b - A @ x
        z = self._apply_preconditioner(A, r)
        p = z.copy()
        rzold = np.dot(r, z)

        self.convergence_history = []
        residual_history = []

        for iteration in range(self.max_iterations):
            Ap = A @ p

            pAp = np.dot(p, Ap)
            if abs(pAp) < 1e-15:
                break

            alpha = rzold / pAp

            # Atualizar solução e resíduo
            x_old = x.copy()
            x = x + alpha * p
            r = r - alpha * Ap

            # Aplicar precondicionador
            z = self._apply_preconditioner(A, r)

            # Verificar convergência
            residual_norm = np.linalg.norm(r)
            error = float(np.linalg.norm(x - x_old, ord=np.inf))
            self.convergence_history.append(error)
            residual_history.append(float(residual_norm))

            if residual_norm < self.tolerance:
                info = {
                    "converged": True,
                    "iterations": iteration + 1,
                    "final_error": error,
                    "final_residual": residual_norm,
                    "method": self.get_method_name(),
                    "convergence_history": self.convergence_history.copy(),
                    "residual_history": residual_history.copy(),
                    "preconditioner": self.preconditioner_type,
                }
                return x.copy(), info

            # Atualizar direção de busca
            rznew = np.dot(r, z)
            beta = rznew / rzold
            p = z + beta * p
            rzold = rznew

        # Não convergiu
        final_residual = np.linalg.norm(b - A @ x)
        info = {
            "converged": False,
            "iterations": self.max_iterations,
            "final_error": self.convergence_history[-1]
            if self.convergence_history
            else float("inf"),
            "final_residual": final_residual,
            "method": self.get_method_name(),
            "convergence_history": self.convergence_history.copy(),
            "residual_history": residual_history.copy(),
            "preconditioner": self.preconditioner_type,
        }
        return x.copy(), info
