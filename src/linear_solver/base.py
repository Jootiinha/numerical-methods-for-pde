"""
Classe abstrata base para resolvedores de sistemas lineares.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class LinearSolver(ABC):
    """
    Classe abstrata base para todos os métodos de resolução de sistemas lineares.

    Define a interface comum que todos os resolvedores devem implementar.
    """

    def __init__(self, tolerance: float = 1e-4, max_iterations: int = 1000):
        """
        Inicializa o resolvedor.

        Args:
            tolerance: Tolerância para critério de convergência (padrão: 1e-4 = 10^(-4))
            max_iterations: Número máximo de iterações
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.convergence_history = []

    @abstractmethod
    def solve(
        self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resolve o sistema linear Ax = b.

        Args:
            A: Matriz de coeficientes (n x n)
            b: Vetor de termos independentes (n,)
            x0: Aproximação inicial (opcional)

        Returns:
            Tupla contendo:
            - x: Solução do sistema
            - info: Dicionário com informações sobre a convergência
        """
        pass

    @abstractmethod
    def get_method_name(self) -> str:
        """Retorna o nome do método."""
        pass

    def get_iteration_matrix(self, A: np.ndarray) -> Optional[np.ndarray]:
        """
        Retorna a matriz de iteração do método, se aplicável.

        Args:
            A: Matriz de coeficientes

        Returns:
            Matriz de iteração ou None se não for aplicável.
        """
        return None

    def _check_convergence(self, x_new: np.ndarray, x_old: np.ndarray) -> bool:
        """
        Verifica critério de convergência baseado na norma da diferença.

        Args:
            x_new: Nova aproximação
            x_old: Aproximação anterior

        Returns:
            True se converged, False caso contrário
        """
        error = np.linalg.norm(x_new - x_old, ord=np.inf)
        return error < self.tolerance

    def _validate_inputs(self, A: np.ndarray, b: np.ndarray) -> None:
        """
        Valida as entradas do sistema linear.

        Args:
            A: Matriz de coeficientes
            b: Vetor de termos independentes

        Raises:
            ValueError: Se as dimensões não são compatíveis ou A não é quadrada
        """
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A matriz A deve ser quadrada")

        if b.ndim != 1:
            raise ValueError("O vetor b deve ser unidimensional")

        if A.shape[0] != b.shape[0]:
            raise ValueError("Dimensões incompatíveis entre A e b")

    def _get_initial_guess(
        self, A: np.ndarray, x0: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Obtém aproximação inicial para o método iterativo.

        Args:
            A: Matriz de coeficientes
            x0: Aproximação inicial fornecida (opcional)

        Returns:
            Aproximação inicial a ser usada
        """
        n = A.shape[0]
        if x0 is None:
            return np.zeros(n)

        if x0.shape[0] != n:
            raise ValueError(f"x0 deve ter dimensão {n}")

        return x0.copy()
