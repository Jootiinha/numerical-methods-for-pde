"""
Utilitários para validação de propriedades de matrizes.
"""

from typing import Any, Dict

import numpy as np


class MatrixValidator:
    """
    Classe para validação de propriedades de matrizes.
    """

    @staticmethod
    def is_diagonally_dominant(A: np.ndarray, strict: bool = False) -> bool:
        """
        Verifica se a matriz é diagonalmente dominante.

        Args:
            A: Matriz a ser verificada
            strict: Se deve verificar dominância estrita

        Returns:
            True se a matriz é diagonalmente dominante
        """
        n = A.shape[0]

        for i in range(n):
            diagonal = abs(A[i, i])
            off_diagonal_sum = sum(abs(A[i, j]) for j in range(n) if j != i)

            if strict:
                if diagonal <= off_diagonal_sum:
                    return False
            else:
                if diagonal < off_diagonal_sum:
                    return False

        return True

    @staticmethod
    def is_symmetric(A: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Verifica se a matriz é simétrica.

        Args:
            A: Matriz a ser verificada
            tolerance: Tolerância para comparação

        Returns:
            True se a matriz é simétrica
        """
        return np.allclose(A, A.T, rtol=tolerance, atol=tolerance)

    @staticmethod
    def is_positive_definite(A: np.ndarray) -> bool:
        """
        Verifica se a matriz é positiva definida.

        Args:
            A: Matriz a ser verificada (deve ser simétrica)

        Returns:
            True se a matriz é positiva definida
        """
        try:
            # Tentar decomposição de Cholesky
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def condition_number(A: np.ndarray) -> float:
        """
        Calcula o número de condição da matriz.

        Args:
            A: Matriz de entrada

        Returns:
            Número de condição
        """
        return np.linalg.cond(A)

    @staticmethod
    def analyze_matrix(A: np.ndarray) -> Dict[str, Any]:
        """
        Análise completa das propriedades da matriz.

        Args:
            A: Matriz a ser analisada

        Returns:
            Dicionário com propriedades da matriz
        """
        analysis = {
            "shape": A.shape,
            "is_square": A.shape[0] == A.shape[1] if A.ndim == 2 else False,
            "rank": np.linalg.matrix_rank(A),
            "determinant": None,
            "condition_number": None,
            "is_symmetric": False,
            "is_positive_definite": False,
            "is_diagonally_dominant": False,
            "is_strictly_diagonally_dominant": False,
            "eigenvalues": None,
            "spectral_radius": None,
        }

        if analysis["is_square"]:
            try:
                analysis["determinant"] = np.linalg.det(A)
                analysis["condition_number"] = MatrixValidator.condition_number(A)
                analysis["is_symmetric"] = MatrixValidator.is_symmetric(A)

                if analysis["is_symmetric"]:
                    analysis[
                        "is_positive_definite"
                    ] = MatrixValidator.is_positive_definite(A)

                analysis[
                    "is_diagonally_dominant"
                ] = MatrixValidator.is_diagonally_dominant(A, strict=False)
                analysis[
                    "is_strictly_diagonally_dominant"
                ] = MatrixValidator.is_diagonally_dominant(A, strict=True)

                # Calcular autovalores
                eigenvals = np.linalg.eigvals(A)
                analysis["eigenvalues"] = eigenvals
                analysis["spectral_radius"] = np.max(np.abs(eigenvals))

            except np.linalg.LinAlgError as e:
                analysis["error"] = f"Erro no cálculo: {str(e)}"

        return analysis
