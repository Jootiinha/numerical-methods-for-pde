"""
Configuração e fixtures compartilhadas para os testes.
"""

import pytest
import numpy as np
from typing import Tuple


@pytest.fixture
def simple_system() -> Tuple[np.ndarray, np.ndarray]:
    """Sistema linear simples para testes."""
    A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]], dtype=float)
    b = np.array([3, 2, 3], dtype=float)
    return A, b


@pytest.fixture
def diagonal_dominant_system() -> Tuple[np.ndarray, np.ndarray]:
    """Sistema com matriz diagonalmente dominante."""
    A = np.array([[10, 1, 1], [1, 10, 1], [1, 1, 10]], dtype=float)
    b = np.array([12, 12, 12], dtype=float)
    return A, b


@pytest.fixture
def symmetric_positive_definite_system() -> Tuple[np.ndarray, np.ndarray]:
    """Sistema com matriz simétrica e positiva definida."""
    A = np.array([[4, 1, 0], [1, 4, 1], [0, 1, 4]], dtype=float)
    b = np.array([5, 6, 5], dtype=float)
    return A, b


@pytest.fixture
def ill_conditioned_system() -> Tuple[np.ndarray, np.ndarray]:
    """Sistema mal condicionado para testes de estabilidade."""
    # Matriz de Hilbert 3x3 (mal condicionada)
    n = 3
    A = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
    b = np.ones(n)
    return A, b


@pytest.fixture
def non_convergent_system() -> Tuple[np.ndarray, np.ndarray]:
    """Sistema que pode não convergir com métodos iterativos."""
    A = np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1]], dtype=float)
    b = np.array([6, 7, 8], dtype=float)
    return A, b
