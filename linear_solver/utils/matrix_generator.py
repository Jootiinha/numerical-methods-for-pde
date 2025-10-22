"""
Gerador de matrizes de teste com propriedades específicas.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path


class MatrixGenerator:
    """
    Gerador de matrizes de teste com propriedades específicas.
    """
    
    @staticmethod
    def diagonally_dominant_matrix(n: int, 
                                 dominance_factor: float = 2.0,
                                 random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera uma matriz diagonalmente dominante e um vetor b.
        
        Args:
            n: Dimensão da matriz
            dominance_factor: Fator de dominância diagonal
            random_seed: Semente para geração aleatória
            
        Returns:
            Tupla (A, b) onde A é diagonalmente dominante
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Gerar elementos off-diagonal
        A = np.random.uniform(-1, 1, (n, n))
        
        # Fazer diagonal dominante
        for i in range(n):
            off_diagonal_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
            A[i, i] = dominance_factor * off_diagonal_sum
            
            # Adicionar sinal aleatório na diagonal
            if np.random.random() > 0.5:
                A[i, i] = -A[i, i]
        
        # Gerar vetor b
        b = np.random.uniform(-10, 10, n)
        
        return A, b
    
    @staticmethod
    def symmetric_positive_definite_matrix(n: int,
                                         condition_number: float = 10.0,
                                         random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera uma matriz simétrica positiva definida.
        
        Args:
            n: Dimensão da matriz
            condition_number: Número de condição desejado
            random_seed: Semente para geração aleatória
            
        Returns:
            Tupla (A, b) onde A é simétrica e positiva definida
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Gerar matriz ortogonal Q
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        
        # Gerar autovalores com número de condição específico
        eigenvals = np.linspace(1.0, condition_number, n)
        
        # Construir matriz A = Q * Λ * Q^T
        A = Q @ np.diag(eigenvals) @ Q.T
        
        # Gerar vetor b
        b = np.random.uniform(-10, 10, n)
        
        return A, b
    
    @staticmethod
    def tridiagonal_matrix(n: int, 
                          diagonal: float = 2.0,
                          off_diagonal: float = -1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera uma matriz tridiagonal.
        
        Args:
            n: Dimensão da matriz
            diagonal: Valor da diagonal principal
            off_diagonal: Valor das diagonais adjacentes
            
        Returns:
            Tupla (A, b) onde A é tridiagonal
        """
        A = np.zeros((n, n))
        
        # Diagonal principal
        np.fill_diagonal(A, diagonal)
        
        # Diagonais adjacentes
        for i in range(n - 1):
            A[i, i + 1] = off_diagonal
            A[i + 1, i] = off_diagonal
        
        # Gerar vetor b que resulte em solução simples
        x_true = np.ones(n)  # Solução verdadeira
        b = A @ x_true
        
        return A, b
    
    @staticmethod
    def ill_conditioned_matrix(n: int,
                              condition_number: float = 1e12,
                              random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera uma matriz mal condicionada para testar estabilidade numérica.
        
        Args:
            n: Dimensão da matriz
            condition_number: Número de condição (alto = mal condicionada)
            random_seed: Semente para geração aleatória
            
        Returns:
            Tupla (A, b) onde A é mal condicionada
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Usar matriz de Hilbert modificada (naturalmente mal condicionada)
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = 1.0 / (i + j + 1)
        
        # Escalar para atingir o número de condição desejado
        current_cond = np.linalg.cond(A)
        scaling_factor = (condition_number / current_cond) ** (1.0 / (2 * n))
        
        # Aplicar perturbação controlada
        U, s, Vt = np.linalg.svd(A)
        s_new = np.linspace(s[0], s[0] / condition_number, n)
        A = U @ np.diag(s_new) @ Vt
        
        # Gerar vetor b
        b = np.random.uniform(-1, 1, n)
        
        return A, b
    
    @staticmethod
    def create_test_suite(output_dir: Optional[str] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Cria um conjunto de matrizes de teste.
        
        Args:
            output_dir: Diretório para salvar as matrizes (opcional)
            
        Returns:
            Dicionário com diferentes tipos de matrizes
        """
        test_matrices = {}
        
        # 1. Matriz diagonalmente dominante pequena
        A1, b1 = MatrixGenerator.diagonally_dominant_matrix(5, 3.0, random_seed=42)
        test_matrices['diagonal_dominant_5x5'] = (A1, b1)
        
        # 2. Matriz simétrica positiva definida
        A2, b2 = MatrixGenerator.symmetric_positive_definite_matrix(4, 50.0, random_seed=42)
        test_matrices['symmetric_pd_4x4'] = (A2, b2)
        
        # 3. Matriz tridiagonal
        A3, b3 = MatrixGenerator.tridiagonal_matrix(6, 4.0, -1.0)
        test_matrices['tridiagonal_6x6'] = (A3, b3)
        
        # 4. Matriz mal condicionada
        A4, b4 = MatrixGenerator.ill_conditioned_matrix(3, 1e6, random_seed=42)
        test_matrices['ill_conditioned_3x3'] = (A4, b4)
        
        # Salvar arquivos se diretório for especificado
        if output_dir:
            from .csv_loader import CSVMatrixLoader
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for name, (A, b) in test_matrices.items():
                filepath = output_path / f"{name}.csv"
                CSVMatrixLoader.save_augmented_matrix(A, b, filepath, add_header=True)
        
        return test_matrices
