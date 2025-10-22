"""
Método do Gradiente Conjugado para resolução de sistemas lineares.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from ..base import LinearSolver


class ConjugateGradientSolver(LinearSolver):
    """
    Método do Gradiente Conjugado para resolução de sistemas lineares.
    
    O método é especialmente eficiente para matrizes simétricas e positivas definidas.
    Converge teoricamente em no máximo n passos para um sistema de dimensão n.
    
    Algoritmo:
    1. Inicializar x₀, calcular r₀ = b - Ax₀, p₀ = r₀
    2. Para k = 0, 1, ...:
       - αₖ = (rₖᵀrₖ) / (pₖᵀApₖ)
       - xₖ₊₁ = xₖ + αₖpₖ  
       - rₖ₊₁ = rₖ - αₖApₖ
       - βₖ = (rₖ₊₁ᵀrₖ₊₁) / (rₖᵀrₖ)
       - pₖ₊₁ = rₖ₊₁ + βₖpₖ
    """
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 1000,
                 check_symmetric: bool = True, check_positive_definite: bool = True):
        """
        Inicializa o método do Gradiente Conjugado.
        
        Args:
            tolerance: Tolerância para convergência
            max_iterations: Número máximo de iterações
            check_symmetric: Se deve verificar se a matriz é simétrica
            check_positive_definite: Se deve verificar se a matriz é positiva definida
        """
        super().__init__(tolerance, max_iterations)
        self.check_symmetric = check_symmetric
        self.check_positive_definite = check_positive_definite
        
    def get_method_name(self) -> str:
        return "Gradiente Conjugado"
    
    def _check_matrix_properties(self, A: np.ndarray) -> None:
        """
        Verifica se a matriz tem as propriedades necessárias para o CG.
        
        Args:
            A: Matriz de coeficientes
            
        Raises:
            ValueError: Se a matriz não satisfaz as condições necessárias
        """
        if self.check_symmetric:
            # Verificar simetria
            if not np.allclose(A, A.T, rtol=1e-10, atol=1e-12):
                raise ValueError("Matriz A deve ser simétrica para o Gradiente Conjugado")
        
        if self.check_positive_definite:
            # Verificar se é positiva definida através dos autovalores
            eigenvalues = np.linalg.eigvals(A)
            if not np.all(eigenvalues > 1e-12):
                raise ValueError("Matriz A deve ser positiva definida para o Gradiente Conjugado")
    
    def solve(self, A: np.ndarray, b: np.ndarray, 
              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resolve o sistema usando o método do Gradiente Conjugado.
        
        Args:
            A: Matriz de coeficientes (deve ser simétrica e positiva definida)
            b: Vetor de termos independentes
            x0: Aproximação inicial (padrão: vetor nulo)
            
        Returns:
            Tupla (solução, informações_convergência)
        """
        self._validate_inputs(A, b)
        self._check_matrix_properties(A)
        
        n = A.shape[0]
        x = self._get_initial_guess(A, x0)
        
        # Inicialização do algoritmo CG
        r = b - A @ x  # Resíduo inicial
        p = r.copy()   # Direção de busca inicial
        rsold = np.dot(r, r)  # |r₀|²
        
        self.convergence_history = []
        residual_history = []
        
        for iteration in range(self.max_iterations):
            Ap = A @ p
            
            # Calcular passo α
            pAp = np.dot(p, Ap)
            if abs(pAp) < 1e-15:
                # Direção p é ortogonal à matriz A, pode indicar convergência
                break
                
            alpha = rsold / pAp
            
            # Atualizar solução e resíduo
            x_old = x.copy()
            x = x + alpha * p
            r = r - alpha * Ap
            
            # Calcular nova norma do resíduo
            rsnew = np.dot(r, r)
            residual_norm = np.sqrt(rsnew)
            
            # Verificar convergência baseada no resíduo
            error = np.linalg.norm(x - x_old, ord=np.inf)
            self.convergence_history.append(error)
            residual_history.append(residual_norm)
            
            if residual_norm < self.tolerance:
                info = {
                    'converged': True,
                    'iterations': iteration + 1,
                    'final_error': error,
                    'final_residual': residual_norm,
                    'method': self.get_method_name(),
                    'convergence_history': self.convergence_history.copy(),
                    'residual_history': residual_history.copy()
                }
                return x.copy(), info
            
            # Calcular novo β e atualizar direção de busca
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
        
        # Não convergiu
        final_residual = np.linalg.norm(b - A @ x)
        info = {
            'converged': False,
            'iterations': self.max_iterations,
            'final_error': self.convergence_history[-1] if self.convergence_history else float('inf'),
            'final_residual': final_residual,
            'method': self.get_method_name(),
            'convergence_history': self.convergence_history.copy(),
            'residual_history': residual_history.copy()
        }
        return x.copy(), info
