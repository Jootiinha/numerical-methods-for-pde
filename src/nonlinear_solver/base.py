"""
Classe abstrata base para resolvedores de sistemas não lineares.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, Callable
import numpy as np


class NonLinearSolver(ABC):
    """
    Classe abstrata base para todos os métodos de resolução de sistemas não lineares.
    
    Define a interface comum que todos os resolvedores devem implementar.
    """
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 1000):
        """
        Inicializa o resolvedor.
        
        Args:
            tolerance: Tolerância para critério de convergência (padrão: 1e-6)
            max_iterations: Número máximo de iterações
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.convergence_history = []
        
    @abstractmethod
    def solve(self, system_func: Callable[[np.ndarray], np.ndarray], 
              jacobian_func: Optional[Callable[[np.ndarray], np.ndarray]], 
              x0: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resolve o sistema não linear F(x) = 0.
        
        Args:
            system_func: Função que retorna F(x) para um dado x
            jacobian_func: Função que retorna o jacobiano J(x) (pode ser None)
            x0: Aproximação inicial
            **kwargs: Argumentos específicos do método
            
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
    
    def _check_function_convergence(self, f_val: np.ndarray) -> bool:
        """
        Verifica critério de convergência baseado na norma da função.
        
        Args:
            f_val: Valor da função F(x)
            
        Returns:
            True se converged, False caso contrário
        """
        return np.linalg.norm(f_val) < self.tolerance
    
    def _validate_inputs(self, x0: np.ndarray) -> None:
        """
        Valida as entradas do sistema não linear.
        
        Args:
            x0: Aproximação inicial
            
        Raises:
            ValueError: Se as dimensões não são válidas
        """
        if x0.ndim != 1:
            raise ValueError("x0 deve ser um vetor unidimensional")
    
    def numerical_jacobian(self, system_func: Callable[[np.ndarray], np.ndarray], 
                          x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        """
        Calcula o jacobiano numericamente usando diferenças finitas.
        
        Args:
            system_func: Função do sistema F(x)
            x: Ponto onde calcular o jacobiano
            h: Passo para diferenças finitas
            
        Returns:
            Matriz jacobiana aproximada
        """
        n = len(x)
        f_x = system_func(x)
        m = len(f_x)
        J = np.zeros((m, n))
        
        for j in range(n):
            x_plus = x.copy()
            x_plus[j] += h
            f_plus = system_func(x_plus)
            J[:, j] = (f_plus - f_x) / h
            
        return J
