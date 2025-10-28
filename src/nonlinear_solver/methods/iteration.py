"""
Método da Iteração (Ponto Fixo) para resolução de sistemas não lineares.
"""

from typing import Tuple, Optional, Dict, Any, Callable
import numpy as np
from ..base import NonLinearSolver


class IterationSolver(NonLinearSolver):
    """
    Método da Iteração de Ponto Fixo para resolução de sistemas não lineares.
    
    Transforma o problema F(x) = 0 em um problema de ponto fixo x = G(x).
    A fórmula iterativa é: x_{k+1} = G(x_k)
    
    Para sistemas F(x) = 0, pode-se usar G(x) = x - α * F(x), onde α é um parâmetro.
    """
    
    def get_method_name(self) -> str:
        return "Iteração de Ponto Fixo"
    
    def solve(self, system_func: Callable[[np.ndarray], np.ndarray], 
              jacobian_func: Optional[Callable[[np.ndarray], np.ndarray]], 
              x0: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resolve o sistema usando o método da iteração.
        
        Args:
            system_func: Função que retorna F(x)
            jacobian_func: Não utilizado neste método
            x0: Aproximação inicial
            **kwargs: Argumentos opcionais:
                - alpha: Parâmetro de relaxação (padrão: 0.1)
                - fixed_point_func: Função G(x) personalizada (se fornecida, ignora alpha)
            
        Returns:
            Tupla (solução, informações_convergência)
        """
        self._validate_inputs(x0)
        
        x = x0.copy()
        self.convergence_history = []
        residual_history = []
        
        # Parâmetros do método
        alpha = kwargs.get('alpha', 0.1)
        fixed_point_func = kwargs.get('fixed_point_func', None)
        
        # Se função de ponto fixo não fornecida, usar G(x) = x - alpha * F(x)
        if fixed_point_func is None:
            def G(x_val):
                return x_val - alpha * system_func(x_val)
        else:
            G = fixed_point_func
        
        for iteration in range(self.max_iterations):
            # Avaliar função original para verificar convergência
            f_val = system_func(x)
            f_norm = np.linalg.norm(f_val)
            residual_history.append(f_norm)
            
            # Verificar convergência da função
            if self._check_function_convergence(f_val):
                info = {
                    'converged': True,
                    'iterations': iteration + 1,
                    'final_function_norm': f_norm,
                    'final_step_norm': 0.0 if iteration == 0 else self.convergence_history[-1],
                    'method': self.get_method_name(),
                    'convergence_history': self.convergence_history.copy(),
                    'residual_history': residual_history.copy(),
                    'solution_x': x.copy(),
                    'alpha': alpha
                }
                return x.copy(), info
            
            # Aplicar iteração de ponto fixo
            try:
                x_new = G(x)
            except Exception as e:
                info = {
                    'converged': False,
                    'iterations': iteration + 1,
                    'final_function_norm': f_norm,
                    'final_step_norm': np.inf,
                    'method': self.get_method_name(),
                    'convergence_history': self.convergence_history.copy(),
                    'residual_history': residual_history.copy(),
                    'error_message': f'Erro na função de iteração: {str(e)}',
                    'solution_x': x.copy(),
                    'alpha': alpha
                }
                return x.copy(), info
            
            # Calcular norma do passo
            step_norm = np.linalg.norm(x_new - x)
            self.convergence_history.append(step_norm)
            
            # Verificar convergência do passo
            if self._check_convergence(x_new, x):
                info = {
                    'converged': True,
                    'iterations': iteration + 1,
                    'final_function_norm': f_norm,
                    'final_step_norm': step_norm,
                    'method': self.get_method_name(),
                    'convergence_history': self.convergence_history.copy(),
                    'residual_history': residual_history.copy(),
                    'solution_x': x_new.copy(),
                    'alpha': alpha
                }
                return x_new.copy(), info
            
            # Verificar divergência
            if step_norm > 1e10:
                info = {
                    'converged': False,
                    'iterations': iteration + 1,
                    'final_function_norm': f_norm,
                    'final_step_norm': step_norm,
                    'method': self.get_method_name(),
                    'convergence_history': self.convergence_history.copy(),
                    'residual_history': residual_history.copy(),
                    'error_message': 'Método divergiu (passo muito grande)',
                    'solution_x': x.copy(),
                    'alpha': alpha
                }
                return x.copy(), info
            
            x = x_new
        
        # Não convergiu
        final_f_val = system_func(x)
        final_f_norm = np.linalg.norm(final_f_val)
        
        info = {
            'converged': False,
            'iterations': self.max_iterations,
            'final_function_norm': final_f_norm,
            'final_step_norm': self.convergence_history[-1] if self.convergence_history else np.inf,
            'method': self.get_method_name(),
            'convergence_history': self.convergence_history.copy(),
            'residual_history': residual_history.copy(),
            'solution_x': x.copy(),
            'alpha': alpha
        }
        return x.copy(), info
