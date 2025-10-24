"""
Método de Newton-Raphson para resolução de sistemas não lineares.
"""

from typing import Tuple, Optional, Dict, Any, Callable
import numpy as np
from ..base import NonLinearSolver


class NewtonSolver(NonLinearSolver):
    """
    Método de Newton-Raphson para resolução de sistemas não lineares.
    
    O método de Newton resolve F(x) = 0 através da fórmula iterativa:
    x_{k+1} = x_k - J(x_k)^{-1} * F(x_k)
    
    onde J(x_k) é o jacobiano de F no ponto x_k.
    """
    
    def get_method_name(self) -> str:
        return "Newton-Raphson"
    
    def solve(self, system_func: Callable[[np.ndarray], np.ndarray], 
              jacobian_func: Optional[Callable[[np.ndarray], np.ndarray]], 
              x0: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resolve o sistema usando o método de Newton-Raphson.
        
        Args:
            system_func: Função que retorna F(x)
            jacobian_func: Função que retorna J(x). Se None, usa diferenças finitas
            x0: Aproximação inicial
            **kwargs: Argumentos adicionais (não utilizados neste método)
            
        Returns:
            Tupla (solução, informações_convergência)
        """
        self._validate_inputs(x0)
        
        x = x0.copy()
        self.convergence_history = []
        residual_history = []
        
        # Se jacobiano não fornecido, usar diferenças finitas
        if jacobian_func is None:
            jacobian_func = lambda x_val: self.numerical_jacobian(system_func, x_val)
        
        for iteration in range(self.max_iterations):
            # Avaliar função e jacobiano
            f_val = system_func(x)
            J = jacobian_func(x)
            
            # Calcular erro da função
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
                    'solution_x': x.copy()
                }
                return x.copy(), info
            
            try:
                # Resolver sistema linear J * delta_x = -f_val
                delta_x = np.linalg.solve(J, -f_val)
            except np.linalg.LinAlgError:
                # Jacobiano singular - tentar com pseudo-inversa
                try:
                    delta_x = -np.linalg.pinv(J) @ f_val
                except:
                    info = {
                        'converged': False,
                        'iterations': iteration + 1,
                        'final_function_norm': f_norm,
                        'final_step_norm': np.inf,
                        'method': self.get_method_name(),
                        'convergence_history': self.convergence_history.copy(),
                        'residual_history': residual_history.copy(),
                        'error_message': 'Jacobiano singular',
                        'solution_x': x.copy()
                    }
                    return x.copy(), info
            
            # Atualizar solução
            x_new = x + delta_x
            
            # Calcular norma do passo
            step_norm = np.linalg.norm(delta_x)
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
                    'solution_x': x_new.copy()
                }
                return x_new.copy(), info
            
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
            'solution_x': x.copy()
        }
        return x.copy(), info
