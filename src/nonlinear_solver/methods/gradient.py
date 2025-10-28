"""
Método do Gradiente para resolução de sistemas não lineares.
"""

from typing import Tuple, Optional, Dict, Any, Callable
import numpy as np
from ..base import NonLinearSolver


class GradientSolver(NonLinearSolver):
    """
    Método do Gradiente para resolução de sistemas não lineares.
    
    Transforma o problema F(x) = 0 em um problema de otimização:
    minimizar f(x) = (1/2) * ||F(x)||²
    
    O gradiente de f é: ∇f(x) = J^T(x) * F(x)
    A fórmula iterativa é: x_{k+1} = x_k - α_k * ∇f(x_k)
    """
    
    def get_method_name(self) -> str:
        return "Método do Gradiente"
    
    def solve(self, system_func: Callable[[np.ndarray], np.ndarray], 
              jacobian_func: Optional[Callable[[np.ndarray], np.ndarray]], 
              x0: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resolve o sistema usando o método do gradiente.
        
        Args:
            system_func: Função que retorna F(x)
            jacobian_func: Função que retorna J(x). Se None, usa diferenças finitas
            x0: Aproximação inicial
            **kwargs: Argumentos opcionais:
                - step_size: Tamanho do passo inicial (padrão: 0.01)
                - adaptive_step: Se True, usa busca linear (padrão: True)
                - min_step: Tamanho mínimo do passo (padrão: 1e-10)
            
        Returns:
            Tupla (solução, informações_convergência)
        """
        self._validate_inputs(x0)
        
        x = x0.copy()
        self.convergence_history = []
        residual_history = []
        step_size_history = []
        
        # Parâmetros do método
        initial_step = kwargs.get('step_size', 0.01)
        adaptive_step = kwargs.get('adaptive_step', True)
        min_step = kwargs.get('min_step', 1e-10)
        
        # Se jacobiano não fornecido, usar diferenças finitas
        if jacobian_func is None:
            jacobian_func = lambda x_val: self.numerical_jacobian(system_func, x_val)
        
        for iteration in range(self.max_iterations):
            # Avaliar função e jacobiano
            f_val = system_func(x)
            J = jacobian_func(x)
            
            # Calcular função objetivo e gradiente
            objective = 0.5 * np.dot(f_val, f_val)
            gradient = J.T @ f_val
            
            # Calcular normas
            f_norm = np.linalg.norm(f_val)
            grad_norm = np.linalg.norm(gradient)
            
            residual_history.append(f_norm)
            
            # Verificar convergência da função
            if self._check_function_convergence(f_val):
                info = {
                    'converged': True,
                    'iterations': iteration + 1,
                    'final_function_norm': f_norm,
                    'final_gradient_norm': grad_norm,
                    'final_objective': objective,
                    'method': self.get_method_name(),
                    'convergence_history': self.convergence_history.copy(),
                    'residual_history': residual_history.copy(),
                    'step_size_history': step_size_history.copy(),
                    'solution_x': x.copy()
                }
                return x.copy(), info
            
            # Verificar convergência do gradiente
            if grad_norm < self.tolerance:
                info = {
                    'converged': True,
                    'iterations': iteration + 1,
                    'final_function_norm': f_norm,
                    'final_gradient_norm': grad_norm,
                    'final_objective': objective,
                    'method': self.get_method_name(),
                    'convergence_history': self.convergence_history.copy(),
                    'residual_history': residual_history.copy(),
                    'step_size_history': step_size_history.copy(),
                    'solution_x': x.copy()
                }
                return x.copy(), info
            
            # Determinar tamanho do passo
            if adaptive_step:
                step_size = self._line_search(system_func, x, gradient, initial_step)
                if step_size < min_step:
                    info = {
                        'converged': False,
                        'iterations': iteration + 1,
                        'final_function_norm': f_norm,
                        'final_gradient_norm': grad_norm,
                        'final_objective': objective,
                        'method': self.get_method_name(),
                        'convergence_history': self.convergence_history.copy(),
                        'residual_history': residual_history.copy(),
                        'step_size_history': step_size_history.copy(),
                        'error_message': 'Passo muito pequeno - possível mínimo local',
                        'solution_x': x.copy()
                    }
                    return x.copy(), info
            else:
                step_size = initial_step
            
            step_size_history.append(step_size)
            
            # Atualizar solução
            x_new = x - step_size * gradient
            
            # Calcular norma do passo
            step_norm = step_size * grad_norm
            self.convergence_history.append(step_norm)
            
            # Verificar convergência do passo
            if self._check_convergence(x_new, x):
                info = {
                    'converged': True,
                    'iterations': iteration + 1,
                    'final_function_norm': f_norm,
                    'final_gradient_norm': grad_norm,
                    'final_objective': objective,
                    'method': self.get_method_name(),
                    'convergence_history': self.convergence_history.copy(),
                    'residual_history': residual_history.copy(),
                    'step_size_history': step_size_history.copy(),
                    'solution_x': x_new.copy()
                }
                return x_new.copy(), info
            
            x = x_new
        
        # Não convergiu
        final_f_val = system_func(x)
        final_f_norm = np.linalg.norm(final_f_val)
        final_grad = jacobian_func(x).T @ final_f_val
        final_grad_norm = np.linalg.norm(final_grad)
        final_objective = 0.5 * np.dot(final_f_val, final_f_val)
        
        info = {
            'converged': False,
            'iterations': self.max_iterations,
            'final_function_norm': final_f_norm,
            'final_gradient_norm': final_grad_norm,
            'final_objective': final_objective,
            'method': self.get_method_name(),
            'convergence_history': self.convergence_history.copy(),
            'residual_history': residual_history.copy(),
            'step_size_history': step_size_history.copy(),
            'solution_x': x.copy()
        }
        return x.copy(), info
    
    def _line_search(self, system_func: Callable[[np.ndarray], np.ndarray], 
                    x: np.ndarray, gradient: np.ndarray, 
                    initial_step: float) -> float:
        """
        Busca linear simples usando backtracking.
        
        Args:
            system_func: Função do sistema F(x)
            x: Ponto atual
            gradient: Gradiente atual
            initial_step: Passo inicial
            
        Returns:
            Tamanho do passo otimizado
        """
        # Parâmetros da busca linear
        c1 = 1e-4  # Parâmetro de Armijo
        rho = 0.5  # Fator de redução
        max_backtracks = 20
        
        # Valor atual da função objetivo
        f_current = system_func(x)
        phi_current = 0.5 * np.dot(f_current, f_current)
        
        # Derivada direcional
        dphi = np.dot(gradient, -gradient)  # direção = -gradient
        
        step = initial_step
        for _ in range(max_backtracks):
            x_new = x - step * gradient
            f_new = system_func(x_new)
            phi_new = 0.5 * np.dot(f_new, f_new)
            
            # Condição de Armijo
            if phi_new <= phi_current + c1 * step * dphi:
                return step
            
            step *= rho
        
        return step
