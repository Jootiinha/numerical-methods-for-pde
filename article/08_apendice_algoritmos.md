# APÊNDICE - ALGORITMOS COMPUTACIONAIS

## A.1. Métodos para Sistemas Lineares

### A.1.1. Método de Jacobi

```python
class JacobiSolver(LinearSolver):
    """
    Método iterativo de Jacobi para resolução de sistemas lineares.
    
    Fórmula iterativa: x^(k+1) = D^(-1) * (b - (L + U) * x^(k))
    onde A = D + L + U (D diagonal, L triangular inferior, U triangular superior)
    """
    
    def __init__(self, tolerance=1e-4, max_iterations=1000, 
                 omega1=1.0, omega2=0.0, omega3=0.0):
        super().__init__(tolerance, max_iterations)
        self.omega1 = omega1
        self.omega2 = omega2
        self.omega3 = omega3
        self.is_order2 = not (omega1 == 1.0 and omega2 == 0.0 and omega3 == 0.0)
    
    def solve(self, A, b, x0=None):
        """
        Resolve o sistema Ax = b usando o método de Jacobi.
        
        Args:
            A: Matriz de coeficientes (n x n)
            b: Vetor de termos independentes (n,)
            x0: Aproximação inicial (opcional)
            
        Returns:
            Tupla (solução, informações_convergência)
        """
        self._validate_inputs(A, b)
        
        # Verificar diagonal não-nula
        diagonal_values = np.diag(A)
        if np.any(np.abs(diagonal_values) < 1e-14):
            raise ValueError("Matriz A deve ter diagonal principal não-nula")
        
        # Inicialização
        x = self._get_initial_guess(A, x0)
        x_prev = x.copy()
        self.convergence_history = []
        residual_history = []
        
        # Matriz L+U (off-diagonal)
        L_plus_U = A - np.diag(diagonal_values)
        
        # Iteração principal
        for iteration in range(self.max_iterations):
            # Fórmula de Jacobi vetorizada
            x_jacobi = (b - (L_plus_U @ x)) / diagonal_values
            
            # Aplicar pesos para ordem 2 se configurado
            if self.is_order2:
                if iteration == 0:
                    x_new = x_jacobi
                else:
                    x_new = (self.omega1 * x_jacobi + 
                            self.omega2 * x + 
                            self.omega3 * x_prev)
            else:
                x_new = x_jacobi
            
            # Calcular erros
            error = float(np.linalg.norm(x_new - x, ord=np.inf))
            residual = float(np.linalg.norm(A @ x_new - b))
            
            self.convergence_history.append(error)
            residual_history.append(residual)
            
            # Verificar convergência
            if self._check_convergence(x_new, x):
                return self._create_convergence_info(
                    converged=True,
                    iterations=iteration + 1,
                    solution=x_new,
                    final_error=error,
                    final_residual=residual,
                    residual_history=residual_history
                )
            
            # Atualizar para próxima iteração
            x_prev, x = x, x_new
        
        # Não convergiu
        final_residual = float(np.linalg.norm(A @ x - b))
        return self._create_convergence_info(
            converged=False,
            iterations=self.max_iterations,
            solution=x,
            final_error=self.convergence_history[-1],
            final_residual=final_residual,
            residual_history=residual_history
        )
```

### A.1.2. Método de Gauss-Seidel

```python
class GaussSeidelSolver(LinearSolver):
    """
    Método iterativo de Gauss-Seidel (e SOR) para sistemas lineares.
    
    Fórmula iterativa: x_i^(k+1) = (1-ω)x_i^(k) + ω/D_ii * (b_i - Σ A_ij*x_j^(k+1) - Σ A_ij*x_j^(k))
    onde ω é o fator de relaxação (ω=1 é Gauss-Seidel puro)
    """
    
    def __init__(self, tolerance=1e-4, max_iterations=1000, 
                 relaxation_factor=1.0, omega1=1.0, omega2=0.0, omega3=0.0):
        super().__init__(tolerance, max_iterations)
        if not 0 < relaxation_factor < 2:
            raise ValueError("Fator de relaxação (ω) deve estar em (0, 2)")
        
        self.relaxation_factor = relaxation_factor
        self.omega1 = omega1
        self.omega2 = omega2
        self.omega3 = omega3
        self.is_sor = relaxation_factor != 1.0
        self.is_order2 = not (omega1 == 1.0 and omega2 == 0.0 and omega3 == 0.0)
    
    def solve(self, A, b, x0=None):
        """
        Resolve o sistema Ax = b usando Gauss-Seidel/SOR.
        """
        self._validate_inputs(A, b)
        
        # Verificar diagonal não-nula
        diagonal = np.diag(A)
        if np.any(np.abs(diagonal) < 1e-14):
            raise ValueError("Matriz A deve ter diagonal principal não-nula")
        
        n = A.shape[0]
        x = self._get_initial_guess(A, x0)
        x_prev = x.copy()
        
        self.convergence_history = []
        residual_history = []
        
        # Iteração principal
        for iteration in range(self.max_iterations):
            x_old = x.copy()
            x_sor = x.copy()
            
            # Loop principal do Gauss-Seidel/SOR
            for i in range(n):
                # Soma dos termos já atualizados (triangular inferior)
                sum_lower = np.dot(A[i, :i], x_sor[:i])
                # Soma dos termos ainda não atualizados (triangular superior)
                sum_upper = np.dot(A[i, i+1:], x_old[i+1:])
                
                # Fórmula de Gauss-Seidel
                x_gs = (b[i] - sum_lower - sum_upper) / A[i, i]
                
                # Aplicar relaxação (SOR)
                x_sor[i] = ((1 - self.relaxation_factor) * x_old[i] + 
                           self.relaxation_factor * x_gs)
            
            # Aplicar pesos para ordem 2 se configurado
            if self.is_order2:
                if iteration == 0:
                    x_new = x_sor
                else:
                    x_new = (self.omega1 * x_sor + 
                            self.omega2 * x + 
                            self.omega3 * x_prev)
            else:
                x_new = x_sor
            
            # Calcular erros
            error = float(np.linalg.norm(x_new - x, ord=np.inf))
            residual = float(np.linalg.norm(A @ x_new - b))
            
            self.convergence_history.append(error)
            residual_history.append(residual)
            
            # Verificar convergência
            if self._check_convergence(x_new, x):
                return self._create_convergence_info(
                    converged=True,
                    iterations=iteration + 1,
                    solution=x_new,
                    final_error=error,
                    final_residual=residual,
                    residual_history=residual_history
                )
            
            # Atualizar para próxima iteração
            x_prev, x = x, x_new
        
        # Não convergiu
        final_residual = float(np.linalg.norm(A @ x - b))
        return self._create_convergence_info(
            converged=False,
            iterations=self.max_iterations,
            solution=x,
            final_error=self.convergence_history[-1],
            final_residual=final_residual,
            residual_history=residual_history
        )
```

### A.1.3. Método do Gradiente Conjugado

```python
class ConjugateGradientSolver(LinearSolver):
    """
    Método do Gradiente Conjugado para sistemas simétricos e definidos positivos.
    
    Algoritmo:
    1. r_0 = b - A*x_0, p_0 = r_0
    2. Para k = 0, 1, 2, ...
       α_k = (r_k^T * r_k) / (p_k^T * A * p_k)
       x_{k+1} = x_k + α_k * p_k
       r_{k+1} = r_k - α_k * A * p_k
       β_k = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)
       p_{k+1} = r_{k+1} + β_k * p_k
    """
    
    def get_method_name(self):
        return "Gradiente Conjugado"
    
    def solve(self, A, b, x0=None):
        """
        Resolve o sistema Ax = b usando o método do Gradiente Conjugado.
        """
        self._validate_inputs(A, b)
        
        # Verificar se A é simétrica
        if not np.allclose(A, A.T, rtol=1e-10):
            raise ValueError("Matriz A deve ser simétrica para o método do Gradiente Conjugado")
        
        n = A.shape[0]
        x = self._get_initial_guess(A, x0)
        
        self.convergence_history = []
        residual_history = []
        
        # Inicialização
        r = b - A @ x  # Resíduo inicial
        p = r.copy()   # Direção de busca inicial
        
        # Iteração principal
        for iteration in range(self.max_iterations):
            # Calcular norma do resíduo
            r_norm_squared = np.dot(r, r)
            residual_history.append(np.sqrt(r_norm_squared))
            
            # Verificar convergência
            if np.sqrt(r_norm_squared) < self.tolerance:
                return self._create_convergence_info(
                    converged=True,
                    iterations=iteration + 1,
                    solution=x,
                    final_error=0.0,
                    final_residual=np.sqrt(r_norm_squared),
                    residual_history=residual_history
                )
            
            # Calcular passo α_k
            Ap = A @ p
            alpha = r_norm_squared / np.dot(p, Ap)
            
            # Atualizar solução
            x_new = x + alpha * p
            
            # Calcular novo resíduo
            r_new = r - alpha * Ap
            
            # Calcular erro
            error = float(np.linalg.norm(x_new - x, ord=np.inf))
            self.convergence_history.append(error)
            
            # Calcular β_k
            r_new_norm_squared = np.dot(r_new, r_new)
            beta = r_new_norm_squared / r_norm_squared
            
            # Atualizar direção de busca
            p = r_new + beta * p
            
            # Atualizar variáveis
            x = x_new
            r = r_new
        
        # Não convergiu
        final_residual = float(np.linalg.norm(b - A @ x))
        return self._create_convergence_info(
            converged=False,
            iterations=self.max_iterations,
            solution=x,
            final_error=self.convergence_history[-1],
            final_residual=final_residual,
            residual_history=residual_history
        )
```

## A.2. Métodos para Sistemas Não Lineares

### A.2.1. Método de Newton-Raphson

```python
class NewtonSolver(NonLinearSolver):
    """
    Método de Newton-Raphson para resolução de sistemas não lineares.
    
    Fórmula iterativa: x_{k+1} = x_k - J(x_k)^{-1} * F(x_k)
    onde J(x_k) é o jacobiano de F no ponto x_k.
    """
    
    def get_method_name(self):
        return "Newton-Raphson"
    
    def solve(self, system_func, jacobian_func, x0, **kwargs):
        """
        Resolve o sistema F(x) = 0 usando o método de Newton-Raphson.
        
        Args:
            system_func: Função que retorna F(x)
            jacobian_func: Função que retorna J(x). Se None, usa diferenças finitas
            x0: Aproximação inicial
            
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
        
        # Iteração principal
        for iteration in range(self.max_iterations):
            # Avaliar função e jacobiano
            f_val = system_func(x)
            J = jacobian_func(x)
            
            # Calcular norma da função
            f_norm = np.linalg.norm(f_val)
            residual_history.append(f_norm)
            
            # Verificar convergência da função
            if self._check_function_convergence(f_val):
                return self._create_convergence_info(
                    converged=True,
                    iterations=iteration + 1,
                    solution=x,
                    final_function_norm=f_norm,
                    final_step_norm=0.0 if iteration == 0 else self.convergence_history[-1]
                )
            
            try:
                # Resolver sistema linear J * delta_x = -f_val
                delta_x = np.linalg.solve(J, -f_val)
            except np.linalg.LinAlgError:
                # Jacobiano singular - tentar com pseudo-inversa
                try:
                    delta_x = -np.linalg.pinv(J) @ f_val
                except:
                    return self._create_convergence_info(
                        converged=False,
                        iterations=iteration + 1,
                        solution=x,
                        final_function_norm=f_norm,
                        final_step_norm=np.inf,
                        error_message='Jacobiano singular'
                    )
            
            # Atualizar solução
            x_new = x + delta_x
            
            # Calcular norma do passo
            step_norm = np.linalg.norm(delta_x)
            self.convergence_history.append(step_norm)
            
            # Verificar convergência do passo
            if self._check_convergence(x_new, x):
                return self._create_convergence_info(
                    converged=True,
                    iterations=iteration + 1,
                    solution=x_new,
                    final_function_norm=f_norm,
                    final_step_norm=step_norm
                )
            
            x = x_new
        
        # Não convergiu
        final_f_val = system_func(x)
        final_f_norm = np.linalg.norm(final_f_val)
        
        return self._create_convergence_info(
            converged=False,
            iterations=self.max_iterations,
            solution=x,
            final_function_norm=final_f_norm,
            final_step_norm=self.convergence_history[-1] if self.convergence_history else np.inf
        )
    
    def numerical_jacobian(self, system_func, x, h=1e-8):
        """
        Calcula o jacobiano usando diferenças finitas.
        
        Args:
            system_func: Função do sistema F(x)
            x: Ponto onde calcular o jacobiano
            h: Tamanho do passo para diferenças finitas
            
        Returns:
            Matriz jacobiana J(x)
        """
        n = len(x)
        f_val = system_func(x)
        m = len(f_val)
        
        J = np.zeros((m, n))
        
        for j in range(n):
            x_plus_h = x.copy()
            x_plus_h[j] += h
            
            f_plus_h = system_func(x_plus_h)
            J[:, j] = (f_plus_h - f_val) / h
        
        return J
```

### A.2.2. Método da Iteração de Ponto Fixo

```python
class IterationSolver(NonLinearSolver):
    """
    Método da Iteração de Ponto Fixo para sistemas não lineares.
    
    Fórmula iterativa: x_{k+1} = x_k - α * F(x_k)
    onde α é um parâmetro de relaxação.
    """
    
    def get_method_name(self):
        return "Iteração de Ponto Fixo"
    
    def solve(self, system_func, jacobian_func, x0, alpha=0.1, **kwargs):
        """
        Resolve o sistema F(x) = 0 usando iteração de ponto fixo.
        
        Args:
            system_func: Função que retorna F(x)
            jacobian_func: Função jacobiana (não utilizada neste método)
            x0: Aproximação inicial
            alpha: Parâmetro de relaxação
            
        Returns:
            Tupla (solução, informações_convergência)
        """
        self._validate_inputs(x0)
        
        x = x0.copy()
        self.convergence_history = []
        residual_history = []
        
        # Iteração principal
        for iteration in range(self.max_iterations):
            # Avaliar função
            f_val = system_func(x)
            f_norm = np.linalg.norm(f_val)
            residual_history.append(f_norm)
            
            # Verificar convergência da função
            if self._check_function_convergence(f_val):
                return self._create_convergence_info(
                    converged=True,
                    iterations=iteration + 1,
                    solution=x,
                    final_function_norm=f_norm,
                    final_step_norm=0.0 if iteration == 0 else self.convergence_history[-1]
                )
            
            # Atualizar solução: x_{k+1} = x_k - α * F(x_k)
            x_new = x - alpha * f_val
            
            # Calcular norma do passo
            step_norm = np.linalg.norm(x_new - x)
            self.convergence_history.append(step_norm)
            
            # Verificar convergência do passo
            if self._check_convergence(x_new, x):
                return self._create_convergence_info(
                    converged=True,
                    iterations=iteration + 1,
                    solution=x_new,
                    final_function_norm=f_norm,
                    final_step_norm=step_norm
                )
            
            x = x_new
        
        # Não convergiu
        final_f_val = system_func(x)
        final_f_norm = np.linalg.norm(final_f_val)
        
        return self._create_convergence_info(
            converged=False,
            iterations=self.max_iterations,
            solution=x,
            final_function_norm=final_f_norm,
            final_step_norm=self.convergence_history[-1] if self.convergence_history else np.inf
        )
```

### A.2.3. Método do Gradiente

```python
class GradientSolver(NonLinearSolver):
    """
    Método do Gradiente para sistemas não lineares.
    
    Minimiza a função objetivo: g(x) = (1/2) * ||F(x)||²
    Fórmula iterativa: x_{k+1} = x_k - α_k * ∇g(x_k)
    onde ∇g(x) = J(x)^T * F(x)
    """
    
    def get_method_name(self):
        return "Gradiente"
    
    def solve(self, system_func, jacobian_func, x0, **kwargs):
        """
        Resolve o sistema F(x) = 0 usando o método do gradiente.
        
        Args:
            system_func: Função que retorna F(x)
            jacobian_func: Função que retorna J(x)
            x0: Aproximação inicial
            
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
        
        # Iteração principal
        for iteration in range(self.max_iterations):
            # Avaliar função e jacobiano
            f_val = system_func(x)
            J = jacobian_func(x)
            
            # Calcular norma da função
            f_norm = np.linalg.norm(f_val)
            residual_history.append(f_norm)
            
            # Verificar convergência da função
            if self._check_function_convergence(f_val):
                return self._create_convergence_info(
                    converged=True,
                    iterations=iteration + 1,
                    solution=x,
                    final_function_norm=f_norm,
                    final_step_norm=0.0 if iteration == 0 else self.convergence_history[-1]
                )
            
            # Calcular gradiente: ∇g(x) = J(x)^T * F(x)
            gradient = J.T @ f_val
            
            # Busca linear para determinar tamanho do passo
            alpha = self.line_search(system_func, x, gradient)
            
            # Atualizar solução
            x_new = x - alpha * gradient
            
            # Calcular norma do passo
            step_norm = np.linalg.norm(x_new - x)
            self.convergence_history.append(step_norm)
            
            # Verificar convergência do passo
            if self._check_convergence(x_new, x):
                return self._create_convergence_info(
                    converged=True,
                    iterations=iteration + 1,
                    solution=x_new,
                    final_function_norm=f_norm,
                    final_step_norm=step_norm
                )
            
            x = x_new
        
        # Não convergiu
        final_f_val = system_func(x)
        final_f_norm = np.linalg.norm(final_f_val)
        
        return self._create_convergence_info(
            converged=False,
            iterations=self.max_iterations,
            solution=x,
            final_function_norm=final_f_norm,
            final_step_norm=self.convergence_history[-1] if self.convergence_history else np.inf
        )
    
    def line_search(self, system_func, x, gradient, alpha0=1.0, c1=1e-4, c2=0.9):
        """
        Busca linear usando condições de Wolfe.
        
        Args:
            system_func: Função do sistema F(x)
            x: Ponto atual
            gradient: Gradiente no ponto atual
            alpha0: Tamanho inicial do passo
            c1: Constante para condição de Armijo
            c2: Constante para condição de curvatura
            
        Returns:
            Tamanho ótimo do passo
        """
        alpha = alpha0
        f_val = system_func(x)
        f_norm_squared = np.dot(f_val, f_val)
        
        # Condição de Armijo: f(x + α*d) ≤ f(x) + c1*α*∇f(x)^T*d
        armijo_rhs = f_norm_squared + c1 * alpha * np.dot(gradient, gradient)
        
        for _ in range(20):  # Máximo 20 tentativas
            x_new = x - alpha * gradient
            f_new_val = system_func(x_new)
            f_new_norm_squared = np.dot(f_new_val, f_new_val)
            
            if f_new_norm_squared <= armijo_rhs:
                return alpha
            
            alpha *= 0.5
        
        return alpha0  # Retornar valor inicial se busca falhar
```

## A.3. Classes Base e Utilitários

### A.3.1. Classe Base para Solucionadores Lineares

```python
class LinearSolver:
    """
    Classe base para solucionadores de sistemas lineares.
    """
    
    def __init__(self, tolerance=1e-4, max_iterations=1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.convergence_history = []
    
    def _validate_inputs(self, A, b):
        """Valida as entradas do sistema."""
        if not isinstance(A, np.ndarray) or A.ndim != 2:
            raise ValueError("A deve ser uma matriz 2D")
        if not isinstance(b, np.ndarray) or b.ndim != 1:
            raise ValueError("b deve ser um vetor 1D")
        if A.shape[0] != A.shape[1]:
            raise ValueError("A deve ser quadrada")
        if A.shape[0] != b.shape[0]:
            raise ValueError("Dimensões de A e b devem ser compatíveis")
    
    def _get_initial_guess(self, A, x0):
        """Obtém aproximação inicial."""
        if x0 is not None:
            return x0.copy()
        return np.zeros(A.shape[0])
    
    def _check_convergence(self, x_new, x_old):
        """Verifica critérios de convergência."""
        # Critério de incremento relativo
        if np.linalg.norm(x_new) > 0:
            relative_error = np.linalg.norm(x_new - x_old) / np.linalg.norm(x_new)
            if relative_error < self.tolerance:
                return True
        
        # Critério de incremento absoluto
        absolute_error = np.linalg.norm(x_new - x_old)
        if absolute_error < self.tolerance:
            return True
        
        return False
    
    def _create_convergence_info(self, converged, iterations, solution, 
                                final_error, final_residual, residual_history):
        """Cria dicionário com informações de convergência."""
        return solution, {
            'converged': converged,
            'iterations': iterations,
            'final_error': final_error,
            'final_residual': final_residual,
            'method': self.get_method_name(),
            'convergence_history': self.convergence_history.copy(),
            'residual_history': residual_history.copy()
        }
    
    def get_method_name(self):
        """Retorna nome do método (deve ser implementado pelas subclasses)."""
        raise NotImplementedError
```

### A.3.2. Classe Base para Solucionadores Não Lineares

```python
class NonLinearSolver:
    """
    Classe base para solucionadores de sistemas não lineares.
    """
    
    def __init__(self, tolerance=1e-6, max_iterations=1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.convergence_history = []
    
    def _validate_inputs(self, x0):
        """Valida a aproximação inicial."""
        if not isinstance(x0, np.ndarray) or x0.ndim != 1:
            raise ValueError("x0 deve ser um vetor 1D")
    
    def _check_function_convergence(self, f_val):
        """Verifica convergência baseada na norma da função."""
        return np.linalg.norm(f_val) < self.tolerance
    
    def _check_convergence(self, x_new, x_old):
        """Verifica critérios de convergência."""
        # Critério de incremento relativo
        if np.linalg.norm(x_new) > 0:
            relative_error = np.linalg.norm(x_new - x_old) / np.linalg.norm(x_new)
            if relative_error < self.tolerance:
                return True
        
        # Critério de incremento absoluto
        absolute_error = np.linalg.norm(x_new - x_old)
        if absolute_error < self.tolerance:
            return True
        
        return False
    
    def _create_convergence_info(self, converged, iterations, solution,
                                final_function_norm, final_step_norm, **kwargs):
        """Cria dicionário com informações de convergência."""
        info = {
            'converged': converged,
            'iterations': iterations,
            'final_function_norm': final_function_norm,
            'final_step_norm': final_step_norm,
            'method': self.get_method_name(),
            'convergence_history': self.convergence_history.copy(),
            'solution_x': solution.copy()
        }
        info.update(kwargs)
        return solution, info
    
    def get_method_name(self):
        """Retorna nome do método (deve ser implementado pelas subclasses)."""
        raise NotImplementedError
```

## A.4. Exemplo de Sistema Não Linear Implementado

### A.4.1. Definição do Sistema

```python
class NonLinearSystemExample:
    """
    Exemplo de sistema não linear tridimensional:
    F₁: (x-1)² + (y-1)² + (z-1)² - 1 = 0
    F₂: 2x² + (y-1)² - 4z = 0
    F₃: 3x² + 2z² - 4y = 0
    """
    
    def system_function(self, x):
        """
        Avalia o sistema de equações F(x) = 0.
        
        Args:
            x: Vetor [x, y, z]
            
        Returns:
            Vetor F(x) = [F₁(x), F₂(x), F₃(x)]
        """
        x_val, y_val, z_val = x[0], x[1], x[2]
        
        f1 = (x_val - 1)**2 + (y_val - 1)**2 + (z_val - 1)**2 - 1
        f2 = 2*x_val**2 + (y_val - 1)**2 - 4*z_val
        f3 = 3*x_val**2 + 2*z_val**2 - 4*y_val
        
        return np.array([f1, f2, f3])
    
    def jacobian_function(self, x):
        """
        Calcula o jacobiano do sistema.
        
        Args:
            x: Vetor [x, y, z]
            
        Returns:
            Matriz jacobiana J(x)
        """
        x_val, y_val, z_val = x[0], x[1], x[2]
        
        # Derivadas parciais de F₁
        df1_dx = 2*(x_val - 1)
        df1_dy = 2*(y_val - 1)
        df1_dz = 2*(z_val - 1)
        
        # Derivadas parciais de F₂
        df2_dx = 4*x_val
        df2_dy = 2*(y_val - 1)
        df2_dz = -4
        
        # Derivadas parciais de F₃
        df3_dx = 6*x_val
        df3_dy = -4
        df3_dz = 4*z_val
        
        return np.array([
            [df1_dx, df1_dy, df1_dz],
            [df2_dx, df2_dy, df2_dz],
            [df3_dx, df3_dy, df3_dz]
        ])
    
    def run_all_methods(self, tolerance=1e-6, max_iterations=1000):
        """
        Executa todos os métodos implementados no sistema.
        
        Args:
            tolerance: Tolerância para convergência
            max_iterations: Máximo de iterações
            
        Returns:
            Dicionário com resultados de todos os métodos
        """
        # Aproximações iniciais para teste
        initial_guesses = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 2.0]),
            np.array([0.5, 0.5, 0.5]),
            np.array([1.5, 1.5, 0.5])
        ]
        
        results = {}
        
        # Testar método de Newton
        newton_solver = NewtonSolver(tolerance=tolerance, max_iterations=max_iterations)
        newton_results = []
        
        for i, x0 in enumerate(initial_guesses):
            solution, info = newton_solver.solve(
                self.system_function, 
                self.jacobian_function, 
                x0
            )
            newton_results.append({
                'initial_guess': x0,
                'solution': solution,
                'converged': info['converged'],
                'iterations': info['iterations'],
                'final_function_norm': info['final_function_norm']
            })
        
        results['newton'] = newton_results
        
        # Testar método da iteração
        iteration_solver = IterationSolver(tolerance=tolerance, max_iterations=max_iterations)
        iteration_results = []
        
        for i, x0 in enumerate(initial_guesses):
            solution, info = iteration_solver.solve(
                self.system_function, 
                None, 
                x0, 
                alpha=0.1
            )
            iteration_results.append({
                'initial_guess': x0,
                'solution': solution,
                'converged': info['converged'],
                'iterations': info['iterations'],
                'final_function_norm': info['final_function_norm']
            })
        
        results['iteration'] = iteration_results
        
        # Testar método do gradiente
        gradient_solver = GradientSolver(tolerance=tolerance, max_iterations=max_iterations)
        gradient_results = []
        
        for i, x0 in enumerate(initial_guesses):
            solution, info = gradient_solver.solve(
                self.system_function, 
                self.jacobian_function, 
                x0
            )
            gradient_results.append({
                'initial_guess': x0,
                'solution': solution,
                'converged': info['converged'],
                'iterations': info['iterations'],
                'final_function_norm': info['final_function_norm']
            })
        
        results['gradient'] = gradient_results
        
        return results
```

Este apêndice apresenta os algoritmos computacionais completos implementados para cada método discutido no trabalho. Os códigos incluem tratamento de erros, validação de entradas, monitoramento de convergência e interfaces padronizadas que facilitam a comparação e uso dos diferentes métodos.
