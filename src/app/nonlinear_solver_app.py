import numpy as np
from pathlib import Path
import time
from typing import Dict, Any, List
from src.nonlinear_solver.methods import (
    NewtonSolver, IterationSolver, GradientSolver
)
import matplotlib.pyplot as plt

class NonLinearSystemExample:
    """
    Classe para resolver o sistema não linear específico.
    """
    
    def __init__(self):
        """Inicializa o exemplo."""
        self.solutions = {}
        self.results_dir = Path("results/nonlinear")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def system_function(self, x: np.ndarray) -> np.ndarray:
        """
        Define o sistema de equações não lineares F(x) = 0.
        
        Args:
            x: Vetor [x, y, z]
            
        Returns:
            Vetor F(x) = [F1, F2, F3]
        """
        x_val, y_val, z_val = x[0], x[1], x[2]
        
        # F₁(x,y,z) = (x-1)² + (y-1)² + (z-1)² - 1
        f1 = (x_val - 1)**2 + (y_val - 1)**2 + (z_val - 1)**2 - 1
        
        # F₂(x,y,z) = 2x² + (y-1)² - 4z
        f2 = 2*x_val**2 + (y_val - 1)**2 - 4*z_val
        
        # F₃(x,y,z) = 3x² + 2z² - 4y
        f3 = 3*x_val**2 + 2*z_val**2 - 4*y_val
        
        return np.array([f1, f2, f3])
    
    def jacobian_function(self, x: np.ndarray) -> np.ndarray:
        """
        Calcula o jacobiano analítico do sistema.
        
        Args:
            x: Vetor [x, y, z]
            
        Returns:
            Matriz jacobiana 3x3
        """
        x_val, y_val, z_val = x[0], x[1], x[2]
        
        # Derivadas de F₁ = (x-1)² + (y-1)² + (z-1)² - 1
        df1_dx = 2*(x_val - 1)
        df1_dy = 2*(y_val - 1)  
        df1_dz = 2*(z_val - 1)
        
        # Derivadas de F₂ = 2x² + (y-1)² - 4z
        df2_dx = 4*x_val
        df2_dy = 2*(y_val - 1)
        df2_dz = -4
        
        # Derivadas de F₃ = 3x² + 2z² - 4y
        df3_dx = 6*x_val
        df3_dy = -4
        df3_dz = 4*z_val
        
        J = np.array([
            [df1_dx, df1_dy, df1_dz],
            [df2_dx, df2_dy, df2_dz],
            [df3_dx, df3_dy, df3_dz]
        ])
        
        return J
    
    def run_methods(self, args: Any, initial_guesses: List[np.ndarray] = None) -> Dict[str, Any]:
        """
        Executa os métodos selecionados para resolver o sistema.

        Args:
            args: Argumentos da linha de comando.
            initial_guesses: Lista de aproximações iniciais para testar.

        Returns:
            Dicionário com resultados dos métodos executados.
        """
        tolerance = args.tolerance
        max_iterations = args.max_iterations

        if initial_guesses is None:
            initial_guesses = [
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 1.0, 1.0]),
                np.array([2.0, 2.0, 2.0]),
                np.array([0.5, 0.5, 0.5]),
                np.array([1.5, 1.5, 0.5]),
            ]

        all_methods = {
            'Newton': NewtonSolver(tolerance=tolerance, max_iterations=max_iterations),
            'Iteracao': IterationSolver(tolerance=tolerance, max_iterations=max_iterations),
            'Gradiente': GradientSolver(tolerance=tolerance, max_iterations=max_iterations)
        }

        methods_to_run = {}
        run_all = not any([args.newton, args.gradient, args.iteration])

        if run_all or args.newton:
            methods_to_run['Newton'] = all_methods['Newton']
        if run_all or args.iteration:
            methods_to_run['Iteracao'] = all_methods['Iteracao']
        if run_all or args.gradient:
            methods_to_run['Gradiente'] = all_methods['Gradiente']

        results = {}

        print(f"\n{'='*70}")
        print(f"RESOLUÇÃO DO SISTEMA NÃO LINEAR")
        print(f"{'='*70}")
        print(f"Sistema:")
        print(f"  F₁: (x-1)² + (y-1)² + (z-1)² - 1 = 0")
        print(f"  F₂: 2x² + (y-1)² - 4z = 0")
        print(f"  F₃: 3x² + 2z² - 4y = 0")
        print(f"")
        print(f"Métodos a executar: {', '.join(methods_to_run.keys())}")
        print(f"Tolerância: {tolerance}")
        print(f"Máximo de iterações: {max_iterations}")
        print(f"{'='*70}\n")

        for method_name, solver in methods_to_run.items():
            print(f"\n{'-'*50}")
            print(f"MÉTODO: {method_name.upper()}")
            print(f"{'-'*50}")
            
            method_results = []
            
            for i, x0 in enumerate(initial_guesses):
                print(f"\nAproximação inicial #{i+1}: {x0}")
                
                start_time = time.time()
                
                try:
                    # Parâmetros específicos para cada método
                    if method_name == 'Iteracao':
                        # Testar diferentes valores de alpha para método da iteração
                        best_result = None
                        best_alpha = None
                        
                        for alpha in [0.01, 0.05, 0.1, 0.2]:
                            solution, info = solver.solve(
                                self.system_function, 
                                self.jacobian_function, 
                                x0.copy(), 
                                alpha=alpha
                            )
                            
                            if info['converged'] and (best_result is None or 
                                                    info['final_function_norm'] < best_result['final_function_norm']):
                                best_result = info
                                best_alpha = alpha
                        
                        if best_result:
                            solution = best_result['solution_x']
                            info = best_result
                            info['best_alpha'] = best_alpha
                        else:
                            # Se nenhum alpha convergiu, usar o último resultado
                            pass
                            
                    elif method_name == 'Gradiente':
                        solution, info = solver.solve(
                            self.system_function, 
                            self.jacobian_function, 
                            x0.copy(),
                            step_size=0.01,
                            adaptive_step=True
                        )
                    else:  # Newton
                        solution, info = solver.solve(
                            self.system_function, 
                            self.jacobian_function, 
                            x0.copy()
                        )
                    
                    elapsed_time = time.time() - start_time
                    
                    # Verificar qualidade da solução
                    f_final = self.system_function(solution)
                    f_norm = np.linalg.norm(f_final)
                    max_f_abs = np.max(np.abs(f_final))
                    
                    result = {
                        'initial_guess': x0.copy(),
                        'solution': solution.copy(),
                        'function_values': f_final,
                        'function_norm': f_norm,
                        'max_f_abs': max_f_abs,
                        'converged': info['converged'],
                        'iterations': info['iterations'],
                        'elapsed_time': elapsed_time,
                        'info': info
                    }
                    
                    if info['converged']:
                        try:
                            det_J = np.linalg.det(self.jacobian_function(solution))
                            result['det_J'] = det_J
                        except np.linalg.LinAlgError:
                            result['det_J'] = float('nan') # Singular matrix

                    method_results.append(result)
                    
                    # Imprimir resultado
                    status = "✓ CONVERGIU" if info['converged'] else "✗ NÃO CONVERGIU"
                    print(f"  Status: {status}")
                    print(f"  Iterações: {info['iterations']}")
                    print(f"  Solução: [{solution[0]:.6f}, {solution[1]:.6f}, {solution[2]:.6f}]")
                    print(f"  Critérios de parada:")
                    print(f"    ||F(x)||: {f_norm:.2e}")
                    print(f"    max|Fᵢ(x)|: {max_f_abs:.2e}")
                    
                    if info['converged']:
                        det_J = result.get('det_J', float('nan'))
                        print(f"  Regularidade da raiz:")
                        print(f"    det(J(x*)): {det_J:.4f}")

                    print(f"  Tempo: {elapsed_time:.4f}s")
                    
                    if method_name == 'Iteracao' and 'best_alpha' in info:
                        print(f"  Melhor α: {info['best_alpha']}")
                    
                except Exception as e:
                    print(f"  ❌ ERRO: {str(e)}")
                    result = {
                        'initial_guess': x0.copy(),
                        'error': str(e),
                        'converged': False,
                        'elapsed_time': time.time() - start_time
                    }
                    method_results.append(result)
            
            results[method_name] = method_results
        
        # Salvar resultados
        self._save_results(results, tolerance)
        
        # Criar visualizações
        self._create_visualizations(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any], tolerance: float):
        """Salva os resultados em arquivo texto."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"nonlinear_results_tol_{tolerance:.0e}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("RESULTADOS DA RESOLUÇÃO DO SISTEMA NÃO LINEAR\n")
            f.write("="*70 + "\n\n")
            
            f.write("Sistema de equações:\n")
            f.write("  F₁: (x-1)² + (y-1)² + (z-1)² - 1 = 0\n")
            f.write("  F₂: 2x² + (y-1)² - 4z = 0\n")
            f.write("  F₃: 3x² + 2z² - 4y = 0\n\n")
            
            f.write(f"Tolerância: {tolerance}\n\n")
            
            for method_name, method_results in results.items():
                f.write(f"MÉTODO: {method_name.upper()}\n")
                f.write("-" * 50 + "\n\n")
                
                converged_count = sum(1 for r in method_results if r.get('converged', False))
                f.write(f"Taxa de convergência: {converged_count}/{len(method_results)} aproximações iniciais\n\n")
                
                for i, result in enumerate(method_results):
                    f.write(f"Aproximação inicial #{i+1}: {result['initial_guess']}\n")
                    
                    if 'error' in result:
                        f.write(f"  ERRO: {result['error']}\n")
                    else:
                        status = "CONVERGIU" if result['converged'] else "NÃO CONVERGIU"
                        f.write(f"  Status: {status}\n")
                        f.write(f"  Iterações: {result['info']['iterations']}\n")
                        f.write(f"  Solução: [{result['solution'][0]:.8f}, {result['solution'][1]:.8f}, {result['solution'][2]:.8f}]\n")
                        f.write(f"  Critérios de parada:\n")
                        f.write(f"    ||F(x)||: {result['function_norm']:.2e}\n")
                        f.write(f"    max|Fᵢ(x)|: {result['max_f_abs']:.2e}\n")
                        
                        if result['converged']:
                            det_J = result.get('det_J', float('nan'))
                            f.write(f"  Regularidade da raiz:\n")
                            f.write(f"    det(J(x*)): {det_J:.4f}\n")

                        f.write(f"  Tempo: {result['elapsed_time']:.6f}s\n")
                        
                        # Verificar a solução
                        x, y, z = result['solution']
                        eq1 = (x-1)**2 + (y-1)**2 + (z-1)**2
                        eq2_left = 2*x**2 + (y-1)**2
                        eq2_right = 4*z
                        eq3_left = 3*x**2 + 2*z**2
                        eq3_right = 4*y
                        
                        f.write(f"  Verificação:\n")
                        f.write(f"    Eq1: (x-1)²+(y-1)²+(z-1)² = {eq1:.8f} (deve ser ≈ 1)\n")
                        f.write(f"    Eq2: 2x²+(y-1)² = {eq2_left:.8f}, 4z = {eq2_right:.8f}\n")
                        f.write(f"    Eq3: 3x²+2z² = {eq3_left:.8f}, 4y = {eq3_right:.8f}\n")
                    
                    f.write("\n")
                
                f.write("\n")
        
        print(f"\n📁 Resultados salvos em: {filename}")
    
    def _create_visualizations(self, results: Dict[str, Any]):
        """Cria visualizações dos resultados."""
            
        try:
            # Gráfico de convergência
            plt.figure(figsize=(15, 10))
            
            # Comparar convergência dos métodos
            plt.subplot(2, 2, 1)
            for method_name, method_results in results.items():
                converged_results = [r for r in method_results if r.get('converged', False)]
                if converged_results:
                    iterations = [r['info']['iterations'] for r in converged_results]
                    plt.hist(iterations, alpha=0.7, label=f'{method_name} ({len(converged_results)}/{len(method_results)} convergiu)', bins=10)
            
            plt.xlabel('Número de Iterações')
            plt.ylabel('Frequência')
            plt.title('Distribuição de Iterações para Convergência')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Tempo de execução
            plt.subplot(2, 2, 2)
            method_names = []
            avg_times = []
            for method_name, method_results in results.items():
                times = [r['elapsed_time'] for r in method_results if not r.get('error')]
                if times:
                    method_names.append(method_name)
                    avg_times.append(np.mean(times))
            
            plt.bar(method_names, avg_times, alpha=0.7)
            plt.xlabel('Método')
            plt.ylabel('Tempo Médio (s)')
            plt.title('Tempo Médio de Execução')
            plt.grid(True, alpha=0.3)
            
            # Norma da função final
            plt.subplot(2, 2, 3)
            for method_name, method_results in results.items():
                converged_results = [r for r in method_results if r.get('converged', False)]
                if converged_results:
                    function_norms = [r['function_norm'] for r in converged_results]
                    plt.semilogy(function_norms, 'o-', label=method_name, alpha=0.7)
            
            plt.xlabel('Aproximação Inicial')
            plt.ylabel('||F(x)|| (log scale)')
            plt.title('Norma da Função na Solução Final')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Taxa de convergência
            plt.subplot(2, 2, 4)
            method_names = []
            success_rates = []
            for method_name, method_results in results.items():
                converged_count = sum(1 for r in method_results if r.get('converged', False))
                total_count = len(method_results)
                method_names.append(method_name)
                success_rates.append(100 * converged_count / total_count)
            
            plt.bar(method_names, success_rates, alpha=0.7)
            plt.xlabel('Método')
            plt.ylabel('Taxa de Convergência (%)')
            plt.title('Taxa de Convergência por Método')
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Salvar gráfico
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plot_filename = self.results_dir / f"nonlinear_comparison_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"📊 Gráficos salvos em: {plot_filename}")
            
        except Exception as e:
            print(f"⚠️  Erro ao criar visualizações: {e}")

    def generate_attraction_basin(self, args: Any):
        """
        Gera e visualiza o mapa de bacias de atração.

        Esta é uma ferramenta de visualização que mostra para qual raiz um
        método iterativo (neste caso, Newton) converge a partir de uma grade
        de diferentes pontos iniciais. Não é um método de resolução em si,
        mas uma análise do comportamento do solucionador.
        """
        print("\n🌊 Gerando Mapa de Bacias de Atração...")
        print("   - Esta é uma ferramenta de visualização para analisar o comportamento do método de Newton.")
        print("   - Cada ponto no mapa representa um chute inicial; a cor indica para qual raiz ele converge.")

        # 1. Encontrar as raízes do sistema
        print("   - Identificando raízes do sistema...")
        roots = self._find_system_roots(args)
        
        if not roots:
            print("   - ⚠️ Nenhuma raiz encontrada. Não é possível gerar o mapa.")
            return
            
        print(f"   - {len(roots)} raízes encontradas:")
        for i, root in enumerate(roots):
            print(f"     Raiz {i+1}: [{root[0]:.6f}, {root[1]:.6f}, {root[2]:.6f}]")

        # 2. Configurar a grade 2D
        # Por enquanto, vamos fixar z=0 e variar x e y
        grid_size = args.basin_resolution
        x_range = np.linspace(-2, 4, grid_size)
        y_range = np.linspace(-2, 4, grid_size)
        z_fixed = 0.0
        
        basin_map = np.zeros((grid_size, grid_size), dtype=int)
        
        solver = NewtonSolver(tolerance=args.tolerance, max_iterations=args.max_iterations)

        # 3. Iterar sobre a grade e classificar cada ponto
        print(f"   - Varrendo uma grade de {grid_size}x{grid_size} pontos...")
        
        for i, y in enumerate(y_range):
            for j, x in enumerate(x_range):
                x0 = np.array([x, y, z_fixed])
                
                try:
                    solution, info = solver.solve(self.system_function, self.jacobian_function, x0)
                    
                    if info['converged']:
                        # Encontrar a raiz mais próxima
                        distances = [np.linalg.norm(solution - root) for root in roots]
                        closest_root_idx = np.argmin(distances)
                        
                        # Verificar se está realmente perto da raiz
                        if distances[closest_root_idx] < 1e-3:
                            basin_map[i, j] = closest_root_idx + 1
                        else:
                            basin_map[i, j] = 0 # Convergiu para um ponto desconhecido
                    else:
                        basin_map[i, j] = -1 # Não convergiu
                except Exception:
                    basin_map[i, j] = -2 # Erro na execução

        # 4. Visualizar o mapa
        print("   - Gerando visualização do mapa...")
        plt.figure(figsize=(10, 8))
        
        # Usar um mapa de cores discreto
        cmap = plt.cm.get_cmap('viridis', len(roots) + 3)
        
        plt.imshow(basin_map, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], 
                   origin='lower', cmap=cmap, interpolation='nearest')
        
        # Adicionar legenda de cores
        cbar = plt.colorbar()
        tick_locs = np.arange(-2, len(roots) + 1) + 0.5 * (len(roots)+2)/(len(roots)+3)
        cbar.set_ticks(tick_locs)
        
        labels = ['Erro', 'Não Convergiu', 'Desconhecido'] + [f'Raiz {i+1}' for i in range(len(roots))]
        cbar.set_ticklabels(labels)

        plt.title(f'Bacias de Atração (plano z={z_fixed})')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # Plotar as raízes no mapa (se estiverem no plano)
        for i, root in enumerate(roots):
            if np.isclose(root[2], z_fixed):
                plt.plot(root[0], root[1], 'r*', markersize=10, label=f'Raiz {i+1}' if i == 0 else "")

        if any(np.isclose(r[2], z_fixed) for r in roots):
            plt.legend()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_filename = self.results_dir / f"basin_map_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   - ✅ Mapa salvo em: {plot_filename}")

    def _find_system_roots(self, args: Any, num_guesses: int = 100) -> List[np.ndarray]:
        """
        Tenta encontrar as raízes do sistema a partir de várias aproximações iniciais.
        """
        solver = NewtonSolver(tolerance=1e-8, max_iterations=args.max_iterations)
        roots = []
        
        # Gerar pontos iniciais aleatórios num espaço maior
        initial_guesses = np.random.uniform(-5, 5, size=(num_guesses, 3))
        
        for x0 in initial_guesses:
            try:
                solution, info = solver.solve(self.system_function, self.jacobian_function, x0)
                
                if info['converged']:
                    # Verificar se a raiz é nova
                    is_new_root = True
                    for existing_root in roots:
                        if np.linalg.norm(solution - existing_root) < 1e-4:
                            is_new_root = False
                            break
                    
                    if is_new_root:
                        roots.append(solution)
            except Exception:
                continue # Ignorar erros durante a busca por raízes
                
        return roots


def solve_nonlinear_system(args: Any):
    """
    Resolve o sistema não linear específico com base nos argumentos fornecidos.
    """
    print("\n🔬 RESOLVEDOR DE SISTEMAS NÃO LINEARES")
    print("=" * 60)
    
    # Executar exemplo não linear
    try:
        example = NonLinearSystemExample()

        if args.basin_map:
            example.generate_attraction_basin(args)
        else:
            # Executar com os argumentos da CLI
            example.run_methods(args)
        
        print(f"\n✅ Sistema não linear processado com sucesso!")
        print(f"📁 Resultados salvos em: ./results/nonlinear/")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erro ao resolver sistema não linear: {e}")
        return False
