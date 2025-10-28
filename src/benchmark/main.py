import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import time
import statistics
from linear_solver import (
    JacobiSolver, GaussSeidelSolver, ConjugateGradientSolver,
    JacobiOrder2Solver, GaussSeidelOrder2Solver,
    PreconditionedConjugateGradientSolver,
    MatrixValidator
)

# Tenta importar o matplotlib para verificar a disponibilidade
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

class BenchmarkResult:
    """Classe para armazenar resultados do benchmark."""
    
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.execution_times = []
        self.iterations_to_converge = []
        self.final_errors = []
        self.convergence_success = []
        self.residuals = []
        
    def add_result(self, execution_time: float, iterations: int, final_error: float, 
                   converged: bool, residual: float):
        """Adiciona um resultado de execução."""
        self.execution_times.append(execution_time)
        self.iterations_to_converge.append(iterations)
        self.final_errors.append(final_error)
        self.convergence_success.append(converged)
        self.residuals.append(residual)
    
    def get_statistics(self) -> Dict:
        """Calcula estatísticas dos resultados."""
        if not self.execution_times:
            return {}
            
        successful_runs = [i for i, success in enumerate(self.convergence_success) if success]
        
        stats = {
            'total_runs': len(self.execution_times),
            'successful_runs': len(successful_runs),
            'success_rate': len(successful_runs) / len(self.execution_times) * 100,
            'avg_execution_time': statistics.mean(self.execution_times),
            'median_execution_time': statistics.median(self.execution_times),
            'min_execution_time': min(self.execution_times),
            'max_execution_time': max(self.execution_times),
            'std_execution_time': statistics.stdev(self.execution_times) if len(self.execution_times) > 1 else 0,
        }
        
        if successful_runs:
            successful_times = [self.execution_times[i] for i in successful_runs]
            successful_iterations = [self.iterations_to_converge[i] for i in successful_runs]
            successful_errors = [self.final_errors[i] for i in successful_runs]
            successful_residuals = [self.residuals[i] for i in successful_runs]
            
            stats.update({
                'avg_time_successful': statistics.mean(successful_times),
                'median_time_successful': statistics.median(successful_times),
                'avg_iterations': statistics.mean(successful_iterations),
                'median_iterations': statistics.median(successful_iterations),
                'min_iterations': min(successful_iterations),
                'max_iterations': max(successful_iterations),
                'avg_final_error': statistics.mean(successful_errors),
                'median_final_error': statistics.median(successful_errors),
                'avg_residual': statistics.mean(successful_residuals),
                'median_residual': statistics.median(successful_residuals),
            })
        
        return stats


class MethodBenchmark:
    """Classe principal para executar benchmarks dos métodos."""
    
    def __init__(self, tolerance: float = 1e-4, max_iterations: int = 5000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.results = {}
        
    def setup_methods(self, A: np.ndarray) -> Dict:
        """Configura os métodos baseado nas propriedades da matriz."""
        methods = {}
        
        # Analisar matriz para determinar métodos aplicáveis
        analysis = MatrixValidator.analyze_matrix(A)
        
        # Métodos básicos (sempre disponíveis)
        methods['Jacobi'] = JacobiSolver(
            tolerance=self.tolerance, 
            max_iterations=self.max_iterations
        )
        
        methods['Gauss-Seidel'] = GaussSeidelSolver(
            tolerance=self.tolerance, 
            max_iterations=self.max_iterations
        )
        
        methods['Jacobi Ordem 2'] = JacobiOrder2Solver(
            tolerance=self.tolerance, 
            max_iterations=self.max_iterations,
            omega1=0.7, omega2=0.2, omega3=0.1
        )
        
        methods['Gauss-Seidel Ordem 2'] = GaussSeidelOrder2Solver(
            tolerance=self.tolerance, 
            max_iterations=self.max_iterations,
            relaxation_factor=1.2, omega1=0.8, omega2=0.15, omega3=0.05
        )
        
        # Métodos para matrizes simétricas positivas definidas
        if analysis['is_symmetric'] and analysis['is_positive_definite']:
            methods['Gradiente Conjugado'] = ConjugateGradientSolver(
                tolerance=self.tolerance
            )
            
            methods['Gradiente Conjugado Precondicionado'] = PreconditionedConjugateGradientSolver(
                tolerance=self.tolerance
            )
        
        print(f"📋 Métodos configurados: {', '.join(methods.keys())}")
        return methods
        
    def run_single_benchmark(self, method_name: str, solver, A: np.ndarray, 
                           b: np.ndarray, x_reference: np.ndarray) -> Tuple[float, int, float, bool, float]:
        """Executa um benchmark individual de um método."""
        try:
            start_time = time.perf_counter()
            
            # Executar o método
            x, info = solver.solve(A, b)
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Calcular métricas de qualidade
            final_error = np.linalg.norm(x - x_reference, ord=np.inf)
            residual = np.linalg.norm(A @ x - b)
            
            return (execution_time, info['iterations'], final_error, 
                   info['converged'], residual)
                   
        except Exception as e:
            print(f"⚠️  Erro no benchmark {method_name}: {str(e)}")
            return (0.0, 0, float('inf'), False, float('inf'))
    
    def run_benchmark_suite(self, A: np.ndarray, b: np.ndarray, 
                           num_runs: int = 10) -> Dict[str, BenchmarkResult]:
        """Executa suite completa de benchmarks."""
        
        print(f"\n🚀 INICIANDO BENCHMARK SUITE")
        print("-" * 50)
        print(f"📊 Matriz: {A.shape[0]}x{A.shape[1]}")
        print(f"🔄 Rodadas por método: {num_runs}")
        print(f"🎯 Tolerância: {self.tolerance}")
        print(f"🔢 Máx. iterações: {self.max_iterations}")
        
        # Calcular solução de referência
        try:
            x_reference = np.linalg.solve(A, b)
            print("✅ Solução de referência calculada")
        except np.linalg.LinAlgError:
            print("❌ Não foi possível calcular solução de referência")
            return {}
        
        # Configurar métodos
        methods = self.setup_methods(A)
        
        if not methods:
            print("❌ Nenhum método disponível")
            return {}
        
        # Inicializar resultados
        results = {}
        for method_name in methods.keys():
            results[method_name] = BenchmarkResult(method_name)
        
        # Executar benchmarks
        print(f"\n⏱️  EXECUTANDO BENCHMARKS...")
        
        for method_name, solver in methods.items():
            print(f"\n🔧 Testando {method_name}...")
            
            for run in range(num_runs):
                print(f"   Rodada {run+1:2d}/{num_runs}", end=" ")
                
                execution_time, iterations, final_error, converged, residual = \
                    self.run_single_benchmark(method_name, solver, A, b, x_reference)
                
                results[method_name].add_result(
                    execution_time, iterations, final_error, converged, residual
                )
                
                status = "✅" if converged else "❌"
                print(f"{status} {execution_time:.4f}s, {iterations} iter, erro: {final_error:.2e}")
        
        self.results = results
        return results
    
    def print_summary(self):
        """Imprime resumo dos resultados do benchmark."""
        
        if not self.results:
            print("❌ Nenhum resultado para exibir")
            return
        
        print(f"\n📈 RESUMO DO BENCHMARK")
        print("=" * 80)
        
        # Cabeçalho da tabela
        header = f"{'Método':<25} {'Taxa':<8} {'Tempo Méd.':<12} {'Tempo Med.':<12} {'Iter Méd.':<10} {'Erro Méd.':<12}"
        print(header)
        print("-" * len(header))
        
        # Dados por método
        for method_name, result in self.results.items():
            stats = result.get_statistics()
            
            if stats:
                success_rate = f"{stats['success_rate']:.1f}%"
                avg_time = f"{stats.get('avg_time_successful', 0):.4f}s"
                median_time = f"{stats.get('median_time_successful', 0):.4f}s"
                avg_iter = f"{stats.get('avg_iterations', 0):.1f}"
                avg_error = f"{stats.get('avg_final_error', 0):.2e}"
                
                print(f"{method_name:<25} {success_rate:<8} {avg_time:<12} {median_time:<12} {avg_iter:<10} {avg_error:<12}")
            else:
                print(f"{method_name:<25} {'N/A':<8} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<12}")
        
        print()
        
    def save_detailed_report(self, matrix_name: str = "Sistema"):
        """Salva relatório detalhado dos benchmarks."""
        
        if not self.results:
            print("❌ Nenhum resultado para salvar")
            return
        
        # Criar diretório de resultados
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Nome do arquivo
        nome_sistema = matrix_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        report_file = results_dir / f"benchmark_timing_{nome_sistema}_tol_{self.tolerance:.0e}".replace("-", "neg") + ".txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BENCHMARK DE TEMPO DE EXECUÇÃO - MÉTODOS ITERATIVOS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Sistema: {matrix_name}\n")
            f.write(f"Tolerância: {self.tolerance}\n")
            f.write(f"Máximo de iterações: {self.max_iterations}\n\n")
            
            f.write("RESUMO EXECUTIVO:\n")
            f.write("-" * 40 + "\n")
            
            # Ranking por tempo médio (apenas métodos que convergiram)
            successful_methods = []
            for method_name, result in self.results.items():
                stats = result.get_statistics()
                if stats and stats.get('successful_runs', 0) > 0:
                    successful_methods.append((
                        method_name, 
                        stats['avg_time_successful'],
                        stats['success_rate'],
                        stats.get('avg_iterations', 0)
                    ))
            
            # Ordenar por tempo médio
            successful_methods.sort(key=lambda x: x[1])
            
            f.write("RANKING DE PERFORMANCE (tempo médio para convergência):\n")
            for i, (method, time, rate, iterations) in enumerate(successful_methods, 1):
                f.write(f"{i:2d}. {method:<30} {time:.4f}s ({rate:.1f}% sucesso, {iterations:.1f} iter)\n")
            
            f.write(f"\nESTATÍSTICAS DETALHADAS POR MÉTODO:\n")
            f.write("-" * 50 + "\n\n")
            
            for method_name, result in self.results.items():
                f.write(f"{method_name.upper()}:\n")
                f.write("-" * len(method_name) + "\n")
                
                stats = result.get_statistics()
                if stats:
                    f.write(f"  Execuções totais: {stats['total_runs']}\n")
                    f.write(f"  Execuções bem-sucedidas: {stats['successful_runs']}\n")
                    f.write(f"  Taxa de sucesso: {stats['success_rate']:.1f}%\n")
                    
                    if stats.get('successful_runs', 0) > 0:
                        f.write(f"  Tempo médio (sucessos): {stats['avg_time_successful']:.6f}s\n")
                        f.write(f"  Tempo mediano (sucessos): {stats['median_time_successful']:.6f}s\n")
                        f.write(f"  Tempo mínimo: {stats['min_execution_time']:.6f}s\n")
                        f.write(f"  Tempo máximo: {stats['max_execution_time']:.6f}s\n")
                        f.write(f"  Desvio padrão: {stats['std_execution_time']:.6f}s\n")
                        f.write(f"  Iterações médias: {stats['avg_iterations']:.1f}\n")
                        f.write(f"  Iterações medianas: {stats['median_iterations']:.1f}\n")
                        f.write(f"  Min/Max iterações: {stats['min_iterations']}/{stats['max_iterations']}\n")
                        f.write(f"  Erro final médio: {stats['avg_final_error']:.2e}\n")
                        f.write(f"  Resíduo médio: {stats['avg_residual']:.2e}\n")
                    else:
                        f.write("  NENHUMA EXECUÇÃO CONVERGIU\n")
                else:
                    f.write("  NENHUM DADO DISPONÍVEL\n")
                
                f.write("\n")
            
            # Estimativas de tempo de máquina
            f.write("ESTIMATIVAS DE TEMPO DE MÁQUINA:\n")
            f.write("-" * 40 + "\n")
            f.write("Para obter resultados aceitáveis (convergência com tolerância especificada):\n\n")
            
            for method, time, rate, iterations in successful_methods:
                # Estimativas considerando diferentes cenários
                estimated_time = time
                
                # Margem de segurança baseada no desvio padrão
                method_stats = self.results[method].get_statistics()
                safety_margin = method_stats.get('std_execution_time', 0) * 2
                conservative_time = estimated_time + safety_margin
                
                f.write(f"{method}:\n")
                f.write(f"  Tempo típico: {estimated_time:.4f}s\n")
                f.write(f"  Tempo conservador (+2σ): {conservative_time:.4f}s\n")
                f.write(f"  Probabilidade de sucesso: {rate:.1f}%\n")
                f.write(f"  Iterações típicas: {iterations:.0f}\n")
                
                # Classificação de velocidade
                if estimated_time < 0.01:
                    classification = "MUITO RÁPIDO"
                elif estimated_time < 0.1:
                    classification = "RÁPIDO"
                elif estimated_time < 1.0:
                    classification = "MODERADO"
                elif estimated_time < 10.0:
                    classification = "LENTO"
                else:
                    classification = "MUITO LENTO"
                
                f.write(f"  Classificação: {classification}\n\n")
        
        print(f"💾 Relatório detalhado salvo: {report_file}")
        return report_file

def create_benchmark_visualizations(benchmark_data: Dict, matrix_name: str):
    """Cria todas as visualizações do benchmark."""
    
    if not HAS_MATPLOTLIB:
        print("⚠️  Matplotlib não disponível - pulando visualizações do benchmark")
        return
    
    print(f"\n📊 GERANDO VISUALIZAÇÕES DO BENCHMARK PARA: {matrix_name}")
    print("-" * 60)
    
    # Criar gráfico de comparação de tempos
    create_timing_comparison(benchmark_data, matrix_name)
    
    # Criar gráfico de recomendações
    create_recommendation_chart(benchmark_data, matrix_name)
    
    # Criar análise de convergência
    create_convergence_analysis(benchmark_data, matrix_name)
    
    print("✅ Todas as visualizações foram geradas com sucesso!")


def create_timing_comparison(benchmark_data: Dict, matrix_name: str):
    """Cria gráfico de comparação de tempos."""
    
    plt.figure(figsize=(15, 10))
    
    # Extrair dados para cada tolerância
    tolerances = sorted(benchmark_data.keys())
    tolerance_labels = [f"{float(tol):.0e}" for tol in tolerances]
    
    # Organizar dados por método
    methods_data = {}
    for tol in tolerances:
        for method_name, stats in benchmark_data[tol].items():
            if method_name not in methods_data:
                methods_data[method_name] = {'times': [], 'iterations': []}
            
            if stats and stats.get('successful_runs', 0) > 0:
                methods_data[method_name]['times'].append(stats.get('avg_time_successful', 0))
                methods_data[method_name]['iterations'].append(stats.get('avg_iterations', 0))
            else:
                methods_data[method_name]['times'].append(0)
                methods_data[method_name]['iterations'].append(0)
    
    # Subplot 1: Tempos de execução
    plt.subplot(2, 2, 1)
    x = np.arange(len(tolerance_labels))
    width = 0.2
    colors = ['green', 'blue', 'orange', 'red', 'purple', 'brown']
    
    for i, (method, data) in enumerate(methods_data.items()):
        offset = (i - len(methods_data)/2) * width
        plt.bar(x + offset, data['times'], width, label=method, 
                color=colors[i % len(colors)], alpha=0.8)
    
    plt.xlabel('Tolerância')
    plt.ylabel('Tempo de Execução (s)')
    plt.title('Comparação de Tempos de Execução por Método')
    plt.xticks(x, tolerance_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Número de iterações
    plt.subplot(2, 2, 2)
    for i, (method, data) in enumerate(methods_data.items()):
        offset = (i - len(methods_data)/2) * width
        plt.bar(x + offset, data['iterations'], width, label=method,
                color=colors[i % len(colors)], alpha=0.8)
    
    plt.xlabel('Tolerância')
    plt.ylabel('Número de Iterações')
    plt.title('Número de Iterações para Convergência')
    plt.xticks(x, tolerance_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Eficiência (tempo por iteração)
    plt.subplot(2, 2, 3)
    for i, (method, data) in enumerate(methods_data.items()):
        efficiency = [t/iter_count if iter_count > 0 else 0 
                     for t, iter_count in zip(data['times'], data['iterations'])]
        offset = (i - len(methods_data)/2) * width
        plt.bar(x + offset, efficiency, width, label=method,
                color=colors[i % len(colors)], alpha=0.8)
    
    plt.xlabel('Tolerância')
    plt.ylabel('Tempo por Iteração (s)')
    plt.title('Eficiência: Tempo por Iteração')
    plt.xticks(x, tolerance_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Speedup relativo (usando método mais lento como referência)
    plt.subplot(2, 2, 4)
    if methods_data:
        # Encontrar método mais lento para cada tolerância
        reference_times = []
        for i, tol in enumerate(tolerances):
            max_time = max([data['times'][i] for data in methods_data.values() if data['times'][i] > 0])
            reference_times.append(max_time if max_time > 0 else 1)
        
        for i, (method, data) in enumerate(methods_data.items()):
            speedup = [ref/time if time > 0 else 0 
                      for time, ref in zip(data['times'], reference_times)]
            offset = (i - len(methods_data)/2) * width
            bars = plt.bar(x + offset, speedup, width, label=method,
                          color=colors[i % len(colors)], alpha=0.8)
            
            # Adicionar valores de speedup
            for j, (bar, speed) in enumerate(zip(bars, speedup)):
                if speed > 0:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                            f'{speed:.1f}x', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Tolerância')
    plt.ylabel('Speedup (vs método mais lento)')
    plt.title('Speedup Relativo por Método')
    plt.xticks(x, tolerance_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar gráfico
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    chart_path = results_dir / f"benchmark_timing_comparison_{matrix_name.lower().replace(' ', '_')}.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico de comparação salvo: {chart_path}")
    plt.show()


def create_recommendation_chart(benchmark_data: Dict, matrix_name: str):
    """Cria gráfico com recomendações de uso."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Usar tolerância 1e-4 como referência
    ref_tolerance = '0.0001'  # 1e-4
    if ref_tolerance not in benchmark_data:
        ref_tolerance = list(benchmark_data.keys())[0]  # Usar primeira disponível
    
    ref_data = benchmark_data[ref_tolerance]
    
    # Gráfico 1: Tempo vs Métodos (configuração recomendada)
    methods = []
    times = []
    colors = ['green', 'blue', 'orange', 'red', 'purple', 'brown']
    
    for method_name, stats in ref_data.items():
        if stats and stats.get('successful_runs', 0) > 0:
            methods.append(method_name.replace(' ', '\n'))
            times.append(stats.get('avg_time_successful', 0))
    
    if methods:
        bars = ax1.bar(methods, times, color=colors[:len(methods)], alpha=0.7)
        ax1.set_ylabel('Tempo de Execução (s)')
        ax1.set_title(f'Tempo para Tolerância {float(ref_tolerance):.0e}\n(Configuração Recomendada)')
        ax1.grid(True, alpha=0.3)
        
        # Adicionar rótulos com classificação
        for bar, time in zip(bars, times):
            if time < 0.01:
                classification = "MUITO RÁPIDO"
            elif time < 0.1:
                classification = "RÁPIDO"
            elif time < 1.0:
                classification = "MODERADO"
            else:
                classification = "LENTO"
            
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                    f'{time:.4f}s\n{classification}', ha='center', va='bottom', fontsize=9)
    
    # Gráfico 2: Cenários de uso recomendados
    if methods:
        # Usar o método mais rápido para as recomendações
        fastest_method = min(ref_data.items(), 
                           key=lambda x: x[1].get('avg_time_successful', float('inf')) 
                           if x[1] and x[1].get('successful_runs', 0) > 0 else float('inf'))
        
        scenarios = ['Tempo Real\n(<5ms)', 'Científico\n(5-10ms)', 'Alta Precisão\n(>10ms)']
        # Estimar tempos para diferentes tolerâncias baseado no método mais rápido
        tolerances_available = sorted([float(t) for t in benchmark_data.keys()])
        recommended_times = []
        tolerances_rec = []
        
        for scenario_idx in range(3):
            if scenario_idx < len(tolerances_available):
                tol_key = str(tolerances_available[scenario_idx])
                if tol_key in benchmark_data and fastest_method[0] in benchmark_data[tol_key]:
                    method_data = benchmark_data[tol_key][fastest_method[0]]
                    if method_data and method_data.get('successful_runs', 0) > 0:
                        recommended_times.append(method_data.get('avg_time_successful', 0))
                        tolerances_rec.append(f"{tolerances_available[scenario_idx]:.0e}")
                    else:
                        recommended_times.append(0)
                        tolerances_rec.append('N/A')
                else:
                    recommended_times.append(0)
                    tolerances_rec.append('N/A')
            else:
                recommended_times.append(0)
                tolerances_rec.append('N/A')
        
        bars2 = ax2.bar(scenarios, recommended_times, color='green', alpha=0.7)
        ax2.set_ylabel('Tempo Estimado (s)')
        ax2.set_title(f'Tempos por Cenário de Uso\n({fastest_method[0]})')
        ax2.grid(True, alpha=0.3)
        
        for bar, time, tol in zip(bars2, recommended_times, tolerances_rec):
            if time > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(recommended_times)*0.02,
                        f'{time:.4f}s\n(tol: {tol})', ha='center', va='bottom', fontsize=9)
        
        # Adicionar linha de referência de 5ms
        ax2.axhline(y=0.005, color='red', linestyle='--', alpha=0.7, label='Limite 5ms')
        ax2.legend()
    
    plt.tight_layout()
    
    # Salvar gráfico
    results_dir = Path("results")
    chart_path = results_dir / f"benchmark_recommendations_{matrix_name.lower().replace(' ', '_')}.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico de recomendações salvo: {chart_path}")
    plt.show()


def create_convergence_analysis(benchmark_data: Dict, matrix_name: str):
    """Cria análise da velocidade de convergência."""
    
    plt.figure(figsize=(12, 8))
    
    # Extrair dados
    tolerances = sorted([float(t) for t in benchmark_data.keys()])
    tolerance_labels = [f"{tol:.0e}" for tol in tolerances]
    
    # Organizar dados por método
    methods_data = {}
    colors = ['green', 'blue', 'orange', 'red', 'purple', 'brown']
    markers = ['o', 's', '^', 'd', 'v', 'p']
    
    for tol_str in benchmark_data.keys():
        tol = float(tol_str)
        for method_name, stats in benchmark_data[tol_str].items():
            if method_name not in methods_data:
                methods_data[method_name] = {'times': [], 'iterations': [], 'tolerances': []}
            
            if stats and stats.get('successful_runs', 0) > 0:
                methods_data[method_name]['times'].append(stats.get('avg_time_successful', 0))
                methods_data[method_name]['iterations'].append(stats.get('avg_iterations', 0))
                methods_data[method_name]['tolerances'].append(tol)
    
    # Subplot 1: Iterações vs Tolerância
    plt.subplot(2, 2, 1)
    for i, (method, data) in enumerate(methods_data.items()):
        if data['iterations'] and data['tolerances']:
            plt.loglog(data['tolerances'], data['iterations'], 
                      f'{markers[i % len(markers)]}-', 
                      color=colors[i % len(colors)], 
                      label=method, linewidth=2)
    
    plt.xlabel('Tolerância')
    plt.ylabel('Número de Iterações')
    plt.title('Convergência: Iterações vs Tolerância')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Tempo vs Tolerância
    plt.subplot(2, 2, 2)
    for i, (method, data) in enumerate(methods_data.items()):
        if data['times'] and data['tolerances']:
            plt.loglog(data['tolerances'], data['times'], 
                      f'{markers[i % len(markers)]}-', 
                      color=colors[i % len(colors)], 
                      label=method, linewidth=2)
    
    plt.xlabel('Tolerância')
    plt.ylabel('Tempo de Execução (s)')
    plt.title('Performance: Tempo vs Tolerância')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Taxa de convergência estimada
    plt.subplot(2, 2, 3)
    if methods_data:
        method_names = list(methods_data.keys())
        # Estimar taxa de convergência baseada na redução de iterações com tolerância
        convergence_rates = []
        
        for method, data in methods_data.items():
            if len(data['iterations']) >= 2:
                # Calcular taxa baseada na mudança de iterações vs tolerância
                iterations = np.array(data['iterations'])
                tolerances_method = np.array(data['tolerances'])
                if len(iterations) > 1:
                    # Estimar taxa de convergência (quanto menor, melhor)
                    rate = np.mean(iterations[:-1] / iterations[1:]) if len(iterations) > 1 else 0.8
                    convergence_rates.append(min(rate, 1.0))
                else:
                    convergence_rates.append(0.8)
            else:
                convergence_rates.append(0.8)
        
        if convergence_rates:
            bars = plt.bar(range(len(method_names)), convergence_rates, 
                          color=colors[:len(method_names)], alpha=0.7)
            plt.xticks(range(len(method_names)), 
                      [name.replace(' ', '\n') for name in method_names])
            plt.ylabel('Taxa de Convergência Estimada')
            plt.title('Taxa de Convergência por Método')
            plt.grid(True, alpha=0.3)
            
            for bar, rate in zip(bars, convergence_rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{rate:.2f}', ha='center', va='bottom')
    
    # Subplot 4: Relação custo-benefício
    plt.subplot(2, 2, 4)
    if methods_data:
        # Usar tolerância intermediária como referência
        ref_tolerance = tolerances[len(tolerances)//2] if tolerances else tolerances[0]
        ref_tol_str = str(ref_tolerance)
        
        if ref_tol_str in benchmark_data:
            times_ref = []
            quality_scores = []  # Baseado na convergência e estabilidade
            method_names_scatter = []
            
            for method, stats in benchmark_data[ref_tol_str].items():
                if stats and stats.get('successful_runs', 0) > 0:
                    time = stats.get('avg_time_successful', 0)
                    success_rate = stats.get('success_rate', 0)
                    # Score de qualidade baseado na taxa de sucesso e erro final
                    quality = success_rate * (1 - min(stats.get('avg_final_error', 1), 1))
                    
                    times_ref.append(time)
                    quality_scores.append(quality)
                    method_names_scatter.append(method)
            
            if times_ref and quality_scores:
                for i, (time, quality, method) in enumerate(zip(times_ref, quality_scores, method_names_scatter)):
                    plt.scatter(time, quality, c=colors[i % len(colors)], s=100, alpha=0.7)
                    plt.annotate(method.replace(' ', '\n'), (time, quality), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                plt.xlabel('Tempo de Execução (s)')
                plt.ylabel('Score de Qualidade')
                plt.title(f'Relação Custo-Benefício\n(Tolerância {ref_tolerance:.0e})')
                plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar gráfico
    results_dir = Path("results")
    chart_path = results_dir / f"benchmark_convergence_analysis_{matrix_name.lower().replace(' ', '_')}.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"📊 Análise de convergência salva: {chart_path}")
    plt.show()

def run_benchmark_mode(A, b, matrix_name, args):
    """Executa o modo benchmark com múltiplas configurações de tolerância."""
    
    print(f"\n⏱️  MODO BENCHMARK ATIVADO PARA: {matrix_name}")
    print("=" * 60)
    
    # Analisar propriedades da matriz
    analysis = MatrixValidator.analyze_matrix(A)
    print(f"📊 Número de condição: {analysis['condition_number']:.2e}")
    print(f"📊 Simétrica: {'Sim' if analysis['is_symmetric'] else 'Não'}")
    print(f"📊 Positiva definida: {'Sim' if analysis['is_positive_definite'] else 'Não'}")
    
    # Configurar múltiplas tolerâncias para benchmark
    tolerances = [1e-6, 1e-5, 1e-4, 1e-3]
    
    # Coletar dados para visualização
    benchmark_data = {}
    
    for tol in tolerances:
        print(f"\n🎯 TESTANDO TOLERÂNCIA: {tol}")
        print("=" * 50)
        
        # Configurar e executar benchmark
        benchmark = MethodBenchmark(tolerance=tol, max_iterations=args.max_iterations)
        
        # Executar múltiplas rodadas
        results = benchmark.run_benchmark_suite(A, b, num_runs=10)
        
        if results:
            benchmark.print_summary()
            
            # Salvar relatório específico para esta tolerância
            tolerance_str = f"{tol:.0e}".replace("-", "neg")
            benchmark.save_detailed_report(f"{matrix_name}_tol_{tolerance_str}")
            
            # Coletar estatísticas para visualização
            benchmark_data[str(tol)] = {}
            for method_name, result in results.items():
                stats = result.get_statistics()
                benchmark_data[str(tol)][method_name] = stats
            
        print("\n" + "~" * 60)
    
    print("\n🎉 BENCHMARK CONCLUÍDO!")
    print("📁 Relatórios salvos na pasta results/")
    
    # Gerar visualizações se solicitado
    if args.visualize_benchmark and benchmark_data:
        create_benchmark_visualizations(benchmark_data, matrix_name)
