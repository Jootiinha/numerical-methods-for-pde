#!/usr/bin/env python3
"""
Aplicação principal para resolver sistemas lineares e não lineares.

Este script processa sistemas lineares da pasta data/ ou resolve sistemas não lineares específicos
usando métodos numéricos selecionados via argumentos de linha de comando.

Exemplos de uso:
    # Sistemas lineares
    python main.py --all                    # Todos os métodos lineares
    python main.py --jacobi                 # Apenas Jacobi
    python main.py --jacobi --gauss-seidel  # Jacobi e Gauss-Seidel
    python main.py --conjugate-gradient     # Apenas Gradiente Conjugado
    python main.py --no-plots               # Sem gráficos
    python main.py --clear-old-data         # Limpar resultados anteriores
    python main.py --benchmark              # Modo benchmark (múltiplas rodadas)
    python main.py --benchmark --visualize-benchmark  # Benchmark com visualizações avançadas
    python main.py --all --save-solutions --clear-old-data  # Execução completa
    
    # Sistemas não lineares
    python main.py --nonlinear              # Resolver sistema não linear específico
    python main.py --nonlinear --tolerance 1e-8  # Com tolerância personalizada
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil
import warnings
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings('ignore')

from linear_solver import (
    JacobiSolver, GaussSeidelSolver, ConjugateGradientSolver,
    JacobiOrder2Solver, GaussSeidelOrder2Solver,
    PreconditionedConjugateGradientSolver,
    CSVMatrixLoader, MatrixValidator
)

# Importar resolvedores não lineares (opcional - só quando necessário)
try:
    from nonlinear_solver import NewtonSolver, IterationSolver, GradientSolver
    HAS_NONLINEAR = True
except ImportError:
    HAS_NONLINEAR = False

# Importar matplotlib apenas se necessário
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# FUNÇÕES DE ANÁLISE DE CONDICIONAMENTO
# ============================================================================

def calcular_matriz_iteracao_jacobi(A: np.ndarray) -> np.ndarray:
    """Calcula a matriz de iteração do método de Jacobi."""
    n = A.shape[0]
    D = np.diag(np.diag(A))
    D_inv = np.linalg.inv(D)
    M = np.eye(n) - D_inv @ A
    return M


def calcular_matriz_iteracao_gauss_seidel(A: np.ndarray) -> np.ndarray:
    """Calcula a matriz de iteração do método de Gauss-Seidel."""
    n = A.shape[0]
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    DL = D + L
    M = np.linalg.solve(DL, U)
    return M


def calcular_matriz_iteracao_jacobi_ordem2(A: np.ndarray, omega1: float = 0.7, 
                                          omega2: float = 0.2, omega3: float = 0.1) -> np.ndarray:
    """Calcula matriz de iteração aproximada para Jacobi de ordem 2."""
    M_jacobi = calcular_matriz_iteracao_jacobi(A)
    n = A.shape[0]
    I = np.eye(n)
    M_j2 = omega1 * M_jacobi + omega2 * I + omega3 * np.linalg.matrix_power(M_jacobi, 2)
    return M_j2


def calcular_matriz_iteracao_gauss_seidel_ordem2(A: np.ndarray, omega_relax: float = 1.2,
                                                omega1: float = 0.8, omega2: float = 0.15, 
                                                omega3: float = 0.05) -> np.ndarray:
    """Calcula matriz de iteração aproximada para Gauss-Seidel de ordem 2 com SOR."""
    n = A.shape[0]
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    D_wL = D + omega_relax * L
    rhs = (1 - omega_relax) * D - omega_relax * U
    M_sor = np.linalg.solve(D_wL, rhs)
    I = np.eye(n)
    M_gs2 = omega1 * M_sor + omega2 * I + omega3 * np.linalg.matrix_power(M_sor, 2)
    return M_gs2


def calcular_normas_matriz(M: np.ndarray) -> Dict[str, float]:
    """Calcula todas as normas solicitadas para uma matriz, incluindo versões ponderadas."""
    normas = {}
    
    # Normas básicas
    try:
        normas['Euclidiana'] = np.linalg.norm(M, 2)
    except:
        normas['Euclidiana'] = np.nan
        
    normas['Soma_Max_Colunas'] = np.linalg.norm(M, 1)
    normas['Soma_Max_Linhas'] = np.linalg.norm(M, np.inf)
    normas['Frobenius'] = np.linalg.norm(M, 'fro')
    normas['Maximo'] = np.max(np.abs(M))
    
    # Raio espectral
    try:
        eigenvals = np.linalg.eigvals(M)
        normas['Raio_Espectral'] = np.max(np.abs(eigenvals))
    except:
        normas['Raio_Espectral'] = np.nan
    
    # Versões ponderadas (usando pesos baseados na posição)
    n = M.shape[0]
    
    # Pesos lineares crescentes
    pesos_linha = np.arange(1, n+1, dtype=float)
    pesos_coluna = np.arange(1, n+1, dtype=float)
    
    # Matriz ponderada por linha
    M_pond_linha = M * pesos_linha.reshape(-1, 1)
    normas['Euclidiana_Pond_Linha'] = np.linalg.norm(M_pond_linha, 2) if M_pond_linha.size > 0 else np.nan
    normas['Frobenius_Pond_Linha'] = np.linalg.norm(M_pond_linha, 'fro')
    
    # Matriz ponderada por coluna  
    M_pond_coluna = M * pesos_coluna.reshape(1, -1)
    normas['Euclidiana_Pond_Coluna'] = np.linalg.norm(M_pond_coluna, 2) if M_pond_coluna.size > 0 else np.nan
    normas['Frobenius_Pond_Coluna'] = np.linalg.norm(M_pond_coluna, 'fro')
    
    # Matriz duplamente ponderada
    Pesos = np.outer(pesos_linha, pesos_coluna)
    M_pond_dupla = M * Pesos
    normas['Euclidiana_Pond_Dupla'] = np.linalg.norm(M_pond_dupla, 2) if M_pond_dupla.size > 0 else np.nan
    normas['Frobenius_Pond_Dupla'] = np.linalg.norm(M_pond_dupla, 'fro')
    
    return normas


def calcular_numero_condicao(A: np.ndarray) -> Dict[str, float]:
    """Calcula número de condição da matriz A para diferentes normas."""
    cond = {}
    
    try:
        cond['Euclidiana'] = np.linalg.cond(A, 2)
    except:
        cond['Euclidiana'] = np.nan
        
    cond['Soma_Max_Colunas'] = np.linalg.cond(A, 1)
    cond['Soma_Max_Linhas'] = np.linalg.cond(A, np.inf)
    
    # Para Frobenius, calcular manualmente: ||A|| * ||A^(-1)||
    try:
        A_inv = np.linalg.inv(A)
        cond['Frobenius'] = np.linalg.norm(A, 'fro') * np.linalg.norm(A_inv, 'fro')
    except:
        cond['Frobenius'] = np.nan
    
    # Para norma máximo
    try:
        A_inv = np.linalg.inv(A)
        cond['Maximo'] = np.max(np.abs(A)) * np.max(np.abs(A_inv))
    except:
        cond['Maximo'] = np.nan
    
    return cond


def analisar_condicionamento_sistema(A: np.ndarray, matrix_name: str):
    """Análise completa de condicionamento da matriz A e das matrizes de iteração."""
    
    print(f"\n📊 ANÁLISE DE CONDICIONAMENTO: {matrix_name}")
    print("-" * 60)
    
    # Calcular matrizes de iteração
    print("🔧 Calculando matrizes de iteração...")
    matrizes_iteracao = {
        'M1 (Jacobi)': calcular_matriz_iteracao_jacobi(A),
        'M2 (Gauss-Seidel)': calcular_matriz_iteracao_gauss_seidel(A),
        'M3 (Jacobi Ordem 2)': calcular_matriz_iteracao_jacobi_ordem2(A),
        'M4 (Gauss-Seidel Ordem 2)': calcular_matriz_iteracao_gauss_seidel_ordem2(A)
    }
    
    # Condicionamento da matriz A
    print("📊 Calculando condicionamento da matriz A...")
    cond_A = calcular_numero_condicao(A)
    
    # Normas das matrizes de iteração (incluindo versões ponderadas)
    print("📐 Calculando normas das matrizes de iteração...")
    normas_resultados = {}
    for nome, M in matrizes_iteracao.items():
        normas_resultados[nome] = calcular_normas_matriz(M)
    
    # Mostrar resultados resumidos no console
    print("\n📈 Condicionamento da Matriz A:")
    for norma, valor in cond_A.items():
        print(f"   {norma:20}: {valor:.2e}")
    
    print("\n🔄 Raio Espectral das Matrizes de Iteração:")
    for nome, normas in normas_resultados.items():
        rho = normas['Raio_Espectral']
        status = "✅" if rho < 1 else "❌"
        print(f"   {nome:25}: ρ = {rho:.6f} {status}")
    
    # Salvar resultados detalhados
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    nome_sistema = matrix_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    
    # 1. Relatório principal comparativo
    tabela_path = results_dir / f"condicionamento_comparativo_{nome_sistema}.txt"
    
    with open(tabela_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"ANÁLISE COMPARATIVA DE CONDICIONAMENTO - {matrix_name}\n")
        f.write("=" * 100 + "\n\n")
        
        # Condicionamento da matriz A
        f.write("CONDICIONAMENTO DA MATRIZ A:\n")
        f.write("-" * 40 + "\n")
        for norma, valor in cond_A.items():
            f.write(f"{norma:25}: {valor:12.6e}\n")
        
        f.write(f"\n\nTABELA COMPARATIVA DAS MATRIZES DE ITERAÇÃO:\n")
        f.write("-" * 80 + "\n")
        
        # Cabeçalho da tabela (normas básicas)
        f.write(f"{'Método':<25} {'Raio Espect.':<12} {'Euclidiana':<12} {'Max Col':<10} {'Max Lin':<10} {'Frobenius':<12} {'Máximo':<10}\n")
        f.write("-" * 95 + "\n")
        
        # Dados da tabela (normas básicas)
        for nome, normas in normas_resultados.items():
            f.write(f"{nome:<25} {normas['Raio_Espectral']:<12.6f} {normas['Euclidiana']:<12.6f} "
                   f"{normas['Soma_Max_Colunas']:<10.6f} {normas['Soma_Max_Linhas']:<10.6f} "
                   f"{normas['Frobenius']:<12.6f} {normas['Maximo']:<10.6f}\n")
        
        f.write(f"\n\nRESUMO DE CONVERGÊNCIA:\n")
        f.write("-" * 25 + "\n")
        for nome, normas in normas_resultados.items():
            rho = normas['Raio_Espectral']
            status = "CONVERGE" if rho < 1 else "NÃO CONVERGE"
            f.write(f"{nome:25}: ρ = {rho:.6f} - {status}\n")
        
        f.write(f"\n\nINTERPRETAÇÃO:\n")
        f.write("- Raio Espectral < 1: Método garante convergência\n")
        f.write("- Menor raio espectral = convergência mais rápida\n")
        f.write("- Número de condição da matriz A indica estabilidade numérica\n")
    
    # 2. Relatório detalhado com normas ponderadas
    relatorio_detalhado_path = results_dir / f"analise_condicionamento_detalhada_{nome_sistema}.txt"
    
    with open(relatorio_detalhado_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"RELATÓRIO DETALHADO - ANÁLISE DE CONDICIONAMENTO\n")
        f.write(f"Sistema: {matrix_name}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("CONDICIONAMENTO DA MATRIZ ORIGINAL A:\n")
        f.write("-" * 40 + "\n")
        for norma, valor in cond_A.items():
            f.write(f"{norma:25}: {valor:.6e}\n")
        
        f.write(f"\n\nNORMAS DAS MATRIZES DE ITERAÇÃO (COMPLETAS):\n")
        f.write("-" * 50 + "\n\n")
        
        for nome, dados in normas_resultados.items():
            f.write(f"{nome}:\n")
            f.write("-" * len(nome) + "\n")
            for norma, valor in dados.items():
                f.write(f"  {norma:25}: {valor:.6e}\n")
            f.write("\n")
        
        f.write("INTERPRETAÇÃO DETALHADA:\n")
        f.write("-" * 25 + "\n")
        f.write("- Raio Espectral < 1: Método converge\n")
        f.write("- Número de Condição alto: Sistema mal condicionado\n") 
        f.write("- Normas das matrizes de iteração indicam velocidade de convergência\n")
        f.write("- Normas ponderadas mostram sensibilidade a diferentes regiões da matriz\n")
        f.write("\nMenores normas das matrizes de iteração = convergência mais rápida\n")
    
    # 3. Arquivo CSV-like para normas das matrizes de iteração
    normas_tabular_path = results_dir / f"normas_matrizes_iteracao_{nome_sistema}.txt"
    
    with open(normas_tabular_path, 'w', encoding='utf-8') as f:
        f.write("NORMAS DAS MATRIZES DE ITERAÇÃO - FORMATO TABULAR\n")
        f.write("=" * 80 + "\n\n")
        
        # Cabeçalho
        if normas_resultados:
            norma_names = list(next(iter(normas_resultados.values())).keys())
            f.write(f"{'Matriz':<25}")
            for norma in norma_names:
                f.write(f"{norma:<18}")
            f.write("\n" + "-" * (25 + 18 * len(norma_names)) + "\n")
            
            # Dados
            for nome, dados in normas_resultados.items():
                f.write(f"{nome:<25}")
                for norma in norma_names:
                    f.write(f"{dados[norma]:<18.6e}")
                f.write("\n")
    
    # 4. Arquivo separado só para condicionamento da matriz A
    cond_matriz_path = results_dir / f"condicionamento_matriz_A_{nome_sistema}.txt"
    
    with open(cond_matriz_path, 'w', encoding='utf-8') as f:
        f.write(f"CONDICIONAMENTO DA MATRIZ A - {matrix_name}\n")
        f.write("=" * 40 + "\n\n")
        for norma, valor in cond_A.items():
            f.write(f"{norma:<25}: {valor:.6e}\n")
        
        f.write(f"\nINTERPRETAÇÃO:\n")
        f.write("-" * 15 + "\n")
        euclidiana = cond_A.get('Euclidiana', float('inf'))
        if euclidiana < 10:
            f.write("Sistema MUITO BEM condicionado (κ < 10)\n")
        elif euclidiana < 100:
            f.write("Sistema BEM condicionado (10 ≤ κ < 100)\n")
        elif euclidiana < 1000:
            f.write("Sistema MODERADAMENTE condicionado (100 ≤ κ < 1000)\n")
        else:
            f.write("Sistema MAL condicionado (κ ≥ 1000)\n")
    
    print(f"\n💾 ANÁLISES SALVAS:")
    print(f"   📋 Relatório principal: {tabela_path}")
    print(f"   📊 Análise detalhada: {relatorio_detalhado_path}")
    print(f"   📄 Normas tabulares: {normas_tabular_path}")
    print(f"   📐 Condicionamento A: {cond_matriz_path}")
    
    return normas_resultados, cond_A


# ============================================================================
# CLASSES DE BENCHMARK
# ============================================================================

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


# ============================================================================
# FUNÇÕES DE VISUALIZAÇÃO DE BENCHMARK
# ============================================================================

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


def clear_old_results():
    """Limpa resultados anteriores das pastas de output."""
    
    print("\n🗑️  LIMPANDO RESULTADOS ANTERIORES")
    print("-" * 50)
    
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("📁 Pasta 'results/' não existe - nada para limpar")
        return
    
    # Listar arquivos/pastas que serão removidos
    items_to_remove = []
    
    # Arquivos na raiz de results/
    for item in results_dir.iterdir():
        if item.is_file():
            items_to_remove.append(("arquivo", item))
    
    # Subpastas específicas
    subdirs_to_clear = ["text_results", "charts"]
    for subdir_name in subdirs_to_clear:
        subdir = results_dir / subdir_name
        if subdir.exists() and subdir.is_dir():
            for item in subdir.iterdir():
                items_to_remove.append(("arquivo", item))
            items_to_remove.append(("pasta", subdir))
    
    if not items_to_remove:
        print("✅ Nenhum resultado anterior encontrado")
        return
    
    # Mostrar o que será removido
    print(f"📋 Itens a serem removidos ({len(items_to_remove)}):")
    
    files_count = 0
    dirs_count = 0
    
    for item_type, item_path in items_to_remove:
        # Usar o caminho como está (já é relativo ao diretório atual)
        display_path = str(item_path)
        
        if item_type == "arquivo":
            files_count += 1
            print(f"   📄 {display_path}")
        elif item_type == "pasta":
            dirs_count += 1
            print(f"   📁 {display_path}/")
    
    # Remover os itens
    removed_files = 0
    removed_dirs = 0
    errors = []
    
    for item_type, item_path in items_to_remove:
        try:
            if item_type == "arquivo" and item_path.exists():
                item_path.unlink()
                removed_files += 1
            elif item_type == "pasta" and item_path.exists():
                # Só remove se estiver vazia (arquivos já foram removidos)
                if not any(item_path.iterdir()):
                    item_path.rmdir()
                    removed_dirs += 1
        except Exception as e:
            errors.append(f"{item_path}: {str(e)}")
    
    # Relatório final
    print(f"\n✅ Limpeza concluída:")
    print(f"   📄 Arquivos removidos: {removed_files}")
    print(f"   📁 Pastas removidas: {removed_dirs}")
    
    if errors:
        print(f"   ⚠️  Erros ({len(errors)}):")
        for error in errors:
            print(f"      {error}")
    
    print("🆕 Pronto para novos resultados!")


def carregar_matriz_brasileira(filepath):
    """Carrega matriz do formato brasileiro (vírgulas decimais, tabs)."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None
        
    matriz = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for linha in file:
            linha = linha.strip()
            if not linha:
                continue
                
            valores = linha.split('\t')
            linha_numerica = []
            
            for valor in valores:
                valor = valor.strip()
                if valor:
                    valor_num = float(valor.replace(',', '.'))
                    linha_numerica.append(valor_num)
            
            if linha_numerica:
                matriz.append(linha_numerica)
    
    return np.array(matriz)


def carregar_vetor_brasileiro(filepath):
    """Carrega vetor do formato brasileiro (vírgulas decimais)."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None
        
    vetor = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for linha in file:
            linha = linha.strip()
            if not linha:
                continue
                
            valor = float(linha.replace(',', '.'))
            vetor.append(valor)
    
    return np.array(vetor)


def descobrir_sistemas_disponiveis():
    """Descobre todos os sistemas lineares disponíveis na pasta data/."""
    
    print("=" * 60)
    print("DESCOBRINDO SISTEMAS NA PASTA ./data/")
    print("=" * 60)
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Pasta 'data/' não encontrada!")
        return []
    
    sistemas = []
    
    # 1. Sistema brasileiro específico (Matriz_A.txt + Vetor_b.txt)
    matriz_a_path = data_dir / "Matriz_A.txt"
    vetor_b_path = data_dir / "Vetor_b.txt"
    
    if matriz_a_path.exists() and vetor_b_path.exists():
        print("✅ Sistema brasileiro encontrado: Matriz_A.txt + Vetor_b.txt")
        sistemas.append(("Sistema Brasileiro 36x36", "brasileiro", matriz_a_path, vetor_b_path))
    
    # 2. Arquivos CSV de matriz aumentada
    for csv_file in data_dir.glob("*.csv"):
        if "36x36" in csv_file.name and not csv_file.name.startswith("matriz_A_") and not csv_file.name.startswith("vetor_b_"):
            nome = csv_file.stem.replace("_36x36", "").replace("_", " ").title()
            print(f"✅ Sistema CSV encontrado: {csv_file.name}")
            sistemas.append((nome, "csv_aumentado", csv_file, None))
    
    # 3. Pares de arquivos CSV separados (matriz_A_* + vetor_b_*)
    matriz_files = list(data_dir.glob("matriz_A_*.csv"))
    for matriz_file in matriz_files:
        nome_base = matriz_file.name.replace("matriz_A_", "").replace(".csv", "")
        vetor_file = data_dir / f"vetor_b_{nome_base}.csv"
        
        if vetor_file.exists():
            nome = nome_base.replace("_36x36", "").replace("_", " ").title()
            print(f"✅ Sistema CSV separado encontrado: {matriz_file.name} + {vetor_file.name}")
            sistemas.append((nome + " (Separado)", "csv_separado", matriz_file, vetor_file))
    
    print(f"\n📊 Total de sistemas encontrados: {len(sistemas)}")
    return sistemas


def parse_arguments():
    """Parse argumentos da linha de comando."""
    
    parser = argparse.ArgumentParser(
        description="Resolver sistemas lineares usando métodos iterativos",
        epilog="""
Exemplos:
  %(prog)s --all                      # Todos os métodos disponíveis
  %(prog)s --jacobi                   # Apenas método de Jacobi
  %(prog)s --jacobi --gauss-seidel    # Jacobi e Gauss-Seidel
  %(prog)s --conjugate-gradient       # Apenas Gradiente Conjugado
  %(prog)s --jacobi-order2            # Jacobi de segunda ordem
  %(prog)s --no-plots                 # Executar sem gráficos
  %(prog)s --clear-old-data           # Limpar resultados anteriores
  %(prog)s --all --save-solutions --clear-old-data # Execução completa limpa
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Métodos disponíveis
    parser.add_argument('--all', action='store_true',
                       help='Executar todos os métodos disponíveis')
    
    parser.add_argument('--jacobi', action='store_true',
                       help='Executar método de Jacobi')
    
    parser.add_argument('--gauss-seidel', action='store_true',
                       help='Executar método de Gauss-Seidel')
    
    parser.add_argument('--conjugate-gradient', action='store_true',
                       help='Executar Gradiente Conjugado (apenas para matrizes simétricas positivas definidas)')
    
    parser.add_argument('--jacobi-order2', action='store_true',
                       help='Executar Jacobi de segunda ordem')
    
    parser.add_argument('--gauss-seidel-order2', action='store_true',
                       help='Executar Gauss-Seidel de segunda ordem')
    
    parser.add_argument('--preconditioned-cg', action='store_true',
                       help='Executar Gradiente Conjugado Precondicionado')
    
    # Opções de execução
    parser.add_argument('--no-plots', action='store_true',
                       help='Não gerar gráficos (útil se matplotlib não estiver disponível)')
    
    parser.add_argument('--tolerance', type=float, default=1e-4,
                       help='Tolerância para convergência (padrão: 1e-4 = 10^(-4))')
    
    parser.add_argument('--max-iterations', type=int, default=5000,
                       help='Número máximo de iterações (padrão: 5000)')
    
    parser.add_argument('--save-solutions', action='store_true',
                       help='Salvar soluções em arquivos')
    
    parser.add_argument('--skip-conditioning', action='store_true',
                       help='Pular análise de condicionamento (mais rápido)')
    
    parser.add_argument('--clear-old-data', action='store_true',
                       help='Limpar resultados anteriores antes de executar')
    
    parser.add_argument('--benchmark', action='store_true',
                       help='Executar benchmark de tempo dos métodos (múltiplas rodadas para estatísticas)')
    
    parser.add_argument('--visualize-benchmark', action='store_true',
                       help='Gerar visualizações avançadas do benchmark (requer --benchmark)')
    
    # Sistemas não lineares
    parser.add_argument('--nonlinear', action='store_true',
                       help='Resolver sistema não linear específico usando Newton, Iteração e Gradiente')
    
    args = parser.parse_args()
    
    # Validar dependências de argumentos
    if args.visualize_benchmark and not args.benchmark:
        parser.error("--visualize-benchmark requer --benchmark")
    
    # Se nenhum método foi especificado E não é sistema não linear, usar --all
    if not args.nonlinear and not any([args.all, args.jacobi, args.gauss_seidel, args.conjugate_gradient,
                                      args.jacobi_order2, args.gauss_seidel_order2, args.preconditioned_cg]):
        print("⚠️  Nenhum método especificado. Usando --all por padrão.")
        args.all = True
    
    # Validar sistemas não lineares
    if args.nonlinear and not HAS_NONLINEAR:
        parser.error("Módulo nonlinear_solver não encontrado. Verifique se foi instalado corretamente.")
    
    return args


def analyze_matrix_properties(A, matrix_name):
    """Analisa e exibe propriedades da matriz."""
    
    print(f"\n📊 ANÁLISE DA MATRIZ: {matrix_name}")
    print("-" * 50)
    
    analysis = MatrixValidator.analyze_matrix(A)
    
    print(f"Dimensões: {analysis['shape']}")
    print(f"Determinante: {analysis['determinant']:.6e}")
    print(f"Número de condição: {analysis['condition_number']:.2e}")
    print(f"Posto: {analysis['rank']}")
    print(f"Raio espectral: {analysis['spectral_radius']:.6f}")
    
    print("\nPropriedades:")
    print(f"  ✅ Simétrica: {'Sim' if analysis['is_symmetric'] else 'Não'}")
    print(f"  ✅ Positiva definida: {'Sim' if analysis['is_positive_definite'] else 'Não'}")  
    print(f"  ✅ Diagonalmente dominante: {'Sim' if analysis['is_diagonally_dominant'] else 'Não'}")
    print(f"  ✅ Estritamente diag. dominante: {'Sim' if analysis['is_strictly_diagonally_dominant'] else 'Não'}")
    
    return analysis


def build_methods_list(args, analysis):
    """Constrói a lista de métodos baseada nos argumentos fornecidos."""
    
    methods = []
    
    # Métodos básicos
    if args.all or args.jacobi:
        methods.append(("Jacobi", JacobiSolver(
            tolerance=args.tolerance, 
            max_iterations=args.max_iterations
        )))
    
    if args.all or args.gauss_seidel:
        methods.append(("Gauss-Seidel", GaussSeidelSolver(
            tolerance=args.tolerance, 
            max_iterations=args.max_iterations
        )))
    
    # Métodos de segunda ordem
    if args.all or args.jacobi_order2:
        methods.append(("Jacobi Ordem 2", JacobiOrder2Solver(
            tolerance=args.tolerance, 
            max_iterations=args.max_iterations,
            omega1=0.7, omega2=0.2, omega3=0.1
        )))
    
    if args.all or args.gauss_seidel_order2:
        methods.append(("Gauss-Seidel Ordem 2", GaussSeidelOrder2Solver(
            tolerance=args.tolerance, 
            max_iterations=args.max_iterations,
            relaxation_factor=1.2, omega1=0.8, omega2=0.15, omega3=0.05
        )))
    
    # Métodos para matrizes simétricas positivas definidas
    if analysis['is_symmetric'] and analysis['is_positive_definite']:
        if args.all or args.conjugate_gradient:
            methods.append(("Gradiente Conjugado", ConjugateGradientSolver(
                tolerance=args.tolerance
            )))
        
        if args.all or args.preconditioned_cg:
            methods.append(("Gradiente Conjugado Precondicionado", 
                           PreconditionedConjugateGradientSolver(
                               tolerance=args.tolerance
                           )))
    else:
        # Avisar se métodos específicos para SPD foram solicitados mas a matriz não é SPD
        if args.conjugate_gradient or args.preconditioned_cg:
            print("⚠️  Métodos de Gradiente Conjugado ignorados: matriz não é simétrica positiva definida")
    
    return methods


def solve_with_selected_methods(A, b, matrix_name, args):
    """Resolve o sistema com os métodos selecionados."""
    
    print(f"\n🔧 RESOLVENDO SISTEMA: {matrix_name}")
    print("-" * 50)
    
    # Analisar matriz para determinar métodos aplicáveis
    analysis = MatrixValidator.analyze_matrix(A)
    
    # Construir lista de métodos baseada nos argumentos
    methods = build_methods_list(args, analysis)
    
    if not methods:
        print("❌ Nenhum método aplicável selecionado!")
        return {}, {}
    
    print(f"📋 Métodos selecionados: {', '.join([name for name, _ in methods])}")
    
    results = {}
    solutions = {}
    
    for method_name, solver in methods:
        try:
            print(f"\n⚡ {method_name}:")
            
            x, info = solver.solve(A, b)
            results[method_name] = info
            solutions[method_name] = x
            
            if info['converged']:
                print(f"   ✅ Convergiu em {info['iterations']} iterações")
                print(f"   📊 Erro final: {info['final_error']:.2e}")
                
                # Verificar qualidade da solução
                residual = np.linalg.norm(A @ x - b)
                print(f"   🎯 Resíduo: {residual:.2e}")
                
                # Mostrar algumas componentes da solução
                print(f"   🔢 x[0:5]: {x[:5] if len(x) >= 5 else x}")
                if len(x) > 5:
                    print(f"   🔢 x[-5:]: {x[-5:]}")
            else:
                print(f"   ❌ Não convergiu após {info['iterations']} iterações")
                print(f"   📊 Erro final: {info['final_error']:.2e}")
                
        except Exception as e:
            print(f"   💥 Erro: {str(e)}")
            results[method_name] = None
    
    return results, solutions


def save_solutions(solutions, A, b, matrix_name, args):
    """Salva as soluções em arquivos se solicitado."""
    
    if not args.save_solutions:
        return
    
    if not solutions:
        print("❌ Nenhuma solução para salvar")
        return
    
    # Criar estrutura de diretórios
    results_dir = Path("results")
    text_results_dir = results_dir / "text_results"
    text_results_dir.mkdir(parents=True, exist_ok=True)
    
    nome_arquivo_base = matrix_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    
    # Calcular solução de referência
    try:
        x_ref = np.linalg.solve(A, b)
        
        # Salvar solução de referência
        arquivo_ref = text_results_dir / f"solucao_{nome_arquivo_base}_referencia.txt"
        with open(arquivo_ref, 'w', encoding='utf-8') as f:
            f.write(f"# Solução de referência: {matrix_name}\n")
            f.write(f"# Método: NumPy (direto)\n")
            f.write(f"# Data: {Path().absolute()}\n")
            f.write("#" + "="*50 + "\n")
            
            for i, valor in enumerate(x_ref):
                f.write(f"{i+1:3d}  {valor:15.10f}\n")
        
        print(f"✅ Solução de referência salva: {arquivo_ref}")
        
        # Salvar soluções dos métodos iterativos
        for method_name, x in solutions.items():
            if x is not None:
                method_filename = method_name.lower().replace(" ", "_").replace("-", "_")
                arquivo_metodo = text_results_dir / f"solucao_{nome_arquivo_base}_{method_filename}.txt"
                
                error = np.linalg.norm(x - x_ref)
                residual = np.linalg.norm(A @ x - b)
                
                with open(arquivo_metodo, 'w', encoding='utf-8') as f:
                    f.write(f"# Solução: {matrix_name}\n")
                    f.write(f"# Método: {method_name}\n")
                    f.write(f"# Erro vs referência: {error:.2e}\n")
                    f.write(f"# Resíduo: {residual:.2e}\n")
                    f.write(f"# Tolerância usada: {args.tolerance}\n")
                    f.write(f"# Máx. iterações: {args.max_iterations}\n")
                    f.write("#" + "="*50 + "\n")
                    
                    for i, valor in enumerate(x):
                        f.write(f"{i+1:3d}  {valor:15.10f}\n")
                
                print(f"✅ Solução {method_name} salva: {arquivo_metodo.name}")
                
    except np.linalg.LinAlgError as e:
        print(f"❌ Não foi possível salvar soluções: {e}")
    
    print(f"📁 Arquivos salvos em: {text_results_dir.absolute()}")


def plot_convergence_comparison(results, matrix_name):
    """Plota comparação de convergência dos métodos."""
    
    if not HAS_MATPLOTLIB:
        print("⚠️  Matplotlib não disponível - pulando gráficos")
        return
    
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: Histórico de convergência  
    plt.subplot(2, 2, 1)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for i, (method_name, info) in enumerate(results.items()):
        if info and info['converged'] and 'convergence_history' in info:
            history = info['convergence_history']
            if history:
                plt.semilogy(
                    range(1, len(history) + 1), history, 
                    label=method_name, 
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    markersize=4, linewidth=2
                )
    
    plt.xlabel('Iteração')
    plt.ylabel('Erro (escala log)')
    plt.title(f'Convergência - {matrix_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Número de iterações
    plt.subplot(2, 2, 2)
    
    method_names = []
    iterations = []
    colors_bar = []
    
    for i, (method_name, info) in enumerate(results.items()):
        if info and info['converged']:
            method_names.append(method_name)
            iterations.append(info['iterations'])
            colors_bar.append(colors[i % len(colors)])
    
    if method_names:
        bars = plt.bar(method_names, iterations, color=colors_bar, alpha=0.7)
        plt.ylabel('Número de Iterações')
        plt.title('Iterações para Convergência')
        plt.xticks(rotation=45, ha='right')
        
        # Adicionar valores nas barras
        for bar, iter_count in zip(bars, iterations):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(iter_count), ha='center', va='bottom')
    
    # Subplot 3: Erro final
    plt.subplot(2, 2, 3)
    
    method_names_error = []
    final_errors = []
    colors_error = []
    
    for i, (method_name, info) in enumerate(results.items()):
        if info and info['converged']:
            method_names_error.append(method_name)
            final_errors.append(info['final_error'])
            colors_error.append(colors[i % len(colors)])
    
    if method_names_error:
        bars = plt.bar(method_names_error, final_errors, color=colors_error, alpha=0.7)
        plt.ylabel('Erro Final')
        plt.yscale('log')
        plt.title('Erro Final por Método')
        plt.xticks(rotation=45, ha='right')
    
    # Subplot 4: Comparação de resíduos (se disponível)
    plt.subplot(2, 2, 4)
    
    if any('residual_history' in info for info in results.values() if info):
        for i, (method_name, info) in enumerate(results.items()):
            if info and 'residual_history' in info and info['residual_history']:
                plt.semilogy(
                    range(1, len(info['residual_history']) + 1), 
                    info['residual_history'],
                    label=method_name,
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    markersize=4, linewidth=2
                )
        plt.xlabel('Iteração')
        plt.ylabel('Resíduo (escala log)')
        plt.title('História do Resíduo')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Dados de resíduo\nnão disponíveis', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    
    # Criar estrutura de diretórios para gráficos
    results_dir = Path("results")
    charts_dir = results_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvar gráfico na pasta apropriada
    chart_filename = f'convergencia_{matrix_name.replace(" ", "_").lower()}.png'
    chart_path = charts_dir / chart_filename
    
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo: {chart_path}")
    plt.show()


def compare_solutions(solutions, A, b):
    """Compara as soluções obtidas pelos diferentes métodos."""
    
    print("\n🔍 COMPARAÇÃO DAS SOLUÇÕES")
    print("-" * 50)
    
    if not solutions:
        print("Nenhuma solução disponível para comparação.")
        return
    
    # Calcular solução de referência (numpy)
    try:
        x_ref = np.linalg.solve(A, b)
        print(f"Solução de referência (NumPy): {x_ref}")
        
        print("\nComparação com solução de referência:")
        for method_name, x in solutions.items():
            if x is not None:
                error = np.linalg.norm(x - x_ref, ord=np.inf)
                print(f"  {method_name:20}: erro = {error:.2e}")
                
    except np.linalg.LinAlgError:
        print("Não foi possível calcular solução de referência (matriz singular)")
        
        # Comparar soluções entre si
        method_names = list(solutions.keys())
        if len(method_names) > 1:
            print("\nComparação entre métodos:")
            ref_method = method_names[0]
            x_ref = solutions[ref_method]
            
            for method_name in method_names[1:]:
                x = solutions[method_name]
                if x is not None and x_ref is not None:
                    error = np.linalg.norm(x - x_ref, ord=np.inf)
                    print(f"  {method_name} vs {ref_method}: erro = {error:.2e}")


def carregar_sistema(nome, tipo, arquivo1, arquivo2=None):
    """Carrega um sistema linear baseado no tipo especificado."""
    
    try:
        if tipo == "brasileiro":
            A = carregar_matriz_brasileira(arquivo1)
            b = carregar_vetor_brasileiro(arquivo2)
            if A is None or b is None:
                raise ValueError("Falha ao carregar arquivos brasileiros")
            
        elif tipo == "csv_aumentado":
            A, b = CSVMatrixLoader.load_augmented_matrix(arquivo1, skip_header=True)
            
        elif tipo == "csv_separado":
            A, b = CSVMatrixLoader.load_separate_files(arquivo1, arquivo2)
            
        else:
            raise ValueError(f"Tipo de sistema desconhecido: {tipo}")
        
        # Verificar dimensões
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matriz A não é quadrada: {A.shape}")
        
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Dimensões incompatíveis: A {A.shape}, b {b.shape}")
        
        return A, b
        
    except Exception as e:
        print(f"❌ Erro ao carregar {nome}: {str(e)}")
        return None, None


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


def create_summary_report(sistemas_processados, args):
    """Cria um relatório resumo da execução."""
    
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = results_dir / "summary_report.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("LINEAR SOLVER - RELATÓRIO DE EXECUÇÃO\n")
        f.write("=" * 60 + "\n\n")
        
        # Configurações usadas
        f.write("CONFIGURAÇÕES DA EXECUÇÃO:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Tolerância: {args.tolerance}\n")
        f.write(f"Máximo de iterações: {args.max_iterations}\n")
        f.write(f"Gráficos habilitados: {'Não' if args.no_plots else 'Sim' if HAS_MATPLOTLIB else 'Matplotlib indisponível'}\n")
        f.write(f"Salvar soluções: {'Sim' if args.save_solutions else 'Não'}\n")
        f.write(f"Análise de condicionamento: {'Desabilitada' if args.skip_conditioning else 'Habilitada'}\n")
        f.write(f"Dados anteriores limpos: {'Sim' if args.clear_old_data else 'Não'}\n")
        
        # Métodos selecionados
        selected_methods = []
        if args.all:
            selected_methods.append("TODOS")
        else:
            if args.jacobi: selected_methods.append("Jacobi")
            if args.gauss_seidel: selected_methods.append("Gauss-Seidel")
            if args.conjugate_gradient: selected_methods.append("Gradiente Conjugado")
            if args.jacobi_order2: selected_methods.append("Jacobi Ordem 2")
            if args.gauss_seidel_order2: selected_methods.append("Gauss-Seidel Ordem 2")
            if args.preconditioned_cg: selected_methods.append("Gradiente Conjugado Precondicionado")
        
        f.write(f"Métodos selecionados: {', '.join(selected_methods)}\n\n")
        
        # Sistemas processados
        f.write("SISTEMAS PROCESSADOS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total de sistemas encontrados: {len(sistemas_processados)}\n\n")
        
        for i, sistema in enumerate(sistemas_processados, 1):
            f.write(f"{i:2d}. {sistema}\n")
        
        f.write(f"\nRelatório gerado em: {Path().absolute()}\n")
    
    print(f"📋 Relatório de execução salvo: {summary_file}")


def solve_nonlinear_system(tolerance: float = 1e-6, max_iterations: int = 1000):
    """
    Resolve o sistema não linear específico.
    
    Sistema:
        F₁: (x-1)² + (y-1)² + (z-1)² - 1 = 0
        F₂: 2x² + (y-1)² - 4z = 0  
        F₃: 3x² + 2z² - 4y = 0
    """
    print("\n🔬 RESOLVEDOR DE SISTEMAS NÃO LINEARES")
    print("=" * 60)
    print("📝 Sistema de equações:")
    print("   F₁: (x-1)² + (y-1)² + (z-1)² = 1")
    print("   F₂: 2x² + (y-1)² = 4z")
    print("   F₃: 3x² + 2z² = 4y")
    print("=" * 60)
    
    # Importar e executar exemplo não linear
    try:
        from nonlinear_example import NonLinearSystemExample
        
        example = NonLinearSystemExample()
        
        # Executar com a tolerância especificada
        results = example.run_all_methods(
            tolerance=tolerance, 
            max_iterations=max_iterations
        )
        
        print(f"\n✅ Sistema não linear processado com sucesso!")
        print(f"📁 Resultados salvos em: ./results/nonlinear/")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erro ao resolver sistema não linear: {e}")
        return False


def main():
    """Função principal da aplicação."""
    
    # Parse argumentos da linha de comando
    args = parse_arguments()
    
    title = "BENCHMARK DE TEMPO DOS MÉTODOS" if args.benchmark else "ANÁLISE DE SISTEMAS REAIS"
    print(f"🔬 LINEAR SOLVER - {title}")
    print("=" * 60)
    print(f"⚙️  Modo: {'BENCHMARK' if args.benchmark else 'ANÁLISE NORMAL'}")
    print(f"⚙️  Tolerância: {args.tolerance}")
    print(f"⚙️  Máx. iterações: {args.max_iterations}")
    if not args.benchmark:
        print(f"⚙️  Gráficos: {'Desabilitados' if args.no_plots else 'Habilitados' if HAS_MATPLOTLIB else 'Indisponíveis'}")
        print(f"⚙️  Salvar soluções: {'Sim' if args.save_solutions else 'Não'}")
        print(f"⚙️  Análise condicionamento: {'Desabilitada' if args.skip_conditioning else 'Habilitada'}")
    else:
        print(f"⚙️  Visualizações benchmark: {'Habilitadas' if args.visualize_benchmark and HAS_MATPLOTLIB else 'Desabilitadas' if not args.visualize_benchmark else 'Matplotlib indisponível'}")
    print(f"⚙️  Limpar dados anteriores: {'Sim' if args.clear_old_data else 'Não'}")
    
    # Limpar dados anteriores se solicitado
    if args.clear_old_data:
        clear_old_results()
    
    # Se sistema não linear foi solicitado, resolver e sair
    if args.nonlinear:
        success = solve_nonlinear_system(
            tolerance=args.tolerance,
            max_iterations=args.max_iterations
        )
        if success:
            print("\n🎉 Processamento concluído com sucesso!")
        else:
            print("\n❌ Falha no processamento.")
        return
    
    # Mostrar métodos selecionados (apenas no modo normal)
    if not args.benchmark:
        selected_methods = []
        if args.all:
            selected_methods.append("TODOS")
        else:
            if args.jacobi: selected_methods.append("Jacobi")
            if args.gauss_seidel: selected_methods.append("Gauss-Seidel")
            if args.conjugate_gradient: selected_methods.append("Gradiente Conjugado")
            if args.jacobi_order2: selected_methods.append("Jacobi Ordem 2")
            if args.gauss_seidel_order2: selected_methods.append("Gauss-Seidel Ordem 2")
            if args.preconditioned_cg: selected_methods.append("Gradiente Conjugado Precondicionado")
        
        print(f"📋 Métodos: {', '.join(selected_methods)}")
    else:
        print(f"📋 Modo benchmark: Todos os métodos aplicáveis serão testados")
    
    # Descobrir sistemas disponíveis na pasta data/
    sistemas = descobrir_sistemas_disponiveis()
    
    if not sistemas:
        print("❌ Nenhum sistema encontrado na pasta data/")
        print("💡 Certifique-se de que há arquivos de dados na pasta ./data/")
        return
    
    # Lista para rastrear sistemas processados com sucesso
    sistemas_processados = []
    
    # Processar cada sistema encontrado
    for i, (nome, tipo, arquivo1, arquivo2) in enumerate(sistemas, 1):
        print(f"\n{'='*60}")
        print(f"PROCESSANDO {i}/{len(sistemas)}: {nome}")
        print(f"{'='*60}")
        
        # Carregar sistema
        A, b = carregar_sistema(nome, tipo, arquivo1, arquivo2)
        if A is None or b is None:
            continue
        
        print(f"✅ Sistema carregado: {A.shape[0]}x{A.shape[1]} + vetor b({b.shape[0]})")
        
        try:
            # Verificar se está no modo benchmark
            if args.benchmark:
                # Executar modo benchmark
                run_benchmark_mode(A, b, nome, args)
                sistemas_processados.append(nome)
            else:
                # Execução normal
                # Analisar propriedades da matriz
                analyze_matrix_properties(A, nome)
                
                # *** ANÁLISE DE CONDICIONAMENTO INTEGRADA ***
                if not args.skip_conditioning:
                    analisar_condicionamento_sistema(A, nome)
                else:
                    print("⚠️  Análise de condicionamento pulada (--skip-conditioning)")
                
                # Resolver com métodos selecionados
                results, solutions = solve_with_selected_methods(A, b, nome, args)
                
                # Comparar soluções obtidas
                compare_solutions(solutions, A, b)
                
                # Salvar soluções se solicitado
                save_solutions(solutions, A, b, nome, args)
                
                # Plotar convergência (apenas para métodos que convergiram)
                if not args.no_plots:
                    converged_results = {name: info for name, info in results.items() 
                                       if info and info['converged']}
                    
                    if converged_results:
                        plot_convergence_comparison(converged_results, nome)
                    else:
                        print("❌ Nenhum método convergiu para plotar")
                
                # Adicionar sistema à lista de processados com sucesso
                sistemas_processados.append(nome)
            
        except Exception as e:
            print(f"💥 Erro ao processar {nome}: {str(e)}")
        
        # Separador entre sistemas
        if i < len(sistemas):
            print("\n" + "~" * 60)
    
    print("\n🎉 ANÁLISE CONCLUÍDA!")
    
    # Criar relatório de execução
    if sistemas_processados:
        create_summary_report(sistemas_processados, args)
    
    # Mostrar estrutura de resultados criada
    results_dir = Path("results")
    if results_dir.exists():
        print(f"📁 Resultados organizados em: {results_dir.absolute()}")
        
        charts_dir = results_dir / "charts"
        if charts_dir.exists() and any(charts_dir.iterdir()):
            chart_count = len(list(charts_dir.glob("*.png")))
            print(f"   📊 Gráficos ({chart_count}): {charts_dir}")
        
        text_results_dir = results_dir / "text_results"
        if text_results_dir.exists() and any(text_results_dir.iterdir()):
            solution_count = len(list(text_results_dir.glob("*.txt")))
            print(f"   💾 Soluções ({solution_count}): {text_results_dir}")
    
    print(f"✅ {len(sistemas_processados)}/{len(sistemas)} sistemas processados com sucesso")


if __name__ == "__main__":
    main()
