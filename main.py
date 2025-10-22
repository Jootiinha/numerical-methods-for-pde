#!/usr/bin/env python3
"""
Exemplo principal de uso da biblioteca linear_solver.

Este script demonstra como usar os diferentes métodos implementados
para resolver sistemas lineares a partir de arquivos CSV.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from linear_solver import (
    JacobiSolver, GaussSeidelSolver, ConjugateGradientSolver,
    JacobiOrder2Solver, GaussSeidelOrder2Solver,
    PreconditionedConjugateGradientSolver,
    CSVMatrixLoader, MatrixValidator, MatrixGenerator
)


def create_and_analyze_example():
    """Cria exemplos de matrizes e analisa suas propriedades."""
    
    print("=" * 60)
    print("CRIANDO ARQUIVOS DE EXEMPLO")
    print("=" * 60)
    
    # Criar diretório de exemplos
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Gerar diferentes tipos de matrizes
    print("\n1. Gerando matrizes de teste...")
    
    # Matriz diagonalmente dominante
    A_dd, b_dd = MatrixGenerator.diagonally_dominant_matrix(4, 3.0, random_seed=42)
    CSVMatrixLoader.save_augmented_matrix(
        A_dd, b_dd, examples_dir / "diagonal_dominante.csv", add_header=True
    )
    
    # Matriz simétrica positiva definida  
    A_spd, b_spd = MatrixGenerator.symmetric_positive_definite_matrix(4, 50.0, random_seed=42)
    CSVMatrixLoader.save_augmented_matrix(
        A_spd, b_spd, examples_dir / "simetrica_pd.csv", add_header=True
    )
    
    # Matriz tridiagonal
    A_tri, b_tri = MatrixGenerator.tridiagonal_matrix(5, 4.0, -1.0)
    CSVMatrixLoader.save_augmented_matrix(
        A_tri, b_tri, examples_dir / "tridiagonal.csv", add_header=True
    )
    
    print(f"✅ Arquivos salvos em {examples_dir}/")
    
    # Criar também os exemplos padrão da biblioteca
    created_files = CSVMatrixLoader.create_example_files(examples_dir)
    print(f"✅ {len(created_files)} arquivos de exemplo criados")
    
    return examples_dir


def analyze_matrix_properties(A, matrix_name):
    """Analisa e exibe propriedades da matriz."""
    
    print(f"\n📊 ANÁLISE DA MATRIZ: {matrix_name}")
    print("-" * 50)
    
    analysis = MatrixValidator.analyze_matrix(A)
    
    print(f"Dimensões: {analysis['shape']}")
    print(f"Determinante: {analysis['determinant']:.6f}")
    print(f"Número de condição: {analysis['condition_number']:.2e}")
    print(f"Posto: {analysis['rank']}")
    print(f"Raio espectral: {analysis['spectral_radius']:.6f}")
    
    print("\nPropriedades:")
    print(f"  ✅ Simétrica: {'Sim' if analysis['is_symmetric'] else 'Não'}")
    print(f"  ✅ Positiva definida: {'Sim' if analysis['is_positive_definite'] else 'Não'}")  
    print(f"  ✅ Diagonalmente dominante: {'Sim' if analysis['is_diagonally_dominant'] else 'Não'}")
    print(f"  ✅ Estritamente diag. dominante: {'Sim' if analysis['is_strictly_diagonally_dominant'] else 'Não'}")
    
    return analysis


def solve_with_all_methods(A, b, matrix_name):
    """Resolve o sistema com todos os métodos disponíveis."""
    
    print(f"\n🔧 RESOLVENDO SISTEMA: {matrix_name}")
    print("-" * 50)
    
    # Definir métodos a testar
    methods = [
        ("Jacobi", JacobiSolver(tolerance=1e-8, max_iterations=1000)),
        ("Gauss-Seidel", GaussSeidelSolver(tolerance=1e-8, max_iterations=1000)),
        ("Jacobi Ordem 2", JacobiOrder2Solver(
            tolerance=1e-8, max_iterations=1000,
            omega1=0.7, omega2=0.2, omega3=0.1
        )),
        ("Gauss-Seidel Ordem 2", GaussSeidelOrder2Solver(
            tolerance=1e-8, max_iterations=1000,
            relaxation_factor=1.2, omega1=0.8, omega2=0.15, omega3=0.05
        )),
    ]
    
    # Adicionar Gradiente Conjugado se a matriz for apropriada
    analysis = MatrixValidator.analyze_matrix(A)
    if analysis['is_symmetric'] and analysis['is_positive_definite']:
        methods.append(("Gradiente Conjugado", ConjugateGradientSolver(tolerance=1e-10)))
    
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
            else:
                print(f"   ❌ Não convergiu após {info['iterations']} iterações")
                print(f"   📊 Erro final: {info['final_error']:.2e}")
                
        except Exception as e:
            print(f"   💥 Erro: {str(e)}")
            results[method_name] = None
    
    return results, solutions


def plot_convergence_comparison(results, matrix_name):
    """Plota comparação de convergência dos métodos."""
    
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
    plt.savefig(f'convergencia_{matrix_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
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


def main():
    """Função principal do exemplo."""
    
    print("🔬 LINEAR SOLVER - DEMONSTRAÇÃO COMPLETA")
    print("=" * 60)
    
    # 1. Criar exemplos se não existirem
    examples_dir = create_and_analyze_example()
    
    # 2. Lista de arquivos para testar
    test_files = [
        ("diagonal_dominante.csv", "Matriz Diagonalmente Dominante"),
        ("simetrica_pd.csv", "Matriz Simétrica Positiva Definida"),
        ("tridiagonal.csv", "Matriz Tridiagonal"),
        ("exemplo_3x3.csv", "Exemplo 3x3 Padrão")
    ]
    
    # 3. Processar cada arquivo
    for filename, matrix_name in test_files:
        filepath = examples_dir / filename
        
        if not filepath.exists():
            print(f"\n⚠️  Arquivo {filepath} não encontrado, pulando...")
            continue
        
        print(f"\n{'='*60}")
        print(f"PROCESSANDO: {filename}")
        print(f"{'='*60}")
        
        try:
            # Carregar sistema
            A, b = CSVMatrixLoader.load_augmented_matrix(filepath)
            print(f"✅ Sistema carregado: {A.shape[0]}x{A.shape[1]} + vetor b({b.shape[0]})")
            
            # Analisar propriedades
            analysis = analyze_matrix_properties(A, matrix_name)
            
            # Resolver com todos os métodos  
            results, solutions = solve_with_all_methods(A, b, matrix_name)
            
            # Comparar soluções
            compare_solutions(solutions, A, b)
            
            # Plotar convergência (apenas para métodos que convergiram)
            converged_results = {name: info for name, info in results.items() 
                               if info and info['converged']}
            
            if converged_results:
                plot_convergence_comparison(converged_results, matrix_name)
            else:
                print("❌ Nenhum método convergiu para plotar")
            
        except Exception as e:
            print(f"💥 Erro ao processar {filename}: {str(e)}")
        
        # Pausa entre arquivos
        input("\n⏸️  Pressione Enter para continuar para o próximo exemplo...")
    
    print(f"\n🎉 DEMONSTRAÇÃO CONCLUÍDA!")
    print("📁 Gráficos salvos como arquivos PNG no diretório atual")
    print("📋 Arquivos de exemplo salvos no diretório 'examples/'")


if __name__ == "__main__":
    main()
